from re import T
from tkinter import N
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from livestockEnvV3 import load_datas

def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    action_mask = None,
    save_output = False# 新增 action_mask 参数
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

   .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :param action_mask: The action mask to be considered during evaluation.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    import pandas as pd
    
    log = []
    Move_in, Move_out, sensibility, manure = load_datas(model.country) # 需要修改
    num_move_out_counties, num_move_in_counties = Move_out.shape[0], Move_in.shape[0]
    amounts_copy = model.amounts_origin.clone()
    NH3_meta = model.NH3_meta_origin.clone()
    Carrying_meta = model.Carrying_meta_origin.clone()
    if action_mask is None:
        action_mask = torch.zeros([model.num_move_out_counties, model.num_move_in_counties, model.action_len], dtype=bool).to(model.device)

    def update_action_mask(move_out_idx, move_in_idx, breed, violation):
        '''
        action mask update function
        '''
        if move_out_idx is None: # action mask initialization
            for in_idx in range(model.num_move_in_counties):
                for breed in range(model.action_len):
                    action_mask[:, in_idx, breed] = True if Move_in.loc[in_idx, '氨排放差距'] < Move_in.iloc[in_idx, model.k+breed+1] else False     
            for out_idx in range(model.num_move_out_counties):
                for breed in range(model.action_len):
                    action_mask[out_idx, :, breed] = True if Move_out.iloc[out_idx, model.k+breed] <= 0 else False
        else:
            action_mask[move_out_idx, :, breed] = torch.logical_or(action_mask[move_out_idx, :, breed].data,detect_violation_move_out(move_out_idx, breed))
            action_mask[:, move_in_idx] = torch.logical_or(action_mask[:, move_in_idx].data, detect_violation_move_in(move_in_idx))
            if violation:
                action_mask[move_out_idx, move_in_idx, breed] = True
                            
    def detect_violation_move_out(move_out_idx, breed):
        '''
        if move out county[x] left amount = 0 or NH3 admissible minus amount < 0, return True
        '''
        # amount = amounts_copy[move_out_idx, breed]
        empty_violation = np.floor(Move_out.iloc[move_out_idx, model.k + breed]) == 0
        return torch.ones(num_move_in_counties).to(action_mask.device) if empty_violation else torch.zeros(num_move_in_counties).to(action_mask.device)
        
    def detect_violation_move_in(move_in_idx):

        NH3_violation = Move_in.iloc[move_in_idx]['氨排放差距'] <  model.Move_in_tensor_NH3[move_in_idx]
        return NH3_violation   
    
    def detect_violation(move_out_idx, move_in_idx, breed, amounts):
        # amounts = amounts_copy[move_out_idx, breed]
        Amounts_violation = Move_out.iloc[move_out_idx, model.k + breed] < amounts
        empty_violation = amounts == 0 or Move_out.iloc[move_out_idx, model.k + breed] == 0

        Move_in_values = Move_in.iloc[move_in_idx]
        NH3_violation = Move_in_values['氨排放差距'] < model.Move_in_tensor_NH3[move_in_idx, breed] * amounts
        # Carrying_violation = Move_in_values['承载力差距'] < model.Move_in_tensor_carrying[move_in_idx] @ amounts
        return NH3_violation, Amounts_violation, empty_violation
    
    def update_Move_df(move_out_idx, move_in_idx, breed, amount):
        # Move_in_values = Move_in.iloc[move_in_idx]
        Move_in.at[move_in_idx,'氨排放差距'] -= NH3_meta[move_out_idx, move_in_idx, breed].item()
        Move_in.at[move_in_idx,'承载力差距'] -= Carrying_meta[move_out_idx, move_in_idx, breed].item()
        Move_out.iloc[move_out_idx, model.k + breed] -= amount
    
    def amount_adapt(move_out_idx, move_in_idx, breed):
        amount = amounts_copy[move_out_idx, breed].item()
        if Move_in.iloc[move_in_idx]['氨排放差距'] < amount * model.Move_in_tensor_NH3[move_in_idx, breed] or amount==0:
            amount = max(0, min(Move_out.iloc[move_out_idx, model.k + breed].astype(np.int64), 
                            (Move_in.iloc[move_in_idx]['氨排放差距'] / model.Move_in_tensor_NH3[move_in_idx, breed]).to(torch.int64).item()
                            ))
        else:
            amount = max(min(amount, Move_out.iloc[move_out_idx, model.k+breed]), 0)
        NH3_meta[move_out_idx, move_in_idx, breed] = amount * model.Move_in_tensor_NH3[move_in_idx, breed]
        Carrying_meta[move_out_idx, move_in_idx, breed] = amount * model.Move_in_tensor_carrying[move_in_idx, breed]
        
        return amount
    
    update_action_mask(*[None]*4)
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    with tqdm(total=100000, desc="进度") as pbar:  # 假设总进度是 100000

        while (episode_counts < episode_count_targets).any():
            actions, states = model.predict(
                observations,  # type: ignore[arg-type]
                state=states,
                episode_start=episode_starts,
                deterministic=deterministic,
                mask=action_mask  # 传递 action_mask
            )
            move_out_idx, move_in_idx, breed = model.action_proj(actions[0])
            amount = amount_adapt(move_out_idx, move_in_idx, breed)
            violation = detect_violation(move_out_idx, move_in_idx, breed, amount)

            while True in violation and action_mask.sum() < num_move_in_counties* num_move_out_counties * model.action_len:
                update_action_mask(move_out_idx, move_in_idx, breed, True)  # 更新 action_mask
                actions, states = model.predict(
                    observations,  # type: ignore[arg-type]
                    state=states,
                    episode_start=episode_starts,
                    deterministic=deterministic,
                    mask=action_mask  # 传递 action_mask
                )
                
                move_out_idx, move_in_idx, breed = model.action_proj(actions[0])
                amount = amount_adapt(move_out_idx, move_in_idx, breed)
                violation = detect_violation(move_out_idx, move_in_idx, breed ,amount)

            # 更新env NH3_meta, Carrying_meah, amounts
            # env.get_attr("amounts", 0)[0] = amounts_copy
            # env.get_attr("NH3_meta", 0)[0] = NH3_meta
            # env.get_attr("Carrying_meta", 0)[0] = Carrying_meta
            
            new_observations, rewards, dones, infos = env.step(actions)
            amounts_copy = env.get_attr("amounts", 0)[0]
            NH3_meta = env.get_attr("NH3_meta", 0)[0]
            Carrying_meta = env.get_attr("Carrying_meta", 0)[0]
            
            if action_mask.sum() == num_move_in_counties* num_move_out_counties * model.action_len:
                dones = [True] * n_envs
                
            update_Move_df(move_out_idx, move_in_idx, breed, amount)  # 更新 Move_in 和 Move_out
            update_action_mask(move_out_idx, move_in_idx, breed, False)  # 更新 action_mask
            env.unwrapped.envs[0].env.env.env.action_mask_left = num_move_in_counties* num_move_out_counties * model.action_len - action_mask.sum()


            current_rewards += rewards
            current_lengths += 1
            
            pbar.set_postfix({"剩余牲畜数量": infos[0]['livestock_left'], "reward":rewards[0], "mask":action_mask.sum(), "bug":(Move_in['氨排放差距'].values != new_observations['NH3_diff'][0]).sum()})
            pbar.update(1)
            log.append([move_out_idx, move_in_idx, breed, amount, rewards[0], action_mask.sum().item()])
            
            for i in range(n_envs):
                if episode_counts[i] < episode_count_targets[i]:
                    # unpack values so that the callback can access the local variables
                    reward = rewards[i]
                    done = dones[i]
                    info = infos[i]
                    episode_starts[i] = done

                    if callback is not None:
                        callback(locals(), globals())

                    if dones[i]:
                        if is_monitor_wrapped:
                            # Atari wrapper can send a "done" signal when
                            # the agent loses a life, but it does not correspond
                            # to the true end of episode
                            if "episode" in info.keys():
                                # Do not trust "done" with episode endings.
                                # Monitor wrapper includes "episode" key in info if environment
                                # has been wrapped with it. Use those rewards instead.
                                episode_rewards.append(info["episode"]["r"])
                                episode_lengths.append(info["episode"]["l"])
                                # Only increment at the real end of an episode
                                episode_counts[i] += 1
                            action_mask = torch.zeros([num_move_out_counties, num_move_in_counties, model.action_len], dtype=bool).to(action_mask.device)
                            Move_in, Move_out, sensibility,manure = load_datas(model.country) # 需要修改
                            amounts_copy = model.amounts_origin.clone()
                            NH3_meta = model.NH3_meta_origin.clone()
                            Carrying_meta = model.Carrying_meta_origin.clone()
                            env.unwrapped.envs[0].env.env.env.action_mask_left  = model.action_mask_sum_origin

                        else:
                            episode_rewards.append(current_rewards[i])
                            episode_lengths.append(current_lengths[i])
                            episode_counts[i] += 1
                        current_rewards[i] = 0
                        current_lengths[i] = 0

            observations = new_observations

            if render:
                env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    if save_output:
        pd.DataFrame(log, columns=["move_out_idx", "move_in_idx", "breed", "amount", "reward", "action_mask.sum()"], index=None).to_excel(f"/home/yanisy/Researchs/Ai4S/畜牧业空间优化/results/{model.country}/PPO1.xlsx")
    return mean_reward, std_reward