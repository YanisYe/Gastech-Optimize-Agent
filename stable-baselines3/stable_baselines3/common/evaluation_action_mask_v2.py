# from re import T
from tkinter import N
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from model import get_conflicts_tech
from scipy.optimize import linprog
import copy
import pandas as pd
import os


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
    save_path = None
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

    numCounty, numTech = model.numCounty, model.numTech
    device = model.device
    action_mask_sum_origin = model.action_mask_sum_origin

    log = []
    # priorityList = torch.ones(model.numCounty, dtype=int) * 4
    
    def update_action_mask(actions=None, county_tech_selected=None):
        for i in range(n_envs):
            if actions is not None:
                countyId, techId = model.decode_action(actions[i])
                _techIds = torch.tensor(techId).to(device)
                
                # 冲突技术掩码 - 只对当前县的冲突技术进行屏蔽
                conflict_mask = detect_violation(_techIds).to(device)
                action_mask[i, countyId, :] = torch.logical_or(
                    action_mask[i, countyId, :].data,
                    conflict_mask
                )
                
                # 局部气体约束 - 如果该县已达到局部目标，屏蔽所有技术
                local_constraint_mask = detect_local_factors(countyId, i)
                if local_constraint_mask:
                    action_mask[i, countyId, :] = True
                    continue
                
                # 满足最大技术添加约束 - 如果达到技术数量上限，屏蔽所有技术
                if county_tech_selected is not None and tech_amount_constraint(countyId, county_tech_selected[i]):
                    action_mask[i, countyId, :] = True
                    continue

                # 标记已选择的技术为不可用
                action_mask[i, countyId, techId] = True

    def detect_local_factors(countyID, id_env):
        gap_NH3 = env.get_attr('gap_NH3', id_env)[0][countyID]
        gap_NO3 = env.get_attr('gap_NO3', id_env)[0][countyID]
        gap_N_runoff = env.get_attr('gap_N_runoff', id_env)[0][countyID]
        gap_CH4 = env.get_attr('gap_CH4', id_env)[0][countyID]
        gap_N2O = env.get_attr('gap_N2O', id_env)[0][countyID]
        # 返回布尔值，表示该县是否已达到三气体局部目标
        return bool(gap_NH3 <= 0 and gap_NO3 <= 0 and gap_N_runoff <= 0)

    def detect_violation(techId):
        conflicts_tech_idx = get_conflicts_tech(techId, model.techSet)
        mask = torch.zeros(numTech, dtype=bool).to(device)
        mask[conflicts_tech_idx] = True
        return mask
    
    def tech_amount_constraint(countyID, tech_constraint_amount):
        type_ID = type(next(iter(model.lp_tech_count.keys())))
        if countyID is None:
            return tech_constraint_amount.sum(axis=-1) >= model.lp_tech_count[type_ID(countyID)] * model.tech_count_relax
        else:
            return tech_constraint_amount[countyID].sum() >= model.lp_tech_count[type_ID(countyID)] * model.tech_count_relax
    
    def detect_violation_batch(techIds):
        conlicts_tect_idx = get_conflicts_tech(techIds, model.techSet)
        mask = torch.zeros(numCounty, numTech, dtype=bool).to(device)
        mask[conlicts_tect_idx] = True
        return mask

    def check_tech_scale_zero(tech_index, county_scale_data, tech_set):
        """
        检查技术对应的作物/牲畜的规模或面积是否为0
        """
        tech_line = tech_set.iloc[tech_index]
        tech_industry = tech_line['Livestock species'] if tech_line['class'] != 'crop' else tech_line['Crop species']
        if tech_line['class'] == 'crop':
            if tech_industry == 'friut':
                industry_scale = county_scale_data['fruittree_sown_area']
            else:
                industry_scale = county_scale_data['{}_sown_area'.format(tech_industry)]
        else:
            tech_industry = tech_industry.lower()
            if tech_industry in county_scale_data.index or tech_industry in county_scale_data.keys():
                industry_scale = county_scale_data[tech_industry]
            else:
                industry_scale = county_scale_data.get(tech_industry.replace(' ', ''), 0)
        return industry_scale == 0

    def reset_action_mask(env, env_idx=None):
        # 将counties_need_tech中为False的县对应的action_mask设置为True（屏蔽不需要技术的县）
        # counties_need_tech是布尔数组，True表示需要技术，False表示不需要
        # 我们要屏蔽不需要技术的县，所以对False的县设置mask为True
        if env_idx is None:
            action_mask = torch.zeros((n_envs, numCounty, numTech), dtype=bool).to(device)
            for i in range(n_envs):
                counties_need_tech = env.get_attr('counties_need_tech', i)[0]
                action_mask[i, ~counties_need_tech, :] = True
                # 初始化规模/面积为0的县-产业技术全部mask
                county_scale_df = env.get_attr('county_scale_original', i)[0]
                tech_set = model.techSet
                for countyId in range(numCounty):
                    for t_idx in range(numTech):
                        if check_tech_scale_zero(t_idx, county_scale_df.iloc[countyId], tech_set):
                            action_mask[i, countyId, t_idx] = True
        else:
            counties_need_tech = env.get_attr('counties_need_tech', env_idx)[0]
            action_mask = torch.zeros((numCounty, numTech), dtype=bool).to(device)
            action_mask[~counties_need_tech, :] = True
            # 初始化规模/面积为0的县-产业技术全部mask
            county_scale_df = env.get_attr('county_scale_original', env_idx)[0]
            tech_set = model.techSet
            for countyId in range(numCounty):
                for t_idx in range(numTech):
                    if check_tech_scale_zero(t_idx, county_scale_df.iloc[countyId], tech_set):
                        action_mask[countyId, t_idx] = True

        return action_mask

    
    if action_mask is None:
        action_mask = reset_action_mask(env)

    episode_starts = np.ones((n_envs,), dtype=bool)

    with tqdm(total=n_eval_episodes, desc="进度") as pbar:  # 假设总进度是 n_eval_episodes
        while (episode_counts < episode_count_targets).any():
            # 将3D action_mask转换为2D，以匹配CategoricalMasked的期望
            flat_action_mask = action_mask.view(n_envs, -1)  # shape: (n_envs, numCounty * numTech)
            
            actions, states = model.predict(
                observations,  # type: ignore[arg-type]
                state=states,
                episode_start=episode_starts,
                deterministic=deterministic,
                mask=flat_action_mask  # 传递扁平化的 action_mask
            )
            dones = [False] * n_envs
            for i in range(n_envs):
                dones[i] = action_mask[i].sum() == model.action_mask_sum_origin
                if dones[i]:
                    env.unwrapped.envs[i].env.env.env.action_mask_left = 0
                else:
                    env.unwrapped.envs[i].env.env.env.action_mask_left = model.action_mask_sum_origin - action_mask[i].sum()

            new_observations, rewards, dones, infos = env.step(actions)
            county_tech_selected = new_observations['Tech_selected']
            update_action_mask(actions, county_tech_selected)
                
            current_rewards += rewards
            current_lengths += 1
            
            pbar.set_postfix({"step": current_lengths, "reward":rewards})
            if save_path:
                # 将单个动作解码为县ID和技术ID，然后记录
                countyId, techId = model.decode_action(actions[0])
                log.append([countyId, techId, rewards[0], action_mask.sum().item()])
            
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
                            # reset
                            action_mask[i] = reset_action_mask(env, i)
                            # 重置后不需要立即调用update_action_mask，因为没有动作需要处理
                            pbar.update(1)
                            env.unwrapped.envs[i].env.env.env.action_mask_left  = model.action_mask_sum_origin - action_mask[i].sum()
                            # priorityList = torch.ones(model.numCounty, dtype=int) * 4
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

    return mean_reward, std_reward
