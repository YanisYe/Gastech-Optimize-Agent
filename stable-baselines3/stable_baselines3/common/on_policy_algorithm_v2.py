import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from tqdm import tqdm
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer, ChunkDictRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from model import CountyDataLoader, get_conflicts_tech
from scipy.optimize import linprog
import copy

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
        kwargs = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps        # 步长
        self.gamma = gamma            # 折扣因子
        self.learning_rate = learning_rate        # 学习率

        self.gae_lambda = gae_lambda           
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}
        
        self.techSet = env.get_attr('tech_set',0)[0]
        self.numCounty, self.numTech =env.get_attr('numCounty',0)[0], env.get_attr('numTech',0)[0]
        self.action_mask = self.reset_action_mask(current_step=0)

        self.tech_count_relax = env.get_attr('tech_count_relax',0)[0]
        self.lp_tech_count = env.get_attr('lp_tech_count',0)[0]
        self.action_mask_sum_origin = self.numCounty * self.numTech
        
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
                # self.rollout_buffer_class = ChunkDictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)
    
    def tech_amount_constraint(self, countyID, tech_constraint_amount):
        type_ID = type(next(iter(self.lp_tech_count.keys())))
        if countyID is None:
            return tech_constraint_amount.sum(axis=-1) >= self.lp_tech_count[type_ID(countyID)] * self.tech_count_relax
        else:
            return tech_constraint_amount[countyID].sum(axis=-1) >= self.lp_tech_count[type_ID(countyID)] * self.tech_count_relax

    def decode_action(self, action):
        # output: countyId, techId
        return action // self.numTech, action % self.numTech

    def get_conflicts_tech(self, techId, techSet):
        import pandas as pd
        nonTech = techSet.shape[0]
        techId = techId.cpu().numpy() if isinstance(techId, th.Tensor) else techId
        techId = techId.reshape(-1) if techId.ndim == 2 else techId
        conditions = [pd.Series([False] * nonTech) for _ in range(len(techId))]
        idx = np.where(techId != nonTech)[0]
        target_rows = techSet.loc[techId[idx]] if hasattr(techId, '__iter__') or isinstance(techId, slice) else techSet.iloc[[techId]]
        industries = target_rows['class'].values
        conflict_flags = target_rows['技术间的冲突'].values
        # 存储同一个产业的冲突技术
        if not hasattr(self, 'conflict_tech_by_industry'):
            self.conflict_tech_by_industry = {}
        for i, idx_value in enumerate(idx):
            industry = industries[i]
            conflict = conflict_flags[i]
            if industry not in self.conflict_tech_by_industry:
                self.conflict_tech_by_industry[industry] = []
            self.conflict_tech_by_industry[industry].append((techId[idx_value], conflict))
        # 构建多条件查询（向量化）
        for i, idx_value in enumerate(idx):
            industry = industries[i]
            conflict = conflict_flags[i]
            condition = (
                (techSet['class'] == industry) & 
                (techSet['技术间的冲突'] == conflict)
            )
            conditions[idx_value] = condition
        conditions = pd.concat(conditions, axis=1).values.T
        return conditions
        
    def update_action_mask(self, actions=None, county_tech_selected=None):
        for i in range(self.n_envs):
            if actions is not None:
                countyId, techId = self.decode_action(actions[i])
                _techIds = th.tensor(techId).to(self.device)
                # 满足冲突技术约束
                self.action_mask[i, countyId, :] = th.logical_or(
                    self.action_mask[i, countyId, :].data,
                    self.detect_violation(_techIds).to(self.device)
                )
                # 满足局部气体约束
                local_constraint_mask = self.detect_local_factors(countyId, i)
                if local_constraint_mask:
                    self.action_mask[i, countyId, :] = True
                    continue

                # 满足最大技术添加约束
                if county_tech_selected is not None:
                    if self.tech_amount_constraint(countyId.item(), county_tech_selected[i]):
                        self.action_mask[i, countyId, :] = True
                        continue
                self.action_mask[i][countyId, techId] = True
        
    def detect_violation(self, techId):
        '''
        1. 根据tectID和countyID找到相斥的tectID, 返回一个bool tensor
        '''
        conflicts_tech_idx = get_conflicts_tech(techId, self.techSet)
        mask = th.zeros(self.numTech, dtype=bool).to(self.device)
        mask[conflicts_tech_idx] = True
        return mask
    
    def detect_local_factors(self, countyId, id_env):
        '''
        检测县是否三气体指标已经低于阈值
        '''
        gap_NH3 = self.env.get_attr('gap_NH3',id_env)[0][countyId]
        gap_NO3 = self.env.get_attr('gap_NO3',id_env)[0][countyId]
        gap_N_runoff = self.env.get_attr('gap_N_runoff',id_env)[0][countyId]
        gap_CH4 = self.env.get_attr('gap_CH4',id_env)[0][countyId]
        gap_N2O = self.env.get_attr('gap_N2O',id_env)[0][countyId]
        return gap_NH3 <= 0 and gap_NO3 <= 0 and gap_N_runoff <= 0
    
    def detect_global_factors(self):
        '''
        检测县的CH4和N2O是否大于等于0（防止过度减排）
        '''
        CH4 = self.env.get_attr('state',0)[0]['CH4']
        N2O = self.env.get_attr('state',0)[0]['N2O']
        return th.logical_and((CH4 >= 0), (N2O >= 0))
    
    def check_tech_scale_zero(self, tech_index, county_scale_data, tech_set):
        """
        检查技术对应的作物/牲畜的规模或面积是否为0
        """
        tech_line = tech_set.iloc[tech_index]
        tech_industry = tech_line['Livestock species'] if tech_line['class'] != 'crop' else tech_line['Crop species']
        if tech_line['class'] == 'crop':
            if tech_industry == 'friut':
                industry_scale = county_scale_data['fruittree_sown_area']
            else:
                industry_scale = county_scale_data.get(f'{tech_industry}_sown_area', 0)
        else:
            tech_industry = tech_industry.lower()
            if tech_industry in county_scale_data.index or tech_industry in county_scale_data.keys():
                industry_scale = county_scale_data[tech_industry]
            else:
                industry_scale = county_scale_data.get(tech_industry.replace(' ', ''), 0)
        return industry_scale == 0
    
    def reset_action_mask(self, env_idx=None, current_step=None):
        """
        重置action mask，支持基于训练步数的课程学习
        直接使用环境的action mask，环境内部已经实现了课程学习逻辑
        
        Args:
            env_idx: 环境索引，None表示所有环境
            current_step: 当前训练步数，用于课程学习（这里不使用，环境内部管理）
        """
        if env_idx is None:
            action_mask = th.zeros((self.n_envs, self.numCounty, self.numTech), dtype=bool).to(self.device)
            for i in range(self.n_envs):
                # 获取counties_need_tech约束
                counties_need_tech = self.env.get_attr('counties_need_tech', i)[0]
                action_mask[i, ~counties_need_tech, :] = True
                
                # 初始化规模/面积为0的县-产业技术全部mask
                county_scale_df = self.env.get_attr('county_scale_original', i)[0]
                tech_set = self.techSet
                for countyId in range(self.numCounty):
                    for t_idx in range(self.numTech):
                        if self.check_tech_scale_zero(t_idx, county_scale_df.iloc[countyId], tech_set):
                            action_mask[i, countyId, t_idx] = True

                # 检查是否启用了课程学习
                only_lp_phase = self.env.get_attr('only_lp_phase', i)[0]
                if only_lp_phase:
                    # 让环境根据训练步数更新action mask并直接返回
                    if current_step is not None:
                        env_action_mask = self.env.env_method('update_action_mask', current_step, indices=[i])[0]
                   
                    # 转换为torch tensor并移到正确的设备
                    if isinstance(env_action_mask, th.Tensor):
                        curriculum_mask = env_action_mask.to(self.device)
                    else:
                        curriculum_mask = th.tensor(env_action_mask, dtype=th.bool).to(self.device)
                    
                    # 合并基础mask和课程学习mask
                    action_mask[i] = th.logical_or(action_mask[i], curriculum_mask).to(self.device)
        else:
            # 获取counties_need_tech约束
            counties_need_tech = self.env.get_attr('counties_need_tech', env_idx)[0]
            action_mask = th.zeros((self.numCounty, self.numTech), dtype=bool).to(self.device)
            action_mask[~counties_need_tech, :] = True

            # 初始化规模/面积为0的县-产业技术全部mask
            county_scale_df = self.env.get_attr('county_scale_original', env_idx)[0]
            tech_set = self.techSet
            for countyId in range(self.numCounty):
                for t_idx in range(self.numTech):
                    if self.check_tech_scale_zero(t_idx, county_scale_df.iloc[countyId], tech_set):
                        action_mask[countyId, t_idx] = True
            
            # 检查是否启用了课程学习
            only_lp_phase = self.env.get_attr('only_lp_phase', env_idx)[0]
            if only_lp_phase:
                # 让环境根据训练步数更新action mask并直接返回
                if current_step is not None:
                    env_action_mask = self.env.env_method('update_action_mask', current_step, indices=[env_idx])[0]
                else:
                    env_action_mask = self.env.get_attr('action_mask', env_idx)[0]
                
                # 转换为torch tensor并移到正确的设备
                if isinstance(env_action_mask, th.Tensor):
                    curriculum_mask = env_action_mask.to(self.device)
                else:
                    curriculum_mask = th.tensor(env_action_mask, dtype=th.bool).to(self.device)
                
                # 合并基础mask和课程学习mask
                action_mask = th.logical_or(action_mask, curriculum_mask).to(self.device)
        
        return action_mask
    
    def get_curriculum_action_mask(self, env_idx, current_step):
        """
        直接从环境获取课程学习的action mask
        
        Args:
            env_idx: 环境索引
            current_step: 当前训练步数（用于调试输出）
            
        Returns:
            action_mask: 布尔张量，True表示被遮蔽的动作
        """
        # 获取环境配置参数
        only_lp_phase = self.env.get_attr('only_lp_phase', env_idx)[0]
        if not only_lp_phase:
            # 如果没有启用课程学习，返回空mask（不遮蔽任何动作）
            return th.zeros((self.numCounty, self.numTech), dtype=bool).to(self.device)
        
        # 让环境根据训练步数更新action mask并直接返回
        env_action_mask = self.env.env_method('update_action_mask', current_step, indices=[env_idx])[0]
        
        # 转换为torch tensor并移到正确的设备
        if isinstance(env_action_mask, th.Tensor):
            action_mask = env_action_mask.to(self.device)
        else:
            action_mask = th.tensor(env_action_mask, dtype=th.bool).to(self.device)
        
        # 打印调试信息
        if env_idx == 0:  # 只为第一个环境打印
            mask_sum = action_mask.sum().item()
            total_actions = self.numCounty * self.numTech
            print(f"Algorithm Step {current_step}: action mask sum: {mask_sum}, total: {total_actions}, masked ratio: {mask_sum / total_actions:.2%}")
        
        return action_mask
    
    def update_curriculum_action_mask(self):
        """
        在rollout过程中更新课程学习的action mask
        使用当前训练步数让环境更新action mask
        """
        for i in range(self.n_envs):
            # 检查是否启用了课程学习
            only_lp_phase = self.env.get_attr('only_lp_phase', i)[0]
            
            if only_lp_phase:
                # 让环境根据当前训练步数更新action mask并直接返回
                env_action_mask = self.env.env_method('update_action_mask', self.num_timesteps, indices=[i])[0]
                
                # 转换为torch tensor并移到正确的设备
                if isinstance(env_action_mask, th.Tensor):
                    curriculum_mask = env_action_mask.to(self.device)
                else:
                    curriculum_mask = th.tensor(env_action_mask, dtype=th.bool).to(self.device)
                
                # 获取counties_need_tech约束
                counties_need_tech = self.env.get_attr('counties_need_tech', i)[0]
                base_mask = th.zeros((self.numCounty, self.numTech), dtype=bool).to(self.device)
                base_mask[~counties_need_tech, :] = True
                
                # 合并基础mask和课程学习mask
                self.action_mask[i] = th.logical_or(base_mask, curriculum_mask).to(self.device)
            else:
                # 如果没有启用课程学习，只应用基础约束
                counties_need_tech = self.env.get_attr('counties_need_tech', i)[0]
                self.action_mask[i] = th.zeros((self.numCounty, self.numTech), dtype=bool).to(self.device)
                self.action_mask[i][~counties_need_tech, :] = True
                
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        total_timesteps: int,
        # heat_rate:float,

    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)
            
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                # policy action
                flat_action_mask = self.action_mask.view(self.n_envs, -1)
                actions, values, log_probs = self.policy(obs_tensor, flat_action_mask)
                # countyId, tectId = self.action_proj(actions[0])

            # Rescale and perform action
            if isinstance(actions, th.Tensor):
                actions = actions.cpu().numpy()
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
        
            for i in range(self.n_envs):
                self.env.unwrapped.envs[i].env.env.env.action_mask_left = self.action_mask_sum_origin - self.action_mask[i].sum() # last column always 0
            
            new_obs, rewards, dones, infos = self.env.step(clipped_actions)
            county_tech_selected = new_obs['Tech_selected']

            self.update_action_mask(actions, county_tech_selected)
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1
            print(f"n_steps: {n_steps} || mask_sum_ratio:{[self.action_mask[i].sum()/self.action_mask_sum_origin for i in range(self.n_envs)]} || actions: {actions} || rewards: {rewards} || dones: {dones} || infos: {infos}")
            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value
                if done:
                    # 使用当前rollout的第一个n_step重置action mask
                    self.action_mask[idx] = self.action_mask_origin[idx].clone()
                    # self.priorityList = th.ones(self.numCounty) * 4

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]s
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,

    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            self.action_mask_origin = self.reset_action_mask(current_step=self.num_timesteps)
            continue_training = self.collect_rollouts(self.env, 
                                                    callback, 
                                                    self.rollout_buffer, 
                                                    n_rollout_steps=self.n_steps, 
                                                    total_timesteps=total_timesteps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
