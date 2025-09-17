import os
import sys
sys.path.append("stable-baselines3")
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import gymnasium as gym
from logger import setup_logger
from GasEnviroment_curriculum_learning import GasEnv, GasEnvConfig
from stable_baselines3 import PPO_action_mask_v2
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from AttentionPolicy import GasEnvPolicy
import pandas as pd
import torch as th
import numpy as np
import torch
import glob

# 注册环境
register(
    id='GasEnviroment_curriculum_learning',
    entry_point='GasEnviroment_curriculum_learning:GasEnv',
)

date = "0916"
version = f"{date}-all"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def exponential_schedule(initial_lr, final_lr):
    k = -np.log(final_lr / initial_lr)
    def func(progress_remaining):
        fraction_completed = 1 - progress_remaining
        return initial_lr * np.exp(-k * fraction_completed)
    return func

def train():
    area_df = pd.read_excel('data/县市亚区.xlsx')
    area_unique = area_df['所属农业亚区'].unique()

    print("开始训练")
    
    for area_idx, area in enumerate(area_unique):
        print(f"\n训练区域 {area_idx+1}/{len(area_unique)}: {area}")

        # 指定训练亚区
        # if area != '青甘牧农区':
        #     continue

        # 检查当前亚区是否已存在训练日志目录
        area_log_path = f'logs/{version}/{area}/'
        if os.path.exists(area_log_path):
            print(f"区域 {area} 的日志目录已存在，跳过训练")
            continue  # 跳过当前亚区


        total_steps = 2**18
        
        config = GasEnvConfig(
            Reward_priority=[7, 5, 3, 2],
            county_df_path='data/基础数据-县级尺度.xlsx',
            livestock_tech_path='data/畜牧业技术列单-经济产量0827.xlsx',
            crop_tech_path="data/种植业技术列单产量产业0803.xlsx",
            soc_df="data/SOC-县尺度.xlsx",
            livestock_scale='data/动物数量.xlsx',
            crop_scale='data/分县种植面积.xlsx',
            area=area,
            IDs_df='data/县市亚区.xlsx',
            save_path=None,
            linear_result_path='results/linear_optimization_results_by_county_5gases_hard_target.xlsx',
            only_lp_phase=True,  # 启用课程学习模式
            total_steps = total_steps,
            lp_phase_ratio = 0.8,  # 线性规划阶段占比，只在线性规划解空间学习
            phase_1_ratio = 0.85,  # 第一阶段占比，放开技术等级 1
            phase_2_ratio = 0.9,  # 第二阶段占比，放开技术等级 2
        )

        try:
            env = make_vec_env('GasEnviroment_curriculum_learning', n_envs=5, env_kwargs={'config': config})
            env = VecCheckNan(env, raise_exception=True)
            
            eval_callback = EvalCallback(
                env, 
                best_model_save_path=f'logs/{version}/{area}/',
                log_path=f'./logs/{version}/{area}/', 
                eval_freq=2**14+1,
                deterministic=False, 
                render=False
            )
            # 从头开始训练
            model = PPO_action_mask_v2(
                GasEnvPolicy,
                env, 
                batch_size=256,  # 批次大小
                verbose=1, 
                tensorboard_log='./board/',
                seed=42,
                learning_rate=exponential_schedule(2e-5, 1e-6),  # 学习率
                n_steps=2**14,  # 每步收集的样本数 
                ent_coef=0.01,  # 熵系数 
                clip_range=0.5,  # PPO裁剪系数（默认0.2）
                device=device,
                policy_kwargs=dict(
                    optimizer_class=th.optim.Adam,
                    optimizer_kwargs=dict(
                        betas=(0.9, 0.999),
                        eps=1e-7,
                        weight_decay=0
                    )
                )
            )

            model.learn(
                total_timesteps=total_steps,
                tb_log_name=f"PPO_Scratch_{version}_{area}",
                callback=eval_callback
            )

            model.save(f'logs/{version}/{area}/final_model')
            print(f"区域 {area} 训练完成")
            del model, env, config

        except Exception as e:
            print(f"区域 {area} 训练失败: {e}")
            continue



if __name__ == "__main__":
    train()