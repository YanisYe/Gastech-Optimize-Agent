import os
import sys
sys.path.append("stable-baselines3")

import gymnasium as gym
import pandas as pd
import torch as th
import numpy as np
import torch
import glob
from typing import Callable

from logger import setup_logger
from GasEnviroment_curriculum_learning import GasEnv, GasEnvConfig
from stable_baselines3 import PPO_action_mask_v2
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from AttentionPolicy import GasEnvPolicy

# Register environment
register(
    id='GasEnviroment_curriculum_learning',
    entry_point='GasEnviroment_curriculum_learning:GasEnv',
)

# Configuration
date = "xxxx"
version = f"{date}-all"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    """
    Train PPO models for all agricultural regions using curriculum learning.
    """
    # Load agricultural region data
    area_df = pd.read_excel('data/县市亚区.xlsx')
    area_unique = area_df['所属农业亚区'].unique()

    print("Starting training...")
    
    for area_idx, area in enumerate(area_unique):
        print(f"\nTraining region {area_idx+1}/{len(area_unique)}: {area}")

        # Specify training region (uncomment to train specific region)
        # if area != '青甘牧农区':
        #     continue

        # Check if training log directory already exists for current region
        area_log_path = f'logs/{version}/{area}/'
        if os.path.exists(area_log_path):
            print(f"Region {area} log directory already exists, skipping training")
            continue  # Skip current region

        # Training configuration
        total_steps = 2**18
        
        # Environment configuration
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
            only_lp_phase=True,  # Enable curriculum learning mode
            total_steps=total_steps,
            lp_phase_ratio=0.8,  # Linear programming phase ratio, learn only in LP solution space
            phase_1_ratio=0.85,  # Phase 1 ratio, release technology level 1
            phase_2_ratio=0.9,  # Phase 2 ratio, release technology level 2
        )

        try:
            # Create vectorized environment
            env = make_vec_env('GasEnviroment_curriculum_learning', n_envs=5, env_kwargs={'config': config})
            env = VecCheckNan(env, raise_exception=True)
            
            # Setup evaluation callback
            eval_callback = EvalCallback(
                env, 
                best_model_save_path=f'logs/{version}/{area}/',
                log_path=f'./logs/{version}/{area}/', 
                eval_freq=2**14+1,
                deterministic=False, 
                render=False
            )
            
            # Initialize PPO model
            model = PPO_action_mask_v2(
                GasEnvPolicy,
                env, 
                batch_size=256,  # Batch size
                verbose=1, 
                tensorboard_log='./board/',
                seed=42,
                learning_rate=exponential_schedule(2e-5, 1e-6),  # Learning rate
                n_steps=2**14,  # Number of samples collected per step
                ent_coef=0.01,  # Entropy coefficient
                clip_range=0.5,  # PPO clipping coefficient (default 0.2)
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

            # Train the model
            model.learn(
                total_timesteps=total_steps,
                tb_log_name=f"PPO_Scratch_{version}_{area}",
                callback=eval_callback
            )

            # Save the final model
            model.save(f'logs/{version}/{area}/final_model')
            print(f"Region {area} training completed")
            
            # Clean up memory
            del model, env, config

        except Exception as e:
            print(f"Region {area} training failed: {e}")
            continue



if __name__ == "__main__":
    train()