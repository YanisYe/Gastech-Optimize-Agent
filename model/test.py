import os
import sys
sys.path.append("stable-baselines3")
from torch.nn.modules.activation import F
import gymnasium as gym
import pandas as pd
import numpy as np
import torch
from pathlib import Path

from logger import setup_logger
from GasEnviroment_curriculum_learning import GasEnv, GasEnvConfig
from stable_baselines3 import PPO_action_mask_v2
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from AttentionPolicy import GasEnvPolicy
from stable_baselines3.common.evaluation_action_mask_v2 import evaluate_policy

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Register environment
register(
    id='GasEnviroment_curriculum_learning',
    entry_point='GasEnviroment_curriculum_learning:GasEnv',
)

# Configuration
date = "xxxx"
version = f"{date}-all"
area_df = pd.read_excel('data/县市亚区.xlsx')
area_unique = area_df['所属农业亚区'].unique()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def exponential_schedule(initial_lr, final_lr):
    """
    Exponential learning rate schedule.
    """
    k = -np.log(final_lr / initial_lr)
    def func(progress_remaining):
        fraction_completed = 1 - progress_remaining
        return initial_lr * np.exp(-k * fraction_completed)
    return func

def test_all_regions():
    """
    Test trained models for all agricultural regions.
    """
    for area in area_unique:
        print(f"Starting evaluation for region: {area}")
        try:
            # Environment configuration for testing
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
                save_path=f"results/{version}/{area}",
                linear_result_path='results/linear_optimization_results_by_county_5gases_hard_target.xlsx',
                only_lp_phase=False,  # Disable linear programming constraint mode for evaluation
            )

            # Create and wrap environment
            env = make_vec_env('GasEnviroment_curriculum_learning', n_envs=1, env_kwargs={'config': config})
            env = VecCheckNan(env, raise_exception=True)

            # Load or create model
            model_path = rf'logs/{version}/{area}/best_model.zip'
            if not os.path.exists(model_path):
                # Create a dummy model if best model doesn't exist
                dummy_model = PPO_action_mask_v2(
                    GasEnvPolicy,
                    env, 
                    batch_size=256,
                    verbose=1, 
                    tensorboard_log='./board/',
                    seed=42,
                    learning_rate=exponential_schedule(2e-5, 1e-6),
                    n_steps=2**11,
                    ent_coef=0.01,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.5,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    n_epochs=10,
                    device=device,
                    policy_kwargs=dict(
                        optimizer_class=torch.optim.Adam,
                        optimizer_kwargs=dict(
                            betas=(0.9, 0.999),
                            eps=1e-7,
                            weight_decay=0
                        )
                    )
                )
                dummy_model.save(model_path)
                print(f"Region {area} has no best_model, created a dummy model.")

            # Load the trained model
            model = PPO_action_mask_v2.load(model_path, env=env)
            
            # Create results directory
            os.makedirs(f"results/{version}/{area}", exist_ok=True)
            
            # Evaluate the model
            evaluate_policy(model, env, n_eval_episodes=1, deterministic=True, render=False, action_mask=None, save_path=f"results/{version}_{date}/{area}")
            print(f"Region {area} evaluation completed, results saved in results/{version}/{area} directory.")

        except Exception as e:
            print(f"Error evaluating region {area}: {e}")
            continue

def merge_results_by_county():
    """
    Merge results from all regions by county.
    """
    base_path = f"results/{version}"
    merged_path = f"results/{version}_merged"
    
    # Create merged results directory
    if not os.path.exists(merged_path):
        os.makedirs(merged_path)
    
    # Get all region folders
    area_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    # Dictionary to store all data, key is filename (without extension), value is list of DataFrames
    all_data = {}
    
    print(f"Starting to merge results from {len(area_folders)} regions...")
    
    for area in area_folders:
        area_path = os.path.join(base_path, area)
        
        # Get all Excel files in this region
        excel_files = [f for f in os.listdir(area_path) if f.endswith('.xlsx')]
        
        for excel_file in excel_files:
            file_path = os.path.join(area_path, excel_file)
            
            try:
                # Read Excel file
                df = pd.read_excel(file_path, index_col=0)
                
                # Get filename (without extension) as key
                file_key = os.path.splitext(excel_file)[0]
                
                # If this file type hasn't been recorded yet, create new list
                if file_key not in all_data:
                    all_data[file_key] = []
                
                # Add to corresponding list
                all_data[file_key].append(df)
                
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue
    
    print(f"Found {len(all_data)} types of files to merge")
    
    # Merge each type of file
    for file_type, df_list in all_data.items():
        if df_list:  # Ensure list is not empty
            try:
                # Merge all DataFrames
                if file_type == "tech_selected_summary":
                    merged_df = pd.concat(df_list, axis=1)
                else:
                    merged_df = pd.concat(df_list, axis=0)
                
                # Save merged file
                output_path = os.path.join(merged_path, f"{file_type}_merged.xlsx")
                merged_df.to_excel(output_path)
                
                print(f"Saved merged file: {file_type}_merged.xlsx (contains {len(merged_df)} counties)")
                
            except Exception as e:
                print(f"Error merging file type {file_type}: {e}")
                continue
    
    print(f"Merge completed! Results saved in: {merged_path}")

if __name__ == "__main__":
    test_all_regions()
    # Uncomment to merge results
    # merge_results_by_county()