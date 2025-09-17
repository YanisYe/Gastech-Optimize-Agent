import os
import sys
sys.path.append("stable-baselines3")
from torch.nn.modules.activation import F
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gymnasium as gym
from logger import setup_logger
from GasEnviroment_curriculum_learning import GasEnv, GasEnvConfig
from stable_baselines3 import PPO_action_mask_v2
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from AttentionPolicy import GasEnvPolicy
from stable_baselines3.common.evaluation_action_mask_v2 import evaluate_policy
import pandas as pd
import numpy as np
import torch

register(
    id='GasEnviroment_curriculum_learning',
    entry_point='GasEnviroment_curriculum_learning:GasEnv',
)

# 参数配置
date = "0916"
version = f"{date}-all"
area_df = pd.read_excel('data/县市亚区.xlsx')
area_unique = area_df['所属农业亚区'].unique()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def exponential_schedule(initial_lr, final_lr):
    k = -np.log(final_lr / initial_lr)
    def func(progress_remaining):
        fraction_completed = 1 - progress_remaining
        return initial_lr * np.exp(-k * fraction_completed)
    return func

for area in area_unique:
    # 如果亚区不在指定列表中，则跳过
    # if area not in ['琼雷及南海诸岛农林区', '青甘牧农区']:
    #     continue

    # 如果亚区已经存在，则跳过
    # if os.path.exists(f"results/{version}/{area}"):
    #     print(f"亚区 {area} 已经存在，跳过")
    #     continue
    
    print(f"开始评估亚区 {area}")
    try:
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
            only_lp_phase=False,  # 评估过程不启用线性规划约束模式
        )

        # 创建并包装环境
        env = make_vec_env('GasEnviroment_curriculum_learning', n_envs=1, env_kwargs={'config': config})
        env = VecCheckNan(env, raise_exception=True)

        model_path = rf'logs/{version}/{area}/best_model.zip'
        if not os.path.exists(model_path):
            # 捏造一个空模型并保存
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
            print(f"亚区 {area} 没有best_model，已生成一个空模型。")

        model = PPO_action_mask_v2.load(model_path, env=env)
        # 预先创建结果目录
        os.makedirs(f"results/{version}/{area}", exist_ok=True)
        evaluate_policy(model, env, n_eval_episodes=1, deterministic=True, render=False, action_mask=None, save_path=f"results/{version}_{date}/{area}")
        print(f"亚区 {area} 评估完成，结果保存在 results/{version}/{area} 目录下。")

    except Exception as e:
        print(f"Error: {e}")
        continue

    
# 合并所有亚区的结果文件
import pandas as pd
import os
from pathlib import Path

def merge_results_by_county():
    """将所有亚区的结果按县合并"""
    base_path = f"results/{version}"
    merged_path = f"results/{version}_merged"
    
    # 创建合并结果的目录
    if not os.path.exists(merged_path):
        os.makedirs(merged_path)
    
    # 获取所有亚区文件夹
    area_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    # 存储所有数据的字典，键为文件名（不含扩展名），值为DataFrame列表
    all_data = {}
    
    print(f"开始合并 {len(area_folders)} 个亚区的结果...")
    
    for area in area_folders:
        area_path = os.path.join(base_path, area)
        
        # 获取该亚区内的所有Excel文件
        excel_files = [f for f in os.listdir(area_path) if f.endswith('.xlsx')]
        
        for excel_file in excel_files:
            file_path = os.path.join(area_path, excel_file)
            
            try:
                # 读取Excel文件
                df = pd.read_excel(file_path, index_col=0)
                
                # 获取文件名（不含扩展名）作为键
                file_key = os.path.splitext(excel_file)[0]
                
                # 如果这个文件类型还没有记录，创建新列表
                if file_key not in all_data:
                    all_data[file_key] = []
                
                # 添加到对应的列表中
                all_data[file_key].append(df)
                
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")
                continue
    
    print(f"找到 {len(all_data)} 种类型的文件需要合并")
    
    # 合并每种类型的文件
    for file_type, df_list in all_data.items():
        if df_list:  # 确保列表不为空
            try:
                # 合并所有DataFrame
                if file_type == "tech_selected_summary":
                    merged_df = pd.concat(df_list, axis=1)
                else:
                    merged_df = pd.concat(df_list, axis=0)
                
                # 保存合并后的文件
                output_path = os.path.join(merged_path, f"{file_type}_merged.xlsx")
                merged_df.to_excel(output_path)
                
                print(f"已保存合并文件: {file_type}_merged.xlsx (包含 {len(merged_df)} 个县)")
                
            except Exception as e:
                print(f"合并文件类型 {file_type} 时出错: {e}")
                continue
    
    print(f"合并完成！结果保存在: {merged_path}")

# 执行合并
# merge_results_by_county()