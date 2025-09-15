#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分步优化：先为所有县应用1级技术，未达标县继续应用2级技术，仍未达标县再应用3级技术
"""

import os
import sys
import pandas as pd
import torch
import numpy as np
from pathlib import Path

# 添加model目录到路径
model_path = Path(__file__).parent / "model"
sys.path.append(str(model_path))

from GasEnviroment_curriculum_learning import GasEnv, GasEnvConfig

def get_conflicts_tech(techId, techSet):
    """
    获取与指定技术冲突的技术索引
    
    Args:
        techId: 技术ID
        techSet: 技术集合DataFrame
        
    Returns:
        pandas.Series: 布尔序列，True表示冲突技术
    """
    if isinstance(techId, torch.Tensor):
        techId = techId.cpu().numpy()
    
    if hasattr(techId, '__len__') and len(techId) > 1:
        techId = techId[0]
    
    target_row = techSet.iloc[techId]
    industry = target_row['class']
    conflict = target_row['技术间的冲突']
    
    # 找到同一产业且同一冲突组的技术
    condition = (
        (techSet['class'] == industry) & 
        (techSet['技术间的冲突'] == conflict)
    )
    
    return condition

def select_optimal_techs_up_to_level(env, max_tech_level):
    """
    从所有≤指定分级的技术中选择最优技术组合，考虑技术冲突和优先级
    
    Args:
        env: 环境实例
        max_tech_level: 最大技术分级 (1, 2, 3)
        
    Returns:
        list: 选中的技术ID列表
    """
    tech_set = env.tech_set
    
    # 获取所有≤指定分级的技术，按优先级排序
    # 优先级：1. 技术分级越小越优先 2. 经济成本越低越优先
    available_techs = tech_set[tech_set['技术分级'] <= max_tech_level].copy()
    available_techs = available_techs.sort_values(['技术分级', '经济成本'], ascending=[True, True])
    
    print(f"找到 {len(available_techs)} 个技术分级≤{max_tech_level}的技术")
    
    selected_techs = []
    conflict_groups_used = set()  # 记录已使用的冲突组
    
    for tech_idx in available_techs.index:
        tech_row = available_techs.loc[tech_idx]
        
        # 检查是否与已选技术冲突
        conflict_key = (tech_row['class'], tech_row['技术间的冲突'])
        
        if conflict_key in conflict_groups_used:
            # 如果冲突组已被使用，检查当前技术是否更优
            # 找到同冲突组中已选择的技术
            existing_tech_idx = None
            for selected_idx in selected_techs:
                selected_row = tech_set.iloc[selected_idx]
                if (selected_row['class'] == tech_row['class'] and 
                    selected_row['技术间的冲突'] == tech_row['技术间的冲突']):
                    existing_tech_idx = selected_idx
                    break
            
            if existing_tech_idx is not None:
                existing_row = tech_set.iloc[existing_tech_idx]
                
                # 比较优先级：技术分级 > 经济成本
                current_priority = (tech_row['技术分级'], tech_row['经济成本'])
                existing_priority = (existing_row['技术分级'], existing_row['经济成本'])
                
                if current_priority < existing_priority:
                    # 当前技术更优，替换现有技术
                    selected_techs.remove(existing_tech_idx)
                    selected_techs.append(tech_idx)
                    print(f"替换技术: {existing_row['Mitigation strategy']} -> {tech_row['Mitigation strategy']} (更优)")
                # else: 保持现有技术
            continue
        else:
            # 新的冲突组，直接添加
            selected_techs.append(tech_idx)
            conflict_groups_used.add(conflict_key)
            print(f"选择技术: {tech_row['Mitigation strategy']} (分级:{tech_row['技术分级']}, 成本:{tech_row['经济成本']:.2f})")
    
    # 统计各分级技术数量
    level_counts = {}
    for tech_idx in selected_techs:
        level = tech_set.iloc[tech_idx]['技术分级']
        level_counts[level] = level_counts.get(level, 0) + 1
    
    print(f"最终选择了 {len(selected_techs)} 个最优技术，分级分布: {level_counts}")
    return selected_techs

def check_counties_meeting_targets(env):
    """
    检查哪些县已经达标
    
    Args:
        env: 环境实例
        
    Returns:
        numpy.ndarray: 布尔数组，True表示该县所有指标都达标
    """
    # 所有指标都达标的县
    all_targets_met = (
        (env.gap_NO3.squeeze() <= 0) & 
        (env.gap_NH3.squeeze() <= 0) & 
        (env.gap_N_runoff.squeeze() <= 0) 
        # (env.gap_CH4.squeeze() <= 0) &
        # (env.gap_N2O.squeeze() <= 0)
    ).cpu().numpy()
    
    return all_targets_met

def apply_techs_to_counties(env, max_tech_level, target_counties=None):
    """
    为指定县应用≤指定分级的最优技术组合
    
    Args:
        env: 环境实例
        max_tech_level: 最大技术分级 (1, 2, 3)
        target_counties: 目标县索引列表，如果为None则应用到所有需要技术的县
        
    Returns:
        tuple: (应用的技术数量, 总执行动作数)
    """
    print(f"\n开始应用≤{max_tech_level}级的最优技术组合...")
    
    # 选择≤该分级的最优技术组合
    selected_tech_ids = select_optimal_techs_up_to_level(env, max_tech_level)
    
    if not selected_tech_ids:
        print(f"没有找到≤{max_tech_level}级的技术，跳过")
        return 0, 0
    
    # 确定目标县
    if target_counties is None:
        counties_need_tech_indices = np.where(env.counties_need_tech)[0]
    else:
        # 只处理指定的县，且这些县需要在counties_need_tech中
        counties_need_tech_indices = np.where(env.counties_need_tech)[0]
        target_counties = np.array(target_counties)
        # 取交集
        counties_need_tech_indices = np.intersect1d(counties_need_tech_indices, target_counties)
    
    print(f"目标县数量: {len(counties_need_tech_indices)}")
    
    total_actions = 0
    applied_techs = 0
    
    # 为每个目标县应用选定的技术组合
    for i, county_idx in enumerate(counties_need_tech_indices):
        county_name = env.IDs['Counties'].iloc[county_idx]
        print(f"\n处理第 {i+1}/{len(counties_need_tech_indices)} 个县: {county_name}")
        
        # 应用选中的技术
        county_applied_techs = 0
        for tech_idx in selected_tech_ids:
            # 检查技术是否已经应用过
            if env.state['Tech_selected'][county_idx, tech_idx] == 0:
                # 编码动作：需要将county_idx转换为在counties_need_tech中的索引
                all_counties_need_tech = np.where(env.counties_need_tech)[0]
                county_need_tech_pos = np.where(all_counties_need_tech == county_idx)[0][0]
                action = county_need_tech_pos * env.numTech + tech_idx
                
                try:
                    state, reward, terminated, truncated, info = env.step(action)
                    total_actions += 1
                    applied_techs += 1
                    county_applied_techs += 1
                except Exception as e:
                    print(f"    应用技术时出错 - 技术ID: {tech_idx}, 错误: {e}")
                    continue
        
        print(f"  县 {county_name} 实际应用了 {county_applied_techs} 个技术")
        
        # 每处理10个县打印一次进度
        if (i + 1) % 10 == 0:
            print(f"\n已处理 {i+1} 个县，总共应用了 {applied_techs} 个技术")
    
    print(f"\n≤{max_tech_level}级技术应用完成！总共应用了 {applied_techs} 个技术")
    print(f"总执行动作数: {total_actions}")
    
    return applied_techs, total_actions

def stepwise_tech_optimization(env):
    """
    分步技术优化：逐级扩展技术选择范围直到达标或无更高级技术
    
    Args:
        env: 环境实例
        
    Returns:
        dict: 各步骤技术应用统计
    """
    print("=" * 60)
    print("开始分步技术优化")
    print("=" * 60)
    
    stats = {
        'step_1': {'applied_techs': 0, 'actions': 0, 'counties_met': 0, 'max_level': 1},
        'step_2': {'applied_techs': 0, 'actions': 0, 'counties_met': 0, 'max_level': 2},
        'step_3': {'applied_techs': 0, 'actions': 0, 'counties_met': 0, 'max_level': 3}
    }
    
    total_counties = env.numCounty
    
    # 第一步：为所有县应用≤1级技术的最优组合
    print(f"\n第一步：为所有县应用≤1级技术的最优组合")
    applied_techs_1, actions_1 = apply_techs_to_counties(env, max_tech_level=1)
    stats['step_1']['applied_techs'] = applied_techs_1
    stats['step_1']['actions'] = actions_1
    
    # 检查第一步后的达标情况
    counties_met_after_step1 = check_counties_meeting_targets(env)
    counties_met_count_1 = counties_met_after_step1.sum()
    stats['step_1']['counties_met'] = counties_met_count_1
    
    print(f"\n第一步应用后达标情况:")
    print(f"达标县数: {counties_met_count_1}/{total_counties} ({counties_met_count_1/total_counties*100:.1f}%)")
    
    # 第二步：为未达标县重新从≤2级技术中选择最优组合
    counties_not_met_after_step1 = np.where(~counties_met_after_step1)[0]
    if len(counties_not_met_after_step1) > 0:
        print(f"\n第二步：为 {len(counties_not_met_after_step1)} 个未达标县重新从≤2级技术中选择最优组合")
        
        # 重置这些县的技术选择状态，重新优化
        for county_idx in counties_not_met_after_step1:
            env.state['Tech_selected'][county_idx, :] = 0
        
        applied_techs_2, actions_2 = apply_techs_to_counties(env, max_tech_level=2, target_counties=counties_not_met_after_step1)
        stats['step_2']['applied_techs'] = applied_techs_2
        stats['step_2']['actions'] = actions_2
        
        # 检查第二步后的达标情况
        counties_met_after_step2 = check_counties_meeting_targets(env)
        counties_met_count_2 = counties_met_after_step2.sum()
        stats['step_2']['counties_met'] = counties_met_count_2
        
        print(f"\n第二步应用后达标情况:")
        print(f"达标县数: {counties_met_count_2}/{total_counties} ({counties_met_count_2/total_counties*100:.1f}%)")
        print(f"新增达标县数: {counties_met_count_2 - counties_met_count_1}")
        
        # 第三步：为仍未达标县重新从≤3级技术中选择最优组合
        counties_not_met_after_step2 = np.where(~counties_met_after_step2)[0]
        if len(counties_not_met_after_step2) > 0:
            print(f"\n第三步：为 {len(counties_not_met_after_step2)} 个仍未达标县重新从≤3级技术中选择最优组合")
            
            # 重置这些县的技术选择状态，重新优化
            for county_idx in counties_not_met_after_step2:
                env.state['Tech_selected'][county_idx, :] = 0
            
            applied_techs_3, actions_3 = apply_techs_to_counties(env, max_tech_level=3, target_counties=counties_not_met_after_step2)
            stats['step_3']['applied_techs'] = applied_techs_3
            stats['step_3']['actions'] = actions_3
            
            # 检查第三步后的达标情况
            counties_met_after_step3 = check_counties_meeting_targets(env)
            counties_met_count_3 = counties_met_after_step3.sum()
            stats['step_3']['counties_met'] = counties_met_count_3
            
            print(f"\n第三步应用后达标情况:")
            print(f"达标县数: {counties_met_count_3}/{total_counties} ({counties_met_count_3/total_counties*100:.1f}%)")
            print(f"新增达标县数: {counties_met_count_3 - counties_met_count_2}")
        else:
            print(f"\n所有县在第二步后已达标，无需进行第三步")
    else:
        print(f"\n所有县在第一步后已达标，无需进行后续步骤")
    
    # 输出总体统计
    print(f"\n" + "=" * 60)
    print("分步优化完成！总体统计:")
    print("=" * 60)
    
    total_applied_techs = stats['step_1']['applied_techs'] + stats['step_2']['applied_techs'] + stats['step_3']['applied_techs']
    total_actions = stats['step_1']['actions'] + stats['step_2']['actions'] + stats['step_3']['actions']
    final_met_counties = stats['step_3']['counties_met'] if stats['step_3']['counties_met'] > 0 else (stats['step_2']['counties_met'] if stats['step_2']['counties_met'] > 0 else stats['step_1']['counties_met'])
    
    print(f"第一步(≤1级技术): 应用 {stats['step_1']['applied_techs']} 个技术, {stats['step_1']['actions']} 个动作")
    print(f"第二步(≤2级技术): 应用 {stats['step_2']['applied_techs']} 个技术, {stats['step_2']['actions']} 个动作")
    print(f"第三步(≤3级技术): 应用 {stats['step_3']['applied_techs']} 个技术, {stats['step_3']['actions']} 个动作")
    print(f"总计: 应用 {total_applied_techs} 个技术, {total_actions} 个动作")
    print(f"最终达标县数: {final_met_counties}/{total_counties} ({final_met_counties/total_counties*100:.1f}%)")
    
    return stats

def output_state_summary(env, output_dir="results", suffix="stepwise_techs"):
    """
    输出环境状态汇总
    
    Args:
        env: 环境实例
        output_dir: 输出目录
        suffix: 文件名后缀
    """
    print(f"\n=== 输出分步优化后的状态 ===")
    
    # 创建输出目录
    full_output_dir = os.path.join(output_dir, suffix)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # 获取县名列表
    county_names = env.IDs['Counties'].tolist()
    
    # 1. 保存观察状态数据（env.state）
    state_data = {}
    for key, value in env.state.items():
        if key == 'Tech_selected':
            # 计算每个县选择的技术数量
            tech_counts = value[:, :env.numTech].sum(axis=1).cpu().numpy()
            state_data['选择技术数量'] = tech_counts
        elif isinstance(value, torch.Tensor):
            if value.dim() == 2 and value.shape[1] == 1:
                state_data[key] = value.squeeze().cpu().numpy()
            elif value.dim() == 1:
                state_data[key] = value.cpu().numpy()
    
    # 创建观察状态DataFrame
    state_df = pd.DataFrame(state_data, index=county_names)
    state_file = os.path.join(full_output_dir, "state_after_stepwise_techs.xlsx")
    state_df.to_excel(state_file)
    print(f"观察状态数据已保存到: {state_file}") 
   
    # 2. 保存stateMapping的所有状态数据（按key分别保存）
    if hasattr(env, 'stateMapping') and env.stateMapping:
        print(f"正在保存 {len(env.stateMapping)} 个状态映射...")
        for key, value in env.stateMapping.items():
            try:
                if isinstance(value, torch.Tensor):
                    # 获取对应的列名
                    if key in env.class_mapping:
                        columns = env.class_mapping[key]
                        if hasattr(columns, 'tolist'):
                            columns = columns.tolist()
                        elif hasattr(columns, '__iter__') and not isinstance(columns, str):
                            columns = list(columns)
                        else:
                            columns = [str(columns)]
                    else:
                        # 如果没有映射，使用默认列名
                        columns = [f"Col_{i}" for i in range(value.shape[1])] if value.dim() > 1 else ["Value"]
                    
                    # 创建DataFrame
                    if value.dim() == 1:
                        value_2d = value.unsqueeze(1)
                        columns = ["Value"]
                    else:
                        value_2d = value
                    
                    state_mapping_df = pd.DataFrame(
                        value_2d.detach().cpu().numpy(), 
                        index=county_names,
                        columns=columns
                    )
                else:
                    # 处理非tensor数据
                    if key in env.class_mapping:
                        columns = env.class_mapping[key]
                        if hasattr(columns, 'tolist'):
                            columns = columns.tolist()
                        elif hasattr(columns, '__iter__') and not isinstance(columns, str):
                            columns = list(columns)
                        else:
                            columns = [str(columns)]
                    else:
                        columns = None
                    
                    state_mapping_df = pd.DataFrame(value, index=county_names, columns=columns)
                
                # 保存文件
                state_mapping_file = os.path.join(full_output_dir, f"{key}.xlsx")
                state_mapping_df.to_excel(state_mapping_file)
                print(f"状态映射 '{key}' 已保存到: {state_mapping_file}")
                    
            except Exception as e:
                print(f"保存状态映射 '{key}' 时出错: {e}")
                continue
    
    # 3. 减排差距数据
    gap_data = {
        'gap_NO3': env.gap_NO3.squeeze().cpu().numpy(),
        'gap_N_runoff': env.gap_N_runoff.squeeze().cpu().numpy(),
        'gap_NH3': env.gap_NH3.squeeze().cpu().numpy(),
        'gap_CH4': env.gap_CH4.squeeze().cpu().numpy(),
        'gap_N2O': env.gap_N2O.squeeze().cpu().numpy()
    }
    
    gap_df = pd.DataFrame(gap_data, index=county_names)
    gap_file = os.path.join(full_output_dir, "gaps_after_stepwise_techs.xlsx")
    gap_df.to_excel(gap_file)
    print(f"减排差距数据已保存到: {gap_file}")
    
    # 4. 统计摘要
    print("\n=== 各指标统计摘要 ===")
    print("观察状态指标统计:")
    print(state_df.describe())
    
    print("\n减排差距统计:")
    print(gap_df.describe())
    
    # 如果保存了stateMapping，输出状态映射文件列表
    if hasattr(env, 'stateMapping') and env.stateMapping:
        print(f"\n已保存的状态映射文件:")
        for key in env.stateMapping.keys():
            print(f"  - {key}.xlsx")
    
    # 5. 达标情况统计
    no3_met = (env.gap_NO3.squeeze() <= 0).sum().item()
    nh3_met = (env.gap_NH3.squeeze() <= 0).sum().item()
    n_runoff_met = (env.gap_N_runoff.squeeze() <= 0).sum().item()
    ch4_met = (env.gap_CH4.squeeze() <= 0).sum().item()
    n2o_met = (env.gap_N2O.squeeze() <= 0).sum().item()
    
    total_counties = len(county_names)
    
    print(f"\n=== 达标情况统计 (总县数: {total_counties}) ===")
    print(f"NO3达标: {no3_met} 县 ({no3_met/total_counties*100:.1f}%)")
    print(f"NH3达标: {nh3_met} 县 ({nh3_met/total_counties*100:.1f}%)")
    print(f"N_runoff达标: {n_runoff_met} 县 ({n_runoff_met/total_counties*100:.1f}%)")
    print(f"CH4达标: {ch4_met} 县 ({ch4_met/total_counties*100:.1f}%)")
    print(f"N2O达标: {n2o_met} 县 ({n2o_met/total_counties*100:.1f}%)")
    
    # 所有指标都达标的县数量
    all_targets_met = (
        (env.gap_NO3.squeeze() <= 0) & 
        (env.gap_NH3.squeeze() <= 0) & 
        (env.gap_N_runoff.squeeze() <= 0) &
        (env.gap_CH4.squeeze() <= 0) &
        (env.gap_N2O.squeeze() <= 0)
    ).sum().item()
    
    print(f"所有指标都达标: {all_targets_met} 县 ({all_targets_met/total_counties*100:.1f}%)")
    
    # 6. 保存统计摘要
    summary_data = {
        '指标': ['总县数', 'NO3达标县数', 'NH3达标县数', 'N_runoff达标县数', 'CH4达标县数', 'N2O达标县数', '全部达标县数'],
        '数量': [total_counties, no3_met, nh3_met, n_runoff_met, ch4_met, n2o_met, all_targets_met],
        '比例(%)': [100.0, no3_met/total_counties*100, nh3_met/total_counties*100, 
                   n_runoff_met/total_counties*100, ch4_met/total_counties*100, 
                   n2o_met/total_counties*100, all_targets_met/total_counties*100]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(full_output_dir, "达标统计摘要.xlsx")
    summary_df.to_excel(summary_file, index=False)
    print(f"达标统计摘要已保存到: {summary_file}")

    # 7. 保存成本数据
    if hasattr(env, 'save_path') and env.save_path is not None:
        # 确保保存路径存在
        if not os.path.exists(env.save_path):
            os.makedirs(env.save_path, exist_ok=True)
        cost_df = env._save_tech_selected_summary()
        # 将成本数据也保存到output_dir
        cost_df.to_excel(os.path.join(full_output_dir, "cost.xlsx"))
        print(f"成本数据已保存到: {os.path.join(full_output_dir, 'cost.xlsx')}")
    else:
        print("环境未设置save_path，跳过成本数据保存")
    
    return state_df, gap_df, summary_df, cost_df if 'cost_df' in locals() else None

def main():
    """
    主函数
    """
    print("=" * 60)
    print("分步技术优化：逐级应用技术直到达标")
    print("=" * 60)
    
    try:
        # 创建环境配置 - 使用松嫩－三江平原农业区作为示例
        config = GasEnvConfig(
            tech_amount_constraint=30,
            Reward_priority=[0.7, 0.5, 0.3, 0.2],
            county_df_path='data/基础数据-县级尺度.xlsx',
            IDs_df='data/县市亚区.xlsx',
            livestock_tech_path='data/畜牧业技术列单-经济产量0803.xlsx',
            crop_tech_path="data/种植业技术列单产量产业0803.xlsx",
            soc_df="data/SOC-县尺度.xlsx",
            livestock_scale="data/动物数量.xlsx",
            crop_scale="data/分县种植面积.xlsx",
            local_target_penalty_factor=50.0,  # 新增：为配置添加惩罚因子
            linear_result_path='results/linear_optimization_results_by_county_5gases_hard_target.xlsx',
            only_lp_phase=True,  # 启用线性规划约束模式
            save_path="temp_yield_output"  # 临时路径，用于启用产量追踪
        )
        print("环境配置创建成功")
        
        # 创建环境
        env = GasEnv(config)
        print(f"环境创建成功，包含 {env.numCounty} 个县，{env.numTech} 种技术")
        
        # 重置环境
        env.reset()
        print("环境已重置到初始状态")
        
        # 输出初始状态信息
        print(f"需要技术添加的县数量: {env.counties_need_tech.sum()}")
        
        # 执行分步技术优化
        optimization_stats = stepwise_tech_optimization(env)
        
        # 输出最终状态
        state_df, gap_df, summary_df, cost_df = output_state_summary(env, output_dir="results", suffix="stepwise_techs")
        
        # 保存优化统计数据
        stats_data = []
        for step, stats in optimization_stats.items():
            stats_data.append({
                '优化步骤': step.replace('step_', '第') + '步',
                '最大技术分级': f"≤{stats['max_level']}级",
                '应用技术数量': stats['applied_techs'],
                '执行动作数': stats['actions'],
                '达标县数': stats['counties_met']
            })
        
        stats_df = pd.DataFrame(stats_data)
        stats_file = os.path.join("results/stepwise_techs", "optimization_stats.xlsx")
        stats_df.to_excel(stats_file, index=False)
        print(f"优化统计数据已保存到: {stats_file}")
        
        # 清理临时目录
        import shutil
        if os.path.exists("temp_yield_output"):
            shutil.rmtree("temp_yield_output")
            print("临时产量输出目录已清理")
        
        print("\n" + "=" * 60)
        print("分步技术优化完成！所有结果已保存到 'results/stepwise_techs' 目录中")
        print("包含的文件：")
        print("  - state_after_stepwise_techs.xlsx: 观察状态数据")
        print("  - gaps_after_stepwise_techs.xlsx: 减排差距数据")  
        print("  - 达标统计摘要.xlsx: 达标情况统计")
        print("  - optimization_stats.xlsx: 分步优化统计")
        print("  - cost.xlsx: 成本数据")
        print("  - 各种状态映射文件 (按key分别保存)")
        print("=" * 60)
        
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()