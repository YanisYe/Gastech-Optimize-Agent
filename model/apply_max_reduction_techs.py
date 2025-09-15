#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分步等级优化：按技术等级分步选择技术组合，每等级内按等级、经济成本、减排量排序

优化逻辑：
1. 使用环境中的技术影响机制计算每个技术的净减排量
2. 技术影响公式：g_after = g_before * (1 + delta)
3. 计算步骤：
   a. 对每个受影响的子产业，计算所有县的总基础排放量
   b. 净减排量 = sum(总基础排放量 * delta) for all affected sub-industries
   c. delta < 0: 减排贡献（正值），delta > 0: 增排贡献（负值）
4. 分步选择策略：
   a. 第一步：为所有县应用≤1级技术包（按等级、经济成本、减排量排序）
   b. 第二步：为仍未达标县重新应用≤2级技术包（按等级、经济成本、减排量排序）
   c. 第三步：为仍未达标县重新应用≤3级技术包（按等级、经济成本、减排量排序）
5. 冲突处理：同一产业同一冲突组的技术，按等级、经济成本、减排量排序选择最优
6. 输出每个技术包包含的技术和每个县使用的技术包分配
"""

import os
import sys
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import functools
import time
import logging

# 添加model目录到路径
model_path = Path(__file__).parent / "model"
sys.path.append(str(model_path))

from GasEnviroment_curriculum_learning import GasEnv, GasEnvConfig

# 配置全局日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 默认输出到控制台
    ]
)
logger = logging.getLogger(__name__)



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

    # 处理NaN值：NaN表示无冲突
    if pd.isna(conflict):
        # 如果目标技术无冲突，则返回False序列（没有冲突的技术）
        return pd.Series(False, index=techSet.index)

    # 找到同一产业且同一冲突组的技术
    condition = (
        (techSet['class'] == industry) &
        (techSet['技术间的冲突'] == conflict)
    )
    
    return condition

def calculate_total_emissions_needing_reduction(env):
    """
    计算所有需要减排的县的三种气体总排放量（使用环境中的Total_*_origin数据）

    Args:
        env: 环境实例

    Returns:
        dict: 包含NH3、NO3、N_runoff总排放量的字典
    """
    # 获取需要技术的县索引
    counties_needing_tech = np.where(env.counties_need_tech)[0]

    # 使用环境中的总排放量数据
    total_emissions = {
        'NH3': 0.0,
        'NO3': 0.0,
        'N_runoff': 0.0
    }

    for county_idx in counties_needing_tech:
        # 使用环境中的Total_*_origin数据，这些是每个县的总排放量
        # Total_NH3_origin = 种植业NH3 + 粪便管理NH3 + 粪肥施用NH3
        # Total_NO3_origin = 氮肥NO3 + 粪肥施用NO3
        # Total_N_runoff_origin = N流失
        if hasattr(env, 'Total_NH3_origin') and env.Total_NH3_origin is not None:
            total_emissions['NH3'] += env.Total_NH3_origin[county_idx].item()

        if hasattr(env, 'Total_NO3_origin') and env.Total_NO3_origin is not None:
            total_emissions['NO3'] += env.Total_NO3_origin[county_idx].item()

        if hasattr(env, 'Total_N_runoff_origin') and env.Total_N_runoff_origin is not None:
            total_emissions['N_runoff'] += env.Total_N_runoff_origin[county_idx].item()

    logger.info(f"需要减排的县总数: {len(counties_needing_tech)}")
    logger.info(f"总排放量 - NH3: {total_emissions['NH3']:.4f}, NO3: {total_emissions['NO3']:.4f}, N_runoff: {total_emissions['N_runoff']:.4f}")

    return total_emissions

def _calculate_tech_impacts_base(tech_row, env):
    """
    计算单个技术对所有子产业的影响的基础函数
    返回每个受影响子产业的详细变化信息

    Args:
        tech_row: 技术数据行
        env: 环境实例

    Returns:
        list: 包含(industry, delta, sub, total_change)的元组列表
    """
    impacts = []

    try:
        # 获取技术ID
        tech_id = tech_row.name  # 假设tech_row的索引就是技术ID

        # 获取技术详情
        line = env._get_line(tech_id)
        class_name = 'crop' if 'crop' in line.index[4].lower() else 'livestock'
        line = line[5:]

        # 获取技术影响的环节（复用环境的方法）
        deltas = env._get_delta(line, class_name)

        # 扩展技术子产业（复用环境的方法）
        deltas = env.expand_tech_subindustries(deltas, relaxed=False)

        # 遍历所有受影响的子产业
        for industry, delta, subIndustry in deltas:
            # 获取技术影响的环节的当前状态
            if industry in env.stateMapping:
                countyState = env.stateMapping[industry]
                subindustry_classes = env.class_mapping[industry]

                # 遍历技术影响的环节的子环节
                for sub in subIndustry:
                    # 获取子环节的索引
                    col = env._get_or_create_industry_mapping(sub, subindustry_classes)

                    # 计算该子产业在所有需要减排的县的总基础排放量
                    total_before_value = 0.0
                    for county_idx in np.where(env.counties_need_tech)[0]:
                        try:
                            before_value = float(countyState[county_idx, col].item())
                            total_before_value += before_value
                        except (IndexError, AttributeError):
                            continue

                    # 使用正确的公式：总变化量 = 总基础排放量 * delta
                    if total_before_value > 0:  # 只计算有排放的子产业
                        total_change = total_before_value * delta
                        impacts.append((industry, delta, sub, total_change))

    except Exception as e:
        logger.warning(f"计算技术影响基础数据失败: {e}")

    return impacts

def calculate_tech_comprehensive_impacts(tech_row, env):
    """
    统一计算单个技术的综合影响：净减排量 + 各种气体和产量的影响

    Args:
        tech_row: 技术数据行
        env: 环境实例

    Returns:
        dict: 包含净减排量和各种气体产量影响的字典
    """
    # 初始化结果字典
    impacts = {
        'net_reduction': 0.0,  # 净减排量
        'NH3_reduction': 0.0,
        'NO3_reduction': 0.0,
        'N_runoff_reduction': 0.0,
        'CH4_reduction': 0.0,
        'N2O_reduction': 0.0,
        'SOC_reduction': 0.0,
        'yield_change': 0.0
    }

    try:
        # 使用基础函数获取所有子产业的影响
        impacts_base = _calculate_tech_impacts_base(tech_row, env)

        # 根据子产业名称分类气体影响
        for industry, delta, sub, total_change in impacts_base:
            sub_lower = sub.lower()

            # 累加净减排量（排除某些特定的子产业）
            if not any(x.lower() in sub_lower for x in ['organic_carbon', 'CH4', 'N2O', 'Yield']):
                impacts['net_reduction'] += total_change

            # 分类累加各种气体和产量的影响
            if 'nh3' in sub_lower:
                impacts['NH3_reduction'] += total_change
            elif 'no3' in sub_lower:
                impacts['NO3_reduction'] += total_change
            elif ('n_runoff' in sub_lower) or ('runoff' in sub_lower and 'n' in sub_lower):
                impacts['N_runoff_reduction'] += total_change
            elif 'ch4' in sub_lower and 'intestine' in sub_lower:
                impacts['CH4_reduction'] += total_change
            elif 'ch4' in sub_lower and 'fecal' in sub_lower:
                impacts['CH4_reduction'] += total_change
            elif 'ch4' in sub_lower and 'rice' in sub_lower:
                impacts['CH4_reduction'] += total_change
            elif 'n2o' in sub_lower:
                impacts['N2O_reduction'] += total_change
            elif 'organic_carbon' in sub_lower or 'soc' in sub_lower:
                impacts['SOC_reduction'] += total_change
            elif 'yield' in sub_lower:
                impacts['yield_change'] += total_change

    except Exception as e:
        logger.warning(f"计算技术综合影响失败: {e}")

    return impacts

def select_optimal_techs_by_level_and_reduction(env, max_level):
    """
    根据技术等级、经济成本和净减排量排序选择最优技术组合，考虑技术冲突
    排序优先级：技术等级升序 > 经济成本升序 > 净减排量降序（负值越小越优先）
    使用环境中的技术影响机制：g_after = g_before * (1 + delta)

    Args:
        env: 环境实例
        max_level: 最大技术等级 (1, 2, 3)

    Returns:
        tuple: (选中的技术ID列表, 技术包信息字典)
    """
    tech_set = env.tech_set

    # 筛选指定等级内的技术
    available_techs = tech_set[tech_set['技术分级'] <= max_level].copy()
    if len(available_techs) == 0:
        logger.warning(f"没有找到≤{max_level}级的技术")
        return [], {}

    logger.info(f"开始选择≤{max_level}级技术，最优技术组合...")

    # 计算需要减排的县的排放总量（用于信息显示）
    total_emissions = calculate_total_emissions_needing_reduction(env)

    # 串行计算所有可用技术的净减排量
    logger.info(f"计算≤{max_level}级技术({len(available_techs)}个)的净减排量...")
    start_time = time.time()
    tech_reductions = []
    for idx, tech_row in available_techs.iterrows():
        impacts = calculate_tech_comprehensive_impacts(tech_row, env)
        net_reduction = impacts['net_reduction']
        tech_reductions.append((idx, net_reduction, tech_row))
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(".2f")

    # 按技术等级、经济成本和净减排量排序：等级升序 > 经济成本升序 > 减排量升序
    tech_reductions.sort(key=lambda x: (x[2]['技术分级'], x[2]['经济成本'], x[1]))

    logger.info(f"计算了 {len(tech_reductions)} 个≤{max_level}级技术的净减排量")
    logger.info("前10个经济成本最低的技术:")
    for i, (idx, net_reduction, row) in enumerate(tech_reductions[:10]):
        reduction_type = "净减排" if net_reduction <= 0 else "净增排"
        value = abs(net_reduction)
        logger.info(f"  {i+1}. {row['Mitigation strategy']} (等级:{row['技术分级']}, 成本:{row['经济成本']:.2f}, {reduction_type}:{value:.4f})")

    selected_techs = []
    conflict_groups_used = set()  # 记录已使用的冲突组

    for tech_idx, net_reduction, tech_row in tech_reductions:
        # 检查技术是否有冲突
        conflict_value = tech_row['技术间的冲突']

        # 如果技术没有冲突（NaN），直接采用
        if pd.isna(conflict_value):
            selected_techs.append(tech_idx)
            reduction_type = "净减排" if net_reduction <= 0 else "净增排"
            logger.info(f"选择技术: {tech_row['Mitigation strategy']}, 种类: {tech_row['Crop species'] if tech_row['class'] == 'crop' else tech_row['Livestock species']} (等级:{tech_row['技术分级']}, {reduction_type}:{abs(net_reduction):.4f}, 成本:{tech_row['经济成本']:.2f}) - 无冲突")
            continue

        # 如果有冲突，进行冲突检测
        conflict_key = (tech_row['class'], conflict_value)

        if conflict_key in conflict_groups_used:
            # 如果冲突组已被使用，检查当前技术是否更优
            existing_tech_idx = None
            for selected_idx in selected_techs:
                selected_row = available_techs.loc[selected_idx] if selected_idx in available_techs.index else tech_set.iloc[selected_idx]
                selected_conflict = selected_row['技术间的冲突']
                if (selected_row['class'] == tech_row['class'] and
                    selected_conflict == conflict_value):
                    existing_tech_idx = selected_idx
                    break

            if existing_tech_idx is not None:
                existing_row = available_techs.loc[existing_tech_idx] if existing_tech_idx in available_techs.index else tech_set.iloc[existing_tech_idx]
                existing_impacts = calculate_tech_comprehensive_impacts(existing_row, env)
                existing_net_reduction = existing_impacts['net_reduction']

                # 比较：等级升序 > 经济成本升序 > 净减排量降序（负值越小越优先）
                current_priority = (tech_row['技术分级'], tech_row['经济成本'], net_reduction)
                existing_priority = (existing_row['技术分级'], existing_row['经济成本'], existing_net_reduction)

                if current_priority < existing_priority:
                    # 当前技术更优，替换现有技术
                    selected_techs.remove(existing_tech_idx)
                    selected_techs.append(tech_idx)
                    reduction_type_existing = "净减排" if existing_net_reduction <= 0 else "净增排"
                    reduction_type_current = "净减排" if net_reduction <= 0 else "净增排"
                    logger.info(f"替换技术: {existing_row['Mitigation strategy']} (等级:{existing_row['技术分级']}, {reduction_type_existing}:{abs(existing_net_reduction):.4f}) -> {tech_row['Mitigation strategy']} (等级:{tech_row['技术分级']}, {reduction_type_current}:{abs(net_reduction):.4f})")
                # else: 保持现有技术
            continue
        else:
            # 新的冲突组，直接添加
            selected_techs.append(tech_idx)
            conflict_groups_used.add(conflict_key)
            reduction_type = "净减排" if net_reduction <= 0 else "净增排"
            logger.info(f"选择技术: {tech_row['Mitigation strategy']}, 种类: {tech_row['Crop species'] if tech_row['class'] == 'crop' else tech_row['Livestock species']} (等级:{tech_row['技术分级']}, {reduction_type}:{abs(net_reduction):.4f}, 成本:{tech_row['经济成本']:.2f})")

    # 统计各分级技术数量
    level_counts = {}
    for tech_idx in selected_techs:
        row = available_techs.loc[tech_idx] if tech_idx in available_techs.index else tech_set.iloc[tech_idx]
        level = row['技术分级']
        level_counts[level] = level_counts.get(level, 0) + 1

    logger.info(f"最终选择了 {len(selected_techs)} 个≤{max_level}级最优技术，分级分布: {level_counts}")

    # 创建技术包信息
    tech_package_info = {
        'max_level': max_level,
        'selected_techs': selected_techs,
        'level_counts': level_counts,
        'tech_details': []
    }

    for tech_idx in selected_techs:
        row = available_techs.loc[tech_idx] if tech_idx in available_techs.index else tech_set.iloc[tech_idx]

        # 使用统一函数计算所有影响
        comprehensive_impacts = calculate_tech_comprehensive_impacts(row, env)
        net_reduction = comprehensive_impacts['net_reduction']
        reduction_type = "净减排" if net_reduction <= 0 else "净增排"

        tech_package_info['tech_details'].append({
            'tech_idx': tech_idx,
            'name': row['Mitigation strategy'],
            'level': row['技术分级'],
            'net_reduction': net_reduction,
            'reduction_type': reduction_type,
            'cost': row['经济成本'],
            'class': row['class'],
            'species': row['Livestock species'] if row['class'] != 'crop' else row['Crop species'],
            'conflict': '' if pd.isna(row['技术间的冲突']) else row['技术间的冲突'],
            # 添加气体和产量影响信息
            'NH3_reduction': comprehensive_impacts['NH3_reduction'],
            'NO3_reduction': comprehensive_impacts['NO3_reduction'],
            'N_runoff_reduction': comprehensive_impacts['N_runoff_reduction'],
            'CH4_reduction': comprehensive_impacts['CH4_reduction'],
            'N2O_reduction': comprehensive_impacts['N2O_reduction'],
            'SOC_reduction': comprehensive_impacts['SOC_reduction'],
            'yield_change': comprehensive_impacts['yield_change']
        })

    return selected_techs, tech_package_info

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



def apply_techs_to_county_batch(county_batch, env, selected_tech_ids):
    """
    为一批县应用选定的技术组合

    Args:
        county_batch: 县批次列表 [(batch_idx, county_idx), ...]
        env: 环境实例
        selected_tech_ids: 选定的技术ID列表

    Returns:
        dict: 该批次的统计结果
    """
    batch_actions = 0
    batch_applied_techs = 0
    batch_results = []

    for batch_idx, county_idx in county_batch:
        county_name = env.IDs['Counties'].iloc[county_idx]
        county_applied_techs = 0

        # 应用选中的技术
        for tech_idx in selected_tech_ids:
            # 检查技术是否已经应用过
            if env.state['Tech_selected'][county_idx, tech_idx] == 0:
                # 编码动作：需要将county_idx转换为在counties_need_tech中的索引
                all_counties_need_tech = np.where(env.counties_need_tech)[0]
                county_need_tech_pos = np.where(all_counties_need_tech == county_idx)[0][0]
                action = county_need_tech_pos * env.numTech + tech_idx

                try:
                    state, reward, terminated, truncated, info = env.step(action)
                    batch_actions += 1
                    batch_applied_techs += 1
                    county_applied_techs += 1
                except Exception as e:
                    logger.error(f"县 {county_name} 应用技术 {tech_idx} 时出错: {e}")
                    continue

        batch_results.append({
            'county_idx': county_idx,
            'county_name': county_name,
            'applied_techs': county_applied_techs,
            'batch_idx': batch_idx
        })

    return {
        'batch_actions': batch_actions,
        'batch_applied_techs': batch_applied_techs,
        'batch_results': batch_results
    }

def apply_techs_to_counties_parallel(env, selected_tech_ids, target_counties=None, num_workers=None):
    """
    使用多线程并行地为县应用选定的技术组合

    Args:
        env: 环境实例
        selected_tech_ids: 选定的技术ID列表
        target_counties: 目标县索引列表，如果为None则应用到所有需要技术的县
        num_workers: 工作线程数，默认自动选择

    Returns:
        tuple: (应用的技术数量, 总执行动作数)
    """
    if not selected_tech_ids:
        logger.warning("没有技术需要应用，跳过")
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

    if len(counties_need_tech_indices) == 0:
        logger.warning("没有需要技术的县，跳过")
        return 0, 0

    logger.info(f"开始并行应用技术到 {len(counties_need_tech_indices)} 个县...")

    # 设置工作线程数
    if num_workers is None:
        num_workers = min(cpu_count(), 24)  # 最多使用8个线程

    # 将县分成多个批次
    county_list = list(enumerate(counties_need_tech_indices))
    batch_size = max(1, len(county_list) // num_workers)
    county_batches = [county_list[i:i + batch_size] for i in range(0, len(county_list), batch_size)]

    logger.info(f"使用 {num_workers} 个线程并行处理 {len(county_list)} 个县")
    logger.info(f"每个线程处理约 {batch_size} 个县")

    start_time = time.time()

    # 创建一个线程安全的计数器来跟踪总进度
    total_actions = 0
    total_applied_techs = 0
    completed_batches = 0

    # 使用线程池并行处理县批次
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有批次任务
        future_to_batch = {
            executor.submit(apply_techs_to_county_batch, batch, env, selected_tech_ids): batch_idx
            for batch_idx, batch in enumerate(county_batches)
        }

        # 收集结果
        for future in as_completed(future_to_batch):
            try:
                batch_result = future.result()
                total_actions += batch_result['batch_actions']
                total_applied_techs += batch_result['batch_applied_techs']
                completed_batches += 1

                # 显示批次完成信息
                batch_idx = future_to_batch[future]
                batch_counties = len(batch_result['batch_results'])
                logger.info(f"完成批次 {completed_batches}/{len(county_batches)} ({batch_counties} 个县, {batch_result['batch_applied_techs']} 个技术)")

            except Exception as e:
                logger.error(f"批次处理失败: {e}")
                completed_batches += 1

    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info("并行技术应用完成！")
    logger.info(f"总共应用了 {total_applied_techs} 个技术")
    logger.info(f"总执行动作数: {total_actions}")
    logger.info(".2f")

    return total_applied_techs, total_actions

def apply_tech_package_to_counties(env, target_counties, selected_tech_ids, tech_package_info, num_workers=None, collect_impacts=True):
    """
    为指定县应用技术包，并记录县的技术包分配和每个技术对每个县的影响

    Args:
        env: 环境实例
        target_counties: 目标县索引列表
        selected_tech_ids: 技术包中的技术ID列表
        tech_package_info: 技术包信息字典
        num_workers: 并行应用技术的线程数，默认自动选择
        collect_impacts: 是否收集技术影响数据，默认True

    Returns:
        tuple: (应用的技术数量, 总执行动作数, 县的技术包分配字典, 技术影响数据字典)
    """
    logger.info(f"开始应用≤{tech_package_info['max_level']}级技术包到 {len(target_counties)} 个县...")

    # 初始化县的技术包分配记录
    county_package_assignment = {}

    # 初始化技术影响数据记录
    tech_county_impacts = {
        'step_level': tech_package_info['max_level'],
        'applied_techs': selected_tech_ids,
        'county_impacts': {}  # county_idx -> {tech_idx -> impact_data}
    } if collect_impacts else None

    if not selected_tech_ids:
        logger.warning("技术包为空，跳过应用")
        return 0, 0, county_package_assignment, tech_county_impacts

    # 确定目标县（只处理指定的县）
    target_counties = np.array(target_counties)

    logger.info(f"目标县数量: {len(target_counties)}")

    # 初始化变量
    total_actions = 0
    applied_techs = 0


    # 使用串行方式
    logger.info("使用串行方式应用技术...")
    total_actions = 0
    applied_techs = 0

    for i, county_idx in enumerate(target_counties):
        county_name = env.IDs['Counties'].iloc[county_idx]
        logger.info(f"处理第 {i+1}/{len(target_counties)} 个县: {county_name}")

        # 记录该县使用的是哪个技术包
        county_package_assignment[county_idx] = tech_package_info['max_level']

        # 初始化该县的影响数据
        if collect_impacts and county_idx not in tech_county_impacts['county_impacts']:
            tech_county_impacts['county_impacts'][county_idx] = {
                'county_name': county_name,
                'applied_techs': [],
                'total_impacts': {
                    'NH3_change': 0.0,
                    'NO3_change': 0.0,
                    'N_runoff_change': 0.0,
                    'CH4_change': 0.0,
                    'N2O_change': 0.0,
                    'SOC_change': 0.0,
                    'yield_change': 0.0,
                    'net_reduction': 0.0
                }
            }

        # 如果需要收集影响数据，记录应用技术前的初始状态
        initial_state = {}
        if collect_impacts and hasattr(env, 'state') and env.state:
            for key, value in env.state.items():
                if key != 'Tech_selected':
                    # 对于二维状态，提取该县的值
                    initial_state[key] = float(value[county_idx].item())
            initial_state['yield'] = float(env.stateMapping['畜牧业产量'][county_idx].sum().item()) + float(env.stateMapping['种植业产量'][county_idx].sum().item())

        # 应用选中的技术
        county_applied_techs = 0
        for tech_idx in selected_tech_ids:
            # 检查技术是否已经应用过
            if env.state['Tech_selected'][county_idx, tech_idx] == 0:
                action = county_idx * env.numTech + tech_idx


                state, reward, terminated, truncated, info = env.step(action)
                total_actions += 1
                applied_techs += 1
                county_applied_techs += 1

                # 如果需要收集影响数据，则从返回的state中提取影响
                if collect_impacts:
                    tech_row = env.tech_set.iloc[tech_idx]
                    tech_name = tech_row['Mitigation strategy']

                    # 从state中提取技术应用后的气体指标
                    impact_data = {
                        'county_name': county_name,
                        'NH3_change': 0.0,
                        'NO3_change': 0.0,
                        'N_runoff_change': 0.0,
                        'CH4_change': 0.0,
                        'N2O_change': 0.0,
                        'SOC_change': 0.0,
                        'yield_change': 0.0,
                        'net_reduction': 0.0
                    }

                    # 计算技术应用后的状态变化
                    for key, value in state.items():
                        if key == 'Tech_selected':
                            continue
                        # 根据状态key分类气体影响
                        key_lower = key.lower()
                        # 对于二维状态，提取该县的当前值
                        current_value = float(value[county_idx].item())

                        # 计算变化量（当前值减去初始值）
                        initial_value = initial_state.get(key, 0.0)
                        change = current_value - initial_value
                        # 只记录有变化的状态，避免累加0值
                        if change != 0:
                            if 'nh3' in key_lower:
                                impact_data['NH3_change'] += change
                                impact_data['net_reduction'] += change
                            
                            elif 'no3' in key_lower:
                                impact_data['NO3_change'] += change
                                impact_data['net_reduction'] += change
                                
                            elif ('n_runoff' in key_lower):
                                impact_data['N_runoff_change'] += change
                                impact_data['net_reduction'] += change

                            elif 'ch4' in key_lower:
                                impact_data['CH4_change'] += change
                            elif 'n2o' in key_lower:
                                impact_data['N2O_change'] += change
                            elif 'organic' in key_lower or 'soc' in key_lower:
                                impact_data['SOC_change'] += change

                    tech_line = env._get_line(tech_idx)
                    if 'Yield' in tech_line:
                        change = float(env.stateMapping['畜牧业产量'][county_idx].sum().item()) + float(env.stateMapping['种植业产量'][county_idx].sum().item()) - initial_state['yield']
                        impact_data['yield_change'] += change

                    # 保存技术影响数据
                    tech_county_impacts['county_impacts'][county_idx]['applied_techs'].append({
                        'tech_idx': tech_idx,
                        'tech_name': tech_name,
                        'impacts': impact_data
                    })

                    # for key, value in impact_data.items():
                    #     if value != 0:
                    #         logger.info(f"{key}: {value}")

                    # 累加到县的总影响
                    for key in tech_county_impacts['county_impacts'][county_idx]['total_impacts']:
                        tech_county_impacts['county_impacts'][county_idx]['total_impacts'][key] += impact_data[key]

                    # 更新initial state，使其反映应用技术后的当前状态
                    # 这样下一次应用技术时会基于前一个技术的最终状态计算影响
                    for key, value in state.items():
                        if key != 'Tech_selected':
                            # 更新该县的初始状态为应用技术后的状态
                            initial_state[key] = float(value[county_idx].item())
                    initial_state['yield'] = float(env.stateMapping['畜牧业产量'][county_idx].sum().item()) + float(env.stateMapping['种植业产量'][county_idx].sum().item())
        
        logger.info(f"县 {county_name} 实际应用了 {county_applied_techs} 个技术")

        # 每处理10个县打印一次进度
        if (i + 1) % 10 == 0:
            logger.info(f"已处理 {i+1} 个县，总共应用了 {applied_techs} 个技术")

        logger.info("串行技术应用完成！")
        logger.info(f"总共应用了 {applied_techs} 个技术")
        logger.info(f"总执行动作数: {total_actions}")

    return applied_techs, total_actions, county_package_assignment, tech_county_impacts

def save_tech_package_to_excel(tech_package, step_name, output_dir="results", suffix="level_based_stepwise_techs"):
    """
    立即保存单个技术包信息到Excel文件

    Args:
        tech_package: 技术包信息字典
        step_name: 步骤名称 (如 'step_1')
        output_dir: 输出目录
        suffix: 文件名后缀
    """
    if not tech_package or 'tech_details' not in tech_package:
        return

    # 创建输出目录
    full_output_dir = os.path.join(output_dir, suffix)
    os.makedirs(full_output_dir, exist_ok=True)

    # 准备技术包数据
    tech_package_data = []
    for tech_detail in tech_package['tech_details']:
        tech_package_data.append({
            '步骤': step_name.upper(),
            '最大等级': tech_package.get('max_level', 0),
            '技术ID': tech_detail['tech_idx'],
            '技术名称': tech_detail['name'],
            '技术等级': tech_detail['level'],
            '净减排量': tech_detail['net_reduction'],
            '减排类型': tech_detail['reduction_type'],
            '经济成本': tech_detail['cost'],
            '产业类型': tech_detail['class'],
            '物种': tech_detail.get('species', ''),
            '冲突组': '' if pd.isna(tech_detail['conflict']) else tech_detail['conflict'],
            # 添加气体和产量影响信息
            'NH3减排量': tech_detail.get('NH3_reduction', 0.0),
            'NO3减排量': tech_detail.get('NO3_reduction', 0.0),
            'N_runoff减排量': tech_detail.get('N_runoff_reduction', 0.0),
            'CH4减排量': tech_detail.get('CH4_reduction', 0.0),
            'N2O减排量': tech_detail.get('N2O_reduction', 0.0),
            'SOC减排量': tech_detail.get('SOC_reduction', 0.0),
            '产量变化': tech_detail.get('yield_change', 0.0)
        })

    # 保存到Excel
    if tech_package_data:
        tech_package_df = pd.DataFrame(tech_package_data)
        tech_package_file = os.path.join(full_output_dir, f"{step_name}_tech_package.xlsx")
        tech_package_df.to_excel(tech_package_file, index=False)
        logger.info(f"{step_name.upper()} 技术包信息已保存到: {tech_package_file}")

def stepwise_level_based_tech_optimization(env_config, use_parallel=False, num_workers=None):
    """
    分步等级优化：按技术等级分步优化，先1级，再1+2级，最后1+2+3级
    每个等级内按等级、经济成本、减排量排序，考虑技术冲突，根据达标情况应用合适的技术包

    Args:
        env: 环境实例
        env_config: 环境配置对象，用于创建新的环境实例
        use_parallel: 是否使用并行应用技术，默认False
        num_workers: 并行应用技术的线程数，默认自动选择

    Returns:
        dict: 优化统计和县的技术包分配信息
    """
    logger.info("=" * 60)
    logger.info("开始分步等级优化（按技术等级分步 + 等级、经济成本、减排量排序）")
    logger.info("=" * 60)

    stats = {
        'step_1': {'applied_techs': 0, 'actions': 0, 'counties_met': 0, 'max_level': 1, 'tech_package': {}},
        'step_2': {'applied_techs': 0, 'actions': 0, 'counties_met': 0, 'max_level': 2, 'tech_package': {}},
        'step_3': {'applied_techs': 0, 'actions': 0, 'counties_met': 0, 'max_level': 3, 'tech_package': {}}
    }

    # 记录所有县的技术包分配
    all_county_assignments = {}

    # 第一步：为所有需要技术的县应用≤1级技术包
    logger.info(f"\n" + "="*50)
    env_step1 = GasEnv(env_config)
    env_step1.reset()
    logger.info("第一步：为所有需要减排的县应用≤1级技术包")
    logger.info("="*50)

    total_counties = env_step1.numCounty
    counties_need_tech_indices = np.where(env_step1.counties_need_tech)[0]
    logger.info(f"总县数: {total_counties}, 需要减排的县数: {len(counties_need_tech_indices)}")

    if len(counties_need_tech_indices) > 0:
        selected_tech_ids_1, tech_package_1 = select_optimal_techs_by_level_and_reduction(env_step1, max_level=1)
        stats['step_1']['tech_package'] = tech_package_1

        if selected_tech_ids_1:
            applied_techs_1, actions_1, assignments_1, _ = apply_tech_package_to_counties(
                env_step1, counties_need_tech_indices, selected_tech_ids_1, tech_package_1,
                num_workers=num_workers, collect_impacts=False
            )
            stats['step_1']['applied_techs'] = applied_techs_1
            stats['step_1']['actions'] = actions_1
            all_county_assignments.update(assignments_1)
        else:
            logger.warning("没有找到1级技术")

    # 检查第一步后的达标情况
    counties_met_after_step1 = check_counties_meeting_targets(env_step1)
    counties_met_count_1 = counties_met_after_step1.sum()
    stats['step_1']['counties_met'] = counties_met_count_1

    logger.info(f"第一步应用后达标情况:")
    logger.info(f"达标县数: {counties_met_count_1}/{total_counties} ({counties_met_count_1/total_counties*100:.1f}%)")

    # 第一步完成后立即保存技术包信息
    if stats['step_1']['tech_package']:
        save_tech_package_to_excel(stats['step_1']['tech_package'], 'step_1')
        logger.info("第一步技术包已保存完成")

    # 第二步：为未达标县应用≤2级技术包
    counties_not_met_after_step1 = np.where(~counties_met_after_step1)[0]
    if len(counties_not_met_after_step1) > 0:
        logger.info(f"\n" + "="*50)
        logger.info(f"第二步：为 {len(counties_not_met_after_step1)} 个未达标县重新应用≤2级技术包")
        logger.info("="*50)

        # 为第二步创建新的环境实例，确保干净的状态
        logger.info("为第二步创建新的环境实例...")
        env_step2 = GasEnv(env_config)
        env_step2.reset()
        logger.info("第二步环境实例创建完成")

        selected_tech_ids_2, tech_package_2 = select_optimal_techs_by_level_and_reduction(env_step2, max_level=2)
        stats['step_2']['tech_package'] = tech_package_2

        if selected_tech_ids_2:
            applied_techs_2, actions_2, assignments_2, _ = apply_tech_package_to_counties(
                env_step2, counties_not_met_after_step1, selected_tech_ids_2, tech_package_2,
                num_workers=num_workers, collect_impacts=False
            )
            stats['step_2']['applied_techs'] = applied_techs_2
            stats['step_2']['actions'] = actions_2
            all_county_assignments.update(assignments_2)

        # 检查第二步后的达标情况
        counties_met_after_step2 = check_counties_meeting_targets(env_step2)
        counties_met_count_2 = counties_met_after_step2.sum()
        stats['step_2']['counties_met'] = counties_met_count_2

        # 计算第一级技术包符合要求的县与1，2技术包的县的并集
        # 即所有在第一步或第二步后达标的县
        counties_met_union = np.logical_or(counties_met_after_step1, counties_met_after_step2)
        counties_met_union_count = counties_met_union.sum()

        logger.info(f"第一级技术包符合要求的县与1，2技术包的县的并集:")
        logger.info(f"并集达标县数: {counties_met_union_count}/{total_counties} ({counties_met_union_count/total_counties*100:.1f}%)")
        logger.info(f"第一步达标县数: {counties_met_count_1}")
        logger.info(f"第二步达标县数: {counties_met_count_2}")
        logger.info(f"并集新增县数: {counties_met_union_count - counties_met_count_1}")

        # 将并集结果也保存到统计中
        stats['step_2']['counties_met_union'] = counties_met_union_count

        logger.info(f"第二步应用后达标情况:")
        logger.info(f"达标县数: {counties_met_count_2}/{total_counties} ({counties_met_count_2/total_counties*100:.1f}%)")
        logger.info(f"新增达标县数: {counties_met_count_2 - counties_met_count_1}")

        # 第二步完成后立即保存技术包信息
        if stats['step_2']['tech_package']:
            save_tech_package_to_excel(stats['step_2']['tech_package'], 'step_2')
            logger.info("第二步技术包已保存完成")

        # 第三步：为不在并集的不满足目标的县应用≤3级技术包
        # 计算不在并集中的县（即仍未达标的县）
        counties_not_in_union = np.where(~counties_met_union)[0]
        if len(counties_not_in_union) > 0:
            logger.info(f"\n" + "="*50)
            logger.info(f"第三步：为 {len(counties_not_in_union)} 个不在并集的不满足目标的县重新应用≤3级技术包")
            logger.info(f"并集达标县数: {counties_met_union_count}, 仍需第三步处理的县数: {len(counties_not_in_union)}")
            logger.info("="*50)

            # 为第三步创建新的环境实例，确保干净的状态
            logger.info("为第三步创建新的环境实例...")
            env_step3 = GasEnv(env_config)
            env_step3.reset()
            logger.info("第三步环境实例创建完成")

            selected_tech_ids_3, tech_package_3 = select_optimal_techs_by_level_and_reduction(env_step3, max_level=3)
            stats['step_3']['tech_package'] = tech_package_3

            if selected_tech_ids_3:
                applied_techs_3, actions_3, assignments_3, _ = apply_tech_package_to_counties(
                    env_step3, counties_not_in_union, selected_tech_ids_3, tech_package_3,
                    num_workers=num_workers, collect_impacts=False
                )
                stats['step_3']['applied_techs'] = applied_techs_3
                stats['step_3']['actions'] = actions_3
                all_county_assignments.update(assignments_3)

            # 检查第三步后的达标情况
            counties_met_after_step3 = check_counties_meeting_targets(env_step3)
            counties_met_count_3 = counties_met_after_step3.sum()
            stats['step_3']['counties_met'] = counties_met_count_3

            # 计算三个技术包的并集（第一级、1-2级、1-3级技术包的并集）
            # 三个技术包的并集 = 第一级达标 ∪ 1-2级达标 ∪ 1-3级达标
            counties_met_three_tech_packages = np.logical_or(
                np.logical_or(counties_met_after_step1, counties_met_after_step2),
                counties_met_after_step3
            )
            counties_met_three_tech_packages_count = counties_met_three_tech_packages.sum()

            logger.info(f"第三步应用后达标情况:")
            logger.info(f"达标县数: {counties_met_three_tech_packages_count}/{total_counties} ({counties_met_three_tech_packages_count/total_counties*100:.1f}%)")
            logger.info(f"第三步新增达标县数: {counties_met_three_tech_packages_count - counties_met_union_count}")

            logger.info(f"\n三个技术包的并集详情:")
            logger.info(f"第一级技术包符合要求的县数: {counties_met_count_1}")
            logger.info(f"1-2级技术包符合要求的县数: {counties_met_union_count}")
            logger.info(f"1-3级技术包符合要求的县数: {counties_met_three_tech_packages_count}")
            logger.info(f"并集相对第一步新增县数: {counties_met_three_tech_packages_count - counties_met_count_1}")
            logger.info(f"并集相对第二步新增县数: {counties_met_three_tech_packages_count - counties_met_union_count}")

            # 将三个技术包的并集结果也保存到统计中
            stats['step_3']['counties_met_three_tech_packages'] = counties_met_three_tech_packages_count

            # 第三步完成后立即保存技术包信息
            if stats['step_3']['tech_package']:
                save_tech_package_to_excel(stats['step_3']['tech_package'], 'step_3')
                logger.info("第三步技术包已保存完成")
        else:
            logger.info(f"所有县都在第一级技术包符合要求的县与1，2技术包的县的并集中，无需进行第三步")

            # 计算三个技术包的并集（此时第三步没有执行，所以并集等于第一级和1-2级的并集）
            counties_met_three_tech_packages_count = counties_met_union_count

            logger.info(f"\n第三步应用后达标情况:")
            logger.info(f"达标县数: {counties_met_three_tech_packages_count}/{total_counties} ({counties_met_three_tech_packages_count/total_counties*100:.1f}%)")
            logger.info(f"总计新增达标县数: {counties_met_three_tech_packages_count - counties_met_count_1}")

            logger.info(f"\n三个技术包的并集详情:")
            logger.info(f"第一级技术包符合要求的县数: {counties_met_count_1}")
            logger.info(f"1-2级技术包符合要求的县数: {counties_met_union_count}")
            logger.info(f"1-3级技术包符合要求的县数: 未执行第三步，无新增")
            logger.info(f"三个技术包并集达标县数: {counties_met_three_tech_packages_count}/{total_counties} ({counties_met_three_tech_packages_count/total_counties*100:.1f}%)")
            logger.info(f"并集相对第一步新增县数: {counties_met_three_tech_packages_count - counties_met_count_1}")

            # 将三个技术包的并集结果也保存到统计中
            stats['step_3']['counties_met_three_tech_packages'] = counties_met_three_tech_packages_count
    else:
        logger.info(f"所有县在第一步后已达标，无需进行后续步骤")

        # 计算三个技术包的并集（此时第二步和第三步都没有执行，所以并集等于第一级的达标情况）
        counties_met_three_tech_packages_count = counties_met_count_1

        logger.info(f"\n第三步应用后达标情况:")
        logger.info(f"达标县数: {counties_met_three_tech_packages_count}/{total_counties} ({counties_met_three_tech_packages_count/total_counties*100:.1f}%)")

        logger.info(f"\n三个技术包的并集详情:")
        logger.info(f"第一级技术包符合要求的县数: {counties_met_count_1}")
        logger.info(f"1-2级技术包符合要求的县数: 未执行第二步，等于第一步")
        logger.info(f"1-3级技术包符合要求的县数: 未执行第三步，等于第一步")
        logger.info(f"三个技术包并集达标县数: {counties_met_three_tech_packages_count}/{total_counties} ({counties_met_three_tech_packages_count/total_counties*100:.1f}%)")

        # 将三个技术包的并集结果也保存到统计中
        stats['step_3']['counties_met_three_tech_packages'] = counties_met_three_tech_packages_count

    # 输出总体统计
    logger.info(f"\n" + "=" * 60)
    logger.info("分步等级优化完成！总体统计:")
    logger.info("=" * 60)

    total_applied_techs = stats['step_1']['applied_techs'] + stats['step_2']['applied_techs'] + stats['step_3']['applied_techs']
    total_actions = stats['step_1']['actions'] + stats['step_2']['actions'] + stats['step_3']['actions']

    # 使用三个技术包的并集达标县数作为最终达标县数
    final_met_counties = stats['step_3']['counties_met_three_tech_packages']

    logger.info(f"第一步(≤1级技术): 应用 {stats['step_1']['applied_techs']} 个技术, {stats['step_1']['actions']} 个动作")
    logger.info(f"第二步(≤2级技术): 应用 {stats['step_2']['applied_techs']} 个技术, {stats['step_2']['actions']} 个动作")
    logger.info(f"第三步(≤3级技术): 应用 {stats['step_3']['applied_techs']} 个技术, {stats['step_3']['actions']} 个动作")
    logger.info(f"总计: 应用 {total_applied_techs} 个技术, {total_actions} 个动作")
    logger.info(f"最终达标县数: {final_met_counties}/{total_counties} ({final_met_counties/total_counties*100:.1f}%)")

    # 添加县的技术包分配信息到统计结果
    stats['county_assignments'] = all_county_assignments

    return stats

def load_tech_packages_from_files(output_dir="results/level_based_stepwise_techs"):
    """
    从Excel文件中加载技术包信息

    Args:
        output_dir: 技术包文件所在的目录

    Returns:
        dict: 包含技术包信息和县分配信息的字典
    """
    logger.info("从文件中加载技术包信息...")

    # 初始化统计结构
    stats = {
        'step_1': {'applied_techs': 0, 'actions': 0, 'counties_met': 0, 'max_level': 1, 'tech_package': {}},
        'step_2': {'applied_techs': 0, 'actions': 0, 'counties_met': 0, 'max_level': 2, 'tech_package': {}},
        'step_3': {'applied_techs': 0, 'actions': 0, 'counties_met': 0, 'max_level': 3, 'tech_package': {}}
    }

    # 加载技术包信息
    tech_package_files = [
        os.path.join(output_dir, "step_1_tech_package.xlsx"),
        os.path.join(output_dir, "step_2_tech_package.xlsx"),
        os.path.join(output_dir, "step_3_tech_package.xlsx")
    ]

    # 加载县分配信息
    county_assignment_file = os.path.join(output_dir, "level_based_stepwise_techs", "county_tech_assignments.xlsx")
    if os.path.exists(county_assignment_file):
        try:
            county_df = pd.read_excel(county_assignment_file)
            # 构建county_assignments字典
            county_assignments = {}
            for _, row in county_df.iterrows():
                county_assignments[row['县索引']] = row['分配技术包等级']
            stats['county_assignments'] = county_assignments
            logger.info(f"已加载县分配信息，共 {len(county_assignments)} 个县")
        except Exception as e:
            logger.error(f"加载县分配文件失败: {e}")
            stats['county_assignments'] = {}
    else:
        logger.warning(f"未找到县分配文件: {county_assignment_file}")
        stats['county_assignments'] = {}

    # 加载每个步骤的技术包
    for i, file_path in enumerate(tech_package_files, 1):
        if os.path.exists(file_path):
            try:
                tech_df = pd.read_excel(file_path)

                # 构建技术包信息
                tech_package = {
                    'max_level': i,
                    'selected_techs': tech_df['技术ID'].tolist() if '技术ID' in tech_df.columns else [],
                    'tech_details': []
                }

                # 构建技术详情，确保字段名与原始格式一致
                for _, row in tech_df.iterrows():
                    tech_detail = {
                        'tech_idx': row.get('技术ID', row.get('tech_idx', 0)),
                        'name': row.get('技术名称', row.get('name', 'Unknown')),
                        'level': row.get('技术等级', row.get('level', 1)),
                        'net_reduction': row.get('净减排量', row.get('net_reduction', 0.0)),
                        'reduction_type': row.get('减排类型', row.get('reduction_type', '净减排')),
                        'cost': row.get('经济成本', row.get('cost', 0.0)),
                        'class': row.get('产业类型', row.get('class', 'unknown')),
                        'species': row.get('物种', row.get('species', '')),
                        'conflict': row.get('冲突组', row.get('conflict', '')),
                        # 添加气体和产量影响信息
                        'NH3_reduction': row.get('NH3减排量', row.get('NH3_reduction', 0.0)),
                        'NO3_reduction': row.get('NO3减排量', row.get('NO3_reduction', 0.0)),
                        'N_runoff_reduction': row.get('N_runoff减排量', row.get('N_runoff_reduction', 0.0)),
                        'CH4_reduction': row.get('CH4减排量', row.get('CH4_reduction', 0.0)),
                        'N2O_reduction': row.get('N2O减排量', row.get('N2O_reduction', 0.0)),
                        'SOC_reduction': row.get('SOC减排量', row.get('SOC_reduction', 0.0)),
                        'yield_change': row.get('产量变化', row.get('yield_change', 0.0))
                    }
                    tech_package['tech_details'].append(tech_detail)

                # 计算技术数量和动作数（近似值）
                tech_count = len(tech_package['selected_techs'])
                # 假设每个县平均应用该技术包中的所有技术
                county_count = len([idx for idx, level in stats['county_assignments'].items() if level == i])
                actions_count = tech_count * county_count

                stats[f'step_{i}']['tech_package'] = tech_package
                stats[f'step_{i}']['applied_techs'] = tech_count
                stats[f'step_{i}']['actions'] = actions_count

                logger.info(f"已加载第{i}步技术包: {tech_count} 个技术, 预计 {actions_count} 个动作")

            except Exception as e:
                logger.error(f"加载技术包文件 {file_path} 时出错: {e}")
                stats[f'step_{i}']['tech_package'] = {}
        else:
            logger.warning(f"未找到技术包文件: {file_path}")
            stats[f'step_{i}']['tech_package'] = {}

    # 计算总体统计
    total_applied_techs = sum(stats[f'step_{i}']['applied_techs'] for i in [1, 2, 3])
    total_actions = sum(stats[f'step_{i}']['actions'] for i in [1, 2, 3])

    # 计算达标县数（使用分配给各级技术包的县数之和作为近似值）
    final_met_counties = len(stats['county_assignments'])
    total_counties = final_met_counties  # 这里无法准确获取总县数，使用分配县数作为近似

    logger.info(f"技术包加载完成:")
    logger.info(f"总计: 应用 {total_applied_techs} 个技术, {total_actions} 个动作")
    logger.info(f"涉及县数: {final_met_counties}")

    # 设置最终达标县数
    stats['step_3']['counties_met_three_tech_packages'] = final_met_counties

    return stats

def save_tech_county_impacts(env, output_dir="results", suffix="level_based_stepwise_techs", tech_impacts_data=None):
    """
    保存每个技术对每个县的气体指标影响数据

    Args:
        env: 环境实例
        output_dir: 输出目录
        suffix: 文件名后缀
    """
    logger.info("正在生成每个技术对各县的影响数据...")

    # 创建输出目录
    full_output_dir = os.path.join(output_dir, suffix)
    os.makedirs(full_output_dir, exist_ok=True)

    # 获取所有技术
    tech_set = env.tech_set
    county_names = env.IDs['Counties'].tolist()

    # 为每个技术创建影响数据文件
    tech_county_impacts_dir = os.path.join(full_output_dir, "tech_county_impacts")
    os.makedirs(tech_county_impacts_dir, exist_ok=True)

    # 如果提供了实际的技术影响数据，使用它；否则重新计算
    if tech_impacts_data:
        logger.info("使用实际应用技术时的影响数据...")
        # 从技术影响数据中提取所有技术的影响
        all_tech_impacts = {}
        for step_key, step_data in tech_impacts_data.items():
            if step_key.startswith('step_') and 'tech_impacts' in step_data:
                step_impacts = step_data['tech_impacts']
                for county_idx, county_data in step_impacts['county_impacts'].items():
                    for tech_data in county_data['applied_techs']:
                        tech_idx = tech_data['tech_idx']
                        if tech_idx not in all_tech_impacts:
                            all_tech_impacts[tech_idx] = {}
                        all_tech_impacts[tech_idx][county_idx] = tech_data['impacts']

        # 为每个技术创建影响数据文件
        for tech_idx, county_impacts in all_tech_impacts.items():
            try:
                tech_row = tech_set.iloc[tech_idx]
                tech_name = tech_row['Mitigation strategy']

                # 准备数据用于保存 - 只处理需要技术的县
                impact_data = []

                # 获取需要技术的县索引
                counties_need_tech_indices = np.where(env.counties_need_tech)[0]

                # 只遍历需要技术的县
                for county_idx in counties_need_tech_indices:
                    if county_idx in county_impacts:
                        impact_data.append({
                            '县名称': county_names[county_idx],
                            '县索引': county_idx,
                            'NH3变化': county_impacts[county_idx]['NH3_change'],
                            'NO3变化': county_impacts[county_idx]['NO3_change'],
                            'N_runoff变化': county_impacts[county_idx]['N_runoff_change'],
                            'CH4变化': county_impacts[county_idx]['CH4_change'],
                            'N2O变化': county_impacts[county_idx]['N2O_change'],
                            'SOC变化': county_impacts[county_idx]['SOC_change'],
                            '产量变化': county_impacts[county_idx]['yield_change'],
                            '净减排量': county_impacts[county_idx]['net_reduction']
                        })
                    else:
                        # 对于需要技术但未应用该技术的县，显示0值
                        impact_data.append({
                            '县名称': county_names[county_idx],
                            '县索引': county_idx,
                            'NH3变化': 0.0,
                            'NO3变化': 0.0,
                            'N_runoff变化': 0.0,
                            'CH4变化': 0.0,
                            'N2O变化': 0.0,
                            'SOC变化': 0.0,
                            '产量变化': 0.0,
                            '净减排量': 0.0
                        })

                # 保存到Excel文件
                safe_tech_name = "".join(c for c in tech_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_tech_name = safe_tech_name.replace(' ', '_').replace('-', '_')

                impact_df = pd.DataFrame(impact_data)
                impact_file = os.path.join(tech_county_impacts_dir, f"tech_{tech_idx}_{safe_tech_name[:50]}.xlsx")
                impact_df.to_excel(impact_file, index=False)

            except Exception as e:
                logger.error(f"处理技术 {tech_idx} 时出错: {e}")
                continue
    else:
        # 如果没有提供技术影响数据，记录警告但不进行计算
        logger.warning("未提供技术影响数据，跳过单个技术影响文件的生成")

    logger.info(f"技术对各县的影响数据已保存到目录: {tech_county_impacts_dir}")

    # 创建汇总文件，显示每个县应用了哪些技术及其影响
    try:
        logger.info("正在生成县技术影响汇总...")

        # 如果提供了实际的技术影响数据，使用它；否则从环境状态计算
        if tech_impacts_data:
            # 使用实际的技术影响数据
            applied_techs = {}
            county_summary_data = []

            # 从技术影响数据中提取县的技术应用信息
            for step_key, step_data in tech_impacts_data.items():
                if 'tech_impacts' in step_data:
                    step_impacts = step_data['tech_impacts']
                    for county_idx, county_data in step_impacts['county_impacts'].items():
                        if county_idx not in applied_techs:
                            applied_techs[county_idx] = []

                        # 记录该县应用的技术
                        for tech_data in county_data['applied_techs']:
                            tech_idx = tech_data['tech_idx']
                            if tech_idx not in applied_techs[county_idx]:
                                applied_techs[county_idx].append(tech_idx)

            # 为每个县创建汇总数据 - 只处理需要技术的县
            # 获取需要技术的县索引
            counties_need_tech_indices = np.where(env.counties_need_tech)[0]

            for county_idx in counties_need_tech_indices:
                county_name = county_names[county_idx]
                applied_tech_list = applied_techs.get(county_idx, [])

                # 计算该县所有应用技术的影响总和
                total_impacts = {
                    'NH3变化': 0.0,
                    'NO3变化': 0.0,
                    'N_runoff变化': 0.0,
                    'CH4变化': 0.0,
                    'N2O变化': 0.0,
                    'SOC变化': 0.0,
                    '产量变化': 0.0,
                    '净减排量': 0.0
                }

                tech_details = []
                for step_key, step_data in tech_impacts_data.items():
                    if 'tech_impacts' in step_data:
                        step_impacts = step_data['tech_impacts']
                        if county_idx in step_impacts['county_impacts']:
                            county_data = step_impacts['county_impacts'][county_idx]

                            for tech_data in county_data['applied_techs']:
                                tech_idx = tech_data['tech_idx']
                                tech_row = tech_set.iloc[tech_idx]
                                tech_name = tech_row['Mitigation strategy']
                                impact = tech_data['impacts']

                                tech_details.append({
                                    '技术ID': tech_idx,
                                    '技术名称': tech_name,
                                    'NH3变化': impact['NH3_change'],
                                    'NO3变化': impact['NO3_change'],
                                    'N_runoff变化': impact['N_runoff_change'],
                                    'CH4变化': impact['CH4_change'],
                                    'N2O变化': impact['N2O_change'],
                                    'SOC变化': impact['SOC_change'],
                                    '产量变化': impact['yield_change'],
                                    '净减排量': impact['net_reduction']
                                })

                                # 累加到县的总影响
                                # 创建键名映射
                                key_mapping = {
                                    'NH3变化': 'NH3_change',
                                    'NO3变化': 'NO3_change',
                                    'N_runoff变化': 'N_runoff_change',
                                    'CH4变化': 'CH4_change',
                                    'N2O变化': 'N2O_change',
                                    'SOC变化': 'SOC_change',
                                    '产量变化': 'yield_change',
                                    '净减排量': 'net_reduction'
                                }

                                for key in total_impacts:
                                    if key in key_mapping:
                                        total_impacts[key] += impact[key_mapping[key]]

                county_summary_data.append({
                    '县名称': county_name,
                    '县索引': county_idx,
                    '应用技术数量': len(applied_tech_list),
                    '技术ID列表': ','.join(map(str, applied_tech_list)),
                    '总NH3变化': total_impacts['NH3变化'],
                    '总NO3变化': total_impacts['NO3变化'],
                    '总N_runoff变化': total_impacts['N_runoff变化'],
                    '总CH4变化': total_impacts['CH4变化'],
                    '总N2O变化': total_impacts['N2O变化'],
                    '总SOC变化': total_impacts['SOC变化'],
                    '总产量变化': total_impacts['产量变化'],
                    '总净减排量': total_impacts['净减排量']
                })
        else:
            # 如果没有提供技术影响数据，创建空的汇总数据
            logger.warning("未提供技术影响数据，无法生成县汇总")
            county_summary_data = []

        # 保存县汇总数据
        county_summary_df = pd.DataFrame(county_summary_data)
        county_summary_file = os.path.join(full_output_dir, "county_tech_impacts_summary.xlsx")
        county_summary_df.to_excel(county_summary_file, index=False)
        logger.info(f"县技术影响汇总已保存到: {county_summary_file}")

        # 保存详细的技术应用数据（每个县每种技术的影响）
        detailed_data = []
        if tech_impacts_data:
            # 使用实际的技术影响数据
            for step_key, step_data in tech_impacts_data.items():
                if 'tech_impacts' in step_data:
                    step_impacts = step_data['tech_impacts']
                    for county_idx, county_data in step_impacts['county_impacts'].items():
                        county_name = county_names[county_idx]

                        for tech_data in county_data['applied_techs']:
                            tech_idx = tech_data['tech_idx']
                            tech_row = tech_set.iloc[tech_idx]
                            impact = tech_data['impacts']

                            detailed_data.append({
                                '县名称': county_name,
                                '县索引': county_idx,
                                '技术ID': tech_idx,
                                '技术名称': tech_row['Mitigation strategy'],
                                '技术等级': tech_row['技术分级'],
                                '产业类型': tech_row['class'],
                                'NH3变化': impact['NH3_change'],
                                'NO3变化': impact['NO3_change'],
                                'N_runoff变化': impact['N_runoff_change'],
                                'CH4变化': impact['CH4_change'],
                                'N2O变化': impact['N2O_change'],
                                'SOC变化': impact['SOC_change'],
                                '产量变化': impact['yield_change'],
                                '净减排量': impact['net_reduction']
                            })
        else:
            # 如果没有提供技术影响数据，跳过详细数据生成
            logger.warning("未提供技术影响数据，跳过详细数据生成")

        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            detailed_file = os.path.join(full_output_dir, "county_tech_impacts_detailed.xlsx")
            detailed_df.to_excel(detailed_file, index=False)
            logger.info(f"县技术影响详细信息已保存到: {detailed_file}")

    except Exception as e:
        logger.error(f"生成县技术影响汇总时出错: {e}")

def output_state_summary(env, output_dir="results", suffix="level_based_stepwise_techs", optimization_stats=None, skip_tech_save=False):
    """
    输出环境状态汇总，包含技术包信息和县的技术包分配

    Args:
        env: 环境实例
        output_dir: 输出目录
        suffix: 文件名后缀
        optimization_stats: 优化统计信息，包含技术包和县分配信息
    """
    logger.info(f"\n=== 输出分步等级优化后的状态 ===")

    # 创建输出目录
    full_output_dir = os.path.join(output_dir, suffix)
    os.makedirs(full_output_dir, exist_ok=True)

    # 输出和保存技术包信息
    if optimization_stats:
        logger.info(f"\n=== 技术包信息 ===")

        # 准备技术包数据用于保存
        tech_package_data = []

        for step_key, step_data in optimization_stats.items():
            if step_key.startswith('step_') and 'tech_package' in step_data:
                tech_package = step_data['tech_package']
                if tech_package and 'tech_details' in tech_package:
                    logger.info(f"\n{step_key.upper()} 技术包 (≤{tech_package.get('max_level', '?')}级):")
                    logger.info(f"  技术数量: {len(tech_package['tech_details'])}")
                    logger.info(f"  分级分布: {tech_package.get('level_counts', {})}")

                    for tech_detail in tech_package['tech_details']:
                        logger.info(f"    - {tech_detail['name']} (等级:{tech_detail['level']}, {tech_detail['reduction_type']}:{abs(tech_detail['net_reduction']):.4f}, 成本:{tech_detail['cost']:.2f})")
                        logger.info(f"      气体影响 - NH3:{tech_detail.get('NH3_reduction', 0):.4f}, NO3:{tech_detail.get('NO3_reduction', 0):.4f}, N_runoff:{tech_detail.get('N_runoff_reduction', 0):.4f}")
                        logger.info(f"      气体影响 - CH4:{tech_detail.get('CH4_reduction', 0):.4f}, N2O:{tech_detail.get('N2O_reduction', 0):.4f}, SOC:{tech_detail.get('SOC_reduction', 0):.4f}, 产量:{tech_detail.get('yield_change', 0):.4f}")

                        # 添加到技术包数据列表
                        tech_package_data.append({
                            '步骤': step_key.upper(),
                            '最大等级': tech_package.get('max_level', 0),
                            '技术ID': tech_detail['tech_idx'],
                            '技术名称': tech_detail['name'],
                            '技术等级': tech_detail['level'],
                            '净减排量': tech_detail['net_reduction'],
                            '减排类型': tech_detail['reduction_type'],
                            '经济成本': tech_detail['cost'],
                            '产业类型': tech_detail['class'],
                            '物种': tech_detail.get('species', ''),
                            '冲突组': '' if pd.isna(tech_detail['conflict']) else tech_detail['conflict']
                        })

        # 保存技术包信息到Excel
        if not skip_tech_save and tech_package_data:
            tech_package_df = pd.DataFrame(tech_package_data)
            tech_package_file = os.path.join(full_output_dir, "tech_packages.xlsx")
            tech_package_df.to_excel(tech_package_file, index=False)
            logger.info(f"技术包信息已保存到: {tech_package_file}")
        elif skip_tech_save:
            logger.info("跳过技术包信息保存（文件已存在）")

        # 输出县的技术包分配信息
        if 'county_assignments' in optimization_stats:
            logger.info(f"\n=== 县的技术包分配 ===")
            county_names = env.IDs['Counties'].tolist()
            assignments = optimization_stats['county_assignments']

            level_1_counties = []
            level_2_counties = []
            level_3_counties = []
            no_tech_counties = []

            for county_idx in range(len(county_names)):
                if county_idx in assignments:
                    level = assignments[county_idx]
                    if level == 1:
                        level_1_counties.append(county_names[county_idx])
                    elif level == 2:
                        level_2_counties.append(county_names[county_idx])
                    elif level == 3:
                        level_3_counties.append(county_names[county_idx])
                else:
                    no_tech_counties.append(county_names[county_idx])

            logger.info(f"使用1级技术包的县 ({len(level_1_counties)}个): {', '.join(level_1_counties[:10])}{'...' if len(level_1_counties) > 10 else ''}")
            logger.info(f"使用2级技术包的县 ({len(level_2_counties)}个): {', '.join(level_2_counties[:10])}{'...' if len(level_2_counties) > 10 else ''}")
            logger.info(f"使用3级技术包的县 ({len(level_3_counties)}个): {', '.join(level_3_counties[:10])}{'...' if len(level_3_counties) > 10 else ''}")
            logger.info(f"无需技术干预的县 ({len(no_tech_counties)}个): {', '.join(no_tech_counties[:10])}{'...' if len(no_tech_counties) > 10 else ''}")

            # 保存县的技术包分配信息到Excel
            county_assignment_data = []
            for county_idx in range(len(county_names)):
                county_name = county_names[county_idx]
                if county_idx in assignments:
                    package_level = assignments[county_idx]
                    county_assignment_data.append({
                        '县名称': county_name,
                        '县索引': county_idx,
                        '分配技术包等级': package_level,
                        '技术包类型': f'≤{package_level}级技术包'
                    })
                else:
                    county_assignment_data.append({
                        '县名称': county_name,
                        '县索引': county_idx,
                        '分配技术包等级': 0,
                        '技术包类型': '无需技术干预'
                    })

            if not skip_tech_save:
                county_assignment_df = pd.DataFrame(county_assignment_data)
                county_assignment_file = os.path.join(full_output_dir, "county_tech_assignments.xlsx")
                county_assignment_df.to_excel(county_assignment_file, index=False)
                logger.info(f"县的技术包分配信息已保存到: {county_assignment_file}")
            else:
                logger.info("跳过县的技术包分配信息保存（文件已存在）")
    
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
    state_file = os.path.join(full_output_dir, "state_after_single_step_techs.xlsx")
    state_df.to_excel(state_file)
    logger.info(f"观察状态数据已保存到: {state_file}")

    # 2. 保存stateMapping的所有状态数据（按key分别保存）
    if hasattr(env, 'stateMapping') and env.stateMapping:
        logger.info(f"正在保存 {len(env.stateMapping)} 个状态映射...")
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
                logger.info(f"状态映射 '{key}' 已保存到: {state_mapping_file}")

            except Exception as e:
                logger.error(f"保存状态映射 '{key}' 时出错: {e}")
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
    gap_file = os.path.join(full_output_dir, "gaps_after_single_step_techs.xlsx")
    gap_df.to_excel(gap_file)
    logger.info(f"减排差距数据已保存到: {gap_file}")

    # 4. 统计摘要
    logger.info("\n=== 各指标统计摘要 ===")
    logger.info("观察状态指标统计:")
    logger.info(f"\n{state_df.describe()}")

    logger.info("\n减排差距统计:")
    logger.info(f"\n{gap_df.describe()}")

    # 如果保存了stateMapping，输出状态映射文件列表
    if hasattr(env, 'stateMapping') and env.stateMapping:
        logger.info(f"\n已保存的状态映射文件:")
        for key in env.stateMapping.keys():
            logger.info(f"  - {key}.xlsx")
    
    # 5. 达标情况统计
    no3_met = (env.gap_NO3.squeeze() <= 0).sum().item()
    nh3_met = (env.gap_NH3.squeeze() <= 0).sum().item()
    n_runoff_met = (env.gap_N_runoff.squeeze() <= 0).sum().item()
    ch4_met = (env.gap_CH4.squeeze() <= 0).sum().item()
    n2o_met = (env.gap_N2O.squeeze() <= 0).sum().item()
    
    total_counties = len(county_names)
    
    logger.info(f"\n=== 达标情况统计 (总县数: {total_counties}) ===")
    logger.info(f"NO3达标: {no3_met} 县 ({no3_met/total_counties*100:.1f}%)")
    logger.info(f"NH3达标: {nh3_met} 县 ({nh3_met/total_counties*100:.1f}%)")
    logger.info(f"N_runoff达标: {n_runoff_met} 县 ({n_runoff_met/total_counties*100:.1f}%)")
    logger.info(f"CH4达标: {ch4_met} 县 ({ch4_met/total_counties*100:.1f}%)")
    logger.info(f"N2O达标: {n2o_met} 县 ({n2o_met/total_counties*100:.1f}%)")
    
    # 所有指标都达标的县数量
    all_targets_met = (
        (env.gap_NO3.squeeze() <= 0) & 
        (env.gap_NH3.squeeze() <= 0) & 
        (env.gap_N_runoff.squeeze() <= 0) 
        # (env.gap_CH4.squeeze() <= 0) &
        # (env.gap_N2O.squeeze() <= 0)
    ).sum().item()
    
    logger.info(f"所有指标都达标: {all_targets_met} 县 ({all_targets_met/total_counties*100:.1f}%)")

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
    logger.info(f"达标统计摘要已保存到: {summary_file}")

    # 7. 保存成本数据
    if hasattr(env, 'save_path') and env.save_path is not None:
        # 确保保存路径存在
        if not os.path.exists(env.save_path):
            os.makedirs(env.save_path, exist_ok=True)
        cost_df = env._save_tech_selected_summary()
        # 将成本数据也保存到output_dir
        cost_df.to_excel(os.path.join(full_output_dir, "cost.xlsx"))
        logger.info(f"成本数据已保存到: {os.path.join(full_output_dir, 'cost.xlsx')}")
    else:
        logger.warning("环境未设置save_path，跳过成本数据保存")

    # 保存每个技术对每个县的影响数据
    try:
        # 如果有技术影响数据，传递给保存函数
        tech_impacts_data = None
        if optimization_stats:
            tech_impacts_data = {}
            for step_key, step_data in optimization_stats.items():
                if step_key.startswith('step_') and 'tech_impacts' in step_data:
                    tech_impacts_data[step_key] = {'tech_impacts': step_data['tech_impacts']}

        save_tech_county_impacts(env, output_dir, suffix, tech_impacts_data)
    except Exception as e:
        logger.error(f"保存技术对各县的影响数据时出错: {e}")

    return state_df, gap_df, summary_df, cost_df if 'cost_df' in locals() else None

def create_tech_impact_summary(output_dir="results/level_based_stepwise_techs"):
    """
    创建技术对气体影响的汇总表
    将所有技术对各县的影响累加汇总
    """
    import pandas as pd
    import os
    import glob
    from pathlib import Path

    logger.info("开始创建技术影响汇总表...")

    # 构建tech_county_impacts目录路径
    tech_impacts_dir = os.path.join(output_dir, "level_based_stepwise_techs", "tech_county_impacts")

    if not os.path.exists(tech_impacts_dir):
        logger.error(f"技术影响目录不存在: {tech_impacts_dir}")
        return

    # 获取所有技术文件
    tech_files = glob.glob(os.path.join(tech_impacts_dir, "tech_*.xlsx"))
    logger.info(f"找到 {len(tech_files)} 个技术文件")

    if not tech_files:
        logger.error("没有找到技术影响文件")
        return

    # 初始化汇总数据
    tech_summary = {}

    # 处理每个技术文件
    for i, file_path in enumerate(tech_files):
        try:
            # 从文件名提取技术ID和名称
            filename = os.path.basename(file_path)
            parts = filename.replace('.xlsx', '').split('_', 2)
            if len(parts) >= 3:
                tech_id = int(parts[1])
                tech_name = parts[2]
            else:
                logger.warning(f"无法解析文件名: {filename}")
                continue

            # 读取技术影响数据
            df = pd.read_excel(file_path)

            # 计算该技术对所有县的影响总和
            total_impacts = {
                'NH3变化': df['NH3变化'].sum(),
                'NO3变化': df['NO3变化'].sum(),
                'N_runoff变化': df['N_runoff变化'].sum(),
                'CH4变化': df['CH4变化'].sum(),
                'N2O变化': df['N2O变化'].sum(),
                'SOC变化': df['SOC变化'].sum(),
                '产量变化': df['产量变化'].sum(),
                '净减排量': df['净减排量'].sum(),
                '影响县数': len(df[df[['NH3变化', 'NO3变化', 'N_runoff变化', 'CH4变化', 'N2O变化', 'SOC变化', '产量变化']].sum(axis=1) != 0])
            }

            tech_summary[tech_id] = {
                'tech_name': tech_name,
                'tech_id': tech_id,
                **total_impacts
            }

        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {e}")
            continue

        if (i + 1) % 50 == 0:
            logger.info(f"已处理 {i + 1}/{len(tech_files)} 个技术文件")

    # 创建汇总DataFrame
    summary_data = []
    for tech_id, data in tech_summary.items():
        summary_data.append(data)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # 按净减排量排序（绝对值最大的排在前面）
        summary_df['净减排量绝对值'] = summary_df['净减排量'].abs()
        summary_df = summary_df.sort_values('净减排量绝对值', ascending=False)
        summary_df = summary_df.drop('净减排量绝对值', axis=1)

        # 保存汇总表
        output_file = os.path.join(output_dir, "level_based_stepwise_techs", "tech_gas_impact_summary.xlsx")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        summary_df.to_excel(output_file, index=False)

        logger.info(f"技术影响汇总表已保存到: {output_file}")
        logger.info(f"汇总了 {len(summary_data)} 个技术的影响数据")

        # 输出统计信息
        logger.info("汇总统计:")
        logger.info(f"- 技术总数: {len(summary_data)}")
        logger.info(f"- 有影响的技术数: {len(summary_df[summary_df['净减排量'] != 0])}")
        logger.info(f"- 平均影响县数: {summary_df['影响县数'].mean():.1f}")
        logger.info(f"- 最大净减排量: {summary_df['净减排量'].max():.4f}")
        logger.info(f"- 最小净减排量: {summary_df['净减排量'].min():.4f}")

    else:
        logger.error("没有有效的技术影响数据")

def main():
    """
    主函数
    """
    # 配置日志文件输出
    log_file = "results/level_based_stepwise_techs/optimization.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 添加文件处理器到全局logger
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("分步等级技术优化：按技术等级分步(1级→1+2级→1+2+3级)，每等级内按等级、经济成本、减排量排序")
    logger.info("=" * 60)

    # 检查是否已经保存过三个技术包，如果是则跳过执行
    output_dir = "results/level_based_stepwise_techs"
    tech_package_files = [
        os.path.join(output_dir, "step_1_tech_package.xlsx"),
        os.path.join(output_dir, "step_2_tech_package.xlsx"),
        os.path.join(output_dir, "step_3_tech_package.xlsx")
    ]

    def check_tech_package_file_valid(file_path):
        """检查技术包文件是否存在且包含有效数据"""
        if not os.path.exists(file_path):
            return False
        try:
            df = pd.read_excel(file_path)
            return len(df) > 0  # 确保文件包含数据
        except Exception as e:
            logger.warning(f"读取技术包文件 {file_path} 时出错: {e}")
            return False

    all_files_valid = all(check_tech_package_file_valid(file) for file in tech_package_files)

    skip_tech_selection = False
    if all_files_valid:
        logger.info("检测到已经存在三个有效的技术包文件，将跳过技术包选取过程")
        logger.info("现有技术包文件：")
        for file in tech_package_files:
            df = pd.read_excel(file)
            logger.info(f"  - {file} (包含 {len(df)} 个技术)")
        logger.info("将基于现有技术包进行后续处理")
        logger.info("=" * 60)
        skip_tech_selection = True

    try:
        # 创建环境配置 - 使用松嫩－三江平原农业区作为示例
        config = GasEnvConfig(
            tech_amount_constraint=30,
            Reward_priority=[0.7, 0.5, 0.3, 0.2],
            county_df_path='data/基础数据-县级尺度.xlsx',
            IDs_df='data/县市亚区.xlsx',
            livestock_tech_path='data/畜牧业技术列单-经济产量0827.xlsx',
            crop_tech_path="data/种植业技术列单产量产业0803.xlsx",
            soc_df="data/SOC-县尺度.xlsx",
            livestock_scale="data/动物数量.xlsx",
            crop_scale="data/分县种植面积.xlsx",
            local_target_penalty_factor=50.0,  # 新增：为配置添加惩罚因子
            linear_result_path='results/linear_optimization_results_by_county_5gases_hard_target.xlsx',
            only_lp_phase=True,  # 启用线性规划约束模式
            save_path="temp_yield_output"  # 临时路径，用于启用产量追踪
        )
        logger.info("环境配置创建成功")

        # 创建环境
        env = GasEnv(config)
        # 保存配置以便后续创建新的环境实例
        env_config = config
        logger.info(f"环境创建成功，包含 {env.numCounty} 个县，{env.numTech} 种技术")

        # 重置环境
        env.reset()
        logger.info("环境已重置到初始状态")

        # 输出初始状态信息
        logger.info(f"需要技术添加的县数量: {env.counties_need_tech.sum()}")
        
        # 执行分步等级技术优化（按技术等级分步 + 等级、经济成本、减排量排序）
        if skip_tech_selection:
            logger.info("跳过技术包选取，从文件中加载技术包信息...")
            optimization_stats = load_tech_packages_from_files("results/level_based_stepwise_techs")
        else:
            optimization_stats = stepwise_level_based_tech_optimization(env, env_config, use_parallel=False)

        # 在原始环境中重新应用所有选择的技术，确保最终状态正确
        logger.info("在原始环境中重新应用所有选择的技术...")

        # 根据county_assignments确定每个步骤应该应用到哪些县
        if 'county_assignments' in optimization_stats and optimization_stats['county_assignments']:
            counties_step1 = [idx for idx, level in optimization_stats['county_assignments'].items() if level == 1]
            counties_step2 = [idx for idx, level in optimization_stats['county_assignments'].items() if level == 2]
            counties_step3 = [idx for idx, level in optimization_stats['county_assignments'].items() if level == 3]

            logger.info(f"第一步技术将应用到 {len(counties_step1)} 个县")
            logger.info(f"第二步技术将应用到 {len(counties_step2)} 个县")
            logger.info(f"第三步技术将应用到 {len(counties_step3)} 个县")
        else:
            logger.warning("未找到有效的县技术包分配信息，将重新运行优化过程来生成分配信息")
            logger.info("开始重新运行分步等级优化...")
            optimization_stats = stepwise_level_based_tech_optimization(env, env_config, use_parallel=False)

            if 'county_assignments' in optimization_stats and optimization_stats['county_assignments']:
                counties_step1 = [idx for idx, level in optimization_stats['county_assignments'].items() if level == 1]
                counties_step2 = [idx for idx, level in optimization_stats['county_assignments'].items() if level == 2]
                counties_step3 = [idx for idx, level in optimization_stats['county_assignments'].items() if level == 3]

                logger.info(f"重新计算后，第一步技术将应用到 {len(counties_step1)} 个县")
                logger.info(f"重新计算后，第二步技术将应用到 {len(counties_step2)} 个县")
                logger.info(f"重新计算后，第三步技术将应用到 {len(counties_step3)} 个县")
            else:
                logger.error("重新运行优化过程后仍未找到县技术包分配信息")
                return

        # 创建新的环境实例
        env = GasEnv(env_config)
        env.reset()
        logger.info("环境已重置到初始状态")

        # 第一步：应用到被分配到第一步的县
        if counties_step1 and optimization_stats['step_1']['tech_package'] and optimization_stats['step_1']['tech_package']['selected_techs']:
            logger.info(f"正在应用第一步技术包到 {len(counties_step1)} 个县...")
            _, _, _, tech_impacts_1 = apply_tech_package_to_counties(env, counties_step1,
                                         sorted(optimization_stats['step_1']['tech_package']['selected_techs']),
                                         optimization_stats['step_1']['tech_package'], collect_impacts=True)
            logger.info("第一步技术应用完成")

        # 第二步：应用到被分配到第二步的县
        if counties_step2 and optimization_stats['step_2']['tech_package'] and optimization_stats['step_2']['tech_package']['selected_techs']:
            logger.info(f"正在应用第二步技术包到 {len(counties_step2)} 个县...")
            _, _, _, tech_impacts_2 = apply_tech_package_to_counties(env, counties_step2,
                                         sorted(optimization_stats['step_2']['tech_package']['selected_techs']),
                                         optimization_stats['step_2']['tech_package'], collect_impacts=True)
            logger.info("第二步技术应用完成")

        # 第三步：应用到被分配到第三步的县
        if counties_step3 and optimization_stats['step_3']['tech_package'] and optimization_stats['step_3']['tech_package']['selected_techs']:
            logger.info(f"正在应用第三步技术包到 {len(counties_step3)} 个县...")
            _, _, _, tech_impacts_3 = apply_tech_package_to_counties(env, counties_step3,
                                         sorted(optimization_stats['step_3']['tech_package']['selected_techs']),
                                         optimization_stats['step_3']['tech_package'], collect_impacts=True)
            logger.info("第三步技术应用完成")
        else:
            logger.warning("未找到县的技术包分配信息，跳过技术重新应用")

        logger.info("技术重新应用完成")

        # 收集所有步骤的影响数据
        final_tech_impacts = {}
        if 'tech_impacts_1' in locals() and tech_impacts_1:
            final_tech_impacts['step_1'] = {'tech_impacts': tech_impacts_1}
        if 'tech_impacts_2' in locals() and tech_impacts_2:
            final_tech_impacts['step_2'] = {'tech_impacts': tech_impacts_2}
        if 'tech_impacts_3' in locals() and tech_impacts_3:
            final_tech_impacts['step_3'] = {'tech_impacts': tech_impacts_3}

        # 将收集的影响数据添加到优化统计中
        for step_key, impacts_data in final_tech_impacts.items():
            if step_key in optimization_stats:
                optimization_stats[step_key]['tech_impacts'] = impacts_data['tech_impacts']

        # 输出最终状态
        # 如果重新运行了优化过程，则不跳过技术保存
        final_skip_tech_save = skip_tech_selection
        if not ('county_assignments' in optimization_stats or optimization_stats['county_assignments']):
            final_skip_tech_save = False  # 重新运行了优化过程，需要保存文件

        state_df, gap_df, summary_df, cost_df = output_state_summary(env, output_dir="results/level_based_stepwise_techs", suffix="level_based_stepwise_techs", optimization_stats=optimization_stats, skip_tech_save=final_skip_tech_save)

        # 创建技术影响汇总表
        try:
            create_tech_impact_summary(output_dir="results/level_based_stepwise_techs")
        except Exception as e:
            logger.error(f"创建技术影响汇总表时出错: {e}")

        # 保存优化统计数据
        if not skip_tech_selection or not ('county_assignments' in optimization_stats and optimization_stats['county_assignments']):
            stats_data = []
            for step, stats in optimization_stats.items():
                if step.startswith('step_'):
                    step_num = step.split('_')[1]
                    stats_data.append({
                        '优化步骤': f'第{step_num}步（≤{stats["max_level"]}级技术包）',
                        '应用技术数量': stats['applied_techs'],
                        '执行动作数': stats['actions'],
                        '达标县数': stats['counties_met']
                    })

            stats_df = pd.DataFrame(stats_data)
            stats_file = os.path.join("results/level_based_stepwise_techs", "optimization_stats.xlsx")
            stats_df.to_excel(stats_file, index=False)
            logger.info(f"优化统计数据已保存到: {stats_file}")
        else:
            logger.info("跳过优化统计数据保存（文件已存在）")

        # 清理临时目录
        import shutil
        if os.path.exists("temp_yield_output"):
            shutil.rmtree("temp_yield_output")
            logger.info("临时产量输出目录已清理")

        logger.info("\n" + "=" * 60)
        if skip_tech_selection:
            logger.info("基于现有技术包的重新优化完成！结果已更新到 'results/level_based_stepwise_techs' 目录中")
        else:
            logger.info("分步等级技术优化完成！所有结果已保存到 'results/level_based_stepwise_techs' 目录中")

        logger.info("包含的文件：")
        if not skip_tech_selection:
            logger.info("  - step_1_tech_package.xlsx: 第一步技术包详细信息")
            logger.info("  - step_2_tech_package.xlsx: 第二步技术包详细信息")
            logger.info("  - step_3_tech_package.xlsx: 第三步技术包详细信息")
            logger.info("  - tech_packages.xlsx: 全部技术包汇总信息")
            logger.info("  - county_tech_assignments.xlsx: 县的技术包分配")
            logger.info("  - optimization_stats.xlsx: 分步等级优化统计")

        logger.info("  - state_after_single_step_techs.xlsx: 观察状态数据")
        logger.info("  - gaps_after_single_step_techs.xlsx: 减排差距数据")
        logger.info("  - 达标统计摘要.xlsx: 达标情况统计")
        logger.info("  - cost.xlsx: 成本数据")
        logger.info("  - county_tech_impacts_summary.xlsx: 县技术影响汇总")
        logger.info("  - county_tech_impacts_detailed.xlsx: 县技术影响详细信息")
        logger.info("  - tech_gas_impact_summary.xlsx: 技术对气体影响汇总表")
        logger.info("  - tech_county_impacts/ 目录: 每个技术的详细影响数据")
        logger.info("  - 各种状态映射文件 (按key分别保存)")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        # 只运行汇总功能
        create_tech_impact_summary()
    else:
        main()