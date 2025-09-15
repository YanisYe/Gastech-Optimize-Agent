"""
强化学习技术采用结果分析 - 按农业大区分析技术减排量
分析每个农业大区内，哪些技术在NH3、NO3、N_runoff方面的减排量最大
"""

import pandas as pd
import os
import glob
from pathlib import Path
import logging

# 设置日志
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def generate_tech_impact_dataframe(tech_id, tech_info_row, target_counties=None):
    """
    动态生成技术影响数据DataFrame
    直接调用 apply_max_reduction_techs.py 中的 apply_tech_package_to_counties 函数

    Args:
        tech_id: 技术ID
        tech_info_row: 技术信息行（从技术ID映射表中获取）
        target_counties: 目标县名列表，如果为None则处理所有县

    Returns:
        pd.DataFrame: 技术影响数据，格式与现有文件一致
    """
    try:
        # 导入必要的模块
        import sys
        from pathlib import Path

        # 添加model目录到路径
        model_path = Path(__file__).parent.parent / "model"
        if str(model_path) not in sys.path:
            sys.path.append(str(model_path))

        # 动态导入环境类和技术应用函数
        from GasEnviroment_curriculum_learning import GasEnv, GasEnvConfig
        from apply_max_reduction_techs import apply_tech_package_to_counties

        # 创建环境配置
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
            local_target_penalty_factor=50.0,
            linear_result_path='results/linear_optimization_results_by_county_5gases_hard_target.xlsx',
            only_lp_phase=True,
            save_path="temp_yield_output"  # 临时路径，用于启用产量追踪
        )

        # 创建环境实例
        env = GasEnv(config)
        env.reset()

        # 获取县名列表
        county_names = env.IDs['Counties'].tolist()
        
        # 创建技术包信息
        tech_package_info = {
            'max_level': 1,  # 默认等级
            'selected_techs': [tech_id],
            'tech_details': []
        }

        # 确定目标县索引
        if target_counties is not None:
            # 将县名转换为县索引
            target_county_indices = []
            for county_name in target_counties:
                if county_name in county_names:
                    county_idx = county_names.index(county_name)
                    target_county_indices.append(county_idx)
            logger.info(f"技术 {tech_id} 将应用到 {len(target_county_indices)} 个指定县")
        else:
            # 获取所有需要技术的县索引
            target_county_indices = list(range(len(county_names)))  # 所有县
            logger.info(f"技术 {tech_id} 将应用到所有县")

        logger.info(f"开始计算技术 {tech_id} 的影响...")

        # 直接调用 apply_tech_package_to_counties 函数
        applied_techs, total_actions, county_assignments, tech_county_impacts = apply_tech_package_to_counties(
            env, target_county_indices, [tech_id], tech_package_info, 
            num_workers=None, collect_impacts=True
        )

        # 从 tech_county_impacts 中提取影响数据
        impact_data = []
        for county_idx, county_name in enumerate(county_names):
            # 初始化该县的影响数据
            county_impacts = {
                '县名称': county_name,
                '县索引': county_idx,
                'NH3变化': 0.0,
                'NO3变化': 0.0,
                'N_runoff变化': 0.0,
                'CH4变化': 0.0,
                'N2O变化': 0.0,
                'SOC变化': 0.0,
                '产量变化': 0.0,
                '净减排量': 0.0
            }

            # 检查该县是否需要技术干预
            if not env.counties_need_tech[county_idx]:
                # 如果不需要技术干预，所有变化为0
                impact_data.append(county_impacts)
                continue

            # 从 tech_county_impacts 中获取该县的影响数据
            if tech_county_impacts and county_idx in tech_county_impacts['county_impacts']:
                county_data = tech_county_impacts['county_impacts'][county_idx]
                total_impacts = county_data['total_impacts']
                
                county_impacts['NH3变化'] = total_impacts['NH3_change']
                county_impacts['NO3变化'] = total_impacts['NO3_change']
                county_impacts['N_runoff变化'] = total_impacts['N_runoff_change']
                county_impacts['CH4变化'] = total_impacts['CH4_change']
                county_impacts['N2O变化'] = total_impacts['N2O_change']
                county_impacts['SOC变化'] = total_impacts['SOC_change']
                county_impacts['产量变化'] = total_impacts['yield_change']
                county_impacts['净减排量'] = total_impacts['net_reduction']

            impact_data.append(county_impacts)

        # 创建DataFrame
        df = pd.DataFrame(impact_data)
        
        # 计算总影响统计
        total_nh3 = df['NH3变化'].sum()
        total_no3 = df['NO3变化'].sum()
        total_net = df['净减排量'].sum()
        
        logger.info(f"成功生成了技术 {tech_id} 的影响数据，共 {len(impact_data)} 个县")
        logger.info(f"技术 {tech_id} 总影响: NH3={total_nh3:.4f}, NO3={total_no3:.4f}, 净减排={total_net:.4f}")

        return df

    except Exception as e:
        logger.error(f"生成技术 {tech_id} 的影响数据时出错: {e}")
        # 返回空的DataFrame作为fallback
        return pd.DataFrame(columns=[
            '县名称', '县索引', 'NH3变化', 'NO3变化', 'N_runoff变化',
            'CH4变化', 'N2O变化', 'SOC变化', '产量变化', '净减排量'
        ])


def analyze_tech_impacts_by_region(output_dir="results",
                                   region_data_path="data/县市亚区.xlsx",
                                   tech_selected_path="results/tech_selected_summary_merged.xlsx",
                                   focus_gases=['NH3变化', 'NO3变化', 'N_runoff变化']):
    """
    按农业大区分析强化学习应用的技术减排量，找出每个农业区内减排效果最好的技术

    Args:
        output_dir: 结果输出目录
        region_data_path: 农业分区数据文件路径
        tech_selected_path: 强化学习应用的技术详情文件路径
        focus_gases: 重点关注的减排气体
    """
    logger.info("开始按农业大区分析强化学习应用的技术减排量...")

    # 构建tech_county_impacts目录路径
    tech_impacts_dir = os.path.join(output_dir, "tech_county_impacts")

    if not os.path.exists(tech_impacts_dir):
       os.makedirs(tech_impacts_dir)
       logger.info(f"创建目录: {tech_impacts_dir}")

    # 检查农业分区数据文件
    if not os.path.exists(region_data_path):
        logger.error(f"农业分区数据文件不存在: {region_data_path}")
        return

    # 检查强化学习技术详情文件
    if not os.path.exists(tech_selected_path):
        logger.error(f"强化学习技术详情文件不存在: {tech_selected_path}")
        return

    # 读取农业分区数据
    try:
        region_df = pd.read_excel(region_data_path)
        logger.info(f"成功读取农业分区数据，共 {len(region_df)} 个县")
    except Exception as e:
        logger.error(f"读取农业分区数据失败: {e}")
        return

    # 读取技术ID映射表
    tech_id_mapping_path = "data/tech_id_mapping.xlsx"
    if not os.path.exists(tech_id_mapping_path):
        logger.error(f"技术ID映射表不存在: {tech_id_mapping_path}")
        return

    try:
        tech_id_mapping_df = pd.read_excel(tech_id_mapping_path)
        logger.info(f"成功读取技术ID映射表，共 {tech_id_mapping_df.shape[0]} 个技术")
    except Exception as e:
        logger.error(f"读取技术ID映射表失败: {e}")
        return

    # 读取强化学习应用的技术详情
    try:
        tech_selected_df = pd.read_excel(tech_selected_path)
        logger.info(f"成功读取强化学习技术详情，共 {tech_selected_df.shape[1]-1} 个县的技术分配")
    except Exception as e:
        logger.error(f"读取强化学习技术详情失败: {e}")
        return

    # 从强化学习结果中提取实际应用的技术名称列表
    applied_tech_names = set()
    
    # tech_selected_df的格式：第一行是县名，从第二行开始是技术名称
    # 遍历所有列（除了第一列可能是索引列）
    for col_idx in range(1, tech_selected_df.shape[1]):
        # 获取该列从第二行开始的所有技术名称
        county_techs = tech_selected_df.iloc[1:, col_idx]
        
        # 收集所有非空的技术名称
        for tech_name in county_techs:
            if pd.notna(tech_name) and isinstance(tech_name, str) and tech_name.strip() and '总经济成本' not in tech_name:
                applied_tech_names.add(tech_name.strip())

    logger.info(f"强化学习实际应用的技术总数: {len(applied_tech_names)}")
    if applied_tech_names:
        logger.info(f"应用的技术示例: {sorted(list(applied_tech_names))[:10]}")

    # 获取所有技术影响文件
    all_tech_files = glob.glob(os.path.join('results/level_based_stepwise_techs/level_based_stepwise_techs/tech_county_impacts', "tech_*.xlsx"))

    # 创建技术ID到文件路径的映射
    tech_id_to_file = {}
    for tech_file in all_tech_files:
        filename = os.path.basename(tech_file)
        # 解析文件名格式: tech_{id}_{tech_name}.xlsx
        parts = filename.replace('.xlsx', '').split('_', 2)
        if len(parts) >= 2:
            try:
                tech_id = int(parts[1])
                tech_id_to_file[tech_id] = tech_file
            except (ValueError, IndexError):
                continue

    logger.info(f"总共找到 {len(tech_id_to_file)} 个技术影响文件")

    # 使用技术ID映射表来匹配技术
    applied_tech_ids = set()

    for applied_tech in applied_tech_names:
        if '_' in applied_tech:
            parts = applied_tech.split('_', 1)
            tech_base_name = parts[0]
            species = parts[1] if len(parts) > 1 else ''

            # 在技术ID映射表中查找匹配的技术
            # 尝试多种匹配方式
            matching_techs = tech_id_mapping_df[
                tech_id_mapping_df['Mitigation strategy'].str.contains(tech_base_name, case=False, na=False, regex=False)
            ]

            if not matching_techs.empty:
                # 找到匹配的技术，取species = species的技术
                matching_techs = matching_techs[matching_techs['species'] == species]
                matched_tech = matching_techs.iloc[0]
                tech_id = matched_tech['技术ID']
                applied_tech_ids.add(tech_id)

    logger.info(f"成功匹配的技术ID总数: {len(applied_tech_ids)}")
    logger.info(f"总共找到 {len(all_tech_files)} 个技术影响文件")

    if not applied_tech_ids:
        logger.error("没有找到强化学习应用的技术")
        return

    # 初始化按区域分组的数据
    region_tech_impacts = {}

    # 处理每个匹配的技术
    for i, tech_id in enumerate(applied_tech_ids):
        try:
            # 从技术ID映射表中获取技术信息
            tech_info_from_mapping = tech_id_mapping_df[tech_id_mapping_df['技术ID'] == tech_id]
            if not tech_info_from_mapping.empty:
                original_tech_name = tech_info_from_mapping.iloc[0]['Mitigation strategy']
                species_info = tech_info_from_mapping.iloc[0]['species']
                species_list = [species_info] if pd.notna(species_info) else []
                tech_class = tech_info_from_mapping.iloc[0]['class']
            else:
                logger.warning(f"技术ID {tech_id} 在映射表中未找到")
                continue

            # 首先确定哪些县在强化学习技术表格中应用了当前技术
            counties_with_current_tech = []
            for col_idx in range(1, tech_selected_df.shape[1]):
                county_name = tech_selected_df.columns[col_idx]
                county_techs = tech_selected_df.iloc[1:, col_idx].dropna().astype(str).tolist()
                
                # 检查该县是否应用了当前技术
                if pd.notna(species_info) and species_info:
                    tech_applied = any(f'{original_tech_name}_{species_info}' in tech for tech in county_techs)
                else:
                    tech_applied = any(original_tech_name in tech for tech in county_techs)
                if tech_applied:
                    counties_with_current_tech.append(county_name)
            
            # 如果没有任何县应用了该技术，跳过
            if not counties_with_current_tech:
                logger.info(f"技术 {tech_id} ({original_tech_name}) 没有县应用，跳过")
                continue
                
            logger.info(f"技术 {tech_id} ({original_tech_name}) 被 {len(counties_with_current_tech)} 个县应用")

            # 获取或生成技术影响数据
            if tech_id in tech_id_to_file:
                # 文件存在，直接读取
                file_path = tech_id_to_file[tech_id]
                df = pd.read_excel(file_path)
                logger.info(f"从文件读取技术 {tech_id} 的影响数据")
                
                # 过滤掉没有应用该技术的县
                county_col = None
                for col in df.columns:
                    if '县' in col or 'County' in col or col in ['Counties', '县名', '县名称']:
                        county_col = col
                        break
                
                if county_col is not None:
                    # 只保留应用了该技术的县
                    df = df[df[county_col].isin(counties_with_current_tech)]
                    logger.info(f"过滤后保留 {len(df)} 个县的数据")
            else:
                # 文件不存在，动态生成
                df = generate_tech_impact_dataframe(tech_id, tech_info_from_mapping.iloc[0], counties_with_current_tech)
                logger.info(f"动态生成了技术 {tech_id} 的影响数据")

            # 检查是否有县名列（如果之前没有找到）
            if county_col is None:
                for col in df.columns:
                    if '县' in col or 'County' in col or col in ['Counties', '县名', '县名称']:
                        county_col = col
                        break

            if county_col is None:
                logger.warning(f"文件 {filename} 中未找到县名列")
                continue

            # 合并农业分区信息
            df_with_region = df.merge(region_df[['Counties', '所属农业区']], left_on=county_col, right_on='Counties', how='left')

            # 按农业区分组统计
            for region, group_df in df_with_region.groupby('所属农业区'):
                if region not in region_tech_impacts:
                    region_tech_impacts[region] = {}

                # 使用原始技术名作为key
                tech_key = f"{original_tech_name}_{'_'.join(species_list)}" if species_list else original_tech_name

                if tech_key not in region_tech_impacts[region]:
                    region_tech_impacts[region][tech_key] = {
                        'tech_name': original_tech_name,
                        'tech_id': tech_id,
                        'tech_class': tech_class,
                        'species': species_list,
                        '影响县数': 0,
                        '总NH3减排量': 0,
                        '总NO3减排量': 0,
                        '总N_runoff减排量': 0,
                        '总净减排量': 0
                    }

                # 计算该技术在该区域的影响
                region_impacts = region_tech_impacts[region][tech_key]

                # 数据已经在前面过滤过了，直接使用
                filtered_group_df = group_df

                # 计算有影响的县数（至少有一种气体有变化）
                affected_counties = filtered_group_df[
                    (filtered_group_df['NH3变化'] != 0) |
                    (filtered_group_df['NO3变化'] != 0) |
                    (filtered_group_df['N_runoff变化'] != 0)
                ]
                region_impacts['影响县数'] += len(affected_counties)

                # 累加减排量（注意：变化量为负值表示减排）
                region_impacts['总NH3减排量'] += filtered_group_df['NH3变化'].sum()
                region_impacts['总NO3减排量'] += filtered_group_df['NO3变化'].sum()
                region_impacts['总N_runoff减排量'] += filtered_group_df['N_runoff变化'].sum()
                region_impacts['总净减排量'] += filtered_group_df['净减排量'].sum()

        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {e}")
            continue

        if (i + 1) % 50 == 0:
            logger.info(f"已处理 {i + 1}/{len(tech_id_to_file)} 个技术文件")

    # 生成每个区域的分析结果
    generate_region_analysis_results(region_tech_impacts, output_dir)

def generate_region_analysis_results(region_tech_impacts, output_dir):
    """生成每个农业区的技术减排量分析结果"""

    output_base_dir = os.path.join(output_dir, "rl_area_tech_analysis")
    os.makedirs(output_base_dir, exist_ok=True)

    # 为每个农业区生成分析报告
    for region, tech_data in region_tech_impacts.items():
        if not tech_data:
            continue

        logger.info(f"正在生成 {region} 的分析结果...")

        # 转换为DataFrame
        summary_data = []
        for tech_key, impacts in tech_data.items():
            summary_data.append({
                '技术ID': impacts['tech_id'],
                '技术名称': impacts['tech_name'],
                '技术类别': impacts['tech_class'],
                '适用物种': ', '.join(impacts['species']) if impacts['species'] else '通用',
                '影响县数': impacts['影响县数'],
                '总NH3减排量': impacts['总NH3减排量'],
                '总NO3减排量': impacts['总NO3减排量'],
                '总N_runoff减排量': impacts['总N_runoff减排量'],
                '总净减排量': impacts['总净减排量'],
                # 计算重点气体的综合减排效果（取绝对值求和）
                '重点气体综合减排': abs(impacts['总NH3减排量']) + abs(impacts['总NO3减排量']) + abs(impacts['总N_runoff减排量'])
            })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)

            # 按重点气体综合减排量排序（绝对值越大越好）
            summary_df = summary_df.sort_values('重点气体综合减排', ascending=False)

            # 保存结果
            safe_region_name = region.replace('/', '_').replace('\\', '_')
            output_file = os.path.join(output_base_dir, f"{safe_region_name}_tech_emission_reduction_analysis.xlsx")

            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # 主要结果表
                summary_df.to_excel(writer, sheet_name='技术减排量排名', index=False)

                # 按单一气体排序的表
                for gas in ['总NH3减排量', '总NO3减排量', '总N_runoff减排量']:
                    gas_df = summary_df.sort_values(gas, ascending=True)  # 负值越大减排越多
                    gas_df.to_excel(writer, sheet_name=f'{gas}排名', index=False)

            logger.info(f"{region} 的分析结果已保存到: {output_file}")

            # 输出该区域的统计信息
            logger.info(f"{region} 统计信息:")
            logger.info(f"  - 技术总数: {len(summary_df)}")
            logger.info(f"  - 有减排效果的技术数: {len(summary_df[summary_df['重点气体综合减排'] > 0])}")
            logger.info(f"  - NH3最大减排量: {summary_df['总NH3减排量'].min():.4f}")
            logger.info(f"  - NO3最大减排量: {summary_df['总NO3减排量'].min():.4f}")
            logger.info(f"  - N_runoff最大减排量: {summary_df['总N_runoff减排量'].min():.4f}")

    # 生成全国汇总表
    generate_national_summary(region_tech_impacts, output_base_dir)

def generate_national_summary(region_tech_impacts, output_dir):
    """生成全国范围内的技术减排量汇总"""

    logger.info("正在生成全国汇总分析...")

    national_summary = {}

    for region, tech_data in region_tech_impacts.items():
        for tech_key, impacts in tech_data.items():
            if tech_key not in national_summary:
                national_summary[tech_key] = {
                    '技术名称': impacts['tech_name'],
                    '技术ID': impacts['tech_id'],
                    '技术类别': impacts['tech_class'],
                    '适用物种': ', '.join(impacts['species']) if impacts['species'] else '通用',
                    '覆盖区域数': 0,
                    '总影响县数': 0,
                    '全国NH3减排量': 0,
                    '全国NO3减排量': 0,
                    '全国N_runoff减排量': 0,
                    '全国净减排量': 0,
                    '覆盖农业区': []
                }

            national_summary[tech_key]['覆盖区域数'] += 1
            national_summary[tech_key]['总影响县数'] += impacts['影响县数']
            national_summary[tech_key]['全国NH3减排量'] += impacts['总NH3减排量']
            national_summary[tech_key]['全国NO3减排量'] += impacts['总NO3减排量']
            national_summary[tech_key]['全国N_runoff减排量'] += impacts['总N_runoff减排量']
            national_summary[tech_key]['全国净减排量'] += impacts['总净减排量']
            national_summary[tech_key]['覆盖农业区'].append(region)

    # 转换为DataFrame
    national_data = []
    for tech_key, data in national_summary.items():
        national_data.append({
            **data,
            '重点气体综合减排': abs(data['全国NH3减排量']) + abs(data['全国NO3减排量']) + abs(data['全国N_runoff减排量']),
            '覆盖农业区列表': ', '.join(data['覆盖农业区'])
        })

    if national_data:
        national_df = pd.DataFrame(national_data)

        # 按重点气体综合减排量排序
        national_df = national_df.sort_values('重点气体综合减排', ascending=False)

        # 保存全国汇总
        output_file = os.path.join(output_dir, "全国技术减排量汇总.xlsx")
        national_df.to_excel(output_file, index=False)

        logger.info(f"全国技术减排量汇总已保存到: {output_file}")
        logger.info(f"汇总了 {len(national_data)} 个技术在全国范围内的减排效果")

if __name__ == "__main__":
    # 运行分析 - 基于强化学习应用的技术
    analyze_tech_impacts_by_region()
