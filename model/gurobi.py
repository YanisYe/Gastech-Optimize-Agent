import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import os
import torch
import time
from dataLoader import TechDataLoader, CountyDataLoader
from fuzzywuzzy import process
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.utils import *

# 设置路径
current_path = os.path.dirname(__file__)
county_df_path = 'data/基础数据-县级尺度.xlsx'
soc_df_path = 'data/SOC-县尺度.xlsx'
livestock_scale_path = 'data/动物数量.xlsx'
crop_scale_path = 'data/分县种植面积.xlsx'
livestock_tech_path = 'data/畜牧业技术列单-经济产量0626.xlsx'
crop_tech_path = 'data/种植业技术列单产量产业0626.xlsx'

# 加载数据
IDs_all, \
NH3_Crop_colunms, \
N2O_nitrogen_fertilizer_columns, \
NO3_nitrogen_fertilizer_columns, \
N_runoff_columns, \
Soc_columns, \
NH3_Fecal_management_columns, \
N2O_Fecal_management_columns, \
NH3_manure_application_columns, \
N2O_manure_application_columns, \
NO3_manure_application_columns, \
Straw_columns, \
CH4_intestine_columns, \
CH4_Fecal_management_columns, \
NH3_manure_application_tensor_all, \
N2O_manure_application_tensor_all, \
NO3_manure_application_tensor_all, \
Rice_CH4_Gg_all, \
NH3_Crop_tensor_all, \
N2O_nitrogen_fertilizer_tensor_all, \
NO3_nitrogen_fertilizer_tensor_all, \
N_runoff_tensor_all, \
Soc_tensor_origin_all, \
NH3_Fecal_management_tensor_all, \
N2O_Fecal_management_tensor_all, \
Straw_tensor_all, \
CH4_intestine_tensor_all, \
CH4_Fecal_management_tensor_all, \
Total_CH4_tensor_origin_all, \
Total_N2O_tensor_origin_all, \
threshold_NO3_PB_all, \
threshold_N_runoff_PB_all, \
threshold_NH3_PB_all, \
threshold_CH4_all, \
threshold_N2O_all, \
county_scale_all = CountyDataLoader(county_df_path, soc_df_path, livestock_scale_path, crop_scale_path)

Feeding, \
Housing, \
slurry_storage, \
soild_storage, \
composting, \
additives_application, \
soild_application, \
slurry_application, \
crop, \
tech_set = TechDataLoader(livestock_tech_path, crop_tech_path)

# 计算所有县的初始气体阈值差口
gap_NO3_origin_all = torch.sum(NO3_nitrogen_fertilizer_tensor_all, axis=1) + \
                            torch.sum(NO3_manure_application_tensor_all, axis=1) - \
                            threshold_NO3_PB_all
# N流失最小减排量
gap_N_runoff_origin_all = torch.sum(N_runoff_tensor_all, axis=1) - threshold_N_runoff_PB_all

# 氨氮最小减排量
gap_NH3_origin_all = torch.sum(NH3_Crop_tensor_all, axis=1) + \
                torch.sum(NH3_Fecal_management_tensor_all, axis=1) + \
                torch.sum(NH3_manure_application_tensor_all, axis=1) - \
                threshold_NH3_PB_all

# 甲烷最小减排量
Total_CH4_origin_all = torch.sum(CH4_intestine_tensor_all, axis=1) + \
                torch.sum(CH4_Fecal_management_tensor_all, axis=1)
gap_CH4_origin_all = Total_CH4_origin_all * 0.4

# 氧化亚氮最小减排量
Total_N2O_origin_all = torch.sum(N2O_nitrogen_fertilizer_tensor_all, axis=1) + \
                torch.sum(N2O_Fecal_management_tensor_all, axis=1) + \
                torch.sum(N2O_manure_application_tensor_all, axis=1)
gap_N2O_origin_all = Total_N2O_origin_all * 0.4

# 标识需要技术添加的县
valid_counties = ~((gap_NO3_origin_all <= 0) & (gap_N_runoff_origin_all <= 0) & (gap_NH3_origin_all <= 0)).numpy()
        
# 筛选需要技术添加的县的数据
IDs = IDs_all[valid_counties].reset_index(drop=True)
NH3_manure_application_tensor = NH3_manure_application_tensor_all[valid_counties]
N2O_manure_application_tensor = N2O_manure_application_tensor_all[valid_counties]
NO3_manure_application_tensor = NO3_manure_application_tensor_all[valid_counties]
Rice_CH4_Gg = Rice_CH4_Gg_all[valid_counties]
NH3_Crop_tensor = NH3_Crop_tensor_all[valid_counties]
N2O_nitrogen_fertilizer_tensor = N2O_nitrogen_fertilizer_tensor_all[valid_counties]
NO3_nitrogen_fertilizer_tensor = NO3_nitrogen_fertilizer_tensor_all[valid_counties]
N_runoff_tensor = N_runoff_tensor_all[valid_counties]
Soc_tensor_origin = Soc_tensor_origin_all[valid_counties]
NH3_Fecal_management_tensor = NH3_Fecal_management_tensor_all[valid_counties]
N2O_Fecal_management_tensor = N2O_Fecal_management_tensor_all[valid_counties]
Straw_tensor = Straw_tensor_all[valid_counties]
CH4_intestine_tensor = CH4_intestine_tensor_all[valid_counties]
CH4_Fecal_management_tensor = CH4_Fecal_management_tensor_all[valid_counties]
threshold_NO3_PB = threshold_NO3_PB_all[valid_counties]
threshold_N_runoff_PB = threshold_N_runoff_PB_all[valid_counties]
threshold_NH3_PB = threshold_NH3_PB_all[valid_counties]

gap_NO3_origin = gap_NO3_origin_all[valid_counties]
gap_N_runoff_origin = gap_N_runoff_origin_all[valid_counties]
gap_NH3_origin = gap_NH3_origin_all[valid_counties]
gap_CH4_origin = gap_CH4_origin_all[valid_counties]
gap_N2O_origin = gap_N2O_origin_all[valid_counties]
county_scale = county_scale_all[valid_counties]

# 预计算所有技术的成本系数，避免重复计算
tech_cost_coefficients = {}
for i in range(len(tech_set)):
    tech_line = tech_set.iloc[i]
    unit_cost = tech_line['标准化经济成本']
    tech_difficulty = 1 + (-1 + int(tech_line['技术分级'])) * 0.5
    tech_cost_coefficients[i] = unit_cost * tech_difficulty

# 预计算将在函数定义后进行

def calculate_tech_cost(tech_index, county_scale_data):
    """
    计算技术成本，使用预计算的系数
    """
    tech_line = tech_set.iloc[tech_index]
    tech_industry = tech_line['Livestock species'] if tech_line['class'] != 'crop' else tech_line['Crop species']
    
    # 根据产业类型获取对应的规模数据
    if tech_line['class'] == 'crop':
        if tech_industry == 'friut':
            industry_scale = county_scale_data['fruittree_sown_area']
        else:
            industry_scale = county_scale_data['{}_sown_area'.format(tech_industry)]
    else:
        tech_industry = tech_industry.lower()
        if tech_industry in county_scale_data.index:
            industry_scale = county_scale_data[tech_industry]
        else:
            industry_scale = county_scale_data[tech_industry.replace(' ', '')]
    
    return tech_cost_coefficients[tech_index] * industry_scale

line_cache = {}
tech_mapping = {
    'Feeding': Feeding,
    'Housing': Housing,
    'slurry_storage': slurry_storage,
    'soild_storage': soild_storage,
    'composting': composting,
    'additives_application': additives_application,
    'soild_application': soild_application,
    'slurry_application': slurry_application,
    'crop': crop
}

tech_shape = [i.shape[0] for i in tech_mapping.values()]
tech_shape = np.cumsum(tech_shape)
        
county_state = {
    'NH3_Crop': NH3_Crop_tensor.clone(),
    'N2O_nitrogen_fertilizer': N2O_nitrogen_fertilizer_tensor.clone(),
    'NO3_nitrogen_fertilizer': NO3_nitrogen_fertilizer_tensor.clone(),
    'CH4_intestine': CH4_intestine_tensor.clone(),
    'CH4_Fecal_management': CH4_Fecal_management_tensor.clone(),
    'N2O_Fecal_management': N2O_Fecal_management_tensor.clone(),
    'NH3_Fecal_management': NH3_Fecal_management_tensor.clone(),
    'N2O_manure_application': N2O_manure_application_tensor.clone(),
    'NH3_manure_application': NH3_manure_application_tensor.clone(),
    'NO3_manure_application': NO3_manure_application_tensor.clone(),
    'N_runoff': N_runoff_tensor.clone(),
    'SOC': Soc_tensor_origin.clone(),
    'Rice_CH4_Gg': Rice_CH4_Gg.clone(),
    'Straw': Straw_tensor.clone(),
}   

industry_mapping = {}
subindustry_lists = {}

stateMapping = {
    '种植业NH3挥发': county_state['NH3_Crop'],
    '氮肥N2O':  county_state['N2O_nitrogen_fertilizer'],
    '氮肥NO3' : county_state['NO3_nitrogen_fertilizer'],
    '肠道CH4': county_state['CH4_intestine'],
    '粪污管理CH4': county_state['CH4_Fecal_management'],
    '粪污管理N2O': county_state['N2O_Fecal_management'],
    '粪污管理NH3': county_state['NH3_Fecal_management'],
    '粪污施用NH3': county_state['NH3_manure_application'],
    '粪污施用N2O': county_state['N2O_manure_application'],
    '粪污施用CH4': county_state['CH4_Fecal_management'],
    '粪污施用NO3': county_state['NO3_manure_application'],
    'Soc': county_state['SOC'],
    'N runoff': county_state['N_runoff'],
    'Rice CH4 Gg': county_state["Rice_CH4_Gg"]
}

class_mapping = {
    '种植业NH3挥发' : NH3_Crop_colunms,
    '氮肥N2O' : N2O_nitrogen_fertilizer_columns,
    '氮肥NO3' : NO3_nitrogen_fertilizer_columns,
    'N runoff' : N_runoff_columns,
    '秸秆焚烧' : Straw_columns,
    '粪污管理N2O': N2O_Fecal_management_columns,
    '粪污管理NH3': NH3_Fecal_management_columns,
    '粪污管理CH4': CH4_Fecal_management_columns,
    '肠道CH4': CH4_intestine_columns,
    '粪污施用N2O': N2O_manure_application_columns,
    '粪污施用NH3': NH3_manure_application_columns,
    '粪污施用NO3': NO3_manure_application_columns,
    'Rice CH4 Gg': pd.Series(['Rice CH4 Gg']),
    'Soc': Soc_columns
}

def _get_or_create_industry_mapping(sub, subindustry_classes):
    """获取预计算的行业映射
    
    Args:
        sub: 子行业名称
        subindustry_classes: 行业分类
        
    Returns:
        int: 行业索引
    """
    sub = sub.strip()
    col = industry_mapping.get(sub)
    if col is None:
        raise ValueError(f"No precomputed mapping found for {sub}. Please check precomputation.")
    return col

def _compute_N2O_delta(Delta, industry):
    """
    Calculate N2O emission reduction based on industry-specific coefficient.
    
    Args:
    Delta: Emission change value
    industry: Industry type (str)
    
    Returns:
        float: N2O emission reduction
    """
    coefficient = GAS_COEFFICIENTS['N2O'].get(industry, GAS_COEFFICIENTS['N2O']['default'])
    return Delta * coefficient


def _compute_CH4_delta(Delta, industry):
    """
    Calculate CH4 emission reduction based on industry-specific coefficient.
    
    Args:
        Delta: Emission change value
        industry: Industry type (str)
    
    Returns:
        float: CH4 emission reduction
    """
    coefficient = GAS_COEFFICIENTS['CH4'].get(industry, GAS_COEFFICIENTS['CH4']['default'])
    return Delta * coefficient

def gas_tech_optimization_gurobi(county_id):
    """
    使用Gurobi为单个县进行气体技术组合优化，目标是最小化技术成本，同时满足所有气体减排要求。
    参数:
        county_id (int): 县的ID，用于选择具体的数据。
    返回:
        dict: 包含优化结果的字典，包括所选技术和总成本。
    """
    start_time = time.time()
    # 获取县数据
    county_scale_data = county_scale.iloc[county_id]
    # 计算初始气体排放缺口
    gap_NO3 = gap_NO3_origin[county_id].item()
    gap_N_runoff = gap_N_runoff_origin[county_id].item()
    gap_NH3 = gap_NH3_origin[county_id].item()
    gap_CH4 = gap_CH4_origin[county_id].item()
    gap_N2O = gap_N2O_origin[county_id].item()

    # 计算初始气体值
    current_NH3 = (torch.sum(NH3_Crop_tensor[county_id]) + 
                   torch.sum(NH3_Fecal_management_tensor[county_id]) + 
                   torch.sum(NH3_manure_application_tensor[county_id])).item()
    current_NO3 = (torch.sum(NO3_nitrogen_fertilizer_tensor[county_id]) + 
                   torch.sum(NO3_manure_application_tensor[county_id])).item()
    current_N_runoff = torch.sum(N_runoff_tensor[county_id]).item()
    current_CH4 = (torch.sum(CH4_intestine_tensor[county_id]) + 
                  torch.sum(CH4_Fecal_management_tensor[county_id])).item()
    current_N2O = (torch.sum(N2O_nitrogen_fertilizer_tensor[county_id]) + 
                  torch.sum(N2O_Fecal_management_tensor[county_id]) + 
                  torch.sum(N2O_manure_application_tensor[county_id])).item()

    try:
        # 创建Gurobi模型
        model = gp.Model(f"Gas_Tech_Optimization_County_{county_id}")
        model.setParam('TimeLimit', 60)
        model.setParam('OutputFlag', 0)
        model.setParam('NonConvex', 2)  # 支持非凸二次约束

        # 创建二元决策变量：是否选择某项技术
        tech_vars = model.addVars(len(tech_set), vtype=GRB.BINARY, name="Tech")

        # 目标函数：最小化技术成本
        obj = gp.LinExpr()
        for i in range(len(tech_set)):
            cost = calculate_tech_cost(i, county_scale_data)
            obj += tech_vars[i] * cost
        model.setObjective(obj, GRB.MINIMIZE)

        # 收集每个技术对每种气体的影响系数（乘数）
        gas_tech_multipliers = {
            'NH3': {},
            'NO3': {},
            'N_runoff': {},
            'CH4': {},
            'N2O': {}
        }

        # 遍历每个技术，计算其对气体的影响
        for tech_index in range(len(tech_set)):
            deltas = tech_deltas_cache[tech_index]
            for industry, delta, subIndustry in deltas:
                if industry not in stateMapping:
                    continue
                countyState = stateMapping[industry]
                subindustry_classes = class_mapping[industry]
                # 遍历技术影响的子环节
                for sub in subIndustry:
                    # 获取子环节索引
                    try:
                        col = _get_or_create_industry_mapping(sub, subindustry_classes)
                    except ValueError:
                        continue
                    # 获取初始值
                    initial_value = countyState[county_id, col].item()
                    if initial_value == 0:
                        continue
                    # 判断影响的气体类型
                    if 'NH3' in sub:
                        if tech_index not in gas_tech_multipliers['NH3']:
                            gas_tech_multipliers['NH3'][tech_index] = 1.0
                        gas_tech_multipliers['NH3'][tech_index] *= (1 + delta)
                    elif 'NO3' in sub:
                        if tech_index not in gas_tech_multipliers['NO3']:
                            gas_tech_multipliers['NO3'][tech_index] = 1.0
                        gas_tech_multipliers['NO3'][tech_index] *= (1 + delta)
                    elif 'N_runoff' in sub or 'runoff' in sub.lower():
                        if tech_index not in gas_tech_multipliers['N_runoff']:
                            gas_tech_multipliers['N_runoff'][tech_index] = 1.0
                        gas_tech_multipliers['N_runoff'][tech_index] *= (1 + delta)
                    elif 'CH4' in sub:
                        if tech_index not in gas_tech_multipliers['CH4']:
                            gas_tech_multipliers['CH4'][tech_index] = 1.0
                        effective_multiplier = (1 + delta)
                        if industry in ['粪污管理CH4', '肠道CH4']:
                            coeff_multiplier = _compute_CH4_delta(initial_value, industry) / initial_value if initial_value != 0 else 1
                            effective_multiplier *= coeff_multiplier
                        gas_tech_multipliers['CH4'][tech_index] *= effective_multiplier
                    elif 'N2O' in sub:
                        if tech_index not in gas_tech_multipliers['N2O']:
                            gas_tech_multipliers['N2O'][tech_index] = 1.0
                        effective_multiplier = (1 + delta)
                        if industry in ['粪污管理NH3', '粪污管理N2O', '粪污管理NO3', '氮肥N2O', '种植业NH3挥发', '氮肥NO3', 'N runoff', '粪肥施用NH3', '粪肥施用NO3', '粪肥施用N2O', '秸秆焚烧']:
                            coeff_multiplier = _compute_N2O_delta(initial_value, industry) / initial_value if initial_value != 0 else 1
                            effective_multiplier *= coeff_multiplier
                        gas_tech_multipliers['N2O'][tech_index] *= effective_multiplier

        # 构建气体约束（使用乘积链）
        def build_gas_constraint(model, tech_vars, current_value, tech_multipliers, gas_type, threshold):
            if not tech_multipliers:
                return None
            p = [model.addVar(name=f"{gas_type}_prod_0", lb=0)]
            model.addConstr(p[0] == current_value, name=f"{gas_type}_init")
            for idx, (tech_idx, multiplier) in enumerate(tech_multipliers.items()):
                next_p = model.addVar(name=f"{gas_type}_prod_{idx+1}", lb=0)
                term = 1 + (multiplier - 1) * tech_vars[tech_idx]  # 若选中则乘以 multiplier，否则乘以 1
                model.addConstr(next_p == p[idx] * term, name=f"{gas_type}_mul_{tech_idx}")
                p.append(next_p)
            final_var = p[-1]
            model.addConstr(final_var <= threshold, name=f"{gas_type}_threshold")
            return final_var

        # 添加气体约束
        gas_thresholds = {
            'NH3': threshold_NH3_PB[county_id].item(),
            'NO3': threshold_NO3_PB[county_id].item(),
            'N_runoff': threshold_N_runoff_PB[county_id].item(),
            'CH4': current_CH4 * 0.6,
            'N2O': current_N2O * 0.6
        }

        gas_vars = {}
        for gas_type, threshold in gas_thresholds.items():
            current_value = {
                'NH3': current_NH3,
                'NO3': current_NO3,
                'N_runoff': current_N_runoff,
                'CH4': current_CH4,
                'N2O': current_N2O
            }[gas_type]
            gap = {
                'NH3': gap_NH3,
                'NO3': gap_NO3,
                'N_runoff': gap_N_runoff,
                'CH4': gap_CH4,
                'N2O': gap_N2O
            }[gas_type]

            if gap > 0 and current_value != 0:
                tech_multipliers = gas_tech_multipliers.get(gas_type, {})
                if not tech_multipliers:
                    continue
                gas_vars[gas_type] = build_gas_constraint(model, tech_vars, current_value, tech_multipliers, gas_type, threshold)

        # 添加技术冲突约束
        for i, condition in enumerate(tech_conflicts_cache):
            if condition.any():
                condition_indexes = np.where(condition)[0]
                for condition_index in condition_indexes:
                    model.addConstr(tech_vars[i] + tech_vars[condition_index] <= 1, f"Conflict_{i}_{condition_index}")

        # 优化模型
        model.optimize()

        # 获取结果
        if model.status == GRB.OPTIMAL:
            selected_techs = [i for i in range(len(tech_set)) if tech_vars[i].x > 0.5]
            total_cost = model.objVal
            status = "Optimal"
        elif model.status == GRB.TIME_LIMIT:
            if model.SolCount > 0:
                selected_techs = [i for i in range(len(tech_set)) if tech_vars[i].x > 0.5]
                total_cost = model.objVal
                status = "Time Limit with Feasible Solution"
            else:
                selected_techs = []
                total_cost = float('inf')
                status = "Time Limit Exceeded, No Solution"
        else:
            selected_techs = []
            total_cost = float('inf')
            status = f"Failed: {model.status}"

        # 计算最终气体值（从乘积链变量中获取）
        final_gases = {}
        if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
            for gas_type in gas_thresholds:
                if gas_type in gas_vars:
                    final_gases[gas_type] = gas_vars[gas_type].x
                else:
                    final_gases[gas_type] = {
                        'NH3': current_NH3,
                        'NO3': current_NO3,
                        'N_runoff': current_N_runoff,
                        'CH4': current_CH4,
                        'N2O': current_N2O
                    }[gas_type]
        else:
            final_gases = {
                'NH3': current_NH3,
                'NO3': current_NO3,
                'N_runoff': current_N_runoff,
                'CH4': current_CH4,
                'N2O': current_N2O
            }

        # 技术ID到技术名称的映射
        selected_tech_names = []
        if selected_techs:
            for tech_id in selected_techs:
                tech_row = tech_set.iloc[tech_id]
                tech_name = tech_row['Mitigation strategy']
                tech_class = tech_row['class']
                species = tech_row.get('Crop species' if tech_class == 'crop' else 'Livestock species', 'N/A')
                grade = tech_row.get('技术分级', 'N/A')
                selected_tech_names.append({
                    'tech_id': tech_id,
                    'tech_name': tech_name,
                    'tech_class': tech_class,
                    'species': species,
                    'grade': grade
                })

        end_time = time.time()
        solve_time = end_time - start_time

        return {
            'county_id': county_id,
            'county_name': IDs.iloc[county_id]['Counties'],
            'selected_techs': selected_techs,
            'selected_tech_details': selected_tech_names,
            'total_cost': total_cost,
            'status': status,
            'solve_time': solve_time,
            'initial_gas_values': {
                'NH3': current_NH3,
                'NO3': current_NO3,
                'N_runoff': current_N_runoff,
                'CH4': current_CH4,
                'N2O': current_N2O
            },
            'final_gas_values': final_gases,
            'gas_thresholds': gas_thresholds,
            'gaps': {
                'NH3': final_gases['NH3'] - gas_thresholds['NH3'],
                'NO3': final_gases['NO3'] - gas_thresholds['NO3'],
                'N_runoff': final_gases['N_runoff'] - gas_thresholds['N_runoff'],
                'CH4': final_gases['CH4'] - gas_thresholds['CH4'],
                'N2O': final_gases['N2O'] - gas_thresholds['N2O']
            }
        }
    except gp.GurobiError as e:
        print(f"Error for county {county_id}: {e}")
        return {
            'county_id': county_id,
            'county_name': IDs.iloc[county_id]['Counties'],
            'selected_techs': [],
            'selected_tech_details': [],
            'total_cost': float('inf'),
            'status': f"Gurobi Error: {e}",
            'solve_time': time.time() - start_time,
            'initial_gas_values': {
                'NH3': current_NH3,
                'NO3': current_NO3,
                'N_runoff': current_N_runoff,
                'CH4': current_CH4,
                'N2O': current_N2O
            },
            'final_gas_values': {
                'NH3': current_NH3,
                'NO3': current_NO3,
                'N_runoff': current_N_runoff,
                'CH4': current_CH4,
                'N2O': current_N2O
            },
            'gas_thresholds': gas_thresholds,
            'gaps': {
                'NH3': gap_NH3,
                'NO3': gap_NO3,
                'N_runoff': gap_N_runoff,
                'CH4': gap_CH4,
                'N2O': gap_N2O
            }
        }

def get_industry_scale(tech, county_scale_data):
    """
    获取技术的产业规模。
    """
    if tech['class'] == 'crop':
        industry = tech['Crop species']
        if industry == 'friut':
            return county_scale_data['fruittree_sown_area']
        else:
            return county_scale_data[f'{industry}_sown_area']
    else:
        industry = tech['Livestock species'].lower()
        if industry in county_scale_data.index:
            return county_scale_data[industry]
        else:
            return county_scale_data[industry.replace(' ', '')]


def _convert_percentage_to_float(percentage_str):
    """
    将百分比字符串转换为float
    
    Args:
        percentage_str (str): 百分比字符串，如 "40.0%"
    
    Returns:
        float: 转换后的浮点数，如 0.4
    """
    if not isinstance(percentage_str, str) and np.isnan(percentage_str):
        return percentage_str
    
    if isinstance(percentage_str, str) and percentage_str.endswith('%'):
        # 去掉百分号并转换为float，然后除以100
        return float(percentage_str[:-1]) / 100
    else:
        # 如果不是百分比格式，直接转换为float
        return float(percentage_str)

def get_delta(tech_index):
    """
    获取技术影响的行业和子行业。
    """
    line = get_line(tech_index)
    class_name = 'crop' if 'crop' in line.index[4].lower() else 'livestock'
    line = line[5:]  # 跳过前5列
    
    # 不包含产量指标
    num_indicators = line.shape[0] // 2 - 1
    deltas = []
    
    for i in range(num_indicators):
        delta = _convert_percentage_to_float(line.iloc[2*i])
        if delta != 0 and not np.isnan(delta):
            industry = line.index[2*i+1].split(' ', 1)[1].replace('.1', '')
            subIndustry = line.iloc[2*i+1]
            if isinstance(subIndustry, str):
                subIndustry = subIndustry.split('、')
            deltas.append((industry, delta, subIndustry))
    return deltas

def get_line(tech_index):
    """
    获取技术的影响行。
    """
    if tech_index in line_cache:
        return line_cache[tech_index]

    index = np.searchsorted(tech_shape, tech_index, side='right')
    if index == 0:
        line = tech_mapping['Feeding'].iloc[tech_index]
    else:
        line = tech_mapping[list(tech_mapping.keys())[index]].iloc[tech_index - tech_shape[index-1]]
            
    # 缓存结果
    line_cache[tech_index] = line
    return line

def get_conflicts_tech_local(tech_indices, tech_set):
    """
    获取技术冲突矩阵，参考环境中的逻辑
    """
    conflicts = []
    for i, tech_idx in enumerate(tech_indices):
        conflict_array = np.zeros(len(tech_indices), dtype=bool)
        tech_i = tech_set.iloc[tech_idx]
        
        for j, other_tech_idx in enumerate(tech_indices):
            if i != j:
                tech_j = tech_set.iloc[other_tech_idx]
                # 使用utils中的冲突逻辑
                if (tech_i['class'] == tech_j['class'] and 
                    tech_i['技术间的冲突'] == tech_j['技术间的冲突'] and
                    pd.notna(tech_i['技术间的冲突'])):
                    conflict_array[j] = True
        
        conflicts.append(conflict_array)
    
    return conflicts

if __name__ == '__main__':
    print(f"数据加载完成，共有 {len(IDs_all)} 个县")
    print(f"需要技术添加的县数量: {len(IDs)}")
    print(f"技术总数: {len(tech_set)}")
    
    # 预计算所有技术的影响，避免每个县重复计算
    print("预计算技术影响...")
    tech_deltas_cache = {}
    for tech_index in range(len(tech_set)):
        tech_deltas_cache[tech_index] = get_delta(tech_index)
    print(f"技术影响预计算完成，共 {len(tech_deltas_cache)} 个技术")

    # 预计算技术冲突矩阵，避免每个县重复计算
    print("预计算技术冲突矩阵...")
    tech_set_indexs_all = np.arange(tech_set.shape[0])
    tech_conflicts_cache = get_conflicts_tech_local(tech_set_indexs_all, tech_set)
    print(f"技术冲突矩阵预计算完成")
    
    # 预计算所有可能的行业映射，避免运行时的字符串匹配
    print("预计算行业映射...")
    all_possible_subs = set()
    for tech_index in range(len(tech_set)):
        deltas = tech_deltas_cache[tech_index]
        for industry, delta, subIndustry in deltas:
            if industry in class_mapping:
                for sub in subIndustry:
                    all_possible_subs.add((sub.strip(), id(class_mapping[industry])))

    # 预计算所有映射
    for sub, subindustry_id in all_possible_subs:
        if sub not in industry_mapping:
            # 找到对应的subindustry_classes
            subindustry_classes = None
            for industry, classes in class_mapping.items():
                if id(classes) == subindustry_id:
                    subindustry_classes = classes
                    break
            
            if subindustry_classes is not None:
                if subindustry_id not in subindustry_lists:
                    subindustry_lists[subindustry_id] = subindustry_classes.tolist()
                
                subindustry_list = subindustry_lists[subindustry_id]
                match = process.extractOne(sub, subindustry_list)
                if match:
                    matched_item = match[0]
                    industry_mapping[sub] = subindustry_classes[subindustry_classes == matched_item].index[0]

    print(f"行业映射预计算完成，共 {len(industry_mapping)} 个映射")
    
    # 为所有县创建结果列表
    result_list = []
    
    # 优化每个需要技术添加的县
    print("处理需要技术添加的县...")
    start_total_time = time.time()
    
    for i in range(len(IDs)):
        county_start_time = time.time()
        result = gas_tech_optimization_gurobi(i)
        county_end_time = time.time()
        
        # 计算平均时间和预估剩余时间
        elapsed_total = county_end_time - start_total_time
        avg_time_per_county = elapsed_total / (i + 1)
        remaining_counties = len(IDs) - (i + 1)
        estimated_remaining_time = avg_time_per_county * remaining_counties
        
        print(f"[{i+1}/{len(IDs)}] 县 {result['county_name']} 完成")
        print(f"  技术数量: {len(result['selected_techs'])}, 状态: {result['status']}")
        print(f"  求解时间: {result['solve_time']:.2f}s, 平均: {avg_time_per_county:.2f}s/县")
        print(f"  预估剩余时间: {estimated_remaining_time/60:.1f}分钟")
        print('------------------------------')
        
        # 构建技术详情字符串
        tech_details_str = ""
        if result['selected_tech_details']:
            tech_details_str = "; ".join([f"{tech['tech_name']} (ID:{tech['tech_id']}, 类别:{tech['tech_class']}, 种类:{tech['species']}, 等级:{tech['grade']})" 
                                        for tech in result['selected_tech_details']])
        
        result_list.append({'id': result['county_id'],
                            'county': result['county_name'], 
                            'techs': result['selected_techs'], 
                            'tech_details': tech_details_str,
                            'tech_amount': len(result['selected_techs']),
                            'cost': result['total_cost'],
                            'status': result['status'],
                            'solve_time': result['solve_time'],
                            # 初始气体值
                            'initial_NH3': result['initial_gas_values']['NH3'],
                            'initial_NO3': result['initial_gas_values']['NO3'],
                            'initial_N_runoff': result['initial_gas_values']['N_runoff'],
                            'initial_CH4': result['initial_gas_values']['CH4'],
                            'initial_N2O': result['initial_gas_values']['N2O'],
                            # 最终气体值
                            'final_NH3': result['final_gas_values']['NH3'],
                            'final_NO3': result['final_gas_values']['NO3'],
                            'final_N_runoff': result['final_gas_values']['N_runoff'],
                            'final_CH4': result['final_gas_values']['CH4'],
                            'final_N2O': result['final_gas_values']['N2O'],
                            # 气体阈值
                            'threshold_NH3': result['gas_thresholds']['NH3'],
                            'threshold_NO3': result['gas_thresholds']['NO3'],
                            'threshold_N_runoff': result['gas_thresholds']['N_runoff'],
                            'threshold_CH4': result['gas_thresholds']['CH4'],
                            'threshold_N2O': result['gas_thresholds']['N2O'],
                            # 气体缺口
                            'NH3_gap': result['gaps']['NH3'],
                            'NO3_gap': result['gaps']['NO3'],
                            'N_runoff_gap': result['gaps']['N_runoff'],
                            'CH4_gap': result['gaps']['CH4'],
                            'N2O_gap': result['gaps']['N2O'],
                            # 减排效果
                            'NH3_reduction': result['initial_gas_values']['NH3'] - result['final_gas_values']['NH3'],
                            'NO3_reduction': result['initial_gas_values']['NO3'] - result['final_gas_values']['NO3'],
                            'N_runoff_reduction': result['initial_gas_values']['N_runoff'] - result['final_gas_values']['N_runoff'],
                            'CH4_reduction': result['initial_gas_values']['CH4'] - result['final_gas_values']['CH4'],
                            'N2O_reduction': result['initial_gas_values']['N2O'] - result['final_gas_values']['N2O'],
                            'needs_tech': True})
    
    # 处理不需要技术添加的县
    print("添加不需要技术添加的县...")
    counties_with_tech = set([result['county'] for result in result_list])
    
    for i in range(len(IDs_all)):
        county_name = IDs_all.iloc[i]['Counties']
        if county_name not in counties_with_tech:
            # 计算该县的气体缺口和当前排放量
            gap_NO3 = gap_NO3_origin_all[i].item()
            gap_N_runoff = gap_N_runoff_origin_all[i].item()
            gap_NH3 = gap_NH3_origin_all[i].item()
            gap_CH4 = gap_CH4_origin_all[i].item()
            gap_N2O = gap_N2O_origin_all[i].item()
            
            # 计算当前排放量
            current_NH3 = (torch.sum(NH3_Crop_tensor_all[i]) + 
                          torch.sum(NH3_Fecal_management_tensor_all[i]) + 
                          torch.sum(NH3_manure_application_tensor_all[i])).item()
            current_NO3 = (torch.sum(NO3_nitrogen_fertilizer_tensor_all[i]) + 
                          torch.sum(NO3_manure_application_tensor_all[i])).item()
            current_N_runoff = torch.sum(N_runoff_tensor_all[i]).item()
            current_CH4 = (torch.sum(CH4_intestine_tensor_all[i]) + 
                          torch.sum(CH4_Fecal_management_tensor_all[i])).item()
            current_N2O = (torch.sum(N2O_nitrogen_fertilizer_tensor_all[i]) + 
                          torch.sum(N2O_Fecal_management_tensor_all[i]) + 
                          torch.sum(N2O_manure_application_tensor_all[i])).item()
            
            result_list.append({
                'id': i,
                'county': county_name,
                'techs': [],  # 不需要技术
                'tech_details': "",
                'tech_amount': 0,
                'cost': 0.0,
                'status': 'No Tech Needed',
                'solve_time': 0.0,
                # 初始气体值（等于最终值，因为没有技术）
                'initial_NH3': current_NH3,
                'initial_NO3': current_NO3,
                'initial_N_runoff': current_N_runoff,
                'initial_CH4': current_CH4,
                'initial_N2O': current_N2O,
                # 最终气体值（等于初始值，因为没有技术）
                'final_NH3': current_NH3,
                'final_NO3': current_NO3,
                'final_N_runoff': current_N_runoff,
                'final_CH4': current_CH4,
                'final_N2O': current_N2O,
                # 气体阈值
                'threshold_NH3': threshold_NH3_PB_all[i].item(),
                'threshold_NO3': threshold_NO3_PB_all[i].item(),
                'threshold_N_runoff': threshold_N_runoff_PB_all[i].item(),
                'threshold_CH4': current_CH4 * 0.6,  # 减排40%后的目标值
                'threshold_N2O': current_N2O * 0.6,  # 减排40%后的目标值
                # 气体缺口
                'NH3_gap': gap_NH3,
                'NO3_gap': gap_NO3,
                'N_runoff_gap': gap_N_runoff,
                'CH4_gap': gap_CH4,
                'N2O_gap': gap_N2O,
                # 减排效果（没有技术，所以都是0）
                'NH3_reduction': 0.0,
                'NO3_reduction': 0.0,
                'N_runoff_reduction': 0.0,
                'CH4_reduction': 0.0,
                'N2O_reduction': 0.0,
                'needs_tech': False
            })
            print(f"县 {county_name} 不需要技术添加")
    
    # 按县ID排序
    result_list.sort(key=lambda x: x['id'])
    
    result_df = pd.DataFrame(result_list)
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    result_df.to_excel('results/gurobi_result_all_counties.xlsx', index=False)
    
    print(f"\n总计处理了 {len(result_list)} 个县")
    print(f"需要技术添加的县: {sum(1 for r in result_list if r['needs_tech'])}")
    print(f"不需要技术添加的县: {sum(1 for r in result_list if not r['needs_tech'])}")
    print("结果已保存到 results/gurobi_result_all_counties.xlsx")