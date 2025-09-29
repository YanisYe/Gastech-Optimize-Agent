import os
import torch
import time
from dataLoader import TechDataLoader, CountyDataLoader, AgriAreaCountyDataLoader
from fuzzywuzzy import process
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.utils import *
from model.logger import setup_logger
import numpy as np
import pandas as pd
from gurobipy import GRB
import gurobipy as gp
import copy
import multiprocessing as mp

# Parallel processing and logging configuration
ENABLE_DEBUG = os.getenv('OPT_DEBUG', '0') == '1'
# ENABLE_DEBUG = True # Enable debug mode
NUM_PROCS = int(os.getenv('OPT_PROCS', str(max(1, (os.cpu_count() or 2) - 1))))
GRB_THREADS_PER_PROC = int(os.getenv('GRB_THREADS_PER_PROC', '1'))

# Setup logger
logger = setup_logger(output=True, name="nonlinear_opt")

# Helper function definitions
def calculate_unaffected_emissions(gas_sub_multipliers, sub_initial, gas_type):
    """Calculate total emissions unaffected by technologies"""
    unaffected_total = 0
    
    for (industry, col), tech_multiplier_dict in gas_sub_multipliers.items():
        if not tech_multiplier_dict:  # Empty dictionary means unaffected by technology
            initial_value = sub_initial[(industry, col)]
            
            # Apply weighting coefficients
            if gas_type == 'N2O':
                weight = GAS_COEFFICIENTS['N2O'].get(industry, GAS_COEFFICIENTS['N2O']['default'])
            elif gas_type == 'CH4':
                weight = GAS_COEFFICIENTS['CH4'].get(industry, GAS_COEFFICIENTS['CH4']['default'])
            else:
                weight = 1.0
            
            unaffected_total += weight * initial_value
    
    return unaffected_total

def adjust_thresholds_for_technical_feasibility(gas_thresholds, gas_sub_multipliers, sub_initial):
    """Adjust thresholds to ensure technical feasibility"""
    adjusted_thresholds = gas_thresholds.copy()
    
    # for gas_type in ['CH4', 'N2O']:
    for gas_type in gas_thresholds.keys():
        if gas_type in gas_sub_multipliers:
            # Calculate emissions unaffected by technology
            unaffected_emissions = calculate_unaffected_emissions(
                gas_sub_multipliers[gas_type], sub_initial, gas_type
            )
            
            # If threshold is less than unaffected emissions, adjust threshold
            if gas_thresholds[gas_type] < unaffected_emissions:
                adjusted_thresholds[gas_type] = unaffected_emissions + 1
                if ENABLE_DEBUG:
                    logger.info(f"Adjust {gas_type} threshold: {gas_thresholds[gas_type]:.4f} -> {adjusted_thresholds[gas_type]:.4f}")
                    logger.info(f"  Unaffected emissions: {unaffected_emissions:.4f}")
    
    return adjusted_thresholds

def build_gas_constraint(model, tech_multipliers, gas_type, threshold, tech_vars, sub_initial):
    """Build gas constraints using improved product chain logic"""
    if not tech_multipliers:
        return None
    
    total_expr = 0  # Used to accumulate all sub-industry weighted emissions
    
    for (industry, col), tech_multiplier_dict in tech_multipliers.items():
        initial_value = sub_initial[(industry, col)]
        
        if gas_type == 'N2O':
            weight = GAS_COEFFICIENTS['N2O'].get(industry, GAS_COEFFICIENTS['N2O']['default'])
        elif gas_type == 'CH4':
            weight = GAS_COEFFICIENTS['CH4'].get(industry, GAS_COEFFICIENTS['CH4']['default'])
        else:
            weight = 1.0
        
        if tech_multiplier_dict:
            # Use list to manage intermediate variables, avoiding variable reference issues
            p = [model.addVar(name=f"{gas_type}_{industry}_{col}_prod_0")]
            model.addConstr(p[0] == initial_value, name=f"{gas_type}_{industry}_{col}_init")
            
            for idx, (tech_idx, multiplier) in enumerate(tech_multiplier_dict.items()):
                next_p = model.addVar(name=f"{gas_type}_{industry}_{col}_prod_{idx+1}")
                term = 1 + (multiplier - 1) * tech_vars[tech_idx]
                model.addConstr(next_p == p[idx] * term, name=f"{gas_type}_{industry}_{col}_mul_{tech_idx}")
                p.append(next_p)
            
            final_var = p[-1]
            total_expr += weight * final_var
        else:
            total_expr += weight * initial_value
    
    # Directly add threshold constraint, no need for additional total emission variables
    model.addConstr(total_expr <= threshold, name=f"{gas_type}_threshold")
    
    return total_expr

# Set paths
current_path = os.path.dirname(__file__)
county_df_path = 'data/基础数据-县级尺度.xlsx'
IDs_df = "data/县市亚区.xlsx"
soc_df_path = 'data/SOC-县尺度.xlsx'
livestock_scale_path = 'data/动物数量.xlsx'
crop_scale_path = 'data/分县种植面积.xlsx'
livestock_tech_path = 'data/畜牧业技术列单-经济产量0827.xlsx'
crop_tech_path = 'data/种植业技术列单产量产业0803.xlsx'

# First load national data to calculate national thresholds and agricultural region emission ratios
IDs_all, \
NH3_Crop_columns, \
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
nitrogen_deposition_N2O_tensor, \
straw_returning_N2O_tensor, \
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
county_scale_original_all, \
county_scale_all = CountyDataLoader(county_df_path, IDs_df, soc_df_path, livestock_scale_path, crop_scale_path)

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

# Calculate total national emissions and thresholds for various gases
# NH3 total emissions
Total_NH3_national = torch.sum(NH3_Crop_tensor_all, axis=1) + \
                     torch.sum(NH3_Fecal_management_tensor_all, axis=1) + \
                     torch.sum(NH3_manure_application_tensor_all, axis=1)

gap_NH3_origin = Total_NH3_national - threshold_NH3_PB_all

# NO3 total emissions
Total_NO3_national = torch.sum(NO3_nitrogen_fertilizer_tensor_all, axis=1) + \
                     torch.sum(NO3_manure_application_tensor_all, axis=1)

gap_NO3_origin = Total_NO3_national - threshold_NO3_PB_all

# N_runoff total emissions
Total_N_runoff_national = torch.sum(N_runoff_tensor_all, axis=1)

gap_N_runoff_origin = Total_N_runoff_national - threshold_N_runoff_PB_all

# CH4 total emissions
coef_CH4 = GAS_COEFFICIENTS['CH4']
Total_CH4_national = coef_CH4['default'] * torch.sum(CH4_intestine_tensor_all, axis=1) + \
                     coef_CH4['default'] * torch.sum(CH4_Fecal_management_tensor_all, axis=1) + \
                     coef_CH4['default'] * torch.sum(Straw_tensor_all[:, :3], axis=1) + \
                     coef_CH4['Rice CH4 Gg'] * torch.sum(Rice_CH4_Gg_all, axis=1)

# N2O total emissions
coef_N2O = GAS_COEFFICIENTS['N2O']
Total_N2O_national = 1 * torch.sum(nitrogen_deposition_N2O_tensor, axis=1) + \
                     coef_N2O['粪污管理N2O'] * torch.sum(N2O_Fecal_management_tensor_all, axis=1) + \
                     coef_N2O['粪污管理NH3'] * torch.sum(NH3_Fecal_management_tensor_all, axis=1) + \
                     coef_N2O['氮肥N2O'] * torch.sum(N2O_nitrogen_fertilizer_tensor_all, axis=1) + \
                     coef_N2O['种植业NH3挥发'] * torch.sum(NH3_Crop_tensor_all, axis=1) + \
                     coef_N2O['氮肥NO3'] * torch.sum(NO3_nitrogen_fertilizer_tensor_all, axis=1) + \
                     coef_N2O['N runoff'] * torch.sum(N_runoff_tensor_all, axis=1) + \
                     coef_N2O['粪污施用NH3'] * torch.sum(NH3_manure_application_tensor_all, axis=1) + \
                     coef_N2O['粪污施用NO3'] * torch.sum(NO3_manure_application_tensor_all, axis=1) + \
                     coef_N2O['粪污施用N2O'] * torch.sum(N2O_manure_application_tensor_all, axis=1) + \
                     coef_N2O['秸秆焚烧'] * torch.sum(Straw_tensor_all[:, 3:], axis=1) + \
                     1 * torch.sum(straw_returning_N2O_tensor, axis=1)


counties_need_tech = ~((gap_NO3_origin <= 0) & (gap_N_runoff_origin <= 0) & (gap_NH3_origin <= 0)).numpy()

# Read exceeded county data
exceed_counties_df = pd.read_excel('data/超标县域.xlsx')
exceed_counties_mask = exceed_counties_df['超标县'] == 1

# Get list of county names in exceeded counties
exceed_county_names = exceed_counties_df[exceed_counties_mask]['所属地州'].tolist()

# Create boolean mask for exceeded counties (based on county name matching)
exceed_counties_bool = IDs_all['Counties'].isin(exceed_county_names)

# Counties exceeding three gases
gas_exceed_mask = ~((gap_NO3_origin <= 0) & 
                    (gap_N_runoff_origin <= 0) & 
                    (gap_NH3_origin <= 0))
gas_exceed_mask = gas_exceed_mask.numpy()  # Convert to numpy array

# Counties needing technology: exceeding three gases OR marked as 1 in exceeded counties Excel
counties_need_tech = (gas_exceed_mask | exceed_counties_bool)

# Total national emissions for various gases
national_NH3_total = torch.sum(Total_NH3_national).item()
national_NO3_total = torch.sum(Total_NO3_national).item()
national_N_runoff_total = torch.sum(Total_N_runoff_national).item()
national_CH4_total = torch.sum(Total_CH4_national).item()
national_N2O_total = torch.sum(Total_N2O_national).item()

# National reduction targets for various gases (using thresholds)
national_NH3_reduction_target = torch.sum(threshold_NH3_PB_all).item()
national_NO3_reduction_target = torch.sum(threshold_NO3_PB_all).item()
national_N_runoff_reduction_target = torch.sum(threshold_N_runoff_PB_all).item()
national_CH4_reduction_target = threshold_CH4_all.item()
national_N2O_reduction_target = threshold_N2O_all.item()

logger.info(f"National NH3 total emissions: {national_NH3_total:.2f}")
logger.info(f"National NO3 total emissions: {national_NO3_total:.2f}")
logger.info(f"National N_runoff total emissions: {national_N_runoff_total:.2f}")
logger.info(f"National CH4 total emissions: {national_CH4_total:.2f}")
logger.info(f"National N2O total emissions: {national_N2O_total:.2f}")
logger.info(f"National NH3 reduction target: {national_NH3_reduction_target:.2f}")
logger.info(f"National NO3 reduction target: {national_NO3_reduction_target:.2f}")
logger.info(f"National N_runoff reduction target: {national_N_runoff_reduction_target:.2f}")
logger.info(f"National CH4 reduction target: {national_CH4_reduction_target:.2f}")
logger.info(f"National N2O reduction target: {national_N2O_reduction_target:.2f}")

IDs_filtered = IDs_all[counties_need_tech].reset_index(drop=True)

# Calculate emission ratio and allocate reduction responsibility based on filtered counties needing technology addition
agri_areas = IDs_filtered['所属农业亚区'].unique()
agri_area_reduction = pd.read_excel('results/reduction_variables.xlsx', sheet_name='Agricultural Region Data')

# Filter all related tensors and data, only keep counties needing technology addition
IDs_filtered = IDs_all[counties_need_tech].reset_index(drop=True)
NH3_Crop_tensor_filtered = NH3_Crop_tensor_all[counties_need_tech]
N2O_nitrogen_fertilizer_tensor_filtered = N2O_nitrogen_fertilizer_tensor_all[counties_need_tech]
NO3_nitrogen_fertilizer_tensor_filtered = NO3_nitrogen_fertilizer_tensor_all[counties_need_tech]
N_runoff_tensor_filtered = N_runoff_tensor_all[counties_need_tech]
NH3_Fecal_management_tensor_filtered = NH3_Fecal_management_tensor_all[counties_need_tech]
N2O_Fecal_management_tensor_filtered = N2O_Fecal_management_tensor_all[counties_need_tech]
NH3_manure_application_tensor_filtered = NH3_manure_application_tensor_all[counties_need_tech]
N2O_manure_application_tensor_filtered = N2O_manure_application_tensor_all[counties_need_tech]
NO3_manure_application_tensor_filtered = NO3_manure_application_tensor_all[counties_need_tech]
CH4_intestine_tensor_filtered = CH4_intestine_tensor_all[counties_need_tech]
CH4_Fecal_management_tensor_filtered = CH4_Fecal_management_tensor_all[counties_need_tech]
Straw_tensor_filtered = Straw_tensor_all[counties_need_tech]
Rice_CH4_Gg_filtered = Rice_CH4_Gg_all[counties_need_tech]
nitrogen_deposition_N2O_tensor_filtered = nitrogen_deposition_N2O_tensor[counties_need_tech]
straw_returning_N2O_tensor_filtered = straw_returning_N2O_tensor[counties_need_tech]
Soc_tensor_origin_filtered = Soc_tensor_origin_all[counties_need_tech]
county_scale_filtered = county_scale_all[counties_need_tech].reset_index(drop=True)

# Calculate initial gas threshold gaps for all counties (maintain original logic for NO3, N_runoff, NH3)
gap_NO3_origin_all = torch.sum(NO3_nitrogen_fertilizer_tensor_all, axis=1) + \
                            torch.sum(NO3_manure_application_tensor_all, axis=1) - \
                            threshold_NO3_PB_all
# Minimum N runoff reduction amount
gap_N_runoff_origin_all = torch.sum(N_runoff_tensor_all, axis=1) - threshold_N_runoff_PB_all

# Minimum NH3 reduction amount
gap_NH3_origin_all = torch.sum(NH3_Crop_tensor_all, axis=1) + \
                torch.sum(NH3_Fecal_management_tensor_all, axis=1) + \
                torch.sum(NH3_manure_application_tensor_all, axis=1) - \
                threshold_NH3_PB_all

# Filter threshold data
threshold_NO3_PB_filtered = threshold_NO3_PB_all[counties_need_tech]
threshold_N_runoff_PB_filtered = threshold_N_runoff_PB_all[counties_need_tech]
threshold_NH3_PB_filtered = threshold_NH3_PB_all[counties_need_tech]

# Filter gap data
gap_NO3_origin_filtered = gap_NO3_origin_all[counties_need_tech]
gap_N_runoff_origin_filtered = gap_N_runoff_origin_all[counties_need_tech]
gap_NH3_origin_filtered = gap_NH3_origin_all[counties_need_tech]

logger.info(f"Filtered county data shape: {IDs_filtered.shape}")
logger.info(f"Filtered county name examples: {IDs_filtered['Counties'].head().tolist()}")
unique_areas = [str(area) for area in IDs_filtered['所属农业亚区'].unique()]
logger.info(f"Agricultural areas needing technology: {sorted(unique_areas)}")


Total_NH3_filtered = torch.sum(NH3_Crop_tensor_filtered, axis=1) + \
                     torch.sum(NH3_Fecal_management_tensor_filtered, axis=1) + \
                     torch.sum(NH3_manure_application_tensor_filtered, axis=1)

Total_NO3_filtered = torch.sum(NO3_nitrogen_fertilizer_tensor_filtered, axis=1) + \
                     torch.sum(NO3_manure_application_tensor_filtered, axis=1)

Total_N_runoff_filtered = torch.sum(N_runoff_tensor_filtered, axis=1)

Total_CH4_filtered = coef_CH4['default'] * torch.sum(CH4_intestine_tensor_filtered, axis=1) + \
                     coef_CH4['default'] * torch.sum(CH4_Fecal_management_tensor_filtered, axis=1) + \
                     coef_CH4['default'] * torch.sum(Straw_tensor_filtered[:, :3], axis=1) + \
                     coef_CH4['Rice CH4 Gg'] * torch.sum(Rice_CH4_Gg_filtered, axis=1)

Total_N2O_filtered = 1 * torch.sum(nitrogen_deposition_N2O_tensor_filtered, axis=1) + \
                     coef_N2O['粪污管理N2O'] * torch.sum(N2O_Fecal_management_tensor_filtered, axis=1) + \
                     coef_N2O['粪污管理NH3'] * torch.sum(NH3_Fecal_management_tensor_filtered, axis=1) + \
                     coef_N2O['氮肥N2O'] * torch.sum(N2O_nitrogen_fertilizer_tensor_filtered, axis=1) + \
                     coef_N2O['种植业NH3挥发'] * torch.sum(NH3_Crop_tensor_filtered, axis=1) + \
                     coef_N2O['氮肥NO3'] * torch.sum(NO3_nitrogen_fertilizer_tensor_filtered, axis=1) + \
                     coef_N2O['N runoff'] * torch.sum(N_runoff_tensor_filtered, axis=1) + \
                     coef_N2O['粪污施用NH3'] * torch.sum(NH3_manure_application_tensor_filtered, axis=1) + \
                     coef_N2O['粪污施用NO3'] * torch.sum(NO3_manure_application_tensor_filtered, axis=1) + \
                     coef_N2O['粪污施用N2O'] * torch.sum(N2O_manure_application_tensor_filtered, axis=1) + \
                     coef_N2O['秸秆焚烧'] * torch.sum(Straw_tensor_filtered[:, 3:], axis=1) + \
                     1 * torch.sum(straw_returning_N2O_tensor_filtered, axis=1)

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
    'NH3_Crop': NH3_Crop_tensor_filtered.clone(),
    'N2O_nitrogen_fertilizer': N2O_nitrogen_fertilizer_tensor_filtered.clone(),
    'NO3_nitrogen_fertilizer': NO3_nitrogen_fertilizer_tensor_filtered.clone(),
    'CH4_intestine': CH4_intestine_tensor_filtered.clone(),
    'CH4_Fecal_management': CH4_Fecal_management_tensor_filtered.clone(),
    'N2O_Fecal_management': N2O_Fecal_management_tensor_filtered.clone(),
    'NH3_Fecal_management': NH3_Fecal_management_tensor_filtered.clone(),
    'N2O_manure_application': N2O_manure_application_tensor_filtered.clone(),
    'NH3_manure_application': NH3_manure_application_tensor_filtered.clone(),
    'NO3_manure_application': NO3_manure_application_tensor_filtered.clone(),
    'N_runoff': N_runoff_tensor_filtered.clone(),
    'SOC': Soc_tensor_origin_filtered.clone(),
    'Rice_CH4_Gg': Rice_CH4_Gg_filtered.clone(),
    'Straw': Straw_tensor_filtered.clone(),
}   

industry_mapping = {}
subindustry_lists = {}
class_mapping = {
    '种植业NH3挥发' : NH3_Crop_columns,
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
    'Soc': Soc_columns,
    '氮沉降': pd.Series(['大气氮沉降引起的N2O间接排放']),
    '秸秆还田': pd.Series(['秸秆还田N2O排放'])
}

def _get_or_create_industry_mapping(sub, subindustry_classes):
    """获取预计算的行业映射
    
    Args:
        sub: 子行业名称
        subindustry_classes: 行业分类
        
    Returns:
        int: 行业索引
    """
    sub_ = sub.strip()
    if not sub:  # 如果字符串为空，返回默认值或抛出异常
        raise ValueError(f"Sub {sub} not found in subindustry_classes")
    
    col = industry_mapping.get(sub_)
    if col is None:
        try:
            return list(subindustry_classes.values).index(sub)
        except:
            raise ValueError(f"Sub {sub} not found in subindustry_classes")
    return col

def _compute_N2O_delta(industry):
    """
    Calculate N2O emission reduction based on industry-specific coefficient.
    
    Args:
    Delta: Emission change value
    industry: Industry type (str)
    
    Returns:
        float: N2O emission reduction
    """
    coefficient = GAS_COEFFICIENTS['N2O'].get(industry, GAS_COEFFICIENTS['N2O']['default'])
    return coefficient


def _compute_CH4_delta(industry):
    """
    Calculate CH4 emission reduction based on industry-specific coefficient.
    
    Args:
        Delta: Emission change value
        industry: Industry type (str)
    
    Returns:
        float: CH4 emission reduction
    """
    coefficient = GAS_COEFFICIENTS['CH4'].get(industry, GAS_COEFFICIENTS['CH4']['default'])
    return coefficient

def check_tech_scale_zero(tech_index, county_scale_data):
    """
    检查技术对应的作物/牲畜的规模或面积是否为0
    
    Args:
        tech_index: 技术索引
        county_scale_data: 县级规模数据
        
    Returns:
        bool: True表示规模或面积为0，不能选择该技术；False表示可以选择
    """
    tech_line = tech_set.iloc[tech_index]
    tech_industry = tech_line['Livestock species'] if tech_line['class'] != 'crop' else tech_line['Crop species']
    
    # 根据产业类型获取对应的规模数据
    if tech_line['class'] == 'crop':
        # 种植业：检查种植面积
        if tech_industry == 'friut':
            industry_scale = county_scale_data['fruittree_sown_area']
        else:
            industry_scale = county_scale_data['{}_sown_area'.format(tech_industry)]
    else:
        # 畜牧业：检查动物数量
        tech_industry = tech_industry.lower()
        if tech_industry in county_scale_data.index:
            industry_scale = county_scale_data[tech_industry]
        else:
            industry_scale = county_scale_data[tech_industry.replace(' ', '')]
    
    # 如果规模或面积为0，则不能选择该技术
    return industry_scale == 0

def county_gas_tech_optimization(county_idx, relaxed=False):
    """
    Perform gas technology combination optimization for a single county.
    Objective is to minimize technology costs while meeting all gas reduction requirements.
    Uses Gurobi solver to handle linear constraints.

    Args:
        county_idx (int): County index in filtered data

    Returns:
        dict: Dictionary containing optimization results, including selected technologies and total cost.
    """
    # try:
    start_time = time.time()

    # Get county data
    county_data = IDs_filtered.iloc[county_idx]
    county_name = county_data['Counties']
    area_name = county_data['所属农业亚区']
    major_region = county_data['所属农业区']
    
    # Get county reduction targets (based on county threshold gaps)
    county_NH3_target = threshold_NH3_PB_filtered[county_idx].item()
    county_NO3_target = threshold_NO3_PB_filtered[county_idx].item()
    county_N_runoff_target = threshold_N_runoff_PB_filtered[county_idx].item()
    
    # For CH4 and N2O, use agricultural region reduction targets allocated by emission ratio
    try:
        area_reduction_row = agri_area_reduction[agri_area_reduction['Agricultural Region'] == area_name]
        if len(area_reduction_row) == 0:
            area_CH4_target = 0
            area_N2O_target = 0
        else:
            area_CH4_target = area_reduction_row['CH4 Reduction Target'].iloc[0]
            area_N2O_target = area_reduction_row['N2O Reduction Target'].iloc[0]
    except Exception as e:
        if ENABLE_DEBUG:
            logger.error(f"Error reading reduction targets for {area_name}: {e}")
        area_CH4_target = 0
        area_N2O_target = 0
    
    # 计算该县在农业亚区中的CH4和N2O排放占比
    area_mask = IDs_filtered['所属农业亚区'] == area_name
    area_indices = area_mask[area_mask].index
    
    current_CH4_area = Total_CH4_filtered[area_indices]
    current_N2O_area = Total_N2O_filtered[area_indices]
    total_CH4_area = torch.sum(current_CH4_area).item()
    total_N2O_area = torch.sum(current_N2O_area).item()
    
    county_CH4 = Total_CH4_filtered[county_idx].item()
    county_N2O = Total_N2O_filtered[county_idx].item()
    
    condition_county_set = [
        '安平县', '保亭黎族苗族自治县', '定安县', '丰县', '峰峰矿区', '富平县', '海盐县', '恒山区', '罗江区', '麻山区', '沛县', '邳州市',
        '琼中黎族苗族自治县', '三亚市', '睢宁县', '随县', '台安县', '腾冲市', '铜山区', '屯昌县', '温江区', '五指山市', '襄城区',
        '新沂市', '延平区', '源汇区', '枣阳市', '安平县', '保亭黎族苗族自治县', '定安县', '丰县', '峰峰矿区', '富平县', '海盐县',
        '恒山区', '罗江区', '麻山区', '沛县', '邳州市', '琼中黎族苗族自治县', '三亚市', '睢宁县', '随县', '台安县', '腾冲市',
        '屯昌县', '温江区', '五指山市', '襄城区', '新沂市', '延平区', '源汇区', '枣阳市', '大丰区', '海宁市', '乐亭县'
    ]

    condition_N2O_set = ['白沙黎族自治县', '安定区', '拜泉县', '北安市', '大方县', '大通回族土族自治县', '大新县', '当雄县', '滴道区', '定安县', '掇刀区',
                         '上思县', '龙州县', '扶绥县', '江州区']
    # Allocate CH4 and N2O reduction targets by emission ratio
    if total_CH4_area > 0:
        county_CH4_ratio = county_CH4 / total_CH4_area
        county_CH4_target = area_CH4_target * county_CH4_ratio
        if county_CH4 > county_CH4_target:
            county_CH4_target = county_CH4_target * 1.5
        if relaxed:
            if county_name in condition_county_set:
                county_CH4_target = county_CH4
            else:
                county_CH4_target = county_CH4_target * 1.5
    else:
        county_CH4_target = 0
        
    if total_N2O_area > 0:
        county_N2O_ratio = county_N2O / total_N2O_area
        county_N2O_target = area_N2O_target * county_N2O_ratio 
        if county_N2O > county_N2O_target:
            county_N2O_target = county_N2O_target * 1.5
        if relaxed:
            if county_name in condition_county_set:
                county_N2O_target = county_N2O
            else:
                county_N2O_target = county_N2O_target * 1.5
            if county_name in condition_N2O_set:
                county_N2O_target = county_N2O_target * 1.5
            
                        
    else:
        county_N2O_target = 0

    # Get single county data (maintain original data structure but only include one county)
    IDs_area = IDs_filtered.iloc[county_idx:county_idx+1].reset_index(drop=True)
    NH3_Crop_tensor_area = NH3_Crop_tensor_filtered[county_idx:county_idx+1]
    N2O_nitrogen_fertilizer_tensor_area = N2O_nitrogen_fertilizer_tensor_filtered[county_idx:county_idx+1]
    NO3_nitrogen_fertilizer_tensor_area = NO3_nitrogen_fertilizer_tensor_filtered[county_idx:county_idx+1]
    N_runoff_tensor_area = N_runoff_tensor_filtered[county_idx:county_idx+1]
    NH3_Fecal_management_tensor_area = NH3_Fecal_management_tensor_filtered[county_idx:county_idx+1]
    N2O_Fecal_management_tensor_area = N2O_Fecal_management_tensor_filtered[county_idx:county_idx+1]
    NH3_manure_application_tensor_area = NH3_manure_application_tensor_filtered[county_idx:county_idx+1]
    N2O_manure_application_tensor_area = N2O_manure_application_tensor_filtered[county_idx:county_idx+1]
    NO3_manure_application_tensor_area = NO3_manure_application_tensor_filtered[county_idx:county_idx+1]
    CH4_intestine_tensor_area = CH4_intestine_tensor_filtered[county_idx:county_idx+1]
    CH4_Fecal_management_tensor_area = CH4_Fecal_management_tensor_filtered[county_idx:county_idx+1]
    Straw_tensor_area = Straw_tensor_filtered[county_idx:county_idx+1]
    Rice_CH4_Gg_area = Rice_CH4_Gg_filtered[county_idx:county_idx+1]
    nitrogen_deposition_N2O_tensor_area = nitrogen_deposition_N2O_tensor_filtered[county_idx:county_idx+1]
    straw_returning_N2O_tensor_area = straw_returning_N2O_tensor_filtered[county_idx:county_idx+1]
    Soc_tensor_origin_area = Soc_tensor_origin_filtered[county_idx:county_idx+1]
    county_scale_area = county_scale_filtered.iloc[county_idx:county_idx+1].reset_index(drop=True)

    # Filter threshold data
    threshold_NO3_PB_area = threshold_NO3_PB_filtered[county_idx:county_idx+1]
    threshold_N_runoff_PB_area = threshold_N_runoff_PB_filtered[county_idx:county_idx+1]
    threshold_NH3_PB_area = threshold_NH3_PB_filtered[county_idx:county_idx+1]

    # Calculate initial gas values (single county)
    current_NH3_area = Total_NH3_filtered[county_idx:county_idx+1]
    current_NO3_area = Total_NO3_filtered[county_idx:county_idx+1]
    current_N_runoff_area = Total_N_runoff_filtered[county_idx:county_idx+1]
    current_CH4_area = Total_CH4_filtered[county_idx:county_idx+1]
    current_N2O_area = Total_N2O_filtered[county_idx:county_idx+1]

    num_counties = 1  # Only one county
    num_techs = len(tech_set)
    tech_deltas_cache_copy = copy.deepcopy(tech_deltas_cache)  # Use pre-calculated results with deep copy
    # --------------------- Create model using Gurobi ---------------------
    model = gp.Model(f"Gas_Tech_Optimization_County_{county_name}")
    model.setParam('TimeLimit', 60 * 20)  # Solving time constraint
    model.setParam('OutputFlag', 0)
    model.setParam('NonConvex', 2)  # Allow non-convex quadratic constraints
    model.setParam('Threads', GRB_THREADS_PER_PROC)
    model.setParam('MIPGap', 0.05)  # Allow 5% optimality gap
    model.setParam('FeasibilityTol', 1e-5)  # Relax feasibility tolerance to improve numerical stability

    # Binary decision variables: whether to select a technology
    tech_vars = model.addVars(num_techs, vtype=GRB.BINARY, name='tech')

    # Objective function: minimize total technology cost
    obj = gp.LinExpr()
    for tech_index in range(num_techs):
        tech_cost = calculate_tech_cost(tech_index, county_scale_area.iloc[0])
        obj += tech_cost * tech_vars[tech_index]
    model.setObjective(obj, GRB.MINIMIZE)

    # Calculate initial total gas values
    initial_total_gas_values = {
        'NH3': torch.sum(current_NH3_area).item(),
        'NO3': torch.sum(current_NO3_area).item(),
        'N_runoff': torch.sum(current_N_runoff_area).item(),
        'CH4': torch.sum(current_CH4_area).item(),
        'N2O': torch.sum(current_N2O_area).item()
    }

    # Build state mapping (maintain original logic)
    area_stateMapping = {
        '种植业NH3挥发': NH3_Crop_tensor_area,
        '氮肥N2O': N2O_nitrogen_fertilizer_tensor_area,
        '氮肥NO3': NO3_nitrogen_fertilizer_tensor_area,
        '肠道CH4': CH4_intestine_tensor_area,
        '粪污管理CH4': CH4_Fecal_management_tensor_area,
        '粪污管理N2O': N2O_Fecal_management_tensor_area,
        '粪污管理NH3': NH3_Fecal_management_tensor_area,
        '粪污施用NH3': NH3_manure_application_tensor_area,
        '粪污施用N2O': N2O_manure_application_tensor_area,
        '粪污施用NO3': NO3_manure_application_tensor_area,
        'N runoff': N_runoff_tensor_area,
        'Rice CH4 Gg': Rice_CH4_Gg_area,
        '秸秆焚烧': Straw_tensor_area,
        'Soc': Soc_tensor_origin_area,
        '氮沉降': nitrogen_deposition_N2O_tensor_area,  # Add missing industry
        '秸秆还田': straw_returning_N2O_tensor_area,    # Add missing industry
    }

    # Preprocessing: collect sub-industry information for each gas
    # Structure: gas_sub_multipliers[gas][sub_key] = {tech_idx: multiplier}
    gas_sub_multipliers = {
        'NH3': {}, 'NO3': {}, 'N_runoff': {}, 'CH4': {}, 'N2O': {}
    }
    sub_initial = {}
    
    # Reference environment step() gas coupling logic, build common impact on multiple gases
    # Ensure all 12 industries are included in N2O calculation
    industries_affecting_N2O = set([
        '粪污管理NH3', '粪污管理N2O',
        '氮肥N2O', '种植业NH3挥发', '氮肥NO3', 
        'N runoff', '粪污施用NH3', '粪污施用NO3', '粪污施用N2O',
        '秸秆焚烧', '氮沉降', '秸秆还田'  # Add missing industries
    ])
    industries_affecting_CH4 = set(['粪污管理CH4', '肠道CH4', 'Rice CH4 Gg'])

    # Step 1: Collect all sub-industries affected by technology
    for tech_index in range(len(tech_set)):
        deltas = tech_deltas_cache_copy[tech_index]  # Use pre-calculated results
        # For counties with no solution, expand the set of sub-industries affected by technology
        deltas = expand_tech_subindustries(deltas, county_name, relaxed)
        tech_deltas_cache_copy[tech_index] = deltas
        for industry, delta, subIndustry in deltas:
            if industry not in area_stateMapping:
                raise ValueError(f"Industry {industry} not found in area_stateMapping")
            if industry == 'Soc':
                continue
            state_tensor = area_stateMapping[industry]
            subindustry_classes = class_mapping[industry]

            # Iterate through sub-industries of technology-affected industries
            for sub in subIndustry:
                try:
                    col = _get_or_create_industry_mapping(sub, subindustry_classes)
                    initial_value = state_tensor[0, col].item()
                except Exception as e:
                    logger.error(f"Sub {sub} not found in {industry} subindustry_classes: {e}")
                    breakpoint()
                    continue
               
                key = (industry, col)
                sub_initial[key] = initial_value

                # Direct gas classification based on sub-industry name (consistent with environment)
                sub_name_str = sub.strip() if isinstance(sub, str) else str(sub)
                if 'NH3' in sub_name_str:
                    direct_gas = 'NH3'
                elif 'NO3' in sub_name_str:
                    direct_gas = 'NO3'
                elif ('N_runoff' in sub_name_str) or ('runoff' in sub_name_str.lower()):
                    direct_gas = 'N_runoff'
                elif 'CH4' in sub_name_str:
                    direct_gas = 'CH4'
                elif 'N2O' in sub_name_str:
                    direct_gas = 'N2O'

                # Record impact on direct gas
                if key not in gas_sub_multipliers[direct_gas]:
                    gas_sub_multipliers[direct_gas][key] = {}
                gas_sub_multipliers[direct_gas][key][tech_index] = (1 + delta)

                # N2O coupling: changes in multiple industries affect N2O (reference environment)
                if (industry in industries_affecting_N2O):
                    if (industry == '秸秆焚烧' and ('N2O' not in sub_name_str)):
                        pass
                    else:
                        if key not in gas_sub_multipliers['N2O'] and not(key[0] == 'Soc'):
                            gas_sub_multipliers['N2O'][key] = {}
                        if 'N2O' in industry or (industry == '秸秆焚烧' and ('N2O' in sub_name_str)):
                            gas_sub_multipliers['N2O'][key][tech_index] = (1 + delta)
                        else:
                            gas_sub_multipliers['N2O'][key][tech_index] = 1 # Only used to calculate final N2O value

                # CH4 coupling: straw burning affects CH4
                if (industry in industries_affecting_CH4) or (industry == '秸秆焚烧' and ('CH4' in sub_name_str)):
                    if key not in gas_sub_multipliers['CH4']:
                        gas_sub_multipliers['CH4'][key] = {}
                    if 'CH4' in industry or (industry == '秸秆焚烧' and ('CH4' in sub_name_str)):
                        gas_sub_multipliers['CH4'][key][tech_index] = (1 + delta)
                    else:
                        gas_sub_multipliers['CH4'][key][tech_index] = 1 # Only used to calculate final CH4 value

    # Step 2: Ensure all sub-industries are included, even if not affected by technology
    # This is important because some sub-industries (like nitrogen deposition, straw returning, etc.) may not be affected by any technology
    for gas_type in ['NH3', 'NO3', 'N_runoff', 'CH4', 'N2O']:
        # Check if all necessary sub-industries are included (completely reference reinforcement learning environment)
        if gas_type == 'NH3':
            required_industries = ['种植业NH3挥发', '粪污管理NH3', '粪污施用NH3']
        elif gas_type == 'NO3':
            required_industries = ['氮肥NO3', '粪污施用NO3']
        elif gas_type == 'N_runoff':
            required_industries = ['N runoff']
        elif gas_type == 'CH4':
            required_industries = ['肠道CH4', '粪污管理CH4', 'Rice CH4 Gg', '秸秆焚烧']
        elif gas_type == 'N2O':
            required_industries = ['粪污管理N2O', '秸秆焚烧', '种植业NH3挥发', '氮肥NO3', '氮沉降', '粪污施用N2O', 'N runoff', '秸秆还田', '粪污管理NH3', '氮肥N2O', '粪污施用NH3', '粪污施用NO3']
        
        for industry in required_industries:
            if industry in area_stateMapping:
                state_tensor = area_stateMapping[industry]
                subindustry_classes = class_mapping[industry]
                
                # Iterate through all sub-industries of this industry
                for col in range(state_tensor.shape[1]):
                    initial_value = state_tensor[0, col].item()
                    key = (industry, col)
                    
                    # If this sub-industry hasn't been included yet, add it (not affected by technology)
                    if key not in gas_sub_multipliers[gas_type] and not((gas_type == 'N2O') and ('CH4' in subindustry_classes[col])) and not((gas_type == 'CH4') and ('N2O' in subindustry_classes[col])): # 
                        gas_sub_multipliers[gas_type][key] = {}  # Empty dictionary means not affected by technology
                        sub_initial[key] = initial_value
        
    gas_thresholds = {
        'NH3': threshold_NH3_PB_area.item(),
        'NO3': threshold_NO3_PB_area.item(),
        'N_runoff': threshold_N_runoff_PB_area.item(),
        'CH4': county_CH4_target,
        'N2O': county_N2O_target
    }
    current_gas_values = {
        'NH3': current_NH3_area[0].item(),
        'NO3': current_NO3_area[0].item(),
        'N_runoff': current_N_runoff_area[0].item(),
        'CH4': current_CH4_area[0].item(),
        'N2O': current_N2O_area[0].item()
    }

    # ============Emission reduction constraints============
    # Adjust thresholds to ensure technical feasibility
    # gas_thresholds = adjust_thresholds_for_technical_feasibility(
    #     gas_thresholds, gas_sub_multipliers, sub_initial
    # )
    
    # for gas in ['NH3', 'NO3', 'N_runoff', 'CH4', 'N2O']: # five gases
    for gas in ['NH3', 'NO3', 'N_runoff', 'CH4', 'N2O']: # three gases
        if gas_thresholds[gas] <= 0 or current_gas_values[gas] == 0 or (gas in ['CH4', 'N2O'] and county_name == '铜山区'):
            continue
        build_gas_constraint(model, gas_sub_multipliers[gas], gas, gas_thresholds[gas], tech_vars, sub_initial)

    # ============Technology conflict constraints============
    apply_conflict_constraints = True  # Set to False to remove conflict constraints
    if apply_conflict_constraints:
        for i, condition in enumerate(tech_conflicts_cache):
            if condition.any():
                condition_indexs = np.where(condition)[0]
                for condition_index in condition_indexs:
                    model.addConstr(
                        tech_vars[i] + tech_vars[condition_index] <= 1,
                        name=f'Conflict_County_{0}_Tech_{i}_{condition_index}'
                    )

    # ============Zero scale constraints============
    # When the scale or area of crops/livestock corresponding to technology is 0, the technology cannot be selected
    apply_zero_scale_constraints = True  # Set to False to remove zero scale constraints
    if apply_zero_scale_constraints:
        for tech_index in range(num_techs):
            # Check if the scale or area of crops/livestock corresponding to technology is 0
            is_scale_zero = check_tech_scale_zero(tech_index, county_scale_area.iloc[0])
            
            # If scale or area is 0, the technology cannot be selected
            if is_scale_zero:
                model.addConstr(
                    tech_vars[tech_index] == 0,
                    name=f'ZeroScale_County_{0}_Tech_{tech_index}'
                )
                if ENABLE_DEBUG:
                    tech_line = tech_set.iloc[tech_index]
                    tech_name = f"{tech_line['Mitigation strategy']}_{tech_line['Crop species'] if tech_line['class'] == 'crop' else tech_line['Livestock species']}"
                    tech_industry = tech_line['Livestock species'] if tech_line['class'] != 'crop' else tech_line['Crop species']
                    logger.info(f"Technology {tech_index} ({tech_name}) corresponding {tech_industry} scale or area is 0, adding constraint to prohibit selection")

    # # Limit how many solutions to collect
    # model.setParam(GRB.Param.PoolSolutions, 10)

    # # Limit the search space by setting a gap for the worst possible solution
    # # that will be accepted
    # model.setParam(GRB.Param.PoolGap, 0.1)

    # # do a systematic search for the k- best solutions
    # model.setParam(GRB.Param.PoolSearchMode, 2)

    # Solve the model
    model.optimize()
    
    solve_time = time.time() - start_time
    
    # Add debug information
    if ENABLE_DEBUG:
        logger.info(f"Debug info for {county_name}:")
        logger.info(f"  NH3 target: {county_NH3_target:.4f}, current: {current_NH3_area[0].item():.4f}")
        logger.info(f"  NO3 target: {county_NO3_target:.4f}, current: {current_NO3_area[0].item():.4f}")
        logger.info(f"  N_runoff target: {county_N_runoff_target:.4f}, current: {current_N_runoff_area[0].item():.4f}")
        logger.info(f"  CH4 target: {county_CH4_target:.4f}, current: {current_CH4_area[0].item():.4f}")
        logger.info(f"  N2O target: {county_N2O_target:.4f}, current: {current_N2O_area[0].item():.4f}")
        logger.info(f"  Model status: {model.status}")
        
        # Detailed status information
        status_mapping = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            # GRB.NUMERICALLY_DIFFICULT: "NUMERICALLY_DIFFICULT",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.NODE_LIMIT: "NODE_LIMIT",
            GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
            GRB.ITERATION_LIMIT: "ITERATION_LIMIT"
        }
        
        logger.info(f"  Model status code: {model.status}")
        logger.info(f"  Model status name: {status_mapping.get(model.status, 'UNKNOWN')}")
        
        # Check constraint count
        logger.info(f"  Number of constraints: {model.NumConstrs}")
        logger.info(f"  Number of variables: {model.NumVars}")
        logger.info(f"  Number of binary variables: {model.NumBinVars}")
        
        if model.status == GRB.INFEASIBLE:
            # Create log filename
            log_filename = f"logs/iis_analysis_{county_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
            os.makedirs('logs', exist_ok=True)
            
            try:
                with open(log_filename, 'w', encoding='utf-8') as log_file:
                    log_file.write(f"IIS Analysis for {county_name} - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_file.write("=" * 60 + "\n\n")
                    
                    # Compute IIS (Irreducible Infeasible Subsystem)
                    model.computeIIS()
                    log_file.write("IIS computation completed\n\n")
                    
                    # Analyze IIS constraints
                    log_file.write("=== IIS Analysis ===\n")

                    # Check if each constraint is in IIS
                    for i in range(model.NumConstrs):
                        constr = model.getConstrs()[i]
                        if constr.IISConstr:
                            log_file.write(f"IIS Constraint {i}: {constr.ConstrName} - {constr}\n")
                            # Get detailed constraint information
                            try:
                                lhs = model.getRow(constr)
                                rhs = constr.RHS
                                sense = constr.Sense
                                log_file.write(f"  LHS: {lhs}\n")
                                log_file.write(f"  Sense: {sense}\n")
                                log_file.write(f"  RHS: {rhs}\n")
                            except Exception as e:
                                log_file.write(f"  Error getting constraint details: {e}\n")
                    
                    # Check variable bounds
                    log_file.write("\n=== Variable Bounds Analysis ===\n")
                    for i in range(model.NumVars):
                        var = model.getVars()[i]
                        if var.IISLB or var.IISUB:
                            log_file.write(f"IIS Variable {i}: {var.VarName}\n")
                            if var.IISLB:
                                log_file.write(f"  Lower bound issue: LB={var.LB}\n")
                            if var.IISUB:
                                log_file.write(f"  Upper bound issue: UB={var.UB}\n")
                    
                    log_file.write(f"\nLog saved to: {log_filename}\n")
                
                logger.info(f"  IIS analysis saved to: {log_filename}")
                
            except Exception as e:
                logger.error(f"  Failed to save IIS analysis: {e}")
                logger.error(f"  Error details: {type(e).__name__}: {str(e)}")
    
        elif model.status == GRB.TIME_LIMIT:
            logger.info(f"  Model hit TIME_LIMIT - solution might be suboptimal")
            if hasattr(model, 'SolCount') and model.SolCount > 0:
                logger.info(f"  Found {model.SolCount} solutions before time limit")
    
    if model.status == GRB.OPTIMAL:
        # Get selected technologies
        selected_techs = []
        tech_details = []
        
        for tech_index in range(num_techs):
            if tech_vars[tech_index].x == 1:
                tech_line = tech_set.iloc[tech_index]
                deltas = tech_deltas_cache_copy[tech_index]
                classes = get_tech_deltas_class(deltas)
                classes.add(tech_line['Livestock species'] if tech_line['class'] != 'crop' else tech_line['Crop species'])
                selected_techs.append(tech_index)
                tech_details.append({
                    'tech_id': tech_index,
                    'tech_name': f"{tech_line['Mitigation strategy']}",
                    'tech_class': tech_line['class'],
                    'species': classes,
                    'grade': tech_line['技术分级']
                })
        
        # Simulate step-by-step state update process of reinforcement learning environment
        if ENABLE_DEBUG:
            logger.info(f"Starting to simulate reinforcement learning environment state updates...")
        # logger.info(f"Selected technologies: {selected_techs}")
        
        # 1. First copy initial state, create working copy (initial state of all sub-industries)
        working_state = {}
        for (industry, col) in sub_initial:
            if industry not in working_state:
                working_state[industry] = {}
            working_state[industry][col] = sub_initial[(industry, col)]
        
        # 2. Apply selected technologies one by one, each technology modifies all sub-industries it affects at once
        for tech_idx in selected_techs:
            tech_line = tech_set.iloc[tech_idx]
            tech_name = f"{tech_line['Mitigation strategy']}_{tech_line['Crop species'] if tech_line['class'] == 'crop' else tech_line['Livestock species']}"
            
            if ENABLE_DEBUG:
                logger.info(f"  Applying technology {tech_idx}: {tech_name}")
            
            # Iterate through all gas types, find all sub-industries affected by this technology
            for gas_type in ['NH3', 'NO3', 'N_runoff', 'CH4', 'N2O']:
                if gas_type in gas_sub_multipliers and gas_sub_multipliers[gas_type]:
                    for (industry, col), tech_multiplier_dict in gas_sub_multipliers[gas_type].items():
                        if tech_idx in tech_multiplier_dict:
                            multiplier = tech_multiplier_dict[tech_idx]
                            if multiplier != 1.0:  # Only modify when multiplier is not 1
                                old_value = working_state[industry][col]
                                # Cumulatively modify sub-industry values (simulate step-by-step updates in reinforcement learning environment)
                                working_state[industry][col] *= multiplier
                                new_value = working_state[industry][col]
                                
                                if ENABLE_DEBUG:
                                    logger.info(f"    {gas_type} - {industry}[{col}]: {old_value} * {multiplier} = {new_value}")
        
        # 3. Calculate final values for each gas based on updated working state
        final_gas_values = {}
        
        for gas_type in ['NH3', 'NO3', 'N_runoff', 'CH4', 'N2O']:
            if gas_type in gas_sub_multipliers and gas_sub_multipliers[gas_type]:
                # Calculate final value for this gas type: first sum by industry, then apply weighting coefficients
                total_emission = 0
                industry_totals = {}
                
                # Sum sub-industry values related to this gas by industry
                for (industry, col), _ in gas_sub_multipliers[gas_type].items():
                    if industry not in industry_totals:
                        industry_totals[industry] = 0
                    industry_totals[industry] += working_state[industry][col]
                
                # Apply industry-level weighting coefficients and accumulate
                for industry, total in industry_totals.items():
                    if gas_type == 'N2O':
                        weight = _compute_N2O_delta(industry)
                    elif gas_type == 'CH4':
                        weight = _compute_CH4_delta(industry)
                    else:
                        weight = 1.0
                    
                    total_emission += weight * total
                    
                    if ENABLE_DEBUG:
                        logger.info(f"  {gas_type} - {industry}: {total} * {weight} = {total * weight}")
                
                final_gas_values[gas_type] = total_emission
                
                if ENABLE_DEBUG:
                    logger.info(f"Final {gas_type} value: {total_emission}")
                    logger.info(f"Initial {gas_type} value: {current_gas_values[gas_type]}")
                    logger.info(f"Reduction: {current_gas_values[gas_type] - total_emission}")
                    logger.info("-" * 50)
            else:
                # If no technology affects, use initial value
                if gas_type == 'NH3':
                    final_gas_values[gas_type] = current_NH3_area[0].item()
                elif gas_type == 'NO3':
                    final_gas_values[gas_type] = current_NO3_area[0].item()
                elif gas_type == 'N_runoff':
                    final_gas_values[gas_type] = current_N_runoff_area[0].item()
                elif gas_type == 'CH4':
                    final_gas_values[gas_type] = current_CH4_area[0].item()
                elif gas_type == 'N2O':
                    final_gas_values[gas_type] = current_N2O_area[0].item()
        

        return {
            'major region': major_region,
            'county_name': county_name,
            'area_name': area_name,
            'status': 'Optimal',
            'solve_time': solve_time,
            'total_cost': model.objVal,
            'selected_techs': selected_techs,
            'tech_details': tech_details,
            'initial_gas_values': initial_total_gas_values,
            'final_gas_values': final_gas_values,
            'targets': {
                'NH3': county_NH3_target,
                'NO3': county_NO3_target,
                'N_runoff': county_N_runoff_target,
                'CH4': county_CH4_target,
                'N2O': county_N2O_target
            }
        }
    else:
        return {
            'major region': major_region,
            'county_name': county_name,
            'area_name': area_name,
            'status': 'Infeasible',
            'solve_time': solve_time,
            'total_cost': float('inf'),
            'selected_techs': [],
            'tech_details': [],
            'initial_gas_values': initial_total_gas_values,
            'final_gas_values': initial_total_gas_values,
            'targets': {
                'NH3': county_NH3_target,
                'NO3': county_NO3_target,
                'N_runoff': county_N_runoff_target,
                'CH4': county_CH4_target,
                'N2O': county_N2O_target
            }
        }
            
def expand_tech_subindustries(tech_deltas, county_name, relaxed=False):
    """
    Expand maize technology to other four crops for failed counties
    When technology affects industries containing any sub-industry with "maize", 
    search for other crop sub-industries that actually exist and apply the same delta value
    
    Args:
        tech_deltas: List of technology impacts [(industry, delta, subIndustry), ...]
        county_name: County name (for debug output)
        relaxed: Whether in relaxed mode
        
    Returns:
        Modified tech_deltas
    """
    if not relaxed:
        return tech_deltas
    
    target_subindustry = {'maize': ['millet', 'sorghum', 'othercereals', 'cotton'],
                          'vegetable': ['beans', 'potato', 'peanut', 'rapeseed', 'sugarbeet'],
                          'fruit': ['flax', 'sugarcane', 'tobacoo'],
                          'sheep': ['horse', 'donkey', 'rabbit']}
    
    def find_corresponding_subindustry(industry, target_subindustry, key, origin):
        """Find corresponding other sub-industries in actual class_mapping"""
        if industry not in class_mapping:
            return []
        
        # Get all sub-industry list for this industry
        all_subindustries = class_mapping[industry].tolist()
        
        # Find corresponding sub-industries for other sub-industries
        other_subindustry = []
        
        # Iterate through target sub-industry list, try to replace key in original sub-industry name with target sub-industry name
        for s in target_subindustry:
            # If key is in original sub-industry name
            if key in sub:
                # Replace key with target sub-industry name, generate new sub-industry name
                new_sub = origin.replace(key, s)
                # Check if new sub-industry name actually exists in all sub-industry list
                if new_sub in all_subindustries:
                    # If exists, add to result list
                    other_subindustry.append(new_sub)

        return other_subindustry
    
    expanded_deltas = []
    for industry, delta, subIndustry in tech_deltas:
        # Keep original technology impact
        expanded_deltas.append((industry, delta, subIndustry))
        
        # Check if contains any sub-industry with target sub-industry
        if isinstance(subIndustry, list):
            for sub in subIndustry:
                if isinstance(sub, str) and any(s in sub.lower() for s in target_subindustry.keys()):
                    # Find other sub-industries that actually exist
                    for s in target_subindustry.keys():
                        if s in sub.lower():
                            corresponding_subindustry = find_corresponding_subindustry(industry, target_subindustry[s], s, sub)
                            for other_subindustry in corresponding_subindustry:
                                # Create new technology impact entry, only containing single crop
                                expanded_deltas.append((industry, delta, [other_subindustry]))
                            
                            if ENABLE_DEBUG:
                                logger.info(f"Expanding {s} technology to other sub-industries for failed county {county_name}:")
                                logger.info(f"  Industry: {industry}, delta: {delta}")
                                logger.info(f"  From {sub} -> expanded to: {corresponding_subindustry}")

    return expanded_deltas

def county_gas_tech_optimization_relaxed(county_idx):
    """
    Relaxed version of county optimization: specifically for retrying failed counties
    In this version, the county has been marked as having no solution, so target sub-industries will be automatically expanded
    """
    # Directly call normal optimization function, at this point the county has been marked as having no solution, technology expansion will be automatically enabled
    return county_gas_tech_optimization(county_idx, relaxed=True)

def get_industry_scale(tech, county_scale_data):
    """
    Get the industry scale for technology.
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
    Convert percentage string to float
    
    Args:
        percentage_str (str): Percentage string, e.g. "40.0%"
    
    Returns:
        float: Converted floating point number, e.g. 0.4
    """
    if not isinstance(percentage_str, str) and np.isnan(percentage_str):
        return percentage_str
    
    if isinstance(percentage_str, str) and percentage_str.endswith('%'):
        # Remove percent sign and convert to float, then divide by 100
        return float(percentage_str[:-1]) / 100
    else:
        # If not percentage format, directly convert to float
        return float(percentage_str)

def get_delta(tech_index):
    """
    Get the industries and sub-industries affected by technology.
    """
    line = get_line(tech_index)
    class_name = 'crop' if 'crop' in line.index[4].lower() else 'livestock'
    line = line[5:]  # Skip first 5 columns
    
    # Exclude yield indicators
    num_indicators = line.shape[0] // 2
    deltas = []
    
    for i in range(num_indicators):
        delta = _convert_percentage_to_float(line.iloc[2*i])
        if delta != 0 and not np.isnan(delta) and (line.index[2*i] != 'Yield'):
            industry = line.index[2*i+1].split(' ', 1)[1].replace('.1', '')
            subIndustry = line.iloc[2*i+1]
            if isinstance(subIndustry, str):
                subIndustry = subIndustry.split('、')
                subIndustry = [s for s in subIndustry if s != '']
                # subIndustry = [s.replace('x000d__x000d_\n', '').strip() for s in subIndustry]
                # subIndustry = [s.replace('/_x000d__x000d_\n', ' ').strip() for s in subIndustry]
                # subIndustry = [s.replace(' _x000d__x000d_\n', ' ').strip() for s in subIndustry]
                for s in subIndustry:
                    if 'x000d_' in s:
                        logger.info(s)

            deltas.append((industry, delta, subIndustry))
    return deltas

def get_line(tech_index):
    """
    Get the impact row for technology.
    """
    if tech_index in line_cache:
        return line_cache[tech_index]

    index = np.searchsorted(tech_shape, tech_index, side='right')
    if index == 0:
        line = tech_mapping['Feeding'].iloc[tech_index]
    else:
        line = tech_mapping[list(tech_mapping.keys())[index]].iloc[tech_index - tech_shape[index-1]]
            
    # Cache result
    line_cache[tech_index] = line
    return line

def get_conflicts_tech_local(tech_indices, tech_set):
    """
    Get technology conflict matrix, referencing logic in environment
    """
    conflicts = []
    for i, tech_idx in enumerate(tech_indices):
        conflict_array = np.zeros(len(tech_indices), dtype=bool)
        tech_i = tech_set.iloc[tech_idx]
        
        for j, other_tech_idx in enumerate(tech_indices):
            if i != j:
                tech_j = tech_set.iloc[other_tech_idx]
                # Use conflict logic from utils
                if (tech_i['class'] == tech_j['class'] and 
                    tech_i['技术间的冲突'] == tech_j['技术间的冲突'] and
                    pd.notna(tech_i['技术间的冲突'])):
                    conflict_array[j] = True
        
        conflicts.append(conflict_array)
    
    return conflicts

# Helper: wrap parallel calls
def _optimize_county(idx: int):
    return county_gas_tech_optimization(idx)


def _optimize_county_relaxed(idx: int):
    return county_gas_tech_optimization_relaxed(idx)


def _build_county_result_row(result: dict) -> dict:
    tech_details_str = ""
    if result.get('tech_details'):
        tech_details_str = "; ".join([
            f"{tech['tech_name']} (ID:{tech['tech_id']}, class:{tech['tech_class']}, species:{tech['species']}, grade:{tech['grade']})"
            for tech in result['tech_details']
        ])

    # Compatible with different return key names
    initial_values = result.get('initial_gas_values') or result.get('initial_total_gas_values') or {}
    final_values = result.get('final_gas_values') or initial_values
    targets = result.get('targets', {})

    # Group by technology category
    tech_by_category = {}
    if result.get('tech_details'):
        for tech in result['tech_details']:
            category = tech['tech_class']
            if category not in tech_by_category:
                tech_by_category[category] = []
            tech_by_category[category].append(f"{tech['tech_name']} (ID:{tech['tech_id']}, species:{tech['species']}, grade:{tech['grade']})")

    row = {
        'major region': result.get('major region', 'Unknown'),
        'area_name': result.get('area_name', 'Unknown'),
        'county': result.get('county_name', 'Unknown'),
        'techs': result.get('selected_techs', []),
        'tech_details': tech_details_str,
        'tech_amount': len(result.get('selected_techs', [])),
        'status': result.get('status', 'Error'),
        'solve_time': result.get('solve_time', 0.0),
        # Grouped technology information by category
        'Feeding_techs': "; ".join(tech_by_category.get('Feeding', [])),
        'Housing_techs': "; ".join(tech_by_category.get('Housing', [])),
        'slurry_storage_techs': "; ".join(tech_by_category.get('slurry storage', [])),
        'soild_storage_techs': "; ".join(tech_by_category.get('soild storage', [])),
        'composting_techs': "; ".join(tech_by_category.get('composting', [])),
        'additives_application_techs': "; ".join(tech_by_category.get('additives application', [])),
        'soild_application_techs': "; ".join(tech_by_category.get('soild application', [])),
        'slurry_application_techs': "; ".join(tech_by_category.get('slurry application', [])),
        'crop_techs': "; ".join(tech_by_category.get('crop', [])),
        # Initial gas values
        'initial_NH3': initial_values.get('NH3', 0.0),
        'initial_NO3': initial_values.get('NO3', 0.0),
        'initial_N_runoff': initial_values.get('N_runoff', 0.0),
        'initial_CH4': initial_values.get('CH4', 0.0),
        'initial_N2O': initial_values.get('N2O', 0.0),
        # Final gas values
        'final_NH3': final_values.get('NH3', initial_values.get('NH3', 0.0)),
        'final_NO3': final_values.get('NO3', initial_values.get('NO3', 0.0)),
        'final_N_runoff': final_values.get('N_runoff', initial_values.get('N_runoff', 0.0)),
        'final_CH4': final_values.get('CH4', initial_values.get('CH4', 0.0)),
        'final_N2O': final_values.get('N2O', initial_values.get('N2O', 0.0)),
        # Emission reduction effects
        'NH3_reduction': (initial_values.get('NH3', 0.0) - final_values.get('NH3', initial_values.get('NH3', 0.0))),
        'NO3_reduction': (initial_values.get('NO3', 0.0) - final_values.get('NO3', initial_values.get('NO3', 0.0))),
        'N_runoff_reduction': (initial_values.get('N_runoff', 0.0) - final_values.get('N_runoff', initial_values.get('N_runoff', 0.0))),
        'CH4_reduction': (initial_values.get('CH4', 0.0) - final_values.get('CH4', initial_values.get('CH4', 0.0))),
        'N2O_reduction': (initial_values.get('N2O', 0.0) - final_values.get('N2O', initial_values.get('N2O', 0.0))),
        # Emission reduction targets
        'NH3_target': targets.get('NH3', 0.0),
        'NO3_target': targets.get('NO3', 0.0),
        'N_runoff_target': targets.get('N_runoff', 0.0),
        'CH4_target': targets.get('CH4', 0.0),
        'N2O_target': targets.get('N2O', 0.0),
    }

    # Gap from targets
    row['gap_NH3'] = row['final_NH3'] - row['NH3_target']
    row['gap_NO3'] = row['final_NO3'] - row['NO3_target']
    row['gap_N_runoff'] = row['final_N_runoff'] - row['N_runoff_target']
    row['gap_CH4'] = row['final_CH4'] - row['CH4_target']
    row['gap_N2O'] = row['final_N2O'] - row['N2O_target']

    # Technology needed flag
    row['needs_tech'] = True

    return row

def get_tech_deltas_class(deltas):
        
    target_subindustry = {'maize': ['millet', 'sorghum', 'othercereals', 'cotton'],
                          'vegetable': ['beans', 'potato', 'peanut', 'rapeseed', 'sugarbeet'],
                          'fruit': ['flax', 'sugarcane', 'tobacoo'],
                          'sheep': ['horse', 'donkey', 'rabbit']}
    classes = set()
    for industry, delta, subIndustry in deltas:
        if len(subIndustry) == 1:
            sub = subIndustry[0]
            for origin_industry, expanded_subindustries in target_subindustry.items():
                for expanded_subindustry in expanded_subindustries:
                    if expanded_subindustry in sub:
                        classes.add(expanded_subindustry)
                        classes.add(origin_industry)
                        break
    return classes

if __name__ == '__main__':
    logger.info(f"Data loading completed, total {len(IDs_all)} counties")
    logger.info(f"Total number of technologies: {len(tech_set)}")

    # Pre-calculate all technology impacts to avoid repeated calculations for each agricultural area
    logger.info("Pre-calculating technology impacts...")
    tech_deltas_cache = {}
    for tech_index in range(len(tech_set)):
        tech_deltas_cache[tech_index] = get_delta(tech_index)
    logger.info(f"Technology impact pre-calculation completed, total {len(tech_deltas_cache)} technologies")

    # Pre-calculate technology conflict matrix to avoid repeated calculations for each agricultural area
    logger.info("Pre-calculating technology conflict matrix...")
    tech_set_indexs_all = np.arange(tech_set.shape[0])
    tech_conflicts_cache = get_conflicts_tech_local(tech_set_indexs_all, tech_set)
    logger.info(f"Technology conflict matrix pre-calculation completed")
    
    # Pre-calculate all possible industry mappings to avoid runtime string matching (most time-consuming part)
    logger.info("Pre-calculating industry mappings...")
    all_possible_subs = set()
    for tech_index in range(len(tech_set)):
        deltas = tech_deltas_cache[tech_index]
        for industry, delta, subIndustry in deltas:
            if industry in class_mapping:
                for sub in subIndustry:
                    # Filter out empty strings and strings containing only whitespace
                    stripped_sub = sub.strip()
                    if stripped_sub:
                        all_possible_subs.add((stripped_sub, id(class_mapping[industry])))
    
    # Pre-calculate all mappings
    for sub, subindustry_id in all_possible_subs:
        if sub not in industry_mapping:
            # Find corresponding subindustry_classes
            subindustry_classes = None
            for industry, classes in class_mapping.items():
                if id(classes) == subindustry_id:
                    subindustry_classes = classes
                    break
            
            if subindustry_classes is not None:
                if subindustry_id not in subindustry_lists:
                    subindustry_lists[subindustry_id] = subindustry_classes.tolist()
                
                subindustry_list = subindustry_lists[subindustry_id]
                # Ensure sub is not empty string
                if sub and sub.strip():
                    match = process.extractOne(sub, subindustry_list)
                    if match:
                        matched_item = match[0]
                        industry_mapping[sub] = subindustry_classes[subindustry_classes == matched_item].index[0]
                        if match[1] != 100:
                            logger.warning(f"Warning: {sub} matched with {matched_item} with confidence {match[1]}")
                    else:
                        raise ValueError(f"No match found for {sub} in {subindustry_list}")

    logger.info(f"Industry mapping pre-calculation completed, total {len(industry_mapping)} mappings")
    
    logger.info("Starting county optimization...")
    total_need = len(IDs_filtered)
    
    if ENABLE_DEBUG:
        # Debug mode: use single process
        logger.info(f"Debug mode: using single process to handle {total_need} counties needing technology...")
        rows = [None] * total_need
        failed_indices = []
        start_total_time = time.time()
        counties_to_debug = all_locations = ['永城市']
        logger.info(f"Debug mode: processing {len(counties_to_debug)} counties")
        for idx in range(total_need):
            try:
                if IDs_filtered['Counties'].iloc[idx] not in counties_to_debug:
                    continue
                result = _optimize_county(idx)
                rows[idx] = _build_county_result_row(result)
                # breakpoint()
                if result['status'] != 'Optimal':
                    failed_indices.append(idx)
                
                # Print progress more frequently in debug mode
                if (idx + 1) % 10 == 0 or (idx + 1) == total_need:
                    elapsed_total = time.time() - start_total_time
                    avg_time_per_county = elapsed_total / (idx + 1)
                    remaining_counties = total_need - (idx + 1)
                    estimated_remaining_time = avg_time_per_county * remaining_counties
                    logger.info(f"[{idx+1}/{total_need}] Average {avg_time_per_county:.2f}s/county, estimated remaining {estimated_remaining_time/60:.1f} minutes")
                    
                    # Show current processing county information in debug mode
                    if ENABLE_DEBUG:
                        logger.info(f"  Currently processing: {result['county_name']}, status: {result['status']}")
                        
            except Exception as e:
                logger.error(f"Error processing county {idx}: {e}")
                if ENABLE_DEBUG:
                    import traceback
                    traceback.print_exc()
                rows[idx] = _build_county_result_row({
                    'county_name': f'County_{idx}',
                    'area_name': 'Unknown',
                    'status': 'Error',
                    'solve_time': 0.0,
                    'error': str(e),
                    'total_cost': float('inf'),
                    'selected_techs': [],
                    'tech_details': [],
                    'initial_gas_values': {},
                    'final_gas_values': {},
                    'targets': {},
                    'major region': 'Unknown'
                })
                failed_indices.append(idx)
        
        # Relaxed optimization in debug mode
        if failed_indices:
            
            logger.info(f"\nDebug mode: starting technology expansion re-optimization for {len(failed_indices)} failed counties...")
            for j, idx in enumerate(failed_indices):
                try:
                    relaxed_result = _optimize_county_relaxed(idx)
                    if relaxed_result['status'] == 'Optimal':
                        rows[idx] = _build_county_result_row(relaxed_result)
                        logger.info(f"  Technology expansion optimization successful: {rows[idx]['county']}")
                    else:
                        logger.info(f"  Technology expansion optimization failed: {relaxed_result['county_name']}, status: {relaxed_result['status']}")
                        if rows[idx] is not None:
                            rows[idx]['relaxed_attempted'] = True
                except Exception as e:
                    logger.error(f"  Error in technology expansion optimization for county {idx}: {e}")
                    if ENABLE_DEBUG:
                        import traceback
                        traceback.print_exc()
        exit()                 
    else:
        # Normal multi-process mode
        logger.info(f"Starting to process {total_need} counties needing technology... (parallel processes: {NUM_PROCS}, Gurobi threads per process: {GRB_THREADS_PER_PROC})")
        rows = [None] * total_need
        failed_indices = []

        ctx = mp.get_context('fork')
        start_total_time = time.time()
        with ctx.Pool(processes=NUM_PROCS) as pool:
            for idx, result in enumerate(pool.imap(_optimize_county, range(total_need), chunksize=1)):
                rows[idx] = _build_county_result_row(result)
                if result['status'] != 'Optimal':
                    failed_indices.append(idx)
                if (idx + 1) % 50 == 0 or (idx + 1) == total_need:
                    elapsed_total = time.time() - start_total_time
                    avg_time_per_county = elapsed_total / (idx + 1)
                    remaining_counties = total_need - (idx + 1)
                    estimated_remaining_time = avg_time_per_county * remaining_counties
                    logger.info(f"[{idx+1}/{total_need}] Average {avg_time_per_county:.2f}s/county, estimated remaining {estimated_remaining_time/60:.1f} minutes")
                
                # logger.info(f"\nStarting technology expansion re-optimization for {len(failed_indices)} failed counties...")
            for j, relaxed_result in enumerate(pool.imap(_optimize_county_relaxed, failed_indices, chunksize=1)):
                idx = failed_indices[j]
                if relaxed_result['status'] == 'Optimal':
                    # breakpoint()
                    rows[idx] = _build_county_result_row(relaxed_result)
                    if ENABLE_DEBUG:
                        logger.info(f"  Technology expansion optimization successful: {rows[idx]['county']}")

                    # Mark as attempted relaxation
                    if rows[idx] is not None:
                        rows[idx]['relaxed_attempted'] = True

    failed_county_records = []
    for idx in sorted(set(failed_indices)):
        # Safely get county name
        if 0 <= idx < len(IDs_filtered):
            cname = IDs_filtered['Counties'].iloc[idx]
        else:
            cname = f"County_{idx}"
        # Read existing information from rows (may be None)
        row = rows[idx] if (idx < len(rows) and rows[idx] is not None) else None
        status = row.get('status') if row else 'Infeasible'
        solve_time = row.get('solve_time') if row else None
        techs = row.get('techs') if row else []
        failed_county_records.append({
            'county_index': idx,
            'county': cname,
            'status': status,
            'solve_time': solve_time,
            'techs': ",".join(map(str, techs)) if techs else ""
        })
    if failed_county_records:
        os.makedirs('results', exist_ok=True)
        pd.DataFrame(failed_county_records).to_excel('results/failed_counties.xlsx', index=False)
        logger.info(f"Failed county list saved to results/failed_counties.xlsx (total {len(failed_county_records)} counties)")
    
    # Add counties that don't need technology
    logger.info("Adding counties that don't need technology...")
    counties_with_tech = set([r['county'] for r in rows if r is not None])
    
    counties_no_tech_indices = np.where(~counties_need_tech)[0]
    
    for i in counties_no_tech_indices:
        county_name = IDs_all.iloc[i]['Counties']
        if county_name not in counties_with_tech:
            gap_NO3 = gap_NO3_origin_all[i].item()
            gap_N_runoff = gap_N_runoff_origin_all[i].item()
            gap_NH3 = gap_NH3_origin_all[i].item()
            
            current_NH3 = (torch.sum(NH3_Crop_tensor_all[i]) + 
                          torch.sum(NH3_Fecal_management_tensor_all[i]) + 
                          torch.sum(NH3_manure_application_tensor_all[i])).item()
            current_NO3 = (torch.sum(NO3_nitrogen_fertilizer_tensor_all[i]) + 
                          torch.sum(NO3_manure_application_tensor_all[i])).item()
            current_N_runoff = torch.sum(N_runoff_tensor_all[i]).item()
            current_CH4 = Total_CH4_national[i].item()
            current_N2O = Total_N2O_national[i].item()
            
            area_name = IDs_all.iloc[i]['所属农业亚区'] if pd.notna(IDs_all.iloc[i]['所属农业亚区']) else 'Unknown'
            major_region = IDs_all.iloc[i]['所属农业区'] if pd.notna(IDs_all.iloc[i]['所属农业区']) else 'Unknown'
            
            rows.append({
                'major region': major_region,
                'area_name': area_name,
                'county': county_name,
                'techs': [],
                'tech_details': "",
                'tech_amount': 0,
                'status': 'No Tech Needed',
                'solve_time': 0.0,
                # Technology information grouped by category (empty for counties not needing technology)
                'Feeding_techs': "",
                'Housing_techs': "",
                'slurry_storage_techs': "",
                'soild_storage_techs': "",
                'composting_techs': "",
                'additives_application_techs': "",
                'soild_application_techs': "",
                'slurry_application_techs': "",
                'crop_techs': "",
                'initial_NH3': current_NH3,
                'initial_NO3': current_NO3,
                'initial_N_runoff': current_N_runoff,
                'initial_CH4': current_CH4,
                'initial_N2O': current_N2O,
                'final_NH3': current_NH3,
                'final_NO3': current_NO3,
                'final_N_runoff': current_N_runoff,
                'final_CH4': current_CH4,
                'final_N2O': current_N2O,
                'NH3_reduction': 0.0,
                'NO3_reduction': 0.0,
                'N_runoff_reduction': 0.0,
                'CH4_reduction': 0.0,
                'N2O_reduction': 0.0,
                'NH3_target': 0.0,
                'NO3_target': 0.0,
                'N_runoff_target': 0.0,
                'CH4_target': 0.0,
                'N2O_target': 0.0,
                'gap_NH3': gap_NH3,
                'gap_NO3': gap_NO3,
                'gap_N_runoff': gap_N_runoff,
                'gap_CH4': 0,
                'gap_N2O': 0,
                'needs_tech': False,
            })
    
    # Save results to Excel file
    county_result_df = pd.DataFrame(rows)
    county_result_df.to_excel('results/linear_optimization_results_by_county_5gases_hard_target.xlsx', index=False)
    logger.info(f"County results saved to results/linear_optimization_results_by_county_5gases_hard_target.xlsx")
    
    logger.info(f"Total processed {len(county_result_df)} counties")
    
    # Statistics
    optimal_counties = len([r for r in rows if r['status'] == 'Optimal'])
    no_tech_counties = len([r for r in rows if r['status'] == 'No Tech Needed'])
    failed_counties_cnt = len([r for r in rows if r['status'] in ['Infeasible', 'Error']])
    
    logger.info(f"County optimization successful: {optimal_counties} counties")
    logger.info(f"Counties not needing technology: {no_tech_counties} counties")
    logger.info(f"County optimization failed: {failed_counties_cnt} counties")
    
    # Summary statistics by agricultural region
    area_summary = county_result_df.groupby('area_name').agg({
        'county': 'count',
        'tech_amount': 'sum',
        'NH3_reduction': 'sum',
        'NO3_reduction': 'sum',
        'N_runoff_reduction': 'sum',
        'CH4_reduction': 'sum',
        'N2O_reduction': 'sum'
    }).rename(columns={'county': 'counties_count', 'tech_amount': 'total_techs'})
    
    area_summary.to_excel('results/linear_optimization_results_area_summary_by_county_5gases_hard_target.xlsx')
    logger.info(f"Agricultural region summary results saved to results/linear_optimization_results_area_summary_by_county_5gases_hard_target.xlsx")
