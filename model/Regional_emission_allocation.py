"""
Regional Emission Allocation Calculation Module

This module is used to calculate gas emission proportions and emission reduction target allocation for agricultural regions nationwide.
Main functions:
1. Load national county-level gas emission data
2. Calculate national total emissions and reduction targets for various gases
3. Filter counties that need technology addition
4. Allocate emission reduction responsibilities by agricultural region
5. Output results to Excel file

Author: [Yanis Ye]
Date: [2025.07.29]
"""

import os
import torch
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import custom modules
from dataLoader import TechDataLoader, CountyDataLoader
from model.utils import GAS_COEFFICIENTS

# ================================
# 1. Configuration and Path Setup
# ================================

print("=" * 60)
print("Regional Emission Allocation Calculation Started")
print("=" * 60)

# Data file path configuration
DATA_PATHS = {
    'county_df': 'data/基础数据-县级尺度.xlsx',
    'soc_df': 'data/SOC-县尺度.xlsx',
    'livestock_scale': 'data/动物数量.xlsx',
    'crop_scale': 'data/分县种植面积.xlsx',
    'livestock_tech': 'data/畜牧业技术列单-经济产量0803.xlsx',
    'crop_tech': 'data/种植业技术列单产量产业0803.xlsx',
    'IDs_df': 'data/县市亚区.xlsx'
}

# Output file path
OUTPUT_FILE = 'results/reduction_variables.xlsx'

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

print(f"Data loading path configuration completed")
print(f"Output file path: {OUTPUT_FILE}")

# ================================
# 2. Data Loading
# ================================

print("\n" + "-" * 40)
print("Starting to load national county-level data...")

# Load national county-level data
county_data = CountyDataLoader(
    DATA_PATHS['county_df'], 
    DATA_PATHS['IDs_df'],
    DATA_PATHS['livestock_scale'], 
    DATA_PATHS['crop_scale'], 
    DATA_PATHS['soc_df']
)

# Unpack county-level data
(IDs_all, NH3_Crop_colunms, N2O_nitrogen_fertilizer_columns, NO3_nitrogen_fertilizer_columns,
 N_runoff_columns, Soc_columns, NH3_Fecal_management_columns, N2O_Fecal_management_columns,
 NH3_manure_application_columns, N2O_manure_application_columns, NO3_manure_application_columns,
 Straw_columns, CH4_intestine_columns, CH4_Fecal_management_columns,
 NH3_manure_application_tensor_all, N2O_manure_application_tensor_all, NO3_manure_application_tensor_all,
 Rice_CH4_Gg_all, nitrogen_deposition_N2O_tensor, straw_returning_N2O_tensor,
 NH3_Crop_tensor_all, N2O_nitrogen_fertilizer_tensor_all, NO3_nitrogen_fertilizer_tensor_all,
 N_runoff_tensor_all, Soc_tensor_origin_all, NH3_Fecal_management_tensor_all,
 N2O_Fecal_management_tensor_all, Straw_tensor_all, CH4_intestine_tensor_all,
 CH4_Fecal_management_tensor_all, Total_CH4_tensor_origin_all, Total_N2O_tensor_origin_all,
 threshold_NO3_PB_all, threshold_N_runoff_PB_all, threshold_NH3_PB_all,
 threshold_CH4_all, threshold_N2O_all, county_scale_all) = county_data

print(f"✓ Successfully loaded data for {len(IDs_all)} counties")

# Load technology data
print("Loading technology data...")
tech_data = TechDataLoader(DATA_PATHS['livestock_tech'], DATA_PATHS['crop_tech'])
(Feeding, Housing, slurry_storage, soild_storage, composting, additives_application,
 soild_application, slurry_application, crop, tech_set) = tech_data

print(f"✓ Successfully loaded {len(tech_set)} technologies")

# ================================
# 3. National Gas Emission Calculation
# ================================

print("\n" + "-" * 40)
print("Calculating total national emissions for various gases...")

# Get gas coefficients
coef_CH4 = GAS_COEFFICIENTS['CH4']
coef_N2O = GAS_COEFFICIENTS['N2O']

# NH3 total emission calculation
print("Calculating NH3 emissions...")
Total_NH3_national = (torch.sum(NH3_Crop_tensor_all, axis=1) + 
                     torch.sum(NH3_Fecal_management_tensor_all, axis=1) + 
                     torch.sum(NH3_manure_application_tensor_all, axis=1))

# NO3 total emission calculation
print("Calculating NO3 emissions...")
Total_NO3_national = (torch.sum(NO3_nitrogen_fertilizer_tensor_all, axis=1) + 
                     torch.sum(NO3_manure_application_tensor_all, axis=1))

# N_runoff total emission calculation
print("Calculating N_runoff emissions...")
Total_N_runoff_national = torch.sum(N_runoff_tensor_all, axis=1)

# CH4 total emission calculation (including coefficients)
print("Calculating CH4 emissions...")
Total_CH4_national = (coef_CH4['default'] * torch.sum(CH4_intestine_tensor_all, axis=1) + 
                     coef_CH4['default'] * torch.sum(CH4_Fecal_management_tensor_all, axis=1) + 
                     coef_CH4['default'] * torch.sum(Straw_tensor_all[:, :3], axis=1) + 
                     coef_CH4['Rice CH4 Gg'] * torch.sum(Rice_CH4_Gg_all, axis=1))

# N2O total emission calculation (including coefficients)
print("Calculating N2O emissions...")
Total_N2O_national = (1 * torch.sum(nitrogen_deposition_N2O_tensor, axis=1) + 
                     coef_N2O['粪污管理N2O'] * torch.sum(N2O_Fecal_management_tensor_all, axis=1) + 
                     coef_N2O['粪污管理NH3'] * torch.sum(NH3_Fecal_management_tensor_all, axis=1) + 
                     coef_N2O['氮肥N2O'] * torch.sum(N2O_nitrogen_fertilizer_tensor_all, axis=1) + 
                     coef_N2O['种植业NH3挥发'] * torch.sum(NH3_Crop_tensor_all, axis=1) + 
                     coef_N2O['氮肥NO3'] * torch.sum(NO3_nitrogen_fertilizer_tensor_all, axis=1) + 
                     coef_N2O['N runoff'] * torch.sum(N_runoff_tensor_all, axis=1) + 
                     coef_N2O['粪污施用NH3'] * torch.sum(NH3_manure_application_tensor_all, axis=1) + 
                     coef_N2O['粪污施用NO3'] * torch.sum(NO3_manure_application_tensor_all, axis=1) + 
                     coef_N2O['粪污施用N2O'] * torch.sum(N2O_manure_application_tensor_all, axis=1) + 
                     coef_N2O['秸秆焚烧'] * torch.sum(Straw_tensor_all[:, 3:], axis=1) + 
                     1 * torch.sum(straw_returning_N2O_tensor, axis=1))

# ================================
# 4. Calculate Emission Gaps and Filter Counties Needing Technology
# ================================

print("\n" + "-" * 40)
print("Calculating emission gaps and filtering counties needing technology...")

# Calculate emission gaps for each gas (current emissions - threshold)
gap_NH3_origin = Total_NH3_national - threshold_NH3_PB_all
gap_NO3_origin = Total_NO3_national - threshold_NO3_PB_all
gap_N_runoff_origin = Total_N_runoff_national - threshold_N_runoff_PB_all

# Filter counties needing technology addition (counties exceeding any gas threshold)
# 读取超标县域数据
exceed_counties_df = pd.read_excel('data/超标县域.xlsx')
exceed_counties_mask = exceed_counties_df['超标县'] == 1

# 获取超标县域中的县名列表
exceed_county_names = exceed_counties_df[exceed_counties_mask]['所属地州'].tolist()

# 创建超标县域的布尔掩码（基于县名匹配）
exceed_counties_bool = IDs_all['Counties'].isin(exceed_county_names)

# 三种气体超标的县
gas_exceed_mask = ~((gap_NO3_origin <= 0) & 
                    (gap_N_runoff_origin <= 0) & 
                    (gap_NH3_origin <= 0))
gas_exceed_mask = gas_exceed_mask.numpy()  # 转换为numpy数组

# 需要技术的县：三种气体超标 或者 在超标县域Excel中标记为1
counties_need_tech = (gas_exceed_mask | exceed_counties_bool)

print(f"✓ Total counties: {len(IDs_all)}")
print(f"✓ Counties needing technology addition: {counties_need_tech.sum()}")
print(f"✓ Percentage of counties needing technology: {counties_need_tech.sum()/len(IDs_all)*100:.1f}%")

# ================================
# 5. National Total Emissions and Reduction Target Statistics
# ================================

print("\n" + "-" * 40)
print("Calculating national total emissions and reduction targets...")

# National total emissions for various gases
national_totals = {
    'NH3': torch.sum(Total_NH3_national).item(),
    'NO3': torch.sum(Total_NO3_national).item(),
    'N_runoff': torch.sum(Total_N_runoff_national).item(),
    'CH4': torch.sum(Total_CH4_national).item(),
    'N2O': torch.sum(Total_N2O_national).item()
}

# National reduction targets for various gases
national_reduction_targets = {
    'NH3': torch.sum(threshold_NH3_PB_all).item(),
    'NO3': torch.sum(threshold_NO3_PB_all).item(),
    'N_runoff': torch.sum(threshold_N_runoff_PB_all).item(),
    'CH4': threshold_CH4_all.item(),
    'N2O': threshold_N2O_all.item()
}

# Print national statistics
print("National total emissions:")
for gas, total in national_totals.items():
    print(f"  {gas}: {total:.2f}")

print("National reduction targets:")
for gas, target in national_reduction_targets.items():
    print(f"  {gas}: {target:.2f}")

# ================================
# 6. Filter Data (Keep Only Counties Needing Technology)
# ================================

print("\n" + "-" * 40)
print("Filtering data, keeping only counties needing technology addition...")

# Filter county basic information
IDs_filtered = IDs_all[counties_need_tech].reset_index(drop=True)

# Filter all related tensor data
filtered_tensors = {
    'NH3_Crop': NH3_Crop_tensor_all[counties_need_tech],
    'N2O_nitrogen_fertilizer': N2O_nitrogen_fertilizer_tensor_all[counties_need_tech],
    'NO3_nitrogen_fertilizer': NO3_nitrogen_fertilizer_tensor_all[counties_need_tech],
    'N_runoff': N_runoff_tensor_all[counties_need_tech],
    'NH3_Fecal_management': NH3_Fecal_management_tensor_all[counties_need_tech],
    'N2O_Fecal_management': N2O_Fecal_management_tensor_all[counties_need_tech],
    'NH3_manure_application': NH3_manure_application_tensor_all[counties_need_tech],
    'N2O_manure_application': N2O_manure_application_tensor_all[counties_need_tech],
    'NO3_manure_application': NO3_manure_application_tensor_all[counties_need_tech],
    'CH4_intestine': CH4_intestine_tensor_all[counties_need_tech],
    'CH4_Fecal_management': CH4_Fecal_management_tensor_all[counties_need_tech],
    'Straw': Straw_tensor_all[counties_need_tech],
    'Rice_CH4_Gg': Rice_CH4_Gg_all[counties_need_tech],
    'nitrogen_deposition_N2O': nitrogen_deposition_N2O_tensor[counties_need_tech],
    'straw_returning_N2O': straw_returning_N2O_tensor[counties_need_tech],
    'Soc_tensor_origin': Soc_tensor_origin_all[counties_need_tech]
}

# Filter threshold and scale data
thresholds_filtered = {
    'NO3_PB': threshold_NO3_PB_all[counties_need_tech],
    'N_runoff_PB': threshold_N_runoff_PB_all[counties_need_tech],
    'NH3_PB': threshold_NH3_PB_all[counties_need_tech]
}

county_scale_filtered = county_scale_all[counties_need_tech].reset_index(drop=True)

# Filter gap data
gaps_filtered = {
    'NO3': gap_NO3_origin[counties_need_tech],
    'N_runoff': gap_N_runoff_origin[counties_need_tech],
    'NH3': gap_NH3_origin[counties_need_tech]
}

print(f"✓ Filtered county data shape: {IDs_filtered.shape}")
print(f"✓ Example filtered county names: {IDs_filtered['Counties'].head().tolist()}")

# Get agricultural regions needing technology addition
agri_areas = IDs_filtered['所属农业亚区'].unique()
# Filter out NaN values and convert to string
agri_areas = [str(area) for area in agri_areas if pd.notna(area)]
print(f"✓ Number of agricultural regions needing technology: {len(agri_areas)}")
print(f"✓ Agricultural region list: {sorted(agri_areas)}")

# ================================
# 7. Calculate Gas Emissions for Filtered Counties
# ================================

print("\n" + "-" * 40)
print("Calculating total gas emissions for filtered counties...")

# Recalculate total emissions for various gases in filtered counties
filtered_totals = {}

# NH3 total emissions
filtered_totals['NH3'] = (torch.sum(filtered_tensors['NH3_Crop'], axis=1) + 
                         torch.sum(filtered_tensors['NH3_Fecal_management'], axis=1) + 
                         torch.sum(filtered_tensors['NH3_manure_application'], axis=1))

# NO3 total emissions
filtered_totals['NO3'] = (torch.sum(filtered_tensors['NO3_nitrogen_fertilizer'], axis=1) + 
                         torch.sum(filtered_tensors['NO3_manure_application'], axis=1))

# N_runoff total emissions
filtered_totals['N_runoff'] = torch.sum(filtered_tensors['N_runoff'], axis=1)

# CH4 total emissions
filtered_totals['CH4'] = (coef_CH4['default'] * torch.sum(filtered_tensors['CH4_intestine'], axis=1) + 
                         coef_CH4['default'] * torch.sum(filtered_tensors['CH4_Fecal_management'], axis=1) + 
                         coef_CH4['default'] * torch.sum(filtered_tensors['Straw'][:, :3], axis=1) + 
                         coef_CH4['Rice CH4 Gg'] * torch.sum(filtered_tensors['Rice_CH4_Gg'], axis=1))

# N2O total emissions
filtered_totals['N2O'] = (1 * torch.sum(filtered_tensors['nitrogen_deposition_N2O'], axis=1) + 
                         coef_N2O['粪污管理N2O'] * torch.sum(filtered_tensors['N2O_Fecal_management'], axis=1) + 
                         coef_N2O['粪污管理NH3'] * torch.sum(filtered_tensors['NH3_Fecal_management'], axis=1) + 
                         coef_N2O['氮肥N2O'] * torch.sum(filtered_tensors['N2O_nitrogen_fertilizer'], axis=1) + 
                         coef_N2O['种植业NH3挥发'] * torch.sum(filtered_tensors['NH3_Crop'], axis=1) + 
                         coef_N2O['氮肥NO3'] * torch.sum(filtered_tensors['NO3_nitrogen_fertilizer'], axis=1) + 
                         coef_N2O['N runoff'] * torch.sum(filtered_tensors['N_runoff'], axis=1) + 
                         coef_N2O['粪污施用NH3'] * torch.sum(filtered_tensors['NH3_manure_application'], axis=1) + 
                         coef_N2O['粪污施用NO3'] * torch.sum(filtered_tensors['NO3_manure_application'], axis=1) + 
                         coef_N2O['粪污施用N2O'] * torch.sum(filtered_tensors['N2O_manure_application'], axis=1) + 
                         coef_N2O['秸秆焚烧'] * torch.sum(filtered_tensors['Straw'][:, 3:], axis=1) + 
                         1 * torch.sum(filtered_tensors['straw_returning_N2O'], axis=1))

# Calculate filtered national totals
filtered_national_totals = {gas: torch.sum(total).item() for gas, total in filtered_totals.items()}

print("Filtered national total emissions:")
for gas, total in filtered_national_totals.items():
    print(f"  {gas}: {total:.2f}")

# ================================
# 8. Calculate Emission Proportions and Allocate Reduction Responsibilities by Agricultural Region
# ================================

print("\n" + "-" * 40)
print("Calculating emission proportions and allocating reduction responsibilities by agricultural region...")

agri_area_gas_ratios = {}
agri_area_reduction_targets = {}

for area in agri_areas:
    print(f"\nProcessing agricultural region: {area}")
    
    # Get county indices for this agricultural region
    area_mask = IDs_filtered['所属农业亚区'] == area
    area_indices = area_mask[area_mask].index
    
    if len(area_indices) == 0:
        print(f"  Warning: No corresponding counties found for agricultural region {area}")
        continue
    
    print(f"  This agricultural region contains {len(area_indices)} counties")
    
    # Calculate gas emissions for this agricultural region
    area_totals = {}
    for gas in ['NH3', 'NO3', 'N_runoff', 'CH4', 'N2O']:
        area_totals[gas] = torch.sum(filtered_totals[gas][area_indices]).item()
    
    # Calculate proportion of this agricultural region's emissions relative to national emissions
    area_ratios = {}
    for gas in ['CH4', 'N2O']:
        if filtered_national_totals[gas] > 0:
            area_ratios[gas] = area_totals[gas] / filtered_national_totals[gas]
        else:
            area_ratios[gas] = 0.0
    
    agri_area_gas_ratios[area] = area_ratios
    
    # Allocate reduction responsibilities based on emission proportions
    # Note: CH4 and N2O use national unified thresholds, NH3/NO3/N_runoff use sum of county-level thresholds
    agri_area_reduction_targets[area] = {
        'CH4_reduction': national_reduction_targets['CH4'] * area_ratios['CH4'],
        'N2O_reduction': national_reduction_targets['N2O'] * area_ratios['N2O']
    }
    
    # Print statistics for this agricultural region
    print(f"  Emission proportions:")
    for gas, ratio in area_ratios.items():
        print(f"    {gas}: {ratio:.4f} ({ratio*100:.2f}%)")
    
    print(f"  Reduction targets:")
    for gas in ['CH4', 'N2O']:
        target = agri_area_reduction_targets[area][f'{gas}_reduction']
        print(f"    {gas}: {target:.2f}")

# ================================
# 9. Data Organization and Output
# ================================

print("\n" + "-" * 40)
print("Organizing data and outputting to Excel file...")

# Organize national data
national_data = {
    'Indicator': ['NH3 Total Emissions', 'NO3 Total Emissions', 'N_runoff Total Emissions', 'CH4 Total Emissions', 'N2O Total Emissions',
            'NH3 Reduction Target', 'NO3 Reduction Target', 'N_runoff Reduction Target', 'CH4 Reduction Target', 'N2O Reduction Target',
            'Filtered NH3 Total Emissions', 'Filtered NO3 Total Emissions', 'Filtered N_runoff Total Emissions', 
            'Filtered CH4 Total Emissions', 'Filtered N2O Total Emissions'],
    'Value': [
        national_totals['NH3'], national_totals['NO3'], national_totals['N_runoff'],
        national_totals['CH4'], national_totals['N2O'],
        national_reduction_targets['NH3'], national_reduction_targets['NO3'], 
        national_reduction_targets['N_runoff'], national_reduction_targets['CH4'], 
        national_reduction_targets['N2O'],
        filtered_national_totals['NH3'], filtered_national_totals['NO3'], 
        filtered_national_totals['N_runoff'], filtered_national_totals['CH4'], 
        filtered_national_totals['N2O']
    ]
}

# Organize agricultural region data
agri_area_data_list = []
for area in sorted(agri_areas):
    if area in agri_area_gas_ratios:
        row = {
            'Agricultural Region': area,
            'CH4 Emission Proportion': agri_area_gas_ratios[area]['CH4'],
            'N2O Emission Proportion': agri_area_gas_ratios[area]['N2O'],
            'CH4 Reduction Target': agri_area_reduction_targets[area]['CH4_reduction'],
            'N2O Reduction Target': agri_area_reduction_targets[area]['N2O_reduction']
        }
        agri_area_data_list.append(row)

# Create DataFrames
national_df = pd.DataFrame(national_data)
agri_area_df = pd.DataFrame(agri_area_data_list)

# Save to Excel file
try:
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
        # Save national data
        national_df.to_excel(writer, sheet_name='National Data', index=False)
        
        # Save agricultural region data
        agri_area_df.to_excel(writer, sheet_name='Agricultural Region Data', index=False)
        
        # Add summary information sheet
        summary_data = {
            'Statistical Item': [
                'Total National Counties', 'Counties Needing Technology Addition', 'Percentage of Counties Needing Technology (%)',
                'Total Agricultural Regions', 'Agricultural Regions Needing Technology Addition'
            ],
            'Value': [
                len(IDs_all), counties_need_tech.sum(), 
                f"{counties_need_tech.sum()/len(IDs_all)*100}",
                len(IDs_all['所属农业亚区'].unique()), len(agri_areas)
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary Information', index=False)
    
    print(f"✓ Data successfully saved to: {OUTPUT_FILE}")
    print(f"✓ Contains 3 worksheets: National Data, Agricultural Region Data, Summary Information")
    
except Exception as e:
    print(f"✗ Error saving file: {e}")

# ================================
# 10. Results Summary
# ================================

print("\n" + "=" * 60)
print("Regional Emission Allocation Calculation Completed")
print("=" * 60)

print(f"Processing results summary:")
print(f"  • Total national counties: {len(IDs_all)}")
print(f"  • Counties needing technology addition: {counties_need_tech.sum()}")
print(f"  • Percentage of counties needing technology: {counties_need_tech.sum()/len(IDs_all)*100:.1f}%")
print(f"  • Agricultural regions needing technology addition: {len(agri_areas)}")
print(f"  • Output file: {OUTPUT_FILE}")

print("\nMain output data:")
print("  • National total emissions and reduction targets for various gases")
print("  • Gas emission proportions for each agricultural region")
print("  • Allocated reduction targets for each agricultural region")
print("  • Statistical data for filtered counties (only those needing technology)")

print("\nCalculation completed!")