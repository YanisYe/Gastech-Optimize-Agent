import os
import pandas as pd
import re
import torch

def CountyDataLoader(
    df_county="data/基础数据-县级尺度.xlsx",
    IDs_df = "data/县市亚区.xlsx",
    soc_df="data/SOC-县尺度.xlsx",
    livestock_scale="data/动物数量.xlsx",
    crop_scale="data/分县种植面积.xlsx",
    *args,
    **kwargs
    ):
    
    target = pd.read_excel(df_county, sheet_name="目标")
    threshold_NO3_PB = torch.tensor(target['NO3 PB (t)'].values)
    threshold_N_runoff_PB = torch.tensor(target['N runoff PB (t)'].values)
    threshold_NH3_PB = torch.tensor(target['NH3 PB 31 (t)'].values)

    unique_NO3_vals = torch.unique(threshold_NO3_PB)
    unique_NO3_vals = unique_NO3_vals[unique_NO3_vals != 9999000]
    if len(unique_NO3_vals) >= 2:
        val_smallest = torch.sort(unique_NO3_vals)[0][-1]
    else:
        val_smallest = torch.tensor(0.0)
    threshold_NO3_PB = torch.where(threshold_NO3_PB > 1000000, val_smallest, threshold_NO3_PB)

    unique_N_runoff_vals = torch.unique(threshold_N_runoff_PB)
    unique_N_runoff_vals = unique_N_runoff_vals[unique_N_runoff_vals != 9999000]
    if len(unique_N_runoff_vals) >= 2:
        val_smallest = torch.sort(unique_N_runoff_vals)[0][-1]
    else:
        val_smallest = torch.tensor(0.0)
    threshold_N_runoff_PB = torch.where(threshold_N_runoff_PB > 1000000, val_smallest, threshold_N_runoff_PB)

    unique_NH3_vals = torch.unique(threshold_NH3_PB)
    unique_NH3_vals = unique_NH3_vals[unique_NH3_vals != 9999000]
    if len(unique_NH3_vals) >= 2:
        val_smallest = torch.sort(unique_NH3_vals)[0][-1]
    else:
        val_smallest = torch.tensor(0.0)
    threshold_NH3_PB = torch.where(threshold_NH3_PB > 1000000, val_smallest, threshold_NH3_PB)

    total_GHG = pd.read_excel(df_county, sheet_name="总GHG")
    Total_CH4_tensor = torch.tensor(total_GHG[["Total CH4 True"]].values)
    Total_N2O_tensor = torch.tensor(total_GHG[["Total N2O True"]].values)

    # # CH4 target reduction ratio is 30%
    # threshold_CH4 = torch.sum(Total_CH4_tensor * 0.3)
    threshold_CH4 = target['全国CH4减排目标'].values[0]
    # # N2O target reduction ratio is 34%
    # threshold_N2O = torch.sum(Total_N2O_tensor * 0.34)
    threshold_N2O = target['全国N2O减排目标'].values[0]
    
    soc = pd.read_excel(soc_df).filter(like='total_organic_carbon_kg')
    # soc fillna with 0
    soc = soc.fillna(0)
    Rice_CH4_Gg_tensor = torch.tensor(total_GHG[["Rice CH4 Gg"]].values)
    nitrogen_deposition_N2O_tensor = torch.tensor(total_GHG[["大气氮沉降引起的N2O间接排放"]].values)
    straw_returning_N2O_tensor = torch.tensor(total_GHG[["秸秆还田N2O排放"]].values)
    NH3_Crop = pd.read_excel(df_county, sheet_name='种植业氮肥-NH3挥发')
    NH3_Fecal_management = pd.read_excel(df_county, sheet_name='粪便管理NH3')
    NH3_manure_application = pd.read_excel(df_county, sheet_name='粪肥施用NH3')
    N2O_Fecal_management = pd.read_excel(df_county, sheet_name='粪便管理N2O')
    N2O_manure_application = pd.read_excel(df_county, sheet_name='粪肥施用N2O')
    N2O_nitrogen_fertilizer = pd.read_excel(df_county, sheet_name='氮肥N2O')
    NO3_manure_application = pd.read_excel(df_county, sheet_name='粪肥施用NO3')
    # N03_Fecal_management = pd.read_excel(df_county, sheet_name='Fecal management NO3')
    NO3_nitrogen_fertilizer = pd.read_excel(df_county, sheet_name='Nitrogen fertilizer NO3')
    N_runoff = pd.read_excel(df_county, sheet_name='N runoff')
    CH4_Fecal_management = pd.read_excel(df_county, sheet_name='Fecal management CH4')
    CH4_intestine = pd.read_excel(df_county, sheet_name='Enteric CH4')
    Straw_burning = pd.read_excel(df_county, sheet_name='Straw burning')

    # Add agricultural region information  
    IDs = pd.read_excel(IDs_df)
    NH3_Crop_columns = pd.Series(NH3_Crop.columns[2:].to_list())
    N2O_nitrogen_fertilizer_columns = pd.Series(N2O_nitrogen_fertilizer.columns[2:].to_list())
    NO3_nitrogen_fertilizer_columns = pd.Series(NO3_nitrogen_fertilizer.columns[2:].to_list())
    N_runoff_columns = pd.Series(N_runoff.columns[2:].to_list())
    Soc_columns = pd.Series(soc.columns.to_list())
    
    NH3_Fecal_management_columns = pd.Series(NH3_Fecal_management.columns[4:].to_list())
    N2O_Fecal_management_columns = pd.Series(N2O_Fecal_management.columns[4:].to_list())
    # N03_Fecal_management_columns = pd.Series(N03_Fecal_management.columns[4:].to_list())

    NH3_manure_application_columns = pd.Series(NH3_manure_application.columns[4:].to_list())
    N2O_manure_application_columns = pd.Series(N2O_manure_application.columns[4:].to_list())
    NO3_manure_application_columns = pd.Series(NO3_manure_application.columns[4:].to_list())
    
    Straw_columns = pd.Series([x for x in Straw_burning.columns[1:].to_list()])

    CH4_intestine_columns = pd.Series(CH4_intestine.columns[3:].to_list())
    CH4_Fecal_management_columns = pd.Series(CH4_Fecal_management.columns[2:].to_list())

    # Manure application data
    NH3_manure_application_tensor = torch.tensor(NH3_manure_application.iloc[:, 4:].values)
    N2O_manure_application_tensor = torch.tensor(N2O_manure_application.iloc[:, 4:].values)
    NO3_manure_application_tensor = torch.tensor(NO3_manure_application.iloc[:, 4:].values)

    # Crop production data
    NH3_Crop_tensor = torch.tensor(NH3_Crop.iloc[:, 2:].values)
    N2O_nitrogen_fertilizer_tensor = torch.tensor(N2O_nitrogen_fertilizer.iloc[:, 2:].values)
    NO3_nitrogen_fertilizer_tensor = torch.tensor(NO3_nitrogen_fertilizer.iloc[:, 2:].values)
    N_runoff_tensor = torch.tensor(N_runoff.iloc[:, 2:].values)
    Soc_tensor = torch.tensor(soc.values)
    # Livestock farming data
    NH3_Fecal_management_tensor = torch.tensor(NH3_Fecal_management.iloc[:, 4:].values)
    N2O_Fecal_management_tensor = torch.tensor(N2O_Fecal_management.iloc[:, 4:].values)
    # N03_Fecal_management_tensor = torch.tensor(N03_Fecal_management.iloc[:, 4:].values)

    # Straw data
    Straw_tensor = torch.tensor(Straw_burning.iloc[:, 1:].values)
    
    # Enteric CH4 data
    CH4_intestine_tensor = torch.tensor(CH4_intestine.iloc[:, 3:].values)

    # Fecal management CH4 data
    CH4_Fecal_management_tensor = torch.tensor(CH4_Fecal_management.iloc[:, 2:].values)

    # Load industry scale data
    livestock_scale = pd.read_excel(livestock_scale)
    crop_scale = pd.read_excel(crop_scale)
    # Merge data
    county_scale = livestock_scale.merge(crop_scale, on=['County', 'Cities', 'Province'], how='left')
    county_scale = county_scale.merge(IDs, left_on=['County', 'Cities'], right_on=['Counties', 'Cities'], how='left')

    # Save original scale (for cost calculation)
    county_scale_original = county_scale.copy()

    # Column 0-1 standardization
    county_scale.iloc[:, 3:-4] = county_scale.iloc[:, 3:-4].apply(lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0) + 1e-6), axis=0)
    return IDs, \
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
            NH3_manure_application_tensor, \
            N2O_manure_application_tensor, \
            NO3_manure_application_tensor, \
            Rice_CH4_Gg_tensor, \
            nitrogen_deposition_N2O_tensor, \
            straw_returning_N2O_tensor, \
            NH3_Crop_tensor, \
            N2O_nitrogen_fertilizer_tensor, \
            NO3_nitrogen_fertilizer_tensor, \
            N_runoff_tensor, \
            Soc_tensor, \
            NH3_Fecal_management_tensor, \
            N2O_Fecal_management_tensor, \
            Straw_tensor, \
            CH4_intestine_tensor, \
            CH4_Fecal_management_tensor, \
            Total_CH4_tensor, \
            Total_N2O_tensor, \
            threshold_NO3_PB, \
            threshold_N_runoff_PB, \
            threshold_NH3_PB, \
            threshold_CH4, \
            threshold_N2O, \
            county_scale_original, \
            county_scale
            

def ProvinceCountyDataLoader(
    df_county="data/基础数据-县级尺度.xlsx",
    IDs_df = "data/县市亚区.xlsx",
    soc_df_path="data/SOC-县尺度.xlsx",
    livestock_scale="data/动物数量.xlsx",
    crop_scale="data/分县种植面积.xlsx",
    province=None,
    *args,
    **kwargs
):
    assert province is not None

    target = pd.read_excel(df_county, sheet_name="目标")
    
    NH3_Crop = pd.read_excel(df_county, sheet_name='种植业氮肥-NH3挥发')
    NH3_Fecal_management = pd.read_excel(df_county, sheet_name='粪便管理NH3')
    NH3_manure_application = pd.read_excel(df_county, sheet_name='粪肥施用NH3')
    N2O_Fecal_management = pd.read_excel(df_county, sheet_name='粪便管理N2O')
    N2O_manure_application = pd.read_excel(df_county, sheet_name='粪肥施用N2O')
    N2O_nitrogen_fertilizer = pd.read_excel(df_county, sheet_name='氮肥N2O')
    NO3_manure_application = pd.read_excel(df_county, sheet_name='粪肥施用NO3')
    NO3_nitrogen_fertilizer = pd.read_excel(df_county, sheet_name='氮肥NO3')
    N_runoff = pd.read_excel(df_county, sheet_name='N runoff')
    CH4_Fecal_management = pd.read_excel(df_county, sheet_name='粪便管理CH4')
    CH4_intestine = pd.read_excel(df_county, sheet_name='肠道CH4')
    Straw_burning = pd.read_excel(df_county, sheet_name='秸秆焚烧')

    # Province filtering
    IDs = pd.read_excel(IDs_df)
    indices = IDs.index

    threshold_NO3_PB = torch.tensor(target['NO3 PB (t)'].iloc[indices].values)
    threshold_N_runoff_PB = torch.tensor(target['N runoff PB (t)'].iloc[indices].values)
    threshold_NH3_PB = torch.tensor(target['NH3 PB 31 (t)'].iloc[indices].values)

    # Handle NO3 threshold outliers
    unique_NO3_vals = torch.unique(threshold_NO3_PB)
    unique_NO3_vals = unique_NO3_vals[unique_NO3_vals != 9999000]
    if len(unique_NO3_vals) >= 2:
        val_smallest = torch.sort(unique_NO3_vals)[0][-1]
    else:
        val_smallest = torch.tensor(0.0)
    threshold_NO3_PB = torch.where(threshold_NO3_PB > 1000000, val_smallest, threshold_NO3_PB)

    # Handle N_runoff threshold outliers
    unique_N_runoff_vals = torch.unique(threshold_N_runoff_PB)
    unique_N_runoff_vals = unique_N_runoff_vals[unique_N_runoff_vals != 9999000]
    if len(unique_N_runoff_vals) >= 2:
        val_smallest = torch.sort(unique_N_runoff_vals)[0][-1]
    else:
        val_smallest = torch.tensor(0.0)
    threshold_N_runoff_PB = torch.where(threshold_N_runoff_PB > 1000000, val_smallest, threshold_N_runoff_PB)

    # Handle NH3 threshold outliers
    unique_NH3_vals = torch.unique(threshold_NH3_PB)
    unique_NH3_vals = unique_NH3_vals[unique_NH3_vals != 9999000]
    if len(unique_NH3_vals) >= 2:
        val_smallest = torch.sort(unique_NH3_vals)[0][-1]
    else:
        val_smallest = torch.tensor(0.0)
    threshold_NH3_PB = torch.where(threshold_NH3_PB > 1000000, val_smallest, threshold_NH3_PB)

    total_GHG = pd.read_excel(df_county, sheet_name="总GHG")
    total_GHG = total_GHG.iloc[indices, :]
    Total_CH4_tensor = torch.tensor(total_GHG[["Total CH4 True"]].values)
    Total_N2O_tensor = torch.tensor(total_GHG[["Total N2O True"]].values)

    # CH4 target reduction ratio is 30%
    threshold_CH4 = target['全国CH4减排目标'].values[0]
    # N2O target reduction ratio is 34%
    # threshold_N2O = torch.sum(Total_N2O_tensor * 0.4)
    threshold_N2O = target['全国N2O减排目标'].values[0]
    
    soc_df = pd.read_excel(soc_df_path).iloc[indices, :]
    soc = soc_df.filter(like='total_organic_carbon_kg')
    # soc fillna with 0
    soc = soc.fillna(0)
    Rice_CH4_Gg_tensor = torch.tensor(total_GHG[["Rice CH4 Gg"]].values)
    nitrogen_deposition_N2O_tensor = torch.tensor(total_GHG[["大气氮沉降引起的N2O间接排放"]].values)
    straw_returning_N2O_tensor = torch.tensor(total_GHG[["秸秆还田N2O排放"]].values)
    # Data column names
    NH3_Crop_columns = pd.Series(NH3_Crop.columns[2:].to_list())
    N2O_nitrogen_fertilizer_columns = pd.Series(N2O_nitrogen_fertilizer.columns[2:].to_list())
    NO3_nitrogen_fertilizer_columns = pd.Series(NO3_nitrogen_fertilizer.columns[2:].to_list())
    N_runoff_columns = pd.Series(N_runoff.columns[2:].to_list())
    Soc_columns = pd.Series(soc.columns.to_list())
    
    NH3_Fecal_management_columns = pd.Series(NH3_Fecal_management.columns[4:].to_list())
    N2O_Fecal_management_columns = pd.Series(N2O_Fecal_management.columns[4:].to_list())
    # N03_Fecal_management_columns = pd.Series(N03_Fecal_management.columns[4:].to_list())

    NH3_manure_application_columns = pd.Series(NH3_manure_application.columns[4:].to_list())
    N2O_manure_application_columns = pd.Series(N2O_manure_application.columns[4:].to_list())
    NO3_manure_application_columns = pd.Series(NO3_manure_application.columns[4:].to_list())
    
    Straw_columns = pd.Series([x for x in Straw_burning.columns[1:].to_list()])
    CH4_intestine_columns = pd.Series(CH4_intestine.columns[3:].to_list())
    CH4_Fecal_management_columns = pd.Series(CH4_Fecal_management.columns[2:].to_list())

    # Manure application data
    NH3_manure_application_tensor = torch.tensor(NH3_manure_application.iloc[indices, 4:].values)
    N2O_manure_application_tensor = torch.tensor(N2O_manure_application.iloc[indices, 4:].values)
    NO3_manure_application_tensor = torch.tensor(NO3_manure_application.iloc[indices, 4:].values)

    # Crop production data
    NH3_Crop_tensor = torch.tensor(NH3_Crop.iloc[indices, 2:].values)
    N2O_nitrogen_fertilizer_tensor = torch.tensor(N2O_nitrogen_fertilizer.iloc[indices, 2:].values)
    NO3_nitrogen_fertilizer_tensor = torch.tensor(NO3_nitrogen_fertilizer.iloc[indices, 2:].values)
    N_runoff_tensor = torch.tensor(N_runoff.iloc[indices, 2:].values)
    Soc_tensor = torch.tensor(soc.values)
    # Livestock farming data
    NH3_Fecal_management_tensor = torch.tensor(NH3_Fecal_management.iloc[indices, 4:].values)
    N2O_Fecal_management_tensor = torch.tensor(N2O_Fecal_management.iloc[indices, 4:].values)
    # N03_Fecal_management_tensor = torch.tensor(N03_Fecal_management.iloc[indices, 4:].values)

    # Straw data
    Straw_tensor = torch.tensor(Straw_burning.iloc[:, 1:].values)

    
    # Enteric CH4 data
    CH4_intestine_tensor = torch.tensor(CH4_intestine.iloc[indices, 3:].values)

    # Fecal management CH4 data
    CH4_Fecal_management_tensor = torch.tensor(CH4_Fecal_management.iloc[indices, 2:].values)

    # Load industry scale data
    livestock_scale_df = pd.read_excel(livestock_scale)
    crop_scale_df = pd.read_excel(crop_scale)
    # Merge data
    county_scale = livestock_scale_df.merge(crop_scale_df, on=['County', 'Cities', 'Province'], how='left')
    # Filter industry scale data by province
    county_scale = county_scale[county_scale['Province'] == province]

    # Save original scale (for cost calculation)
    county_scale_original = county_scale.copy()

    # Column 0-1 standardization
    county_scale.iloc[:, 3:] = county_scale.iloc[:, 3:].apply(lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0) + 1e-6), axis=0)

    IDs = IDs.reset_index(drop=True)
    return IDs, \
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
            NH3_manure_application_tensor, \
            N2O_manure_application_tensor, \
            NO3_manure_application_tensor, \
            Rice_CH4_Gg_tensor, \
            nitrogen_deposition_N2O_tensor, \
            straw_returning_N2O_tensor, \
            NH3_Crop_tensor, \
            N2O_nitrogen_fertilizer_tensor, \
            NO3_nitrogen_fertilizer_tensor, \
            N_runoff_tensor, \
            Soc_tensor, \
            NH3_Fecal_management_tensor, \
            N2O_Fecal_management_tensor, \
            Straw_tensor, \
            CH4_intestine_tensor, \
            CH4_Fecal_management_tensor, \
            Total_CH4_tensor, \
            Total_N2O_tensor, \
            threshold_NO3_PB, \
            threshold_N_runoff_PB, \
            threshold_NH3_PB, \
            threshold_CH4, \
            threshold_N2O, \
            county_scale_original, \
            county_scale

def AgriAreaCountyDataLoader(
    df_county="data/基础数据-县级尺度.xlsx",
    IDs_df = "data/县市亚区.xlsx",
    soc_df_path="data/SOC-县尺度.xlsx",
    livestock_scale="data/动物数量.xlsx",
    crop_scale="data/分县种植面积.xlsx",
    area=None,
    *args,
    **kwargs):

    target = pd.read_excel(df_county, sheet_name="目标")
    
    NH3_Crop = pd.read_excel(df_county, sheet_name='种植业氮肥-NH3挥发')
    NH3_Fecal_management = pd.read_excel(df_county, sheet_name='粪便管理NH3')
    NH3_manure_application = pd.read_excel(df_county, sheet_name='粪肥施用NH3')
    N2O_Fecal_management = pd.read_excel(df_county, sheet_name='粪便管理N2O')
    N2O_manure_application = pd.read_excel(df_county, sheet_name='粪肥施用N2O')
    N2O_nitrogen_fertilizer = pd.read_excel(df_county, sheet_name='氮肥N2O')
    NO3_manure_application = pd.read_excel(df_county, sheet_name='粪肥施用NO3')
    NO3_nitrogen_fertilizer = pd.read_excel(df_county, sheet_name='氮肥NO3')
    N_runoff = pd.read_excel(df_county, sheet_name='N runoff')
    CH4_Fecal_management = pd.read_excel(df_county, sheet_name='粪便管理CH4')
    CH4_intestine = pd.read_excel(df_county, sheet_name='肠道CH4')
    Straw_burning = pd.read_excel(df_county, sheet_name='秸秆焚烧')

    # Agricultural region filtering

    IDs = pd.read_excel(IDs_df)
    IDs = IDs[IDs['所属农业亚区'] == area]
    indices = IDs.index

    threshold_NO3_PB = torch.tensor(target['NO3 PB (t)'].iloc[indices].values)
    threshold_N_runoff_PB = torch.tensor(target['N runoff PB (t)'].iloc[indices].values)
    threshold_NH3_PB = torch.tensor(target['NH3 PB 31 (t)'].iloc[indices].values)

    # Handle NO3 threshold outliers
    unique_NO3_vals = torch.unique(threshold_NO3_PB)
    unique_NO3_vals = unique_NO3_vals[unique_NO3_vals != 9999000]
    if len(unique_NO3_vals) >= 2:
        val_smallest = torch.tensor(408395.156056929)
    else:
        val_smallest = torch.tensor(0.0)
    threshold_NO3_PB = torch.where(threshold_NO3_PB > 1000000, val_smallest, threshold_NO3_PB)

    # Handle N_runoff threshold outliers
    unique_N_runoff_vals = torch.unique(threshold_N_runoff_PB)
    unique_N_runoff_vals = unique_N_runoff_vals[unique_N_runoff_vals != 9999000]
    if len(unique_N_runoff_vals) >= 2:
        val_smallest = torch.tensor(297733.242802793)
    else:
        val_smallest = torch.tensor(0.0)
    threshold_N_runoff_PB = torch.where(threshold_N_runoff_PB > 1000000, val_smallest, threshold_N_runoff_PB)

    # Handle NH3 threshold outliers
    unique_NH3_vals = torch.unique(threshold_NH3_PB)
    unique_NH3_vals = unique_NH3_vals[unique_NH3_vals != 9999000]
    if len(unique_NH3_vals) >= 2:
        val_smallest = torch.tensor(372825.2553681)
    else:
        val_smallest = torch.tensor(0.0)
    threshold_NH3_PB = torch.where(threshold_NH3_PB > 1000000, val_smallest, threshold_NH3_PB)

    total_GHG = pd.read_excel(df_county, sheet_name="总GHG")
    total_GHG = total_GHG.iloc[indices, :]
    Total_CH4_tensor = torch.tensor(total_GHG[["Total CH4 True"]].values)
    Total_N2O_tensor = torch.tensor(total_GHG[["Total N2O True"]].values)

    # CH4 target reduction ratio is 30%
    # threshold_CH4 = torch.sum(Total_CH4_tensor * 0.4)
    threshold_CH4 = target['全国CH4减排目标'].values[0]
    # N2O target reduction ratio is 34%
    # threshold_N2O = torch.sum(Total_N2O_tensor * 0.4)
    threshold_N2O = target['全国N2O减排目标'].values[0]
    soc_df = pd.read_excel(soc_df_path).iloc[indices, :]
    soc = soc_df.filter(like='total_organic_carbon_kg')
    # soc fillna with 0
    soc = soc.fillna(0)
    Rice_CH4_Gg_tensor = torch.tensor(total_GHG[["Rice CH4 Gg"]].values)
    nitrogen_deposition_N2O_tensor = torch.tensor(total_GHG[["大气氮沉降引起的N2O间接排放"]].values)
    straw_returning_N2O_tensor = torch.tensor(total_GHG[["秸秆还田N2O排放"]].values)
    # Data column names
    NH3_Crop_columns = pd.Series(NH3_Crop.columns[2:].to_list())
    N2O_nitrogen_fertilizer_columns = pd.Series(N2O_nitrogen_fertilizer.columns[2:].to_list())
    NO3_nitrogen_fertilizer_columns = pd.Series(NO3_nitrogen_fertilizer.columns[2:].to_list())
    N_runoff_columns = pd.Series(N_runoff.columns[2:].to_list())
    Soc_columns = pd.Series(soc.columns.to_list())
    
    NH3_Fecal_management_columns = pd.Series(NH3_Fecal_management.columns[4:].to_list())
    N2O_Fecal_management_columns = pd.Series(N2O_Fecal_management.columns[4:].to_list())
    # N03_Fecal_management_columns = pd.Series(N03_Fecal_management.columns[4:].to_list())

    NH3_manure_application_columns = pd.Series(NH3_manure_application.columns[4:].to_list())
    N2O_manure_application_columns = pd.Series(N2O_manure_application.columns[4:].to_list())
    NO3_manure_application_columns = pd.Series(NO3_manure_application.columns[4:].to_list())
    
    Straw_columns = pd.Series([x for x in Straw_burning.columns[1:].to_list()])
    CH4_intestine_columns = pd.Series(CH4_intestine.columns[3:].to_list())
    CH4_Fecal_management_columns = pd.Series(CH4_Fecal_management.columns[2:].to_list())

    # Manure application data
    NH3_manure_application_tensor = torch.tensor(NH3_manure_application.iloc[indices, 4:].values)
    N2O_manure_application_tensor = torch.tensor(N2O_manure_application.iloc[indices, 4:].values)
    NO3_manure_application_tensor = torch.tensor(NO3_manure_application.iloc[indices, 4:].values)

    # Crop production data
    NH3_Crop_tensor = torch.tensor(NH3_Crop.iloc[indices, 2:].values)
    N2O_nitrogen_fertilizer_tensor = torch.tensor(N2O_nitrogen_fertilizer.iloc[indices, 2:].values)
    NO3_nitrogen_fertilizer_tensor = torch.tensor(NO3_nitrogen_fertilizer.iloc[indices, 2:].values)
    N_runoff_tensor = torch.tensor(N_runoff.iloc[indices, 2:].values)
    Soc_tensor = torch.tensor(soc.values)
    # Livestock farming data
    NH3_Fecal_management_tensor = torch.tensor(NH3_Fecal_management.iloc[indices, 4:].values)
    N2O_Fecal_management_tensor = torch.tensor(N2O_Fecal_management.iloc[indices, 4:].values)
    # N03_Fecal_management_tensor = torch.tensor(N03_Fecal_management.iloc[indices, 4:].values)

    # Straw data
    Straw_tensor = torch.tensor(Straw_burning.iloc[indices, 1:].values)
    
    # Enteric CH4 data
    CH4_intestine_tensor = torch.tensor(CH4_intestine.iloc[indices, 3:].values)

    # Fecal management CH4 data
    CH4_Fecal_management_tensor = torch.tensor(CH4_Fecal_management.iloc[indices, 2:].values)

    # Load industry scale data
    livestock_scale_df = pd.read_excel(livestock_scale)
    crop_scale_df = pd.read_excel(crop_scale)
    # Merge data
    county_scale = livestock_scale_df.merge(crop_scale_df, on=['County', 'Cities', 'Province'], how='left')
    county_scale = county_scale.merge(IDs, left_on=['County', 'Cities'], right_on=['Counties', 'Cities'], how='left')

    # Filter industry scale data by agricultural region
    county_scale = county_scale[county_scale['所属农业亚区'] == area]

    # Save original scale (for cost calculation)
    county_scale_original = county_scale.copy()

    # Column 0-1 standardization
    county_scale.iloc[:, 3:-4] = county_scale.iloc[:, 3:-4].apply(lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0) + 1e-6), axis=0)


    IDs = IDs.reset_index(drop=True)
    return IDs, \
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
            NH3_manure_application_tensor, \
            N2O_manure_application_tensor, \
            NO3_manure_application_tensor, \
            Rice_CH4_Gg_tensor, \
            nitrogen_deposition_N2O_tensor, \
            straw_returning_N2O_tensor, \
            NH3_Crop_tensor, \
            N2O_nitrogen_fertilizer_tensor, \
            NO3_nitrogen_fertilizer_tensor, \
            N_runoff_tensor, \
            Soc_tensor, \
            NH3_Fecal_management_tensor, \
            N2O_Fecal_management_tensor, \
            Straw_tensor, \
            CH4_intestine_tensor, \
            CH4_Fecal_management_tensor, \
            Total_CH4_tensor, \
            Total_N2O_tensor, \
            threshold_NO3_PB, \
            threshold_N_runoff_PB, \
            threshold_NH3_PB, \
            threshold_CH4, \
            threshold_N2O, \
            county_scale_original, \
            county_scale


def PriorityTechDataLoader(livestock_tech = "data/畜牧业技术列单.xlsx", crop_tech = "data/种植业技术列单.xlsx", priority = 4):
    # Livestock technology sheets
    Feeding = pd.read_excel(livestock_tech, sheet_name="Feeding")
    Feeding = Feeding[Feeding['经济分级'] == priority]

    Housing = pd.read_excel(livestock_tech, sheet_name="Housing")
    Housing = Housing[Housing['经济分级'] == priority]
        
    # slurry storage
    slurry_storage = pd.read_excel(livestock_tech, sheet_name="slurry storage")
    slurry_storage = slurry_storage[slurry_storage['经济分级'] == priority]

    # soild storage
    soild_storage = pd.read_excel(livestock_tech, sheet_name="soild storage")
    soild_storage = soild_storage[soild_storage['经济分级'] == priority]

    # composting
    composting = pd.read_excel(livestock_tech, sheet_name="composting")
    composting = composting[composting['经济分级'] == priority]

    # additives application
    additives_application = pd.read_excel(livestock_tech, sheet_name="additives application")
    additives_application = additives_application[additives_application['经济分级'] == priority]

    # soild application
    soild_application = pd.read_excel(livestock_tech, sheet_name="soild application")
    soild_application = soild_application[soild_application['经济分级'] == priority]

    # slurry application
    slurry_application = pd.read_excel(livestock_tech, sheet_name="slurry application")
    slurry_application = slurry_application[slurry_application['经济分级'] == priority]

    # Crop technology sheets
    crop = pd.read_excel(crop_tech)
    crop = crop[crop['经济分级'] == priority]

    Feeding_select = Feeding[['技术间的冲突','Mitigation strategy', '经济分级', 'Livestock species']]
    Feeding_select['class'] = 'Feeding'

    Housing_select = Housing[['技术间的冲突','Mitigation strategy', '经济分级', 'Livestock species']]
    Housing_select['class'] = 'Housing'

    slurry_storage_select = slurry_storage[['技术间的冲突','Mitigation strategy', '经济分级', 'Livestock species']]
    slurry_storage_select['class'] = 'slurry storage'

    soild_storage_select = soild_storage[['技术间的冲突','Mitigation strategy', '经济分级', 'Livestock species']]
    soild_storage_select['class'] = 'soild storage'

    composting_select = composting[['技术间的冲突','Mitigation strategy', '经济分级', 'Livestock species']]
    composting_select['class'] = 'composting'

    additives_application_select = additives_application[['技术间的冲突','Mitigation strategy', '经济分级', 'Livestock species']]
    additives_application_select['class'] = 'additives application'

    soild_application_select = soild_application[['技术间的冲突','Mitigation strategy', '经济分级', 'Livestock species']]
    soild_application_select['class'] = 'soild application'

    slurry_application_select = slurry_application[['技术间的冲突','Mitigation strategy', '经济分级', 'Livestock species']]
    slurry_application_select['class'] = 'slurry application'

    crop_select = crop[['技术间的冲突','Mitigation strategy', '经济分级', 'Crop species']]
    crop_select['class'] = 'crop'

    tech_set = pd.concat([Feeding_select, Housing_select, slurry_storage_select, soild_storage_select, composting_select, additives_application_select, soild_application_select, slurry_application_select, crop_select], ignore_index=True)
    return Feeding, \
            Housing, \
            slurry_storage, \
            soild_storage, \
            composting, \
            additives_application, \
            soild_application, \
            slurry_application, \
            crop, \
            tech_set, \
            Feeding_select, \
            Housing_select, \
            slurry_storage_select, \
            soild_storage_select, \
            composting_select, \
            additives_application_select, \
            soild_application_select, \
            slurry_application_select, \
            crop_select

def TechDataLoader(livestock_tech = "data/畜牧业技术列单-经济产量0626.xlsx", crop_tech = "data/种植业技术列单产量产业0626.xlsx"):
    # Livestock technology sheets
    Feeding = pd.read_excel(livestock_tech, sheet_name="Feeding")
    Housing = pd.read_excel(livestock_tech, sheet_name="Housing")
    # slurry storage
    slurry_storage = pd.read_excel(livestock_tech, sheet_name="slurry storage")
    # soild storage
    soild_storage = pd.read_excel(livestock_tech, sheet_name="soild storage")
    # composting
    composting = pd.read_excel(livestock_tech, sheet_name="composting")
    # additives application
    additives_application = pd.read_excel(livestock_tech, sheet_name="additives application")
    # soild application
    soild_application = pd.read_excel(livestock_tech, sheet_name="soild application")
    # slurry application
    slurry_application = pd.read_excel(livestock_tech, sheet_name="slurry application")

    # Crop technology sheets
    crop = pd.read_excel(crop_tech)

    Feeding_select = Feeding[['技术间的冲突','Mitigation strategy', '技术分级', '经济成本','Livestock species']].copy()
    Feeding_select['class'] = 'Feeding'

    Housing_select = Housing[['技术间的冲突','Mitigation strategy', '技术分级', '经济成本','Livestock species']].copy()
    Housing_select['class'] = 'Housing'

    slurry_storage_select = slurry_storage[['技术间的冲突','Mitigation strategy', '技术分级', '经济成本','Livestock species']].copy()
    slurry_storage_select['class'] = 'slurry storage'

    soild_storage_select = soild_storage[['技术间的冲突','Mitigation strategy', '技术分级', '经济成本','Livestock species']].copy()
    soild_storage_select['class'] = 'soild storage'

    composting_select = composting[['技术间的冲突','Mitigation strategy', '技术分级', '经济成本','Livestock species']].copy()
    composting_select['class'] = 'composting'

    additives_application_select = additives_application[['技术间的冲突','Mitigation strategy', '技术分级', '经济成本','Livestock species']].copy()
    additives_application_select['class'] = 'additives application'

    soild_application_select = soild_application[['技术间的冲突','Mitigation strategy', '技术分级', '经济成本','Livestock species']].copy()
    soild_application_select['class'] = 'soild application'

    slurry_application_select = slurry_application[['技术间的冲突','Mitigation strategy', '技术分级', '经济成本','Livestock species']].copy()
    slurry_application_select['class'] = 'slurry application'

    crop_select = crop[['技术间的冲突','Mitigation strategy', '经济成本','技术分级', 'Crop species']].copy()
    crop_select['class'] = 'crop'

    tech_set = pd.concat([Feeding_select,
                          Housing_select,
                          slurry_storage_select,
                          soild_storage_select,
                          composting_select,
                          additives_application_select,
                          soild_application_select,
                          slurry_application_select,
                          crop_select],
                          ignore_index=True)

    # For crop technologies: economic cost * 1000, for livestock technologies: economic cost * 10000
    tech_set['经济成本'] = tech_set.apply(lambda row: row['经济成本'] * 1000 if row['class'] == 'crop' else row['经济成本'] * 10000, axis=1)
    # 0-1 standardization of economic cost
    tech_set['标准化经济成本'] = (tech_set['经济成本'] - tech_set['经济成本'].min()) / (tech_set['经济成本'].max() - tech_set['经济成本'].min() + 1e-6)

    # Check if technology ID mapping table already exists
    output_path = "data/tech_id_mapping.xlsx"
    if os.path.exists(output_path):
        # If file already exists, read directly
        tech_id_mapping = pd.read_excel(output_path)
        print(f"Loading technology ID mapping table from file: {output_path}")
    else:
        # Generate technology ID mapping table
        tech_id_mapping = tech_set[['技术间的冲突', 'Mitigation strategy', 'class']].copy()
        tech_id_mapping['技术ID'] = tech_id_mapping.index
        # Add species information
        # Use 'Crop species' for crop technologies, 'Livestock species' for livestock technologies
        tech_id_mapping['species'] = tech_set.apply(
            lambda row: row['Crop species'] if row['class'] == 'crop' else row['Livestock species'],
            axis=1
        )
        tech_id_mapping = tech_id_mapping[['技术ID', 'Mitigation strategy', 'class', '技术间的冲突', 'species']]
        # Save technology ID mapping table to file
        tech_id_mapping.to_excel(output_path, index=False)
        print(f"Technology ID mapping table saved to: {output_path}")

    return  Feeding, \
            Housing, \
            slurry_storage, \
            soild_storage, \
            composting, \
            additives_application, \
            soild_application, \
            slurry_application, \
            crop, \
            tech_set


# def find_duplicate_column_pairs(df):
#     pairs = []
#     columns = df.columns
#     for col in columns:
#         if col.endswith('.1'):
#             base_col = col[:-2]
#             if base_col in columns:
#                 pairs.append((base_col, col))
#     return pairs

# def merge_duplicate_columns(df):
#     col_pairs = find_duplicate_column_pairs(df)
#     for base_col, col_with_dot1 in col_pairs:
#         df[base_col] = df[[base_col, col_with_dot1]].apply(lambda x: x.dropna().iloc[0] if not x.dropna().empty else pd.NA, axis=1)
#     df = df.drop([col_with_dot1 for _, col_with_dot1 in col_pairs], axis=1)
#     return df