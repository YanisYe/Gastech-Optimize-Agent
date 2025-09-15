import torch
import pandas as pd
import numpy as np

GAS_COEFFICIENTS = {
    'N2O': {
        '粪污管理NH3': 0.01 * 14 / 17 * 310 * 44 / 28 / 1000,
        '粪污管理N2O': 44 / 28 * 310 / 1000,
        '氮肥N2O': 310 / 28 * 44 / 1000,
        '种植业NH3挥发': 0.01 * 310 / 1000 * 44 / 28,
        '氮肥NO3': 0.0075 * 310 / 1000 * 44 / 28,
        'N runoff': 0.0075 * 310 / 1000 * 44 / 28,
        '粪污施用NO3': 0.0075 * 310 / 1000 * 44 / 28,
        # '粪污管理NO3': 0.0075 * 310 / 1000 * 44 / 28,
        '粪污施用NH3': 0.01 * 44 / 28 * 310 / 1000,
        '粪污施用N2O': 1.0,
        '秸秆焚烧': 310.0,
        'default': 1.0
    },
    'CH4': {
        'Rice CH4 Gg': 1.0,
        'default': 21.0
    }
}
def CH4_intestine_sum(countyID, CH4_intestine_data):
    return torch.sum(CH4_intestine_data[countyID]) * 21

def Fecal_management_CH4_sum(countyID, CH4_Fecal_management_data):
    return torch.sum(CH4_Fecal_management_data[countyID]) * 21

def CH4_straw_sum(countyID, crops_data):
    return torch.sum(crops_data[countyID, :3]) * 21
    
def Fecal_management_N2O_direct_sum(countyID, Livestocks_data):
    return torch.sum(Livestocks_data[countyID]) /1000 * 44 / 28 * 310

def Fecal_management_N2O_indirect_sum(countyID, NH3_Fecal_management_data, NO3_Fecal_management_data):
    return torch.sum(NH3_Fecal_management_data[countyID]) * 0.01 * 14 / 17 * 310 * 44 / 28 / 1000 + torch.sum(NO3_Fecal_management_data[countyID]) * 0.0075 * 310 / 1000 * 44 / 28

def N_fertilizer_N2O_direct_sum(countyID, crops_data):
    return torch.sum(crops_data[countyID]) * 310 / 28 * 44 / 1000

def N_fertilizer_N2O_indirect_sum(countyID, NH3_crops_data, NO3_crops_data, N_runoff_data):
    return torch.sum(NH3_crops_data[countyID]) * 0.01 * 44 / 28 * 310 / 1000 + (torch.sum(NO3_crops_data[countyID]) + torch.sum(N_runoff_data[countyID])) * 0.0075 * 44 / 28 * 310 / 1000

# =粪肥施用NH3!D5*0.01*310/1000*44/28+粪肥施用NO3!D5*0.0075*44/28*310/1000
def N2O_manure_application_indirect_sum(countyID, NH3_manure_application_data, NO3_manure_application_data):
    return torch.sum(NH3_manure_application_data[countyID]) * 0.01 * 310 / 1000 * 44 / 28 + torch.sum(NO3_manure_application_data[countyID]) * 0.0075 * 44 / 28 * 310 / 1000

# =粪肥施用N2O!D5
def N2O_manure_application_direct_sum(countyID, N2O_manure_application_data):
    return torch.sum(N2O_manure_application_data[countyID])

# =SUM(秸秆焚烧!E6:G6)*310
def N2O_straw_sum(countyID, N2O_straw_data):
    return torch.sum(N2O_straw_data[countyID, 3:]) * 310


def get_conflicts_tech(techId, techSet):
    techId = techId.cpu().numpy() if isinstance(techId, torch.Tensor) else techId
    techId = techId.reshape(-1) if techId.ndim == 2 else techId
    n = len(techId) if techId.ndim != 0 else 1
    target_rows = techSet.iloc[techId]
    industry = target_rows['class']
    conflict = target_rows['技术间的冲突']
    condition = (
        (techSet['class'] == industry) & 
        (techSet['技术间的冲突'] is conflict)
    )
    
    return condition
