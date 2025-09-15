'''
动作空间为每一次选择一个县添加一个技术
'''
from re import L
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import sys
import os
from fuzzywuzzy import process
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
np.random.seed(0) # 为了保证每次运行结果一致，设置随机种子
import torch
from scipy.optimize import linprog
from dataLoader import *
import copy
from utils import *
import ast

class GasEnvConfig:
    def __init__(self,
                  Reward_priority: list,
                  county_df_path,
                  livestock_tech_path,
                  crop_tech_path,
                  soc_df,
                  livestock_scale,
                  crop_scale,
                  province:str = None,
                  area:str = None,
                  IDs_df:str = None,
                  save_path:str = None,
                  linear_result_path:str = None,
                  lp_targets_path:str = None,
                  ch4_n2o_target_mode:str = 'emission_threshold',  # 'emission_threshold' 或 'reduction_amount'
                  debug_step_logging: bool = False,
                  debug_log_dir: str = None,
                  only_lp_phase:bool = False,
                  total_steps: int = 2**13,  # 总训练步数
                  lp_phase_ratio: float = 0.5,  # 线性规划阶段占比，前10%只在线性规划解空间学习
                  phase_1_ratio: float = 0.7,  # 第一阶段占比，达到 30% 放开技术等级 1
                  phase_2_ratio: float = 0.9,  # 第二阶段占比，达到 60% 放开技术等级 2
                  ):
        '''
        初始化环境配置
        Reward_priority: 奖励优先级
        county_df_path: 县数据路径
        livestock_tech_path: 畜牧业技术路径
        crop_tech_path: 种植业技术路径
        soc_df: 土壤有机碳数据路径
        livestock_scale: 畜牧业规模数据路径
        crop_scale: 种植业规模数据路径
        province: 省份
        area: 亚区
        area_df_path: 亚区数据路径
        linear_result_path: 线性规划结果路径
        reduction_variables_path: 减排目标路径
        lp_targets_path: LP结果读取每县CH4/N2O减排目标路径
        only_lp_phase: 是否只在线性规划解空间学习
        total_steps: 总训练步数
        lp_phase_ratio: 线性规划阶段占比，前10%只在线性规划解空间学习
        phase_1_ratio: 第一阶段占比，达到 30% 放开技术等级 1
        phase_2_ratio: 第二阶段占比，达到 60% 放开技术等级 2
        '''
        self.Reward_priority = Reward_priority
        self.county_df_path = county_df_path
        self.livestock_tech_path = livestock_tech_path
        self.crop_tech_path = crop_tech_path
        self.soc_df_path = soc_df
        self.province = province
        self.livestock_scale = livestock_scale
        self.crop_scale = crop_scale
        self.area = area
        self.IDs_df = IDs_df
        self.save_path = save_path
        self.linear_result_path = linear_result_path
        self.lp_targets_path = lp_targets_path
        assert ch4_n2o_target_mode in ['emission_threshold', 'reduction_amount']
        self.ch4_n2o_target_mode = ch4_n2o_target_mode
        self.debug_step_logging = debug_step_logging
        self.debug_log_dir = debug_log_dir
        self.only_lp_phase = only_lp_phase
        self.total_steps = total_steps
        self.lp_phase_ratio = lp_phase_ratio
        self.phase_1_ratio = phase_1_ratio
        self.phase_2_ratio = phase_2_ratio

class GasEnv(gym.Env):
    def __init__(self, config: GasEnvConfig):
        super(GasEnv, self).__init__()
        
        self.reward_priority = config.Reward_priority # 初始化优先级权重
        self.county_df_path = config.county_df_path
        self.IDs_df = config.IDs_df
        self.save_path = config.save_path
        self.total_steps = config.total_steps
        self.lp_phase_ratio = config.lp_phase_ratio
        self.phase_1_ratio = config.phase_1_ratio
        self.phase_2_ratio = config.phase_2_ratio
        self.area = config.area
        self.current_phase = 0  # 当前课程学习阶段

        # 初始化数据加载器，根据配置选择不同的加载器
        if config.province == None:  # 如果没有指定省份，默认使用县级数据加载器
            StateDataLoader = CountyDataLoader
            area = None  # 默认不指定区域
            if config.area != None:  # 如果指定了亚区，则使用亚区县级数据加载器
                StateDataLoader = AgriAreaCountyDataLoader
                area = config.area  # 设置区域为指定的亚区
        else:
            # 如果指定了省份，则使用省级县级数据加载器
            StateDataLoader = ProvinceCountyDataLoader
            area = config.province  # 设置区域为指定的省份

        self.IDs, \
        self.NH3_Crop_colunms, \
        self.N2O_nitrogen_fertilizer_columns, \
        self.NO3_nitrogen_fertilizer_columns, \
        self.N_runoff_columns, \
        self.Soc_columns, \
        self.NH3_Fecal_management_columns, \
        self.N2O_Fecal_management_columns, \
        self.NH3_manure_application_columns, \
        self.N2O_manure_application_columns, \
        self.NO3_manure_application_columns, \
        self.Straw_columns, \
        self.CH4_intestine_columns, \
        self.CH4_Fecal_management_columns, \
        self.NH3_manure_application_tensor, \
        self.N2O_manure_application_tensor, \
        self.NO3_manure_application_tensor, \
        self.Rice_CH4_Gg, \
        self.nitrogen_deposition_N2O_tensor, \
        self.straw_returning_N2O_tensor, \
        self.NH3_Crop_tensor, \
        self.N2O_nitrogen_fertilizer_tensor, \
        self.NO3_nitrogen_fertilizer_tensor, \
        self.N_runoff_tensor, \
        self.Soc_tensor_origin, \
        self.NH3_Fecal_management_tensor, \
        self.N2O_Fecal_management_tensor, \
        self.Straw_tensor, \
        self.CH4_intestine_tensor, \
        self.CH4_Fecal_management_tensor, \
        self.Total_CH4_tensor_origin, \
        self.Total_N2O_tensor_origin, \
        self.threshold_NO3_PB, \
        self.threshold_N_runoff_PB, \
        self.threshold_NH3_PB, \
        self.threshold_CH4, \
        self.threshold_N2O, \
        self.county_scale_original, \
        self.county_scale, \
        = StateDataLoader(self.county_df_path, self.IDs_df, config.soc_df_path, config.livestock_scale, config.crop_scale, area=area)     


        if self.save_path is not None:
            counties = self.IDs['Counties'].to_list()
            self.livestock_yield_origin = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'animal_production_yield.csv'), encoding='gbk')
            self.crop_yield_origin = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'crop_production_yield.csv'), encoding='gbk')
            self.livestock_yield_origin = self.livestock_yield_origin[self.livestock_yield_origin['County'].isin(counties)]
            self.crop_yield_origin = self.crop_yield_origin[self.crop_yield_origin['County'].isin(counties)]
            # sort like self.IDs
            self.livestock_yield_origin = self.livestock_yield_origin.set_index('County').loc[counties]
            self.crop_yield_origin = self.crop_yield_origin.set_index('County').loc[counties]

        # 氮肥NO3最小减排量
        self.Total_NO3_origin = torch.sum(self.NO3_nitrogen_fertilizer_tensor, axis=1) + \
                            torch.sum(self.NO3_manure_application_tensor, axis=1)
        self.gap_NO3_origin = self.Total_NO3_origin - self.threshold_NO3_PB

        # N流失最小减排量
        self.Total_N_runoff_origin = torch.sum(self.N_runoff_tensor, axis=1)
        self.gap_N_runoff_origin = self.Total_N_runoff_origin - self.threshold_N_runoff_PB

        # 氨氮最小减排量
        self.Total_NH3_origin = torch.sum(self.NH3_Crop_tensor, axis=1) + \
                                torch.sum(self.NH3_Fecal_management_tensor, axis=1) + \
                                torch.sum(self.NH3_manure_application_tensor, axis=1)
        self.gap_NH3_origin = self.Total_NH3_origin - self.threshold_NH3_PB

        # 初始化线性规划相关变量
        self.CH4_target = None
        self.N2O_target = None
        self.lp_tech_ids = None

        # 计算甲烷和氧化亚氮的原始排放量（目标值将在加载线性规划结果后设置）
        coef_CH4 = GAS_COEFFICIENTS['CH4']
        self.Total_CH4_origin = coef_CH4['default'] * torch.sum(self.CH4_intestine_tensor, axis=1) + \
                                coef_CH4['default'] * torch.sum(self.CH4_Fecal_management_tensor, axis=1) + \
                                coef_CH4['default'] * torch.sum(self.Straw_tensor[:, :3], axis=1) + \
                                coef_CH4['Rice CH4 Gg'] * torch.sum(self.Rice_CH4_Gg, axis=1)

        # 氧化亚氮原始排放量
        coef_N2O = GAS_COEFFICIENTS['N2O']
        self.Total_N2O_origin = 1 * torch.sum(self.nitrogen_deposition_N2O_tensor, axis=1) + \
                                coef_N2O['粪污管理N2O'] * torch.sum(self.N2O_Fecal_management_tensor, axis=1) + \
                                coef_N2O['粪污管理NH3'] * torch.sum(self.NH3_Fecal_management_tensor, axis=1) + \
                                coef_N2O['氮肥N2O'] * torch.sum(self.N2O_nitrogen_fertilizer_tensor, axis=1) + \
                                coef_N2O['种植业NH3挥发'] * torch.sum(self.NH3_Crop_tensor, axis=1) + \
                                coef_N2O['氮肥NO3'] * torch.sum(self.NO3_nitrogen_fertilizer_tensor, axis=1) + \
                                coef_N2O['N runoff'] * torch.sum(self.N_runoff_tensor, axis=1) + \
                                coef_N2O['粪污施用NH3'] * torch.sum(self.NH3_manure_application_tensor, axis=1) + \
                                coef_N2O['粪污施用NO3'] * torch.sum(self.NO3_manure_application_tensor, axis=1) + \
                                coef_N2O['粪污施用N2O'] * torch.sum(self.N2O_manure_application_tensor, axis=1) + \
                                coef_N2O['秸秆焚烧'] * torch.sum(self.Straw_tensor[:, 3:], axis=1) + \
                                1 * torch.sum(self.straw_returning_N2O_tensor, axis=1)
       
        # 保留所有县的数据，不进行过滤
        self.Total_SOC = self.Soc_tensor_origin.clone().sum(axis=-1).reshape(-1, 1)

        # 计算初始产量
        if self.save_path is not None:
            # 如果有保存路径，使用详细的yield数据
            counties = self.IDs['Counties'].to_list()
            self.livestock_yield_origin = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'animal_production_yield.csv'), encoding='gbk')
            self.crop_yield_origin = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'crop_production_yield.csv'), encoding='gbk')
            self.livestock_yield_origin = self.livestock_yield_origin[self.livestock_yield_origin['County'].isin(counties)]
            self.crop_yield_origin = self.crop_yield_origin[self.crop_yield_origin['County'].isin(counties)]
            # sort like self.IDs
            self.livestock_yield_origin = self.livestock_yield_origin.set_index('County').loc[counties]
            self.crop_yield_origin = self.crop_yield_origin.set_index('County').loc[counties]

            # 计算初始产量数据（合并所有产量列）
            livestock_yield_data = self.livestock_yield_origin.iloc[:, 3:].values
            crop_yield_data = self.crop_yield_origin.iloc[:, 3:].values
            # 计算各列的总和
            livestock_total = livestock_yield_data.sum(axis=1)
            crop_total = crop_yield_data.sum(axis=1)
            self.Total_Yield_origin = torch.tensor(livestock_total + crop_total, dtype=torch.float32).reshape(-1, 1)
        else:
            # 保证在未提供 save_path 时也有 Total_Yield_origin（全零），避免 observation_space / state 不一致
            self.Total_Yield_origin = torch.zeros((self.IDs.shape[0], 1), dtype=torch.float32)
   
        if self.save_path is not None:
            # 保留所有县的产量数据
            county_names = self.IDs['Counties'].tolist()
            self.livestock_yield = self.livestock_yield_origin.loc[self.livestock_yield_origin.index.isin(county_names)]
            self.crop_yield = self.crop_yield_origin.loc[self.crop_yield_origin.index.isin(county_names)]
            # 重新排序，确保与self.IDs顺序一致
            self.livestock_yield = self.livestock_yield.reindex(county_names)
            self.crop_yield = self.crop_yield.reindex(county_names)

        self.logs = {
            'county': self.IDs['Counties'].to_list(),
            'CH4': self.Total_CH4_tensor_origin.detach().squeeze().numpy(),
            'N2O': self.Total_N2O_tensor_origin.detach().squeeze().numpy(),
            'NO3': (torch.sum(self.NO3_nitrogen_fertilizer_tensor, axis=1) + \
                    torch.sum(self.NO3_manure_application_tensor, axis=1)).numpy(),
            'N_runoff': torch.sum(self.N_runoff_tensor, axis=1).numpy(),
            'NH3': (torch.sum(self.NH3_Crop_tensor, axis=1) + \
                    torch.sum(self.NH3_Fecal_management_tensor, axis=1) + \
                    torch.sum(self.NH3_manure_application_tensor, axis=1)).numpy(),
            'SOC': self.Soc_tensor_origin.clone().sum(axis=-1).numpy(),
            'Yield': self.Total_Yield_origin.squeeze().numpy()
        }
        if not os.path.exists('GasState_origin.xlsx'):
            pd.DataFrame(self.logs).to_excel('GasState_origin.xlsx', index=False)

        # load tech data
        self.Feeding, \
        self.Housing, \
        self.slurry_storage, \
        self.soild_storage, \
        self.composting, \
        self.additives_application, \
        self.soild_application, \
        self.slurry_application, \
        self.crop, \
        self.tech_set = TechDataLoader(config.livestock_tech_path, config.crop_tech_path)
        self.industry_mapping = {} # 行业映射
        
        self.tech_mapping = {
            'Feeding': self.Feeding,
            'Housing': self.Housing,
            'slurry_storage': self.slurry_storage,
            'soild_storage': self.soild_storage,
            'composting': self.composting,
            'additives_application': self.additives_application,
            'soild_application': self.soild_application,
            'slurry_application': self.slurry_application,
            'crop': self.crop
        }

        self.tech_shape = [i.shape[0] for i in self.tech_mapping.values()]
        self.tech_shape = np.cumsum(self.tech_shape)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.numCounty = self.IDs.shape[0]
        self.numTech = self.tech_set.shape[0]
        self.current_step = 0

        # 调试日志设置（逐步保存每步的县级状态与子行业影响）
        self.debug_step_logging = config.debug_step_logging
        if self.debug_step_logging:
            base_dir = config.debug_log_dir or self.save_path or os.path.join(os.path.dirname(__file__), '..', 'results')
            self.debug_dir = os.path.join(base_dir, 'debug_step_logs')
            os.makedirs(self.debug_dir, exist_ok=True)
            self.debug_summary_csv = os.path.join(self.debug_dir, 'per_step_summary.csv')
            self.debug_impacts_csv = os.path.join(self.debug_dir, 'per_step_subindustry_impacts.csv')
            # 初始化CSV头
            if not os.path.exists(self.debug_summary_csv):
                pd.DataFrame(columns=[
                    'step','countyID','county','techID','tech_name',
                    'before_NH3','before_NO3','before_N_runoff','before_CH4','before_N2O',
                    'after_NH3','after_NO3','after_N_runoff','after_CH4','after_N2O',
                    'before_gap_NH3','before_gap_NO3','before_gap_N_runoff','before_gap_CH4','before_gap_N2O',
                    'after_gap_NH3','after_gap_NO3','after_gap_N_runoff','after_gap_CH4','after_gap_N2O',
                    'Reward_cost','Reward_sim','Reward_gas','Reward_tech_count','terminated'
                ]).to_csv(self.debug_summary_csv, index=False)
            if not os.path.exists(self.debug_impacts_csv):
                pd.DataFrame(columns=[
                    'step','countyID','county','techID','tech_name',
                    'industry','sub_name','col_index','before_value','delta','after_value',
                    'gas_type','weighted_coeff','gas_delta_contribution'
                ]).to_csv(self.debug_impacts_csv, index=False)

        # 加载线性规划结果和动作掩码设置
        self.linear_result_path = config.linear_result_path
        self.only_lp_phase = config.only_lp_phase
        self.action_mask = None
        
        # 统一加载线性规划结果（包括目标值和技术约束）
        if self.linear_result_path and os.path.exists(self.linear_result_path):
            self._load_linear_programming_results()
            # 初始化时使用步数0
            self.action_mask = self._get_action_mask(training_step=0)
        else:
            pass
                
        # 动作空间
        self.action_space = spaces.Discrete(self.numCounty * self.numTech)
       
        self.observation_space = spaces.Dict({
            # 县产业数据
            'CH4': spaces.Box(low=0, high=np.inf, shape=(self.numCounty, 1), dtype=np.float32),
            'N2O': spaces.Box(low=0, high=np.inf, shape=(self.numCounty, 1), dtype=np.float32),
            "NO3": spaces.Box(low=0, high=np.inf, shape=(self.numCounty, 1), dtype=np.float32),
            "N_runoff": spaces.Box(low=0, high=np.inf, shape=(self.numCounty, 1), dtype=np.float32),
            "NH3": spaces.Box(low=0, high=np.inf, shape=(self.numCounty, 1), dtype=np.float32),
            'SOC': spaces.Box(low=0, high=np.inf, shape=(self.numCounty, 1), dtype=np.float32),
            # 当前采用技术集合, shape是num_county * num_tech, 元素是0或1
            "Tech_selected" : spaces.Box(low=0, high=1, shape=(self.numCounty, self.numTech), dtype=np.int32),
            # 'Yield': spaces.Box(low=0, high=np.inf, shape=(self.numCounty, 1), dtype=np.float32)
        })
        
        self.class_mapping = {
            '种植业NH3挥发' : self.NH3_Crop_colunms,
            '氮肥N2O' : self.N2O_nitrogen_fertilizer_columns,
            '氮肥NO3' : self.NO3_nitrogen_fertilizer_columns,
            'N runoff' : self.N_runoff_columns,
            '秸秆焚烧' : self.Straw_columns,
            '粪污管理N2O': self.N2O_Fecal_management_columns,
            '粪污管理NH3': self.NH3_Fecal_management_columns,
            '粪污管理CH4': self.CH4_Fecal_management_columns,
            '肠道CH4': self.CH4_intestine_columns,
            '粪污施用N2O': self.N2O_manure_application_columns,
            '粪污施用NH3': self.NH3_manure_application_columns,
            '粪污施用NO3': self.NO3_manure_application_columns,
            'Rice CH4 Gg': pd.Series(['Rice CH4 Gg']),
            'Soc': self.Soc_columns
        }
        
        # 当save_path不为空时，添加产量相关的映射
        if self.save_path is not None:
            # 添加畜牧业产量相关的映射
            livestock_yield_columns = self.livestock_yield_origin.columns[3:]  # 排除County列
            crop_yield_columns = self.crop_yield_origin.columns[3:]  # 排除County列
            # livestock_yield_columns 转换为Series
            livestock_yield_columns = pd.Series(livestock_yield_columns)
            crop_yield_columns = pd.Series(crop_yield_columns)
            self.class_mapping.update({
                '畜牧业产量': livestock_yield_columns,
                '种植业产量': crop_yield_columns
            })
        
        # Precompute neighbor indices
        self.city_groups = self.IDs.groupby('Cities').apply(lambda x: x.index.values).to_dict()        
        self.area_groups = self.IDs.groupby('所属农业亚区').apply(lambda x: x.index.values).to_dict()
        
        # 关键：用三种气体的正向 gap 重新定义哪些县"需要技术添加"
        self.counties_need_tech = (
            (self.gap_NO3_origin > 0) |
            (self.gap_N_runoff_origin > 0) |
            (self.gap_NH3_origin > 0) 
        ).numpy()

        # 读取超标县域数据
        exceed_counties_df = pd.read_excel('data/超标县域.xlsx')
        exceed_counties_mask = exceed_counties_df['超标县'] == 1
        # 获取超标县域中的县名列表
        exceed_county_names = exceed_counties_df[exceed_counties_mask]['所属地州'].tolist()
        # 创建超标县域的布尔掩码（基于县名匹配）
        exceed_counties_bool = self.IDs['Counties'].isin(exceed_county_names)
        # 需要技术的县：三种气体超标 或者 在超标县域Excel中标记为1
        self.counties_need_tech = (self.counties_need_tech | exceed_counties_bool)

        self.reset()
        self.current_per_county_similarity_scores = torch.zeros(self.numCounty, device=self.device)

    def _load_linear_programming_results(self):
        """
        统一加载线性规划结果，包括目标值和技术约束
        """
        # 读取线性规划结果
        self.lp_results = pd.read_excel(self.linear_result_path)
        
        # 获取当前环境中的县名列表
        county_names = self.IDs['Counties'].tolist()

        # 筛选出当前亚区的县
        lp_results_filtered = self.lp_results[self.lp_results['county'].isin(county_names)]
        # 重排线性规划的列表
        lp_results_filtered = lp_results_filtered.set_index('county')
        lp_results_filtered = lp_results_filtered.reindex(county_names)
        lp_results_filtered = lp_results_filtered.reset_index()  # 如需恢复默认索引
        # 获取线性规划中的县名列表
        lp_counties = lp_results_filtered['county'].tolist()
        # 直接构建县名->tech id 列表的字典（兼容缺失值）
        self.lp_tech_count = {i: len(ast.literal_eval(row['techs'])) if 'techs' in row and pd.notna(row['techs']) else [] 
                              for i, (_, row) in enumerate(lp_results_filtered.iterrows())}
        self.tech_count_relax = 2.0  # 允许的技术数量放宽比例

        # lp_counties 是线性规划结果中的县名列表，self.IDs 是环境中的县名列表
        for i, (lp_name, env_name) in enumerate(zip(lp_counties, county_names)):
            if lp_name != env_name:
                print(f"县顺序不一致：第{i}个, LP结果={lp_name}, 环境={env_name}")
                break
        else:
            print("线性规划县顺序与环境县顺序完全一致")
        
        # 提取目标值
        self.CH4_target = torch.tensor(lp_results_filtered['CH4_target'].values, dtype=torch.float32)
        self.N2O_target = torch.tensor(lp_results_filtered['N2O_target'].values, dtype=torch.float32)
        self.NO3_target = torch.tensor(lp_results_filtered['NO3_target'].values, dtype=torch.float32)
        self.N_runoff_target = torch.tensor(lp_results_filtered['N_runoff_target'].values, dtype=torch.float32)
        self.NH3_target = torch.tensor(lp_results_filtered['NH3_target'].values, dtype=torch.float32)
        
        # 计算CH4和N2O的减排缺口（在目标值设置之后）
        self.gap_CH4_origin = self.Total_CH4_origin - self.CH4_target
        self.gap_N2O_origin = self.Total_N2O_origin - self.N2O_target
        # 初始化每个县的技术ID集合
        self.lp_tech_ids = {}
        
        # 遍历每个县
        for county_idx, county_name in enumerate(county_names):
            allowed_tech_ids = []
            
            # 在线性规划结果中查找该县
            if county_name in self.lp_results['county'].values:
                tech_id = self.lp_results[self.lp_results['county']==county_name]['techs']
                allowed_tech_ids = ast.literal_eval(tech_id.values[0])
            
            self.lp_tech_ids[county_idx] = allowed_tech_ids

        # 失败县名列表
        self.failed_counties = lp_results_filtered[lp_results_filtered['relaxed_attempted'] == 1]['county']
        self.failed_counties = self.failed_counties.astype(str).str.strip().tolist()

        print(f"成功加载线性规划结果，共{len(self.lp_tech_ids)}个县的技术约束")
        print(f"加载了 {len(self.CH4_target)} 个县的 CH4 目标值")
        print(f"加载了 {len(self.N2O_target)} 个县的 N2O 目标值")
        print(f"线性规划失败县个数: {len(self.failed_counties)}")

    def _get_action_mask(self, training_step=None):
        """
        返回一个布尔型一维数组，长度为 numCounty * numTech
        True 表示该动作被遮蔽（不可选），False 表示可选
        
        Args:
            training_step: 从训练代码传入的当前训练步数，如果为None则使用环境内部步数（用于测试）
        """
        mask = torch.ones((self.numCounty, self.numTech), dtype=torch.bool)
        
        # 使用传入的训练步数，如果没有则使用环境内部步数（主要用于测试）
        current_step = training_step if training_step is not None else self.current_step
        print(f"当前步数: {current_step} (训练步数)" if training_step is not None else f"当前步数: {current_step} (环境步数)")

        # 如果没有启用课程学习，直接返回全False的mask
        if not self.only_lp_phase:
            return torch.zeros((self.numCounty, self.numTech), dtype=torch.bool)
        
        # 计算各阶段的步数阈值
        lp_phase_step = int(self.total_steps * self.lp_phase_ratio)
        phase_1_step = int(self.total_steps * self.phase_1_ratio)
        phase_2_step = int(self.total_steps * self.phase_2_ratio)

        print(f"阶段阈值: LP={lp_phase_step}, Phase1={phase_1_step}, Phase2={phase_2_step}")

        # 线性规划阶段：只在线性规划的动作空间里学
        if self.lp_tech_ids is not None and current_step < lp_phase_step:
            print("🔵 线性规划阶段：只允许线性规划解中的技术")
            mask[:] = True  # 先全部遮蔽
            lp_valid_actions = 0
            
            for county_id in range(self.numCounty):
                if county_id in self.lp_tech_ids:
                    allowed_techs = self.lp_tech_ids[county_id]
                    for tech_id in allowed_techs:
                        if tech_id < self.numTech:
                            mask[county_id, tech_id] = False
                            lp_valid_actions += 1
            
            print(f"线性规划阶段有效动作数: {lp_valid_actions}")
            
        else:
            # 根据当前步数确定阶段
            if current_step < lp_phase_step:
                current_phase = 0
                phase_name = "线性规划阶段"
            elif current_step < phase_1_step:
                current_phase = 1
                phase_name = "阶段1 (技术等级1)"
            elif current_step < phase_2_step:
                current_phase = 2
                phase_name = "阶段2 (技术等级1-2)"
            else:
                current_phase = 3
                phase_name = "阶段3 (所有技术等级)"

            print(f"🟡 {phase_name}：当前阶段 {current_phase}")

            # 根据阶段放开动作空间（累积放开，即等级2时包含等级1和2）
            for county_id in range(self.numCounty):
                for tech_id in range(self.numTech):
                    tech_line = self.tech_set.iloc[tech_id]
                    tech_level = int(tech_line['技术分级'])
                    
                    # 累积放开：阶段1放开等级1，阶段2放开等级1和2，阶段3放开所有
                    if (current_phase >= 1 and tech_level <= 1) or \
                        (current_phase >= 2 and tech_level <= 2) or \
                        (current_phase >= 3):
                        mask[county_id, tech_id] = False

        # 打印 action mask 的统计信息
        mask_sum = mask.sum().item()
        total_actions = self.numCounty * self.numTech
        valid_actions = total_actions - mask_sum
        print(f"📊 动作掩码统计: 总动作数={total_actions}, 有效动作数={valid_actions}, 被遮蔽动作数={mask_sum}")
        print(f"📊 被遮蔽动作比例: {mask_sum / total_actions:.2%}")

        # 如果有效动作太少，给出警告
        if valid_actions < 10:
            print("⚠️  警告：有效动作数量过少，可能影响学习效果！")
            print("建议：检查线性规划结果或调整课程学习参数")

        return mask
    
    def update_action_mask(self, training_step):
        """
        根据训练步数更新并返回action mask
        
        Args:
            training_step: 当前训练步数
            
        Returns:
            action_mask: 布尔张量，True表示被遮蔽的动作
        """
        if self.only_lp_phase:
            self.action_mask = self._get_action_mask(training_step=training_step)
            return self.action_mask
        else:
            # 如果没有启用课程学习，返回全False的mask
            return torch.zeros((self.numCounty, self.numTech), dtype=torch.bool)
        
    def _decode_action(self, action):
        """
        解码动作为县ID和技术ID
        Args:
            action: 整数动作，范围[0, numCounty * numTech]
        Returns:
            tuple: (countyID, techID) - countyID是在所有县中的实际索引
        """
        # 先解码为需要技术添加的县索引和技术ID
        countyID = action // self.numTech
        techID = action % self.numTech
        return countyID, techID
        
    def _get_line(self, techID):
        # 添加缓存机制
        if not hasattr(self, 'line_cache'):
            self.line_cache = {}
            # 预构建映射键列表，避免重复计算
            self.tech_mapping_keys = list(self.tech_mapping.keys())
            
        if techID in self.line_cache:
            return self.line_cache[techID]
            
        # 二分查找 输出techID所属的产业的技术详情
        index = np.searchsorted(self.tech_shape, techID, side='right')
        if index == 0:
            line = self.tech_mapping['Feeding'].iloc[techID]
        else:
            line = self.tech_mapping[self.tech_mapping_keys[index]].iloc[techID - self.tech_shape[index-1]]
            
        # 缓存结果
        self.line_cache[techID] = line
        return line


    def _compute_N2O_delta(self, Delta, industry):
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

    def _compute_CH4_delta(self, Delta, industry):
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
    
    def expand_tech_subindustries(self, tech_deltas, relaxed=False):
        """
        为失败的县扩展玉米技术到其他四种作物
        当技术作用的产业包含任何带"maize"的子产业时，检索实际存在的其他作物子产业并应用相同的delta值
        
        Args:
            tech_deltas: 技术影响的列表 [(industry, delta, subIndustry), ...]
            county_name: 县名（用于调试输出）
            relaxed: 是否为放宽模式
            
        Returns:
            修改后的tech_deltas
        """
        if not relaxed:
            return tech_deltas
        
        target_subindustry = {'maize': ['millet', 'sorghum', 'othercereals', 'cotton'],
                              'vegetable': ['beans', 'potato', 'peanut', 'rapeseed', 'sugarbeet'],
                              'fruit': ['flax', 'sugarcane', 'tobacoo'],
                              'sheep': ['horse', 'donkey', 'rabbit']}
        
        def find_corresponding_subindustry(industry, target_subindustry, key, origin):
            """在实际的class_mapping中查找对应的其他子产业"""
            if industry not in self.class_mapping:
                return []
            
            # 获取该产业的所有子产业列表
            all_subindustries = self.class_mapping[industry].tolist()
            
            # 查找其他子产业的对应子产业
            other_subindustry = []
            
            # 遍历目标子产业列表，尝试将原始子产业名中的key替换为目标子产业名
            for s in target_subindustry:
                # 如果key在原始子产业名中
                if key in sub:
                    # 替换key为目标子产业名，生成新的子产业名
                    new_sub = origin.replace(key, s)
                    # 检查新子产业名是否实际存在于所有子产业列表中
                    if new_sub in all_subindustries:
                        # 如果存在，则加入到结果列表
                        other_subindustry.append(new_sub)

            return other_subindustry
        
        expanded_deltas = []
        for industry, delta, subIndustry in tech_deltas:
            # 保持原有的技术影响
            expanded_deltas.append((industry, delta, subIndustry))
            
            # 检查是否包含任何带目标子产业的子产业
            if isinstance(subIndustry, list):
                for sub in subIndustry:
                    if isinstance(sub, str) and any(s in sub.lower() for s in target_subindustry.keys()):
                        # 查找实际存在的其他子产业
                        for s in target_subindustry.keys():
                            if s in sub.lower():
                                corresponding_subindustry = find_corresponding_subindustry(industry, target_subindustry[s], s, sub)
                                for other_subindustry in corresponding_subindustry:
                                    # 创建新的技术影响条目，只包含单个作物
                                    expanded_deltas.append((industry, delta, [other_subindustry]))

        return expanded_deltas

    def _update_state(self, countyID, techID):
        '''
        output: 更新后的状态 
                CH4减排量
                N2O减排量
                NO3减排量
                N流失减排量
                NH3减排量
        '''        
        # 获取技术详情
        line = self._get_line(techID)
        class_name = 'crop' if 'crop' in line.index[4].lower() else 'livestock'
        line = line[5:]
        # 初始化气体变化量
        gas_deltas = {
            'NH3': 0.0,
            'NO3': 0.0,
            'N_runoff': 0.0,
            'SOC': 0.0,
            'CH4': 0.0,
            'N2O': 0.0 
        }
        # 当指定 save_path 时，也需要追踪作物产量（yield）的变化
        if self.save_path is not None:
            gas_deltas['yield'] = 0.0
            
        # 若启用调试，准备子行业影响记录容器
        impact_records = [] if getattr(self, 'debug_step_logging', False) else None

        # 获取技术影响的环节
        deltas = self._get_delta(line, class_name)
        # 仅对已标记为 failed 的县使用 relaxed 扩展（避免对所有县扩展）
        county_name = self.IDs['Counties'].iloc[countyID]
        relaxed_flag = county_name in self.failed_counties
        # 判断是否需要扩展（如某些县或某些条件下）
        deltas = self.expand_tech_subindustries(deltas, relaxed=relaxed_flag)

        # 获取技术影响的环节
        for industry, delta, subIndustry in deltas:
            # 获取技术影响的环节的当前状态
            countyState = self.stateMapping[industry] # 索引得到该技术影响的环节的当前状态 ('粪污管理NH3', -0.836574818682584, ['NH3_Housing_LayingHen', 'NH3_Housing_OtherPoultry'])
            subindustry_classes = self.class_mapping[industry]

            # 遍历技术影响的环节的子环节 
            for sub in subIndustry:
                # 获取子环节的索引
                col = self._get_or_create_industry_mapping(sub, subindustry_classes)
                # 计算技术影响子环节的气体变化量
                before_value = float(countyState[countyID, col].item())
                Delta_tensor = countyState[countyID, col] * delta
                Delta = float(Delta_tensor.item())
                countyState[countyID, col] += Delta # 更新子环节的当前状态
                after_value = float(countyState[countyID, col].item())
                if Delta != 0:
                # 遍历气体变化量
                    for gas_type in gas_deltas:
                        # 如果子环节包含气体变化量
                        if gas_type in sub:
                            # 更新气体变化量
                            if gas_type in ['NH3', 'NO3', 'N_runoff']:
                                gas_deltas[gas_type] += Delta # 更新气体变化量  
                            if industry in ['粪污管理NH3', '粪污管理N2O', '氮肥N2O', '种植业NH3挥发', '氮肥NO3', 'N runoff', '粪污施用NH3', '粪污施用NO3', '粪污施用N2O']:
                                gas_deltas['N2O'] += self._compute_N2O_delta(Delta, industry)
                            elif industry in ['粪污管理CH4', '肠道CH4', 'Rice CH4 Gg']:
                                gas_deltas['CH4'] += self._compute_CH4_delta(Delta, industry)
                            elif industry in ['秸秆焚烧']:
                                if 'N2O' in sub:
                                    gas_deltas['N2O'] += self._compute_N2O_delta(Delta, industry)
                                elif 'CH4' in sub:
                                    gas_deltas['CH4'] += self._compute_CH4_delta(Delta, industry)
                            break

                    # 调试：记录子行业影响
                    if impact_records is not None:
                        # 判定主要气体类别与加权系数、贡献
                        sub_name_str = sub.strip() if isinstance(sub, str) else str(sub)
                        if 'NH3' in sub_name_str:
                            gas_flag = 'NH3'; coeff = 1.0; contrib = Delta
                        elif 'NO3' in sub_name_str:
                            gas_flag = 'NO3'; coeff = 1.0; contrib = Delta
                        elif ('N_runoff' in sub_name_str) or ('runoff' in sub_name_str.lower()):
                            gas_flag = 'N_runoff'; coeff = 1.0; contrib = Delta
                        elif ('CH4' in sub_name_str) or (industry in ['粪污管理CH4','肠道CH4','Rice CH4 Gg']):
                            gas_flag = 'CH4'
                            coeff = GAS_COEFFICIENTS['CH4'].get(industry, GAS_COEFFICIENTS['CH4']['default'])
                            contrib = float(self._compute_CH4_delta(Delta, industry))
                        else:
                            gas_flag = 'N2O'
                            coeff = GAS_COEFFICIENTS['N2O'].get(industry, GAS_COEFFICIENTS['N2O']['default'])
                            contrib = float(self._compute_N2O_delta(Delta, industry))
                        impact_records.append({
                            'industry': industry,
                            'sub_name': sub_name_str,
                            'col_index': int(col),
                            'before_value': before_value,
                            'delta': Delta,
                            'after_value': after_value,
                            'gas_type': gas_flag,
                            'weighted_coeff': float(coeff),
                            'gas_delta_contribution': float(contrib),
                        })

                    # 更新土壤有机碳
                    if 'total_organic_carbon_kg' in sub:
                        gas_deltas['SOC'] += Delta
                    
                    if 'yield' in sub:
                        gas_deltas['yield'] += Delta

        # 更新气体差距值
        self._update_gas_gaps(gas_deltas, countyID, techID)
        if impact_records is not None:
            self._last_impact_records = impact_records
        
        return gas_deltas

    def _update_gas_gaps(self, gas_deltas, countyID=None, techID=None):
        """更新气体差距值
        
        Args:
            countyID: 县级行政区ID
            gas_deltas: 气体变化量字典
        """
        for gas in gas_deltas.keys():
            # 更新状态
            if gas in self.state:
                self.state[gas][countyID] += torch.tensor(gas_deltas[gas]).item()

            # 更新差距值（对于气体指标）
            if gas == 'NH3':
                self.gap_NH3[countyID] += torch.tensor(gas_deltas[gas]).item()
            elif gas == 'NO3':
                self.gap_NO3[countyID] += torch.tensor(gas_deltas[gas]).item()
            elif gas == 'N_runoff':
                self.gap_N_runoff[countyID] += torch.tensor(gas_deltas[gas])
            elif gas == 'CH4':
                self.gap_CH4[countyID] += torch.tensor(gas_deltas[gas])
            elif gas == 'N2O':
                self.gap_N2O[countyID] += torch.tensor(gas_deltas[gas])

        # 更新技术选择  
        self.state['Tech_selected'][countyID, techID] = 1
        
    def _calculate_per_county_similarity_scores(self, county_idx, tech_idx, tech_selection_matrix) -> torch.Tensor:
        """
        Calculate the per-county weighted similarity score for a given tech selection matrix.
        Returns a tensor of shape (numCounty,) with scores for each county.
        """
        if self.numCounty == 0:
            return torch.empty(0, device=self.device)

        tech_selection_matrix_float = tech_selection_matrix.float() # Ensure float for matmul

        county_vec = tech_selection_matrix_float[county_idx, :] 

        # City similarity
        city_name = self.IDs.iloc[county_idx]['Cities']
        city_neighbor_indices = [idx for idx in self.city_groups.get(city_name, []) if idx != county_idx and idx < self.numCounty]
        current_county_city_sim = 0.0
        if city_neighbor_indices:
            neighbor_tech_vectors = tech_selection_matrix_float[city_neighbor_indices, :]
            if neighbor_tech_vectors.shape[0] != 0:
                dot_products = torch.matmul(neighbor_tech_vectors, county_vec)
                # 如果dot_products不为空
                current_county_city_sim = torch.mean(dot_products).item() # .item() to get float
        
        # area similarity
        area_name = self.IDs.iloc[county_idx]['所属农业亚区']
        area_neighbor_indices = [idx for idx in self.area_groups.get(area_name, []) if idx != county_idx and idx < self.numCounty]
        current_county_area_sim = 0.0
        if area_neighbor_indices:
            neighbor_tech_vectors_area = tech_selection_matrix_float[area_neighbor_indices, :]
            if neighbor_tech_vectors_area.shape[0] != 0:
                dot_products_area = torch.matmul(neighbor_tech_vectors_area, county_vec)
                # 如果dot_products_area不为空
                current_county_area_sim = torch.mean(dot_products_area).item() # .item() to get float
        
        per_county_weighted_sims = (self.reward_priority[2] * current_county_city_sim + self.reward_priority[3] * current_county_area_sim)
        
        return per_county_weighted_sims, current_county_city_sim, current_county_area_sim
       

    def _evaluate_sim(self, countyID, techID): 
        '''
        Retrieves per-county similarity scores for state before and after current actions.
        Uses cached scores for 'before' state. Calculates 'after' state scores from current self.state['Tech_selected'].
        Returns:
            scores_after_tensor (Tensor): Per-county similarity scores for the current state.
            scores_before_tensor (Tensor): Per-county similarity scores for the previous state.
        '''
        # 1. Get per-county similarity scores of the state before current actions were applied (cached)
        scores_before = self.current_per_county_similarity_scores[countyID]

        # 2. Calculate per-county similarity scores of the current state 
        # (after actions have been applied by _update_state_batch)
        scores_after, _, _ = self._calculate_per_county_similarity_scores(countyID, techID, self.state['Tech_selected'])

        return scores_after, scores_before

    def _evaluate_techSum(self):
        '''
        计算技术组合总数
        '''
        return self.state['Tech_selected'].sum()
    
    def _convert_percentage_to_float(self, percentage_str):
        """
        将百分比字符串转换为float
        
        Args:
            percentage_str (str): 百分比字符串，如 "40.0%"
        
        Returns:
            float: 转换后的浮点数，如 0.4
        """
        # 方法1：使用字符串处理

        if not isinstance(percentage_str, str) and np.isnan(percentage_str) or isinstance(percentage_str, np.float64) or isinstance(percentage_str, float):
            return percentage_str
        
        
        if percentage_str.endswith('%'):
            # 去掉百分号并转换为float，然后除以100
            return float(percentage_str[:-1]) / 100
        else:
            # 如果不是百分比格式，直接转换为float
            return float(percentage_str)
        
    def _get_delta(self, line, class_name):
        '''
        获取技术影响的行业和子行业
        '''
        # 如果提供 save_path，则说明需要将产量(yield) 也纳入计算，
        # 否则保持原先逻辑（去掉最后一组产量指标）。
        if self.save_path:
            num_indicators = line.shape[0] // 2
        else:
            num_indicators = line.shape[0] // 2 - 1
        deltas = []
        
        # 预处理行业索引
        industry_indices = [(2*i, 2*i+1) for i in range(num_indicators)]
        
        for delta_idx, industry_idx in industry_indices:
            delta = self._convert_percentage_to_float(line.values[delta_idx])
            if delta != 0 and not np.isnan(delta):
                # 如果delta_idx包含yield，则industry为yield
                if class_name == 'crop' and 'yield' in line.index[delta_idx].lower():
                    industry = '种植业产量'
                elif class_name == 'livestock' and 'yield' in line.index[delta_idx].lower():
                    industry = '畜牧业产量'
                else:
                    industry = line.index[industry_idx].split(' ', 1)[1].replace('.1', '')
                subIndustry = line.values[industry_idx]
                if isinstance(subIndustry, str):
                    subIndustry = subIndustry.split('、')
                # 去除空字符串
                if isinstance(subIndustry, list):
                    subIndustry = [s for s in subIndustry if s.strip() != '']
                if np.isnan(delta):
                    delta = 0
                    raise ValueError('delta is nan')
                deltas.append((industry, delta, subIndustry))
                
        return deltas
    
    def _get_or_create_industry_mapping(self, sub, subindustry_classes):
        """获取或创建行业映射
        
        Args:
            sub: 子行业名称
            subindustry_classes: 行业分类
            
        Returns:
            int: 行业索引
        """
        sub = sub.strip()
        col = self.industry_mapping.get(sub)
        if col is None:
            # 预构建子行业类别列表（只在第一次执行）
            if not hasattr(self, '_subindustry_lists'):
                self._subindustry_lists = {}
            
            # 为每个子行业分类缓存一个列表版本，避免重复转换
            subindustry_id = id(subindustry_classes)
            if subindustry_id not in self._subindustry_lists:
                self._subindustry_lists[subindustry_id] = subindustry_classes.tolist()
            
            subindustry_list = self._subindustry_lists[subindustry_id]
            
            # 使用process.extractOne代替extract，直接获取最佳匹配
            match = process.extractOne(sub, subindustry_list)
            if match:
                if match[1] <= 0.9:
                    print(f"No match found for {sub}")
                matched_item = match[0]
                self.industry_mapping[sub] = subindustry_classes[subindustry_classes == matched_item].index[0]
                col = self.industry_mapping[sub]
            else:
                raise ValueError(f"No match found for {sub}")
        return col

    def _evaluate_gas_lp_phase(self, delta_total_CH4, delta_total_N2O, delta_total_NO3, delta_total_N_runoff, delta_total_NH3):
        """
        线性规划阶段的简化奖励函数，专注于基础减排目标
        使用更简单、更直接的奖励信号
        
        Args:
            delta_total_CH4: 本次动作导致的CH4变化量
            delta_total_N2O: 本次动作导致的N2O变化量
            delta_total_NO3: 本次动作导致的NO3变化量
            delta_total_N_runoff: 本次动作导致的N_runoff变化量
            delta_total_NH3: 本次动作导致的NH3变化量
            
        Returns:
            float: 简化的减排奖励分数
        """
        # 简化的奖励计算：只关注减排效果，不关注超额完成
        rewards = []
        
        # CH4减排奖励（简化版）
        if delta_total_CH4 < 0:  # 减排
            ch4_reward = abs(delta_total_CH4) * 2.0  # 正奖励
        else:  # 增排
            ch4_reward = delta_total_CH4 * 1.0  # 负奖励
        rewards.append(ch4_reward)
        
        # N2O减排奖励（简化版）
        if delta_total_N2O < 0:  # 减排
            n2o_reward = abs(delta_total_N2O) * 2.0  # 正奖励
        else:  # 增排
            n2o_reward = delta_total_N2O * 1.0  # 负奖励
        rewards.append(n2o_reward)
        
        # 其他气体的简化奖励
        for delta in [delta_total_NO3, delta_total_N_runoff, delta_total_NH3]:
            if delta < 0:  # 减排
                rewards.append(abs(delta) * 1.5)
            else:  # 增排
                rewards.append(delta * 0.5)
        
        # 返回平均奖励
        return sum(rewards) / len(rewards) * self.reward_priority[0]

    def _evaluate_gap_progress_reward(self, countyID, delta_total_CH4, delta_total_N2O, delta_total_NO3, delta_total_N_runoff, delta_total_NH3):
        """
        稠密的约束推进奖励：只奖励"把正的未达标差距往下推"的部分。
        用归一化后（按初始gap最大值）的正差距减少量作为奖励信号。
        """
        # 当前新gap
        new_NO3 = float(self.gap_NO3[countyID])
        new_N_runoff = float(self.gap_N_runoff[countyID])
        new_NH3 = float(self.gap_NH3[countyID])
        new_CH4 = float(self.gap_CH4[countyID])
        new_N2O = float(self.gap_N2O[countyID])

        # 旧gap = 新gap - 本步变化量
        old_NO3 = new_NO3 - float(delta_total_NO3)
        old_N_runoff = new_N_runoff - float(delta_total_N_runoff)
        old_NH3 = new_NH3 - float(delta_total_NH3)
        old_CH4 = new_CH4 - float(delta_total_CH4)
        old_N2O = new_N2O - float(delta_total_N2O)

        # 归一化分母，避免除零
        eps = 1e-8
        d_NO3 = float(self.gap_NO3_origin.max()) if torch.is_tensor(self.gap_NO3_origin) else float(np.max(self.gap_NO3_origin))
        d_N_runoff = float(self.gap_N_runoff_origin.max()) if torch.is_tensor(self.gap_N_runoff_origin) else float(np.max(self.gap_N_runoff_origin))
        d_NH3 = float(self.gap_NH3_origin.max()) if torch.is_tensor(self.gap_NH3_origin) else float(np.max(self.gap_NH3_origin))
        d_CH4 = float(self.gap_CH4_origin.max()) if torch.is_tensor(self.gap_CH4_origin) else float(np.max(self.gap_CH4_origin))
        d_N2O = float(self.gap_N2O_origin.max()) if torch.is_tensor(self.gap_N2O_origin) else float(np.max(self.gap_N2O_origin))

        def pos(x):
            return x if x > 0 else 0.0

        progress = 0.0
        progress += (pos(old_NO3) - pos(new_NO3)) / max(d_NO3, eps)
        progress += (pos(old_N_runoff) - pos(new_N_runoff)) / max(d_N_runoff, eps)
        progress += (pos(old_NH3) - pos(new_NH3)) / max(d_NH3, eps)
        progress += (pos(old_CH4) - pos(new_CH4)) / max(d_CH4, eps)
        progress += (pos(old_N2O) - pos(new_N2O)) / max(d_N2O, eps)

        # 放大系数可按需调整；这里与整体权重相关联
        return self.reward_priority[0] * progress

    def _evaluate_tech_count(self):
        """
        对每个县, 计算RL选择的技术数量与线性规划采用的技术数量的比值, 平均后作为奖励。
        比值越接近1奖励越高, 超过1惩罚。
        """
        tech_selected = self.state['Tech_selected']
        county_tech_counts = (tech_selected[:, :self.numTech]).sum(axis=1).cpu().numpy()
        rewards = []
        for county_idx in range(self.numCounty):
            rl_count = county_tech_counts[county_idx]
            lp_count = self.lp_tech_count.get(county_idx, 1)  # 防止除0
            if lp_count == 0:
                ratio = 1.0 if rl_count == 0 else 0.0  # LP没选技术且RL也没选，视为完美
            else:
                ratio = rl_count / lp_count
            reward = 1 - ratio
            rewards.append(reward)
        return float(np.mean(rewards))

    def _evaluate_cost(self, countyID, techID):
        """
        计算单个县-技术对的成本
        
        Args:
            countyID: 县ID
            techID: 技术ID
            
        Returns:
            tuple: (成本奖励分数, 正向成本)
        """
        line = self.tech_set.iloc[techID]
        techName = line['Mitigation strategy']
        
        # 预先构建索引（只需执行一次）
        if not hasattr(self, 'tech_strategy_index'):
            self.tech_strategy_index = self.tech_set.groupby('Mitigation strategy').groups
            
        sameTechIndex = list(self.tech_strategy_index.get(techName, []))
        if techID in sameTechIndex:
            sameTechIndex.remove(techID)
            
        # 获取技术对应的产业类型
        tech_industry = line['Livestock species'] if line['class'] != 'crop' else line['Crop species']
        
        # 根据产业类型获取对应的规模数据
        if line['class'] == 'crop':
            if tech_industry == 'friut':
                industry_scale = self.county_scale.iloc[countyID]['fruittree_sown_area']
            else:
                industry_scale = self.county_scale.iloc[countyID]['{}_sown_area'.format(tech_industry)]
        else:
            tech_industry = tech_industry.lower()
            if tech_industry in self.county_scale.columns:
                industry_scale = self.county_scale.iloc[countyID][tech_industry]
            else:
                industry_scale = self.county_scale.iloc[countyID][tech_industry.replace(' ', '')]
                
        # 计算成本：单位成本 * 技术困难度 * 产业规模
        unit_cost = line['标准化经济成本']
        tech_difficulty = 1 + (-1 + int(line['技术分级'])) * 0.5
        county_action_positive_cost = unit_cost * tech_difficulty * industry_scale
        
        cost = -county_action_positive_cost  # 从奖励分数中减去正成本
        
        # 如果存在相同技术，则成本奖励
        tech_selected = self.state['Tech_selected']
        if sameTechIndex and (tech_selected[countyID, sameTechIndex] == 1).any():
            cost += self.reward_priority[1] * county_action_positive_cost
            
        return cost, county_action_positive_cost

    def _save_tech_selected_summary(self):
        """
        保存技术选择情况的汇总表
        - 第一行是县名
        - 每列显示选中的技术名称
        - 最后一行显示每个县选中技术的总经济成本
        """
        tech_selected_matrix = self.state['Tech_selected'].detach().numpy()
        
        # 创建结果数据结构
        result_data = []
        county_names = self.IDs['Counties'].tolist()
        
        # 为每个县创建一列数据
        for county_idx, county_name in enumerate(county_names):
            column_data = [county_name]  # 第一行是县名
            total_cost = 0.0
            
            # 找到该县选择的所有技术
            selected_techs = []
            for tech_idx in range(self.numTech):
                if tech_selected_matrix[county_idx, tech_idx] == 1:
                    # 获取技术名称
                    line = self.tech_set.iloc[tech_idx]
                    if line['class'] == 'crop':
                        tech_name = f"{line['Mitigation strategy']}_{line['Crop species']}"
                    else:
                        tech_name = f"{line['Mitigation strategy']}_{line['Livestock species']}"
                    selected_techs.append(tech_name)
                    
                    # 计算该技术的经济成本
                    tech_line = self.tech_set.iloc[tech_idx]
                    tech_industry = tech_line['Livestock species'] if tech_line['class'] != 'crop' else tech_line['Crop species']
                    
                    # 根据产业类型获取对应的规模数据（使用原始未归一化的规模）
                    if tech_line['class'] == 'crop':
                        if tech_industry == 'friut':
                            industry_scale = self.county_scale_original.iloc[county_idx]['fruittree_sown_area']
                        else:
                            industry_scale = self.county_scale_original.iloc[county_idx]['{}_sown_area'.format(tech_industry)]
                    else:
                        tech_industry = tech_industry.lower()
                        if tech_industry in self.county_scale_original.columns:
                            industry_scale = self.county_scale_original.iloc[county_idx][tech_industry]
                        else:
                            industry_scale = self.county_scale_original.iloc[county_idx][tech_industry.replace(' ', '')]
                    
                    # 计算成本：单位成本 * 产业规模
                    cost = tech_line['经济成本'] * industry_scale
                    total_cost += cost
            
            # 添加选中的技术名称到列数据
            column_data.extend(selected_techs)
            
            # 添加总成本到最后
            column_data.append(f"总经济成本: {total_cost:.2f}")
            
            result_data.append(column_data)
        
        # 找到最长的列，用于确定DataFrame的行数
        max_length = max(len(col) for col in result_data) if result_data else 0
        
        # 将所有列填充到相同长度
        for col in result_data:
            while len(col) < max_length:
                col.insert(-1, "")  # 在总成本前添加空字符串
        
        # 创建DataFrame（转置，使得每个县成为一列）
        tech_selected_df = pd.DataFrame(result_data).T
        tech_selected_df.columns = county_names
        
        # 创建行索引
        row_labels = ["县名"]
        for i in range(max_length - 2):  # 减去县名和总成本两行
            row_labels.append(f"技术{i+1}")
        row_labels.append("总经济成本")
        tech_selected_df.index = row_labels
        
        # 保存到Excel
        if self.save_path is not None:
            tech_selected_df.to_excel(os.path.join(self.save_path, "tech_selected_summary.xlsx"))
        return tech_selected_df
    
    def step(self, action):
        # 解码动作为县ID和技术ID
        countyID, techID = self._decode_action(action)
        
        # 检查动作是否有效
        if countyID >= self.numCounty or techID > self.numTech:
            raise ValueError(f"无效动作: countyID={countyID}, techID={techID}")

        # 调试：动作前本县气体与gap
        if getattr(self, 'debug_step_logging', False):
            before_vals = {
                'NH3': float(self.state['NH3'][countyID]),
                'NO3': float(self.state['NO3'][countyID]),
                'N_runoff': float(self.state['N_runoff'][countyID]),
                'CH4': float(self.state['CH4'][countyID]),
                'N2O': float(self.state['N2O'][countyID]),
                # 'Yield': float(self.state['Yield'][countyID]),  # 记录产量总和
            }
            before_gaps = {
                'NH3': float(self.gap_NH3[countyID]),
                'NO3': float(self.gap_NO3[countyID]),
                'N_runoff': float(self.gap_N_runoff[countyID]),
                'CH4': float(self.gap_CH4[countyID]),
                'N2O': float(self.gap_N2O[countyID]),
            }

        gas_deltas = self._update_state(countyID, techID)

        # 从 gas_deltas 中提取对应气体的变化量 gas_deltas = {'NH3','NO3','N_runoff','SOC','CH4','N2O'}
        delta_total_NH3 = gas_deltas['NH3']
        delta_total_NO3 = gas_deltas['NO3']
        delta_total_N_runoff = gas_deltas['N_runoff']
        delta_total_CH4 = gas_deltas['CH4']
        delta_total_N2O = gas_deltas['N2O']
        
        # 计算成本奖励
        reward_from_cost_evaluation, positive_cost = self._evaluate_cost(countyID, techID)
        
        # 计算相似度奖励
        scores_after_tensor, scores_before_tensor = self._evaluate_sim(countyID, techID)
        marginal_score = scores_after_tensor - scores_before_tensor
        #update current_per_county_similarity_scores
        self.current_per_county_similarity_scores[countyID] = scores_after_tensor
        total_reward_sim_component = marginal_score * positive_cost
        
        # 计算减排奖励（稠密的约束推进信号）
        reward_gas = self._evaluate_gap_progress_reward(
            countyID,
            delta_total_CH4,
            delta_total_N2O,
            delta_total_NO3,
            delta_total_N_runoff,
            delta_total_NH3,
        )
        
        # 总奖励
        reward = reward_from_cost_evaluation + total_reward_sim_component + reward_gas
        # print(f"reward_from_cost_evaluation: {reward_from_cost_evaluation}, reward_gas: {reward_gas}, total_reward: {reward}")

        # 更新步数并检查终止条件
        self.current_step += 1
        # 注意：这里不更新action_mask，因为应该由训练代码传入步数来更新
        
        # 检查本地条件：只检查需要技术添加的县是否都达到阈值
        counties_need_tech_indices = torch.tensor(self.counties_need_tech)
        if counties_need_tech_indices.any():
            local_condition = (self.gap_NO3.squeeze()[counties_need_tech_indices] <= 0).all() and \
                            (self.gap_NH3.squeeze()[counties_need_tech_indices] <= 0).all() and \
                            (self.gap_N_runoff.squeeze()[counties_need_tech_indices] <= 0).all() and \
                            (self.gap_CH4.squeeze()[counties_need_tech_indices] <= 0).all() and \
                            (self.gap_N2O.squeeze()[counties_need_tech_indices] <= 0).all()
        else:
            local_condition = True  # 如果没有县需要技术添加，则认为已达成目标
        
        # 检查终止条件：达到目标或达到最大步数
        terminated = local_condition or (self.action_mask_left == 0)

        # 调试：写入每步汇总与子行业影响
        if getattr(self, 'debug_step_logging', False):
            line = self.tech_set.iloc[techID]
            tech_name_dbg = f"{line['Mitigation strategy']}_{line['Crop species'] if line['class'] == 'crop' else line['Livestock species']}"
            after_vals = {
                'NH3': float(self.state['NH3'][countyID]),
                'NO3': float(self.state['NO3'][countyID]),
                'N_runoff': float(self.state['N_runoff'][countyID]),
                'CH4': float(self.state['CH4'][countyID]),
                'N2O': float(self.state['N2O'][countyID]),
                # 'Yield': float(self.state['Yield'][countyID]),
            }
            after_gaps = {
                'NH3': float(self.gap_NH3[countyID]),
                'NO3': float(self.gap_NO3[countyID]),
                'N_runoff': float(self.gap_N_runoff[countyID]),
                'CH4': float(self.gap_CH4[countyID]),
                'N2O': float(self.gap_N2O[countyID]),
            }
            pd.DataFrame([{
                'step': int(self.current_step),
                'countyID': int(countyID),
                'county': self.IDs['Counties'].iloc[countyID],
                'techID': int(techID),
                'tech_name': tech_name_dbg,
                'before_NH3': before_vals['NH3'],
                'before_NO3': before_vals['NO3'],
                'before_N_runoff': before_vals['N_runoff'],
                'before_CH4': before_vals['CH4'],
                'before_N2O': before_vals['N2O'],
                # 'before_Yield': before_vals['Yield'],
                'after_NH3': after_vals['NH3'],
                'after_NO3': after_vals['NO3'],
                'after_N_runoff': after_vals['N_runoff'],
                'after_CH4': after_vals['CH4'],
                'after_N2O': after_vals['N2O'],
                # 'after_Yield': after_vals['Yield'],
                'before_gap_NH3': before_gaps['NH3'],
                'before_gap_NO3': before_gaps['NO3'],
                'before_gap_N_runoff': before_gaps['N_runoff'],
                'before_gap_CH4': before_gaps['CH4'],
                'before_gap_N2O': before_gaps['N2O'],
                'after_gap_NH3': after_gaps['NH3'],
                'after_gap_NO3': after_gaps['NO3'],
                'after_gap_N_runoff': after_gaps['N_runoff'],
                'after_gap_CH4': after_gaps['CH4'],
                'after_gap_N2O': after_gaps['N2O'],
                'Reward_cost': float(reward_from_cost_evaluation),
                'Reward_sim': float(total_reward_sim_component),
                'Reward_gas': float(reward_gas),
                'terminated': bool(terminated),
            }]).to_csv(self.debug_summary_csv, index=False, mode='a', header=False)

            if hasattr(self, '_last_impact_records') and self._last_impact_records:
                rows = []
                for rec in self._last_impact_records:
                    rows.append({
                        'step': int(self.current_step),
                        'countyID': int(countyID),
                        'county': self.IDs['Counties'].iloc[countyID],
                        'techID': int(techID),
                        'tech_name': tech_name_dbg,
                        'industry': rec['industry'],
                        'sub_name': rec['sub_name'],
                        'col_index': rec['col_index'],
                        'before_value': rec['before_value'],
                        'delta': rec['delta'],
                        'after_value': rec['after_value'],
                        'gas_type': rec['gas_type'],
                        'weighted_coeff': rec['weighted_coeff'],
                        'gas_delta_contribution': rec['gas_delta_contribution'],
                    })
                pd.DataFrame(rows).to_csv(self.debug_impacts_csv, index=False, mode='a', header=False)
        
        # 在最后一轮添加技术数量奖励和局部目标惩罚
        if terminated:
            # 技术数量奖励
            tech_count_reward = self._evaluate_tech_count()
            reward += tech_count_reward

            # 取两位小数
            _gap_NO3 = torch.round(self.gap_NO3 * 100)/100
            _gap_NH3 = torch.round(self.gap_NH3 * 100)/100
            _gap_N_runoff = torch.round(self.gap_N_runoff * 100)/100
            _gap_CH4 = torch.round(self.gap_CH4 * 100)/100
            _gap_N2O = torch.round(self.gap_N2O * 100)/100

            no3_met = (_gap_NO3.squeeze()[counties_need_tech_indices] <= 0) if isinstance(_gap_NO3, torch.Tensor) else (torch.tensor(_gap_NO3).squeeze() <= 0)
            nh3_met = (_gap_NH3.squeeze()[counties_need_tech_indices] <= 0) if isinstance(_gap_NH3, torch.Tensor) else (torch.tensor(_gap_NH3).squeeze() <= 0)
            n_runoff_met = (_gap_N_runoff.squeeze()[counties_need_tech_indices] <= 0) if isinstance(_gap_N_runoff, torch.Tensor) else (torch.tensor(_gap_N_runoff).squeeze() <= 0)
            ch4_met = (_gap_CH4.squeeze()[counties_need_tech_indices] <= 0) if isinstance(_gap_CH4, torch.Tensor) else (torch.tensor(_gap_CH4).squeeze() <= 0)
            n2o_met = (_gap_N2O.squeeze()[counties_need_tech_indices] <= 0) if isinstance(_gap_N2O, torch.Tensor) else (torch.tensor(_gap_N2O).squeeze() <= 0)
            # 逻辑与操作得到每个县是否都满足了所有三个局部气体指标
            counties_based_targets_met = no3_met & nh3_met & n_runoff_met
            counties_all_targets_met = no3_met & nh3_met & n_runoff_met & ch4_met & n2o_met
            num_counties_based_met_target = counties_based_targets_met.sum().item()
            num_counties_all_met_target = counties_all_targets_met.sum().item()

            if self.numCounty > 0:
                if int(self.counties_need_tech.sum()) != 0:
                    reward_penalty_3 = (int(self.counties_need_tech.sum()) - num_counties_based_met_target) / int(self.counties_need_tech.sum()) * 100
                else:
                    reward_penalty_3 = 0.0
            else:
                reward_penalty_3 = 0.0
            
            penalty_local_requirements = 0.0 - reward_penalty_3
            reward += penalty_local_requirements
            print(f"县级目标达成情况: {num_counties_based_met_target}/{int(self.counties_need_tech.sum())} 个县达成基于氮的目标, {num_counties_all_met_target}/{int(self.counties_need_tech.sum())} 个县达成所有目标")
            print(f"penalty_local_requirements: {penalty_local_requirements}, reward: {reward}")

            if self.save_path is not None:
                # 确保目标目录存在
                save_dir = os.path.dirname(self.save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # 保存stateMapping
                for key, value in self.stateMapping.items():
                    if isinstance(value, torch.Tensor):
                        state_df = pd.DataFrame(value.detach().numpy(), index=self.IDs['Counties'], columns=self.class_mapping[key])
                    else:
                        state_df = pd.DataFrame(value, index=self.IDs['Counties'], columns=self.class_mapping[key])
                    state_df.to_excel(os.path.join(self.save_path, f"{key}.xlsx"))
                # 保存gap_NO3, gap_N_runoff, gap_NH3, gap_CH4, gap_N2O
                gap_df = pd.DataFrame({
                    'gap_NO3': self.gap_NO3.squeeze(),
                    'gap_N_runoff': self.gap_N_runoff.squeeze(),
                    'gap_NH3': self.gap_NH3.squeeze(),
                    'gap_CH4': self.gap_CH4.squeeze(),
                    'gap_N2O': self.gap_N2O.squeeze()}, index=self.IDs['Counties'])
                gap_df.to_excel(os.path.join(self.save_path, "gap.xlsx"))
                # 保存observation
                observation_df = pd.DataFrame([obs.squeeze().numpy() for obs in list(self.state.values())[:-1]], columns=self.IDs['Counties'], index=list(self.state.keys())[:-1]).T
                observation_df.to_excel(os.path.join(self.save_path, "observation.xlsx"))
                # 保存技术选择情况
                cost_df = self._save_tech_selected_summary()

                # 提取最终变化情况
                final_reduction = {
                    'NO3': self.Total_NO3_origin - self.state['NO3'].squeeze(),
                    'N_runoff': self.Total_N_runoff_origin - self.state['N_runoff'].squeeze(),
                    'NH3': self.Total_NH3_origin - self.state['NH3'].squeeze(),
                    'CH4': self.Total_CH4_origin - self.state['CH4'].squeeze(),
                    'N2O': self.Total_N2O_origin - self.state['N2O'].squeeze(),
                    # 'Yield': self.state['Yield'].squeeze() - self.Total_Yield_origin.squeeze(),
                }
                # 初始化结果表
                results = []
                # 遍历每个县
                county_names = self.IDs['Counties'].tolist()
                # 构建线性规划技术选择矩阵
                lp_selection = torch.zeros((self.numCounty, self.numTech))
                for county_idx in range(self.numCounty):
                    lp_techs = self.lp_tech_ids.get(county_idx, [])
                    for tech_id in lp_techs:
                        if tech_id < self.numTech:
                            lp_selection[county_idx, tech_id] = 1
                # 存储每个县的信息
                for county_idx, county_name in enumerate(county_names):
                    # RL方案成本
                    rl_cost = 0.0
                    # 相似度分数
                    _, rl_city_sim, rl_area_sim = self._calculate_per_county_similarity_scores(county_idx, 0, self.state['Tech_selected'])
                    # 成本分数
                    tech_selected_matrix = self.state['Tech_selected'].detach().numpy()
                    for tech_idx in range(self.numTech):
                        if tech_selected_matrix[county_idx, tech_idx] == 1:
                            tech_line = self.tech_set.iloc[tech_idx]
                            tech_industry = tech_line['Livestock species'] if tech_line['class'] != 'crop' else tech_line['Crop species']
                            if tech_line['class'] == 'crop':
                                if tech_industry == 'friut':
                                    industry_scale = self.county_scale_original.iloc[county_idx]['fruittree_sown_area']
                                else:
                                    industry_scale = self.county_scale_original.iloc[county_idx]['{}_sown_area'.format(tech_industry)]
                            else:
                                tech_industry = tech_industry.lower()
                                if tech_industry in self.county_scale_original.columns:
                                    industry_scale = self.county_scale_original.iloc[county_idx][tech_industry]
                                else:
                                    industry_scale = self.county_scale_original.iloc[county_idx][tech_industry.replace(' ', '')]
                            rl_cost += tech_line['经济成本'] * industry_scale

                    # LP方案成本和相似度
                    lp_cost = 0.0
                    # 相似度分数
                    _, lp_city_sim, lp_area_sim = self._calculate_per_county_similarity_scores(county_idx, 0, lp_selection)
                    # 成本分数
                    lp_techs = self.lp_tech_ids.get(county_idx, [])
                    for tech_idx in lp_techs:
                        if tech_idx >= self.numTech:
                            continue
                        tech_line = self.tech_set.iloc[tech_idx]
                        tech_industry = tech_line['Livestock species'] if tech_line['class'] != 'crop' else tech_line['Crop species']
                        if tech_line['class'] == 'crop':
                            if tech_industry == 'friut':
                                industry_scale = self.county_scale_original.iloc[county_idx]['fruittree_sown_area']
                            else:
                                industry_scale = self.county_scale_original.iloc[county_idx]['{}_sown_area'.format(tech_industry)]
                        else:
                            tech_industry = tech_industry.lower()
                            if tech_industry in self.county_scale_original.columns:
                                industry_scale = self.county_scale_original.iloc[county_idx][tech_industry]
                            else:
                                industry_scale = self.county_scale_original.iloc[county_idx][tech_industry.replace(' ', '')]
                        lp_cost += tech_line['经济成本'] * industry_scale

                    # 检查是否满足三气体减排目标
                    is_met = all([
                        self.gap_NO3[county_idx].item() <= 0,
                        self.gap_N_runoff[county_idx].item() <= 0,
                        self.gap_NH3[county_idx].item() <= 0,
                    ])
                    # 获取使用的技术
                    selected_techs = []
                    tech_selected_matrix = self.state['Tech_selected'].detach().numpy()
                    for tech_idx in range(self.numTech):
                        if tech_selected_matrix[county_idx, tech_idx] == 1:
                            selected_techs.append(str(tech_idx))
                    # 获取线性规划允许的技术
                    lp_techs = self.lp_tech_ids.get(county_idx, [])
            
                    # 添加到结果表
                    results.append({
                        '县名': county_name,
                        '是否满足目标': 1 if is_met else 0,
                        'gap_NO3_origin': self.gap_NO3_origin[county_idx].item(),
                        'gap_N_runoff_origin': self.gap_N_runoff_origin[county_idx].item(),
                        'gap_NH3_origin': self.gap_NH3_origin[county_idx].item(),
                        'gap_CH4_origin': self.gap_CH4_origin[county_idx].item(),
                        'gap_N2O_origin': self.gap_N2O_origin[county_idx].item(),
                        'gap_NO3': self.gap_NO3[county_idx].item(),
                        'gap_N_runoff': self.gap_N_runoff[county_idx].item(),
                        'gap_NH3': self.gap_NH3[county_idx].item(),
                        'gap_CH4': self.gap_CH4[county_idx].item(),
                        'gap_N2O': self.gap_N2O[county_idx].item(),
                        'NO3减排': final_reduction['NO3'][county_idx].item(),
                        'N_runoff减排': final_reduction['N_runoff'][county_idx].item(),
                        'NH3减排': final_reduction['NH3'][county_idx].item(),
                        'CH4减排': final_reduction['CH4'][county_idx].item(),
                        'N2O减排': final_reduction['N2O'][county_idx].item(),
                        # '产量总和变化': final_reduction['Yield_sum'][county_idx].item(),
                        '使用的技术数量': int(tech_selected_matrix[county_idx, :self.numTech].sum()),
                        '线性规划技术数量': len(lp_techs),
                        '使用的技术': ', '.join(selected_techs),
                        '线性规划技术': ', '.join(map(str, lp_techs)),
                        'RL方案总经济成本': rl_cost,
                        'LP方案总经济成本': lp_cost,
                        'RL方案相似度分数': rl_city_sim,
                        'RL方案区域相似度分数': rl_area_sim,
                        'LP方案相似度分数': lp_city_sim,
                        'LP方案区域相似度分数': lp_area_sim,
                    })
                
                # 保存到 Excel 文件
                results_df2 = pd.DataFrame(results)
                results_df2.to_excel(os.path.join(self.save_path, 'termination_results.xlsx'), index=False)
                print(f"终止情况已保存到 {os.path.join(self.save_path, 'termination_results.xlsx')}")

            current_info = {
                'Reward_cost': reward_from_cost_evaluation, 
                'Reward_sim': total_reward_sim_component,
                'Reward_gas': reward_gas,
                'penalty_local_requirements': penalty_local_requirements,
                'countyID': countyID,
                'techID': techID
            }

        else:
            current_info = {
                'Reward_cost': reward_from_cost_evaluation,
                'Reward_sim': total_reward_sim_component,
                'Reward_gas': reward_gas,
                'countyID': countyID,
                'techID': techID
            }
        
        assert not torch.isnan(reward), f"reward is nan: {reward}"
        return self.state, reward, terminated, False, current_info
    
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.action_mask_left = self.numCounty * self.numTech
        self.gap_NO3 = self.gap_NO3_origin.clone().reshape(-1, 1)
        self.gap_N_runoff = self.gap_N_runoff_origin.clone().reshape(-1, 1)
        self.gap_NH3 = self.gap_NH3_origin.clone().reshape(-1, 1)
        self.gap_CH4 = self.gap_CH4_origin.clone().reshape(-1, 1)
        self.gap_N2O = self.gap_N2O_origin.clone().reshape(-1, 1)
        
        # 合并所有产量列的数据
        self.state = {
            'CH4': self.Total_CH4_tensor_origin.clone().reshape(-1, 1),
            'N2O': self.Total_N2O_tensor_origin.clone().reshape(-1, 1),
            'NO3': self.Total_NO3_origin.clone().reshape(-1, 1),
            'N_runoff': self.Total_N_runoff_origin.clone().reshape(-1, 1),
            'NH3': self.Total_NH3_origin.clone().reshape(-1, 1),
            'SOC': self.Total_SOC.clone(),
            'Tech_selected': torch.zeros(self.numCounty, self.numTech),
            # 'Yield': self.Total_Yield_origin.clone()
        }

        self.stateMapping = {
            '种植业NH3挥发': self.NH3_Crop_tensor.clone(),
            '氮肥N2O':  self.N2O_nitrogen_fertilizer_tensor.clone(),
            '氮肥NO3' : self.NO3_nitrogen_fertilizer_tensor.clone(),
            '肠道CH4': self.CH4_intestine_tensor.clone(),
            '粪污管理CH4': self.CH4_Fecal_management_tensor.clone(),
            '粪污管理N2O': self.N2O_Fecal_management_tensor.clone(),
            '粪污管理NH3': self.NH3_Fecal_management_tensor.clone(),
            '粪污施用NH3': self.NH3_manure_application_tensor.clone(),
            '粪污施用N2O': self.N2O_manure_application_tensor.clone(),
            '粪污施用NO3': self.NO3_manure_application_tensor.clone(),
            'Soc': self.Soc_tensor_origin.clone(),
            'N runoff': self.N_runoff_tensor.clone(),
            'Rice CH4 Gg': self.Rice_CH4_Gg.clone(),
            '秸秆焚烧': self.Straw_tensor.clone(),
        }
        
        # 添加产量相关的状态映射
        if self.save_path is not None:
            # 保留所有县的产量数据
            county_names = self.IDs['Counties'].tolist()
            self.livestock_yield = self.livestock_yield_origin.loc[self.livestock_yield_origin.index.isin(county_names)]
            self.crop_yield = self.crop_yield_origin.loc[self.crop_yield_origin.index.isin(county_names)]
            # 重新排序，确保与self.IDs顺序一致
            self.livestock_yield = self.livestock_yield.reindex(county_names)
            self.crop_yield = self.crop_yield.reindex(county_names)

            # 将产量数据转换为tensor格式
            livestock_yield_tensor = torch.tensor(self.livestock_yield.iloc[:, 3:].values, dtype=torch.float32)
            crop_yield_tensor = torch.tensor(self.crop_yield.iloc[:, 3:].values, dtype=torch.float32)
            self.stateMapping.update({
                '畜牧业产量': livestock_yield_tensor.clone(),
                '种植业产量': crop_yield_tensor.clone()
            })

        # 初始化当前区域和区域计数器
        if not hasattr(self, 'current_area'):
            self.current_area = None
        if not hasattr(self, 'area_counter'):
            self.area_counter = 0

        # 遍历每个区域
        if self.area != self.current_area:
            # 区域切换时初始化数据
            self.current_area = self.area
            self.area_counter += 1

        # 直接返回 state 作为 observation
        # Calculate and cache initial per-county similarity scores after state is fully reset
        self.current_per_county_similarity_scores = torch.zeros(self.numCounty, device=self.device)
        return self.state, {}

    def render(self, mode='human'):
        # Render environment (optional)
        # print(f"step:{env.current_step}, || Action: {action}, Reward: {reward}, terminated: {terminated}, truncated:{truncated},info:{info}")
        pass
    
if __name__ == "__main__":
    import cProfile
    
    # 改进的配置：更好的课程学习设置
    config = GasEnvConfig(
        Reward_priority=[7, 5, 3, 2],
        county_df_path='data/基础数据-县级尺度.xlsx',
        IDs_df='data/县市亚区.xlsx',
        livestock_tech_path='data/畜牧业技术列单-经济产量0827.xlsx',
        crop_tech_path="data/种植业技术列单产量产业0803.xlsx",
        soc_df="data/SOC-县尺度.xlsx",
        livestock_scale="data/动物数量.xlsx",
        crop_scale="data/分县种植面积.xlsx",
        # province="Guangdong",
        area="闽南粤中农林水产区",
        # reduction_variables_path='results/reduction_variables.xlsx',
        linear_result_path='results/linear_optimization_results_by_county_5gases_hard_target.xlsx',
        only_lp_phase=True,  # 启用线性规划约束模式
        save_path='results/debug_log',
        debug_step_logging=True,
        debug_log_dir='results/debug_log',

        # 🎯 改进的课程学习参数
        total_steps=2**13,      # 总训练步数
        lp_phase_ratio=0.8,     # 线性规划阶段占比80% - 给足够时间学习基础解
        phase_1_ratio=0.85,      # 第一阶段占比85% - 放开技术等级1
        phase_2_ratio=0.9,     # 第二阶段占比90% - 放开技术等级1和2
    )
    
    print("🚀 改进的课程学习配置:")
    print(f"   - 线性规划阶段: 0-{int(config.total_steps * config.lp_phase_ratio)} 步")
    print(f"   - 阶段1 (等级1): {int(config.total_steps * config.lp_phase_ratio)}-{int(config.total_steps * config.phase_1_ratio)} 步")
    print(f"   - 阶段2 (等级1-2): {int(config.total_steps * config.phase_1_ratio)}-{int(config.total_steps * config.phase_2_ratio)} 步")
    print(f"   - 阶段3 (所有等级): {int(config.total_steps * config.phase_2_ratio)}-{config.total_steps} 步")
    print()
    
    env = GasEnv(config)

    def run_env():
        state, _ = env.reset()
        terminated, truncated = False, False
        total_reward = 0
 
        print("开始按亚区推理/回放...")
        print(f"环境初始化完成（亚区={env.area}），共有 {env.numCounty} 个县，{env.numTech} 个技术")
 
        # 优先回放LP方案（逐县逐技术），便于与LP结果严格比对
        if env.lp_tech_ids is not None and isinstance(env.lp_tech_ids, dict) and len(env.lp_tech_ids) > 0:
            print("🔵 使用LP回放模式：按当前环境的县序列依次施用LP选定技术（如有）")
            for county_idx in range(env.numCounty):
                county_name = env.IDs['Counties'].iloc[county_idx]
                # 指定县调试
                # if county_name != '鼎湖区':
                #     continue
                tech_list = env.lp_tech_ids.get(county_idx, [])
                if not tech_list:
                    continue
                print(f"县 {county_name}：回放 {len(tech_list)} 个技术")
                for tech_id in tech_list:
                    if tech_id >= env.numTech:
                        print(f"  跳过非法技术ID: {tech_id}")
                        continue
                    action = county_idx * env.numTech + tech_id
                    state, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    tech_info = env.tech_set.iloc[tech_id]
                    tech_name = f"{tech_info['Mitigation strategy']}_{tech_info['Crop species'] if tech_info['class'] == 'crop' else tech_info['Livestock species']}"
                    # print(f"步骤 {env.current_step}: 县={county_name}, 技术={tech_name}, 奖励={float(reward):.2f}, 结束={terminated}")
                    if terminated or truncated:
                        break
                if terminated or truncated:
                    break
        else:
            print("🟡 未检测到LP选定技术，退回随机有效动作测试模式（受课程学习与掩码约束）")
            # 测试时不传入training_step，使用环境内部步数
            action_mask = env._get_action_mask()
            valid_actions = (~action_mask).sum().item()
            total_actions = action_mask.shape[-2] * action_mask.shape[-1]
            print(f"动作掩码统计: 总动作数={total_actions}, 有效动作数={valid_actions}, 被遮蔽动作数={total_actions-valid_actions}")

            for i in range(10000):
                # 从有效动作中随机选择，测试时可以传入模拟的训练步数
                simulated_training_step = i * 5  # 模拟训练步数增长
                action_mask = env._get_action_mask(training_step=simulated_training_step)
                valid_actions_indices = torch.where(~action_mask)[0]
                if len(valid_actions_indices) == 0:
                    print("没有有效动作可选，提前结束")
                    break
                action = valid_actions_indices[torch.randint(0, len(valid_actions_indices), (1,))].item()
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                # 解码动作以便显示
                countyID, techID = env._decode_action(action)
                county_name = env.IDs['Counties'].iloc[countyID]
                tech_info = env.tech_set.iloc[techID]
                tech_name = f"{tech_info['Mitigation strategy']}_{tech_info['Crop species'] if tech_info['class'] == 'crop' else tech_info['Livestock species']}"
                
                print(f"步骤 {env.current_step}: 县={county_name}, 技术={tech_name}, 奖励={float(reward):.2f}, 结束={terminated}")
                print(f"  奖励详情: {info}")
                print()

        print(f"测试完成，总奖励: {float(total_reward):.2f}")

        # 打印亚区不满足减排的县
        counties_need_tech_indices = torch.tensor(env.counties_need_tech)
        _gap_NO3 = torch.round(env.gap_NO3 * 100)/100
        _gap_NH3 = torch.round(env.gap_NH3 * 100)/100
        _gap_N_runoff = torch.round(env.gap_N_runoff * 100)/100
        _gap_CH4 = torch.round(env.gap_CH4 * 100)/100
        _gap_N2O = torch.round(env.gap_N2O * 100)/100

        no3_met = (_gap_NO3.squeeze()[counties_need_tech_indices] <= 0) if isinstance(_gap_NO3, torch.Tensor) else (torch.tensor(_gap_NO3).squeeze() <= 0)
        nh3_met = (_gap_NH3.squeeze()[counties_need_tech_indices] <= 0) if isinstance(_gap_NH3, torch.Tensor) else (torch.tensor(_gap_NH3).squeeze() <= 0)
        n_runoff_met = (_gap_N_runoff.squeeze()[counties_need_tech_indices] <= 0) if isinstance(_gap_N_runoff, torch.Tensor) else (torch.tensor(_gap_N_runoff).squeeze() <= 0)
        ch4_met = (_gap_CH4.squeeze()[counties_need_tech_indices] <= 0) if isinstance(_gap_CH4, torch.Tensor) else (torch.tensor(_gap_CH4).squeeze() <= 0)
        n2o_met = (_gap_N2O.squeeze()[counties_need_tech_indices] <= 0) if isinstance(_gap_N2O, torch.Tensor) else (torch.tensor(_gap_N2O).squeeze() <= 0)

        counties_based_targets_met = no3_met & nh3_met & n_runoff_met
        counties_all_targets_met = no3_met & nh3_met & n_runoff_met & ch4_met & n2o_met
        num_counties_based_met_target = counties_based_targets_met.sum().item()
        num_counties_all_met_target = counties_all_targets_met.sum().item()

        need_indices = torch.where(counties_need_tech_indices)[0].cpu().numpy().tolist()
        # 三气体
        based_met_arr = counties_based_targets_met.cpu().numpy().astype(bool)
        unmet_relative_idx = [i for i, met in enumerate(based_met_arr) if not met]
        unmet_global_idx = [need_indices[i] for i in unmet_relative_idx]
        unmet_names = [env.IDs['Counties'].iloc[i] for i in unmet_global_idx]
        # 五气体
        based_met_arr_all = counties_all_targets_met.cpu().numpy().astype(bool)
        unmet_relative_idx_all = [i for i, met in enumerate(based_met_arr_all) if not met]
        unmet_global_idx_all = [need_indices[i] for i in unmet_relative_idx_all]
        unmet_names_all = [env.IDs['Counties'].iloc[i] for i in unmet_global_idx_all]

        county_names = env.IDs['Counties'].tolist()
        tech_matrix = env.state['Tech_selected'].detach().cpu().numpy()

        def tensor_to_np(t):
            if torch.is_tensor(t):
                return t.squeeze().cpu().numpy()
            return np.array(t).squeeze()

        gap_NO3 = tensor_to_np(env.gap_NO3)
        gap_NH3 = tensor_to_np(env.gap_NH3)
        gap_N_runoff = tensor_to_np(env.gap_N_runoff)
        gap_CH4 = tensor_to_np(env.gap_CH4)
        gap_N2O = tensor_to_np(env.gap_N2O)

        for idx, cname in enumerate(county_names):
            sel = np.where(tech_matrix[idx] == 1)[0].tolist()
            sel_str = ','.join(map(str, sel)) if sel else ''
            print(f"{idx} {cname}: [{sel_str}]")

        for idx, cname in enumerate(county_names):
            no3_gap = float(gap_NO3[idx])
            nh3_gap = float(gap_NH3[idx])
            nrun_gap = float(gap_N_runoff[idx])
            ch4_gap = float(gap_CH4[idx])
            n2o_gap = float(gap_N2O[idx])

            three_met = (no3_gap <= 0 and nh3_gap <= 0 and nrun_gap <= 0)
            all_met = (three_met and ch4_gap <= 0 and n2o_gap <= 0)

            print(f"{idx} {cname} gaps —— NH3:{nh3_gap:.3f}, NO3:{no3_gap:.3f}, Nrun:{nrun_gap:.3f}, CH4:{ch4_gap:.4f}, N2O:{n2o_gap:.3f}")
        
        print(f"未满足三气体的县{len(unmet_names)}: {', '.join(map(str, unmet_names[:100]))}")
        print(f"未满足五气体的县{len(unmet_names_all)}: {', '.join(map(str, unmet_names_all[:100]))}")
        print(f"县级目标达成情况: {num_counties_based_met_target}/{int(env.counties_need_tech.sum())} 个县达成基于氮的目标, {num_counties_all_met_target}/{int(env.counties_need_tech.sum())} 个县达成所有目标")
        
        return total_reward
    
    # cProfile.run('run_env()')  # 注释掉性能分析，直接运行
    run_env()
