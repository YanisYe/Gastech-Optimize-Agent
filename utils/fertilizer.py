import pandas as pd
import numpy as np

# manure数据中的作物关键词映射表
MANURE_CROP_KEYWORDS = {
    'wheat': 'wheat',
    'rice': 'rice',
    'maize': 'maize',
    'bean': 'beans',
    'fruit': 'fruittree',
    'vegetable': 'vegetable',
    'other crops': 'other crops'
}

# fertilizer数据中的作物关键词映射表（包含更多作物）
FERTILIZER_CROP_KEYWORDS = {
    'wheat': 'wheat',
    'rice': 'rice',
    'maize': 'maize',
    'bean': 'beans',
    'potato': 'potato',
    'tuber': 'potato',
    'cotton': 'cotton',
    'sugarcane': 'sugarcane',
    'sugarbeet': 'sugarbeet',
    'sugar beet': 'sugarbeet',
    'fruit': 'fruittree',
    'vegetable': 'vegetable'
}

# 一刀切技术包映射
TECH_PACKAGE_MAPPING = {
    0: [],  # 无需技术干预 - 不采用任何技术
    1: ['STEP_1'],  # ≤1级技术包 - 采用STEP_1的所有技术
    2: ['STEP_2'],  # ≤2级技术包 - 采用STEP_1和STEP_2的所有技术
    3: ['STEP_3']  # ≤3级技术包 - 采用所有步骤的技术
}

def get_crop_name_from_tech(tech_str: str, crop_keywords=None) -> str:
    """从技术名称中提取作物名称"""
    if crop_keywords is None:
        crop_keywords = FERTILIZER_CROP_KEYWORDS
    return next((crop for keyword, crop in crop_keywords.items() if keyword in tech_str), None)

def get_crop_name_from_species(species: str) -> str:
    """从物种名称中获取对应的作物名称"""
    if pd.isna(species):
        return None

    species = str(species).lower().strip()

    # 作物物种映射
    species_to_crop = {
        'rice': 'rice',
        'wheat': 'wheat',
        'maize': 'maize',
        'vegetable': 'vegetable',
        'friut': 'fruittree',  # 注意拼写错误
        'fruit': 'fruittree',
        'potato': 'potato',
        'tuber': 'potato',
        'bean': 'beans',
        'cotton': 'cotton',
        'sugarcane': 'sugarcane',
        'sugarbeet': 'sugarbeet',
        'sugar beet': 'sugarbeet'
    }

    return species_to_crop.get(species, None)

def load_tech_package_assignments(assignment_file_path: str = "results/level_based_stepwise_techs/level_based_stepwise_techs/county_tech_assignments.xlsx") -> pd.DataFrame:
    """加载县技术包分配信息"""
    try:
        return pd.read_excel(assignment_file_path)
    except FileNotFoundError:
        print(f"警告: 找不到技术包分配文件 {assignment_file_path}")
        return None

def load_tech_packages(tech_file_path: str = "results/level_based_stepwise_techs/level_based_stepwise_techs/tech_packages.xlsx") -> pd.DataFrame:
    """加载技术包详细信息"""
    try:
        return pd.read_excel(tech_file_path)
    except FileNotFoundError:
        print(f"警告: 找不到技术包文件 {tech_file_path}")
        return None

def get_techs_for_package_level(package_level: int, tech_packages_df: pd.DataFrame) -> list:
    """根据技术包等级获取对应的技术列表"""
    if package_level not in TECH_PACKAGE_MAPPING:
        return []

    steps = TECH_PACKAGE_MAPPING[package_level]
    if not steps:
        return []

    # 从技术包数据中筛选出对应步骤的技术
    techs_in_package = []
    step_techs = tech_packages_df[tech_packages_df['步骤'] == steps[0]][['技术名称', '物种']].values.tolist()
    techs_in_package.extend(step_techs)

    return techs_in_package  # 去重

def get_county_package_level(county_name: str, assignment_df: pd.DataFrame) -> int:
    """获取县的技术包等级"""
    if assignment_df is None:
        return 0

    county_row = assignment_df[assignment_df['县名称'] == county_name]
    if county_row.empty:
        return 0

    return county_row['分配技术包等级'].iloc[0]

def get_package_levels_with_tech(target_tech: str, assignment_df: pd.DataFrame, tech_packages_df: pd.DataFrame) -> list:
    """获取包含指定技术的技术包等级列表"""
    if assignment_df is None or tech_packages_df is None:
        return []

    package_levels_with_tech = []

    # 检查每个技术包等级是否包含目标技术
    for package_level in TECH_PACKAGE_MAPPING.keys():
        techs_in_package = get_techs_for_package_level(package_level, tech_packages_df)
        if target_tech in techs_in_package:
            package_levels_with_tech.append(package_level)

    return package_levels_with_tech

def check_county_has_tech(county_name: str, target_tech: str, assignment_df: pd.DataFrame, tech_packages_df: pd.DataFrame) -> bool:
    """检查县是否采用了指定的技术（基于一刀切技术包分配）"""
    if assignment_df is None or tech_packages_df is None:
        return False

    # 首先获取该县的技术包等级
    package_level = get_county_package_level(county_name, assignment_df)

    # 获取该等级技术包包含的所有技术
    techs_in_package = get_techs_for_package_level(package_level, tech_packages_df)

    # 检查目标技术是否在技术包中
    return target_tech in techs_in_package

def step_one(df: pd.DataFrame, tech_selected, use_one_cut_analysis=False, assignment_df=None, tech_packages_df=None):
    """
    筛选包含apply right rate这一技术的县，并根据最优氮应用率更新肥料用量

    参数:
    df: 基础数据DataFrame
    tech_selected: 技术选择数据（原有方式使用）
    use_one_cut_analysis: 是否使用一刀切分析方式
    assignment_df: 县技术包分配数据（一刀切方式使用）
    tech_packages_df: 技术包详细信息（一刀切方式使用）
    """
    # 最优氮应用率表 (kg N/ha)
    optimal_n_rates = {
        'rice': 150,
        'wheat': 155, 
        'maize': 152,
        'beans': 35,
        'potato': 135,  # tubers
        'cotton': 219,
        'sugarcane': 314,
        'sugarbeet': 273,
        'fruittree': 304,  # fruits (apple)
        'vegetable': 283
    }
    
    # 创建df的副本
    updated_df = df.copy()
    counties_with_right_rate = []
    
    print("开始筛选包含Right rate技术的县并更新肥料用量...")
    if use_one_cut_analysis:
        print("使用一刀切分析方式")
        if assignment_df is None or tech_packages_df is None:
            print("错误: 使用一刀切分析需要提供assignment_df和tech_packages_df")
            return updated_df
    else:
        print(f"tech_selected表格形状: {tech_selected.shape}")

    # 遍历所有县
    for county_name in updated_df['County'].unique():
        has_right_rate_tech = False
        applied_crops = []

        if use_one_cut_analysis:
            # 一刀切方式：首先获取县的技术包等级，然后检查该包是否包含Right rate技术
            county_package_level = get_county_package_level(county_name, assignment_df)
            county_techs = get_techs_for_package_level(county_package_level, tech_packages_df)

            if county_package_level == 0:
                continue

            # 检查该县的技术包中是否有Right rate技术
            for tech_name, species in county_techs:  
                if pd.notna(tech_name) and 'Right rate' in str(tech_name):
                    # 一刀切方式：从技术包数据中获取物种信息来确定作物
                    crop_name = get_crop_name_from_species(species)
                    if crop_name and crop_name in optimal_n_rates:
                        has_right_rate_tech = True
                        applied_crops.append((crop_name, tech_name))
        else:
            # 原有方式：遍历tech_selected中的技术
            # 找到该县在tech_selected中的列索引
            county_col_idx = None
            for col_idx in range(1, tech_selected.shape[1]):
                if tech_selected.iloc[0, col_idx] == county_name:
                    county_col_idx = col_idx
                    break

            if county_col_idx is None:
                continue

            county_techs = tech_selected.iloc[1:, county_col_idx]

            # 检查该县是否有包含"Right rate"的技术，并识别作用的作物
            for tech in county_techs:
                if pd.notna(tech) and 'Right rate' in str(tech):
                    crop_name = get_crop_name_from_tech(str(tech), FERTILIZER_CROP_KEYWORDS)
                    if crop_name and crop_name in optimal_n_rates:
                        has_right_rate_tech = True
                        applied_crops.append((crop_name, tech))

        if has_right_rate_tech:
            counties_with_right_rate.append(county_name)
            print(f"找到县 {county_name} 使用Right rate技术:")

            # 找到该县在df中的行索引
            county_row_idx = updated_df[updated_df['County'] == county_name].index
            if len(county_row_idx) > 0:
                county_row_idx = county_row_idx[0]

                for crop_name, tech in applied_crops:
                    print(f"  对 {crop_name} 使用技术: {tech}")

                    # 获取该作物的种植面积
                    area_col = f'{crop_name}_sown_area'
                    if area_col in updated_df.columns:
                        crop_area = updated_df.loc[county_row_idx, area_col]

                        if pd.notna(crop_area) and crop_area > 0:
                            # 计算新的肥料用量 = 最优氮应用率 * 种植面积 / 1000 (转换为kt)
                            new_fertilizer_amount = optimal_n_rates[crop_name] * crop_area / 1000

                            # 更新肥料用量
                            fertilizer_col = f'{crop_name}_fertilizer_amount'
                            if fertilizer_col in updated_df.columns:
                                old_amount = updated_df.loc[county_row_idx, fertilizer_col]
                                updated_df.loc[county_row_idx, fertilizer_col] = new_fertilizer_amount

                                print(f"    更新 {county_name} 的 {crop_name} 肥料用量: {old_amount:.3f} -> {new_fertilizer_amount:.3f} kt")      
    
    # 去重县名列表
    counties_with_right_rate = list(set(counties_with_right_rate))
    
    print(f"\n包含Right rate技术的县数量: {len(counties_with_right_rate)}")
    
    # updated_df["化肥氮量(kt N）"] = \sum
    updated_df["化肥氮量\n(kt N）"] = updated_df.filter(regex='.*amount.*').sum(axis=1)

    return updated_df

def step_two(manure_df: pd.DataFrame, tech_selected, fertilizer_amount, use_one_cut_analysis=False, assignment_df=None, tech_packages_df=None):
    '''
    处理compost和fresh manure技术，重新分配肥料氮和manure氮

    参数:
    manure_df: manure数据DataFrame
    tech_selected: 技术选择数据（原有方式使用）
    fertilizer_amount: 肥料用量数据
    use_one_cut_analysis: 是否使用一刀切分析方式
    assignment_df: 县技术包分配数据（一刀切方式使用）
    tech_packages_df: 技术包详细信息（一刀切方式使用）

    技术逻辑:
    1. 遍历所有县,
    2. 如果该县使用了compost,fresh manure(注意包含高中低三个技术的话):
       - low技术: 比例<30%，肥料氮和manure N进行替换，肥料N=（manureN+肥料N）*0.7；manure N= （manure N +肥料N）* 0.3
       - medium技术: 比例50%，肥料N=（manureN+肥料N）*0.5；manure N= （manure N +肥料N）* 0.5
       - high技术: 比例70%，肥料N=（manureN+肥料N）*0.3；manure N= （manure N +肥料N）* 0.7
    '''
    # 创建manure_df的副本
    updated_manure_df = manure_df.copy()
    
    print("开始处理compost和fresh manure技术...")
    print(f"manure数据形状: {manure_df.shape}")
    print(f"manure数据列: {list(manure_df.columns)}")

    if use_one_cut_analysis:
        print("使用一刀切分析方式")
        if assignment_df is None or tech_packages_df is None:
            print("错误: 使用一刀切分析需要提供assignment_df和tech_packages_df")
            return updated_manure_df, fertilizer_amount
    else:
        print(f"tech_selected表格形状: {tech_selected.shape}")

    # 获取第2到第8列（索引1到7，因为列索引从0开始）
    selected_columns = manure_df.columns[2:9]
    print(f"选择的第2-8列: {list(selected_columns)}")

    # 遍历所有县
    for county_id in range(manure_df.shape[0]):
        county_name = manure_df.iloc[county_id]['Counties']  # manure_df中县名列是'Counties'

        # 先遍历manure_df中的作物fraction列，确定需要处理的作物
        for manure_frac_col in selected_columns:
            # 从列名中提取作物名称
            crop_name = manure_frac_col.replace('_manure_frac', '')

            # 检查该县是否有针对该作物的compost/manure技术
            has_compost_manure_tech = False
            tech_level = None
            applied_tech = None

            if use_one_cut_analysis:
                # 一刀切方式：首先获取县的技术包等级，然后检查该包是否包含compost/manure技术
                county_package_level = get_county_package_level(county_name, assignment_df)
                county_techs = get_techs_for_package_level(county_package_level, tech_packages_df)

                # 检查该县的技术包中是否有针对该作物的compost/manure技术
                for tech_name, species in county_techs:
                    if pd.notna(tech_name):
                        tech_str = str(tech_name).lower()
                        # 一刀切方式：从技术包数据中获取物种信息来确定作物
                        tech_crop_name = get_crop_name_from_species(species)

                        if ('compost' in tech_str or 'fresh manure' in tech_str or 'manure' in tech_str) and tech_crop_name == crop_name:
                            has_compost_manure_tech = True
                            applied_tech = tech_name

                            # 检查技术级别
                            tech_name_lower = str(tech_name).lower()
                            if 'low' in tech_name_lower:
                                tech_level = 'low'
                            elif 'medium' in tech_name_lower or 'mid' in tech_name_lower:
                                tech_level = 'medium'
                            elif 'high' in tech_name_lower:
                                tech_level = 'high'
                            break
            else:
                # 原有方式：在tech_selected中找到对应的县
                county_col_idx = None
                for col_idx in range(1, tech_selected.shape[1]):
                    if tech_selected.iloc[0, col_idx] == county_name:
                        county_col_idx = col_idx
                        break

                if county_col_idx is None:
                    continue

                # 获取该县的所有技术
                county_techs = tech_selected.iloc[1:, county_col_idx]

                for tech in county_techs:
                    if pd.notna(tech):
                        tech_str = str(tech).lower()
                        if ('compost' in tech_str or 'fresh manure' in tech_str or 'manure' in tech_str) and crop_name in tech_str:
                            has_compost_manure_tech = True
                            applied_tech = tech

                            # 检查技术级别并找到对应的百分比列
                            if 'low' in tech_str:
                                tech_level = 'low'
                            elif 'medium' in tech_str or 'mid' in tech_str:
                                tech_level = 'medium'
                            elif 'high' in tech_str:
                                tech_level = 'high'
                            break

            if has_compost_manure_tech and tech_level:
                # 首先读取manure_ratio
                manure_ratio_raw = updated_manure_df.loc[county_id, f'{crop_name}_manure %']
                # 处理百分比单位（如果数据是以百分号形式存储，需要转换为小数）
                if isinstance(manure_ratio_raw, str) and '%' in manure_ratio_raw:
                    manure_ratio = float(manure_ratio_raw.strip('%')) / 100
                else:
                    manure_ratio = float(manure_ratio_raw)

                print(f"处理县 {county_name}: 发现{tech_level}级别的compost/manure技术作用于{crop_name}，比例 {manure_ratio:.1f}%")

                # 获取对应的实际manure量
                manure_n = updated_manure_df.loc[county_id, f'{crop_name}_manure']

                if pd.notna(manure_n) and manure_n > 0:
                    # 根据作物名称获取对应的肥料氮量
                    fertilizer_col = f'{crop_name}_fertilizer_amount'

                    if fertilizer_col in fertilizer_amount.columns:
                        fertilizer_n = fertilizer_amount.loc[county_id, fertilizer_col]
                    else:
                        # 如果找不到对应的肥料列，跳过
                        print(f"  警告: 找不到 {county_name} 的 {crop_name} 肥料数据列 {fertilizer_col}")
                        continue

                    if pd.isna(fertilizer_n) or fertilizer_n <= 0:
                        print(f"  警告: {county_name} 的 {crop_name} 肥料氮量为空或为0")
                        continue

                    total_n = fertilizer_n + manure_n

                    # 根据技术级别和读取的manure_ratio重新分配
                    if tech_level == 'low' and manure_ratio < 0.3:
                        # low技术: 肥料N = 总量 * (1-manure_ratio)，manure N = 总量 * manure_ratio
                        new_fertilizer_n = total_n * (1 - 0.3)
                        new_manure_n = total_n * 0.3
                        updated_manure_df.loc[county_id, f'{crop_name}_manure %'] = 0.3
                    elif tech_level == 'medium' and manure_ratio < 0.5:
                        # medium技术: 使用读取的manure_ratio
                        new_fertilizer_n = total_n * (1 - 0.5)
                        new_manure_n = total_n * 0.5
                        updated_manure_df.loc[county_id, f'{crop_name}_manure %'] = 0.5
                    elif tech_level == 'high' and manure_ratio < 0.7:
                        # high技术: 使用读取的manure_ratio
                        new_fertilizer_n = total_n * (1 - 0.7)
                        new_manure_n = total_n * 0.7
                        updated_manure_df.loc[county_id, f'{crop_name}_manure %'] = 0.7
                    else:
                        # 如果low技术但比例>=30%，不进行调整
                        continue

                    # 更新manure N
                    updated_manure_df.loc[county_id, f'{crop_name}_manure'] = new_manure_n
                    # 更新肥料N
                    fertilizer_amount.loc[county_id, fertilizer_col] = new_fertilizer_n
                    # 同时更新manure fraction（如果需要的话）
                    # 这里可以根据需要更新fraction数据

                    print(f"  更新 {county_name} {crop_name} manure: manure N {manure_n:.3f} -> {new_manure_n:.3f}")
                    print(f"    对应肥料N应从 {fertilizer_n:.3f} -> {new_fertilizer_n:.3f}")
                    
    # 更新manure_df中的manure N
    updated_manure_df['manure N'] = updated_manure_df.filter(['rice_manure', 'wheat_manure', 'maize_manure', 'beans_manure', 'potato_manure', 'cotton_manure', 'sugarcane_manure', 'sugarbeet_manure', 'fruittree_manure', 'vegetable_manure']).sum(axis=1)
    # 更新肥料N
    fertilizer_amount['化肥氮量(kt N）_after'] = fertilizer_amount.filter(like='_amount').sum(axis=1)
    return updated_manure_df, fertilizer_amount

def main(use_one_cut_analysis=False):
    """
    主函数，支持原有方式和一刀切分析方式

    参数:
    use_one_cut_analysis: 是否使用一刀切分析方式
    """
    df = pd.read_excel("data/基础数据-县级尺度-latest.xlsx", sheet_name='Sheet2')

    if use_one_cut_analysis:
        # 一刀切分析方式
        print("使用一刀切分析方式")
        assignment_df = load_tech_package_assignments()
        tech_packages_df = load_tech_packages()

        if assignment_df is None or tech_packages_df is None:
            print("错误: 无法加载一刀切分析所需的数据文件")
            return

        updated_df = step_one(df, None, use_one_cut_analysis=True, assignment_df=assignment_df, tech_packages_df=tech_packages_df)

        # 存储step1的肥料用量数据
        manure_df = pd.read_excel("data/基础数据-县级尺度-latest.xlsx", sheet_name='manure 施用')
        updated_manure_df, updated_fertilizer_amount = step_two(manure_df, None, updated_df,
                                                               use_one_cut_analysis=True,
                                                               assignment_df=assignment_df,
                                                               tech_packages_df=tech_packages_df)
    else:
        # 原有方式
        print("使用原有分析方式")
        tech_selected = pd.read_excel("results/rl_opt_result/tech_selected_summary_merged.xlsx")
        updated_df = step_one(df, tech_selected)

        # 存储step1的肥料用量数据
        manure_df = pd.read_excel("data/基础数据-县级尺度-latest.xlsx", sheet_name='manure 施用')
        updated_manure_df, updated_fertilizer_amount = step_two(manure_df, tech_selected, updated_df)

    print(f"\n更新后的数据形状: {updated_df.shape}")
    print(f"\n更新后的manure数据形状: {updated_manure_df.shape}")

    # 保存更新后的数据到新的Excel文件
    suffix = "_one_cut" if use_one_cut_analysis else ""
    updated_df.to_excel(f"results/step1_fertilizer_data{suffix}.xlsx", index=False)
    updated_manure_df.to_excel(f"results/step2_manure_data{suffix}.xlsx", index=False)
    updated_fertilizer_amount.to_excel(f"results/step2_fertilizer_data{suffix}.xlsx", index=False)

if __name__ == "__main__":
    import sys

    # 检查命令行参数
    use_one_cut = len(sys.argv) > 1 and sys.argv[1] == "--one-cut"
    # use_one_cut = True
    if use_one_cut:
        print("运行一刀切分析...")
        main(use_one_cut_analysis=True)
    else:
        print("运行原有分析方式...")
        main(use_one_cut_analysis=False)
