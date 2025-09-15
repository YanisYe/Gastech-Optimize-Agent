"""
课程学习配置文件
用于管理不同阶段的训练参数
"""

class CurriculumLearningConfig:
    """课程学习配置类"""
    
    @staticmethod
    def get_basic_config():
        """基础配置 - 适合初学者"""
        return {
            'total_steps': 2**12,      # 4096步
            'lp_phase_ratio': 0.4,     # 40%时间在线性规划解空间学习
            'phase_1_ratio': 0.7,      # 70%时间放开技术等级1
            'phase_2_ratio': 0.9,      # 90%时间放开技术等级1-2
            'phase_3_ratio': 1.0,      # 最后10%时间放开所有等级
            'reward_priority': [0.8, 0.3, 0.2, 0.1],  # 更重视减排奖励
            'local_target_penalty_factor': 30.0,  # 降低惩罚因子
        }
    
    @staticmethod
    def get_advanced_config():
        """进阶配置 - 适合有经验的模型"""
        return {
            'total_steps': 2**13,      # 8192步
            'lp_phase_ratio': 0.2,     # 20%时间在线性规划解空间学习
            'phase_1_ratio': 0.5,      # 50%时间放开技术等级1
            'phase_2_ratio': 0.8,      # 80%时间放开技术等级1-2
            'phase_3_ratio': 1.0,      # 最后20%时间放开所有等级
            'reward_priority': [0.6, 0.5, 0.3, 0.2],  # 平衡的奖励权重
            'local_target_penalty_factor': 50.0,  # 标准惩罚因子
        }
    
    @staticmethod
    def get_expert_config():
        """专家配置 - 适合复杂场景"""
        return {
            'total_steps': 2**14,      # 16384步
            'lp_phase_ratio': 0.1,     # 10%时间在线性规划解空间学习
            'phase_1_ratio': 0.3,      # 30%时间放开技术等级1
            'phase_2_ratio': 0.6,      # 60%时间放开技术等级1-2
            'phase_3_ratio': 1.0,      # 最后40%时间放开所有等级
            'reward_priority': [0.5, 0.6, 0.4, 0.3],  # 更重视成本优化
            'local_target_penalty_factor': 70.0,  # 高惩罚因子
        }
    
    @staticmethod
    def get_focused_lp_config():
        """专注线性规划配置 - 专门用于学习线性规划解"""
        return {
            'total_steps': 2**12,      # 4096步
            'lp_phase_ratio': 0.8,     # 80%时间在线性规划解空间学习
            'phase_1_ratio': 0.9,      # 90%时间放开技术等级1
            'phase_2_ratio': 0.95,     # 95%时间放开技术等级1-2
            'phase_3_ratio': 1.0,      # 最后5%时间放开所有等级
            'reward_priority': [0.9, 0.2, 0.1, 0.05],  # 极度重视减排奖励
            'local_target_penalty_factor': 20.0,  # 低惩罚因子
        }
    
    @staticmethod
    def get_balanced_config():
        """平衡配置 - 推荐配置"""
        return {
            'total_steps': 2**13,      # 8192步
            'lp_phase_ratio': 0.3,     # 30%时间在线性规划解空间学习
            'phase_1_ratio': 0.6,      # 60%时间放开技术等级1
            'phase_2_ratio': 0.85,     # 85%时间放开技术等级1-2
            'phase_3_ratio': 1.0,      # 最后15%时间放开所有等级
            'reward_priority': [0.7, 0.5, 0.3, 0.2],  # 平衡的奖励权重
            'local_target_penalty_factor': 50.0,  # 标准惩罚因子
        }
    
    @staticmethod
    def print_config_info(config_name, config):
        """打印配置信息"""
        print(f"\n📋 {config_name} 配置:")
        print(f"   总训练步数: {config['total_steps']}")
        print(f"   线性规划阶段: 0-{int(config['total_steps'] * config['lp_phase_ratio'])} 步")
        print(f"   阶段1 (等级1): {int(config['total_steps'] * config['lp_phase_ratio'])}-{int(config['total_steps'] * config['phase_1_ratio'])} 步")
        print(f"   阶段2 (等级1-2): {int(config['total_steps'] * config['phase_1_ratio'])}-{int(config['total_steps'] * config['phase_2_ratio'])} 步")
        print(f"   阶段3 (所有等级): {int(config['total_steps'] * config['phase_2_ratio'])}-{config['total_steps']} 步")
        print(f"   奖励权重: {config['reward_priority']}")
        print(f"   惩罚因子: {config['local_target_penalty_factor']}")

def get_recommended_config(problem_type="balanced"):
    """
    获取推荐的配置
    
    Args:
        problem_type: 问题类型
            - "basic": 基础问题
            - "advanced": 进阶问题  
            - "expert": 专家问题
            - "focused_lp": 专注线性规划学习
            - "balanced": 平衡配置（推荐）
    
    Returns:
        dict: 配置字典
    """
    configs = {
        'basic': CurriculumLearningConfig.get_basic_config,
        'advanced': CurriculumLearningConfig.get_advanced_config,
        'expert': CurriculumLearningConfig.get_expert_config,
        'focused_lp': CurriculumLearningConfig.get_focused_lp_config,
        'balanced': CurriculumLearningConfig.get_balanced_config,
    }
    
    if problem_type not in configs:
        print(f"⚠️  未知的问题类型: {problem_type}，使用平衡配置")
        problem_type = 'balanced'
    
    config = configs[problem_type]()
    CurriculumLearningConfig.print_config_info(problem_type.title(), config)
    
    return config

if __name__ == "__main__":
    # 测试所有配置
    print("🎯 课程学习配置测试")
    print("=" * 50)
    
    configs = [
        ("基础配置", CurriculumLearningConfig.get_basic_config()),
        ("进阶配置", CurriculumLearningConfig.get_advanced_config()),
        ("专家配置", CurriculumLearningConfig.get_expert_config()),
        ("专注线性规划配置", CurriculumLearningConfig.get_focused_lp_config()),
        ("平衡配置", CurriculumLearningConfig.get_balanced_config()),
    ]
    
    for name, config in configs:
        CurriculumLearningConfig.print_config_info(name, config)
        print() 