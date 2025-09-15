# 动态导入，避免在模块初始化时的导入错误
def __getattr__(name):
    if name in ['CountyDataLoader', 'TechDataLoader']:
        from .dataLoader import CountyDataLoader, TechDataLoader
        if name == 'CountyDataLoader':
            return CountyDataLoader
        elif name == 'TechDataLoader':
            return TechDataLoader
    elif name in ['GasEnv', 'GasEnvConfig']:
        from .GasEnviroment_v2 import GasEnv, GasEnvConfig
        if name == 'GasEnv':
            return GasEnv
        elif name == 'GasEnvConfig':
            return GasEnvConfig
    elif name == 'get_conflicts_tech':
        from .utils import get_conflicts_tech
        return get_conflicts_tech
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['GasEnvConfig', 'GasEnv', 'CountyDataLoader', 'TechDataLoader', 'get_conflicts_tech']