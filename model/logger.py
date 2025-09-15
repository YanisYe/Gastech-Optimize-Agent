import logging
import os

def setup_logger(output=True, name="tmp"):

    logger = logging.getLogger(name)
    
    logger.setLevel(logging.INFO)
    # logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if output:
        # logger 存储在log文件夹下
        if not os.path.exists('log'):
            os.makedirs('log')

        file_handler = logging.FileHandler(filename=f'log/{name}.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
