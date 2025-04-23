import logging
import os
import sys

def setup_logger(name="cvdproc", log_file=None, level=logging.DEBUG):
    """
    设置并返回一个统一的 logger 对象。
    
    :param name: str, logger 名称
    :param log_file: str, 日志文件路径（可选，如果提供，则日志也会输出到文件）
    :param level: logging.LEVEL, 日志级别（默认 DEBUG）
    :return: logging.Logger 对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 定义日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # 控制台日志 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件日志 handler（如果提供了日志文件路径）
    if log_file:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# 全局 logger 配置（可以直接使用）
logger = setup_logger()
