# main.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    # 创建日志记录器
    logger = logging.getLogger("apiServer")
    logger.setLevel(logging.DEBUG)  # 设置全局日志级别

    # 创建日志格式
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 创建文件处理器（支持日志滚动）
    file_handler = RotatingFileHandler("log/app.log", maxBytes=1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    return logger
