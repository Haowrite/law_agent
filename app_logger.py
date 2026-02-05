from loguru import logger
import os
from config import LOG_DIR
import sys
from typing import Optional
import functools
import time

# 记录已配置的logger
_configured_loggers = set()


def setup_logger(name: str, console_level: str = "INFO") -> logger:
    """
    配置并返回一个异步日志记录器
    """
    global _configured_loggers
    
    # 确保日志目录存在
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 定义日志格式
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{file}</cyan>:<cyan>{line}</cyan> | "
        "<magenta>{function}</magenta> | "
        "<level>{message}</level>"
    )
    
    # 绑定特定类型的logger
    api_logger = logger.bind(type=name)
    
    # 只在第一次调用时添加控制台handler
    if not _configured_loggers:
        # 移除默认的handler
        logger.remove()
        
        # 添加控制台handler
        logger.add(
            sys.stdout,
            level=console_level,
            format=log_format,
            colorize=True,
            enqueue=True,
            backtrace=True,
            diagnose=True
        )
    
    # 为每个logger单独添加文件handler
    if name not in _configured_loggers:
        logger.add(
            os.path.join(LOG_DIR, f"{name}.log"),
            rotation="00:00",
            retention="10 days",
            compression="zip",
            level="DEBUG",
            format=log_format,
            encoding="utf-8",
            enqueue=True,
            filter=lambda record: record["extra"].get("type") == name
        )
        _configured_loggers.add(name)
    
    return api_logger

# 创建不同的日志记录器实例
app_logger = setup_logger("app")
llm_logger = setup_logger("llm")
database_logger = setup_logger("database")
run_time_logger = setup_logger("run_time")

def timer(node_name):
    """计时装饰器，记录节点执行时间并异步写入loguru日志"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            run_time_logger.info(f"[{node_name}] 开始执行")
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            run_time_logger.info(f"[{node_name}] 执行完成，耗时: {execution_time:.2f}秒")
            
            return result
        return wrapper
    return decorator