"""
配置文件
包含系统运行所需的各种配置参数
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

#主模型
MODEL = os.getenv("MAIN_MODEL", "")
TEMPERATURE = os.getenv("TEMPERATURE", 0.1)
API_KEY = os.getenv("API_KEY", "")

#数据配置
DATABASE_URL = os.getenv("DATABASE_URL", "")
MILVUS_URL = os.getenv("MILVUS_URL", "")
# 日志记录地址
LOG_DIR = os.getenv("LOG_DIR", "./log")

# HTTP请求配置
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))
HTTP_MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "3"))
HTTP_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "MultiAgentCustomerService/1.0.0"
}

# 系统配置
SYSTEM_NAME = "多智能体客服系统"
VERSION = "1.0.0"

#本地知识库路径
BASE_PATH = os.getenv("__BASE_PATH", "")
FILE_PATH = os.getenv("__FILE_PATH", "")  # 本地知识库路径
VECTOR_COLLECTION_NAME = os.getenv("vector_collection", "")  # 向量集合名
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")  #嵌入模型
EMBED_DIM = os.getenv("EMBED_DIM", "")     #嵌入维度
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "")  #摘要模型
RE_BUILD = os.getenv("RE_BUILD", 'False') == 'True'  #是否重建向量库