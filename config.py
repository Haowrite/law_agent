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
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
API_KEY = os.getenv("API_KEY", "")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "")  #摘要模型
TOKENIZER_MODEL = os.getenv("TOKENIZER_MODEL", "")  

#数据库配置
DATABASE_URL = os.getenv("DATABASE_URL", "")
MILVUS_URL = os.getenv("MILVUS_URL", "")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# 日志记录地址
LOG_DIR = os.getenv("LOG_DIR", "./log")

# 系统配置
SYSTEM_NAME = "法律AI助手"
VERSION = "1.0.0"


#本地知识库路径
BASE_PATH = os.getenv("__BASE_PATH", "")
FILE_PATH = os.getenv("__FILE_PATH", "")  # 本地知识库路径
VECTOR_COLLECTION_NAME = os.getenv("vector_collection", "")  # 向量集合名
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")  #嵌入模型
EMBED_DIM = int(os.getenv("EMBED_DIM", ""))     #嵌入维度
RE_BUILD = os.getenv("RE_BUILD", 'False') == 'True'  #是否重建向量库