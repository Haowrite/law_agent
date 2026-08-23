"""
配置文件
包含系统运行所需的各种配置参数
"""

import os
from dotenv import load_dotenv
import pathlib

# 加载环境变量
load_dotenv()


def _optional_int_env(name: str, default: str):
    value = os.getenv(name, default)
    if value == "" or value.lower() == "none":
        return None
    return int(value)


def _select_provider_model(provider: str, api_env: str, local_env: str, legacy_env: str, api_default: str, local_default: str):
    api_model = os.getenv(api_env, os.getenv(legacy_env, api_default))
    local_model = os.getenv(local_env, os.getenv(legacy_env, local_default))
    return local_model if provider == "local" else api_model

# 主模型
MODEL = os.getenv("MAIN_MODEL", "")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
API_KEY = os.getenv("API_KEY", "")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "")  # 摘要模型
TOKENIZER_MODEL = os.getenv("TOKENIZER_MODEL", "")  

# 数据库配置
DATABASE_URL = os.getenv("DATABASE_URL", "")
MILVUS_URL = os.getenv("MILVUS_URL", "")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# 日志记录地址
LOG_DIR = os.getenv("LOG_DIR", "./log")
# 确保日志目录存在
os.makedirs(LOG_DIR, exist_ok=True)

# 系统配置
SYSTEM_NAME = "法律AI助手"
VERSION = "1.0.0"
DOCUMENT_KEY_SECRET = os.getenv("DOCUMENT_KEY_SECRET") or API_KEY or SYSTEM_NAME

# 1. 读取环境变量中的路径
FILE_PATH = os.getenv("FILE_PATH", "")
# 向量库相关配置
VECTOR_COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "law_collection")  # 默认表名
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "api").lower()
EMBEDDING_API_MODEL = os.getenv("EMBEDDING_API_MODEL", os.getenv("EMBEDDING_MODEL", "text-embedding-v4"))
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"))
EMBEDDING_MODEL = _select_provider_model(
    EMBEDDING_PROVIDER,
    "EMBEDDING_API_MODEL",
    "LOCAL_EMBEDDING_MODEL",
    "EMBEDDING_MODEL",
    "text-embedding-v4",
    "BAAI/bge-small-zh-v1.5",
)
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", API_KEY)
EMBEDDING_API_BASE_URL = os.getenv(
    "EMBEDDING_API_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
EMBEDDING_API_DIMENSIONS = _optional_int_env("EMBEDDING_API_DIMENSIONS", "1024")
EMBEDDING_API_BATCH_SIZE = int(os.getenv("EMBEDDING_API_BATCH_SIZE", "32"))
LOCAL_EMBEDDING_BATCH_SIZE = int(os.getenv("LOCAL_EMBEDDING_BATCH_SIZE", "8"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", str(EMBEDDING_API_DIMENSIONS or 1024)))  # Milvus 向量维度
RE_BUILD = os.getenv("RE_BUILD", 'False') == 'True'  # 是否重建向量库
RAG_CACHE_FILE = os.getenv("RAG_CACHE_FILE", "./RAG_DB/cache.json")  # 分割文档缓存路径
# 确保缓存目录存在
os.makedirs(os.path.dirname(RAG_CACHE_FILE), exist_ok=True)

# 重排序模型（增加默认值）
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "True") == "True"
RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "api").lower()
RERANKER_API_MODEL = os.getenv("RERANKER_API_MODEL", os.getenv("RERANKER_MODEL", "qwen3-rerank"))
LOCAL_RERANKER_MODEL = os.getenv("LOCAL_RERANKER_MODEL", os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"))
RERANKER_MODEL = _select_provider_model(
    RERANKER_PROVIDER,
    "RERANKER_API_MODEL",
    "LOCAL_RERANKER_MODEL",
    "RERANKER_MODEL",
    "qwen3-rerank",
    "BAAI/bge-reranker-v2-m3",
)
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY", API_KEY)
RERANKER_API_BASE_URL = os.getenv(
    "RERANKER_API_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-api/v1",
)
RERANKER_API_BATCH_SIZE = int(os.getenv("RERANKER_API_BATCH_SIZE", "500"))
RERANKER_API_TIMEOUT = int(os.getenv("RERANKER_API_TIMEOUT", "60"))
RERANKER_INSTRUCT = os.getenv("RERANKER_INSTRUCT", "")
LOCAL_RERANKER_USE_FP16 = os.getenv("LOCAL_RERANKER_USE_FP16", "True") == "True"



# --- 邮箱 SMTP 配置（QQ 邮箱） ---
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.qq.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "")            # 你的 QQ 邮箱地址，如 123456@qq.com
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")     # QQ 邮箱的 SMTP 授权码（非登录密码）
SMTP_SENDER_NAME = os.getenv("SMTP_SENDER_NAME", "AI法律助手")

# --- 验证码有效期（秒） ---
VERIFY_CODE_EXPIRE = int(os.getenv("VERIFY_CODE_EXPIRE", "300"))  # 默认 5 分钟
