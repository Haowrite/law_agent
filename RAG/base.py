import pathlib
from config import (
    EMBEDDING_API_BASE_URL,
    EMBEDDING_API_BATCH_SIZE,
    EMBEDDING_API_KEY,
    EMBEDDING_API_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    FILE_PATH,
    LOCAL_EMBEDDING_BATCH_SIZE,
    RERANKER_API_BASE_URL,
    RERANKER_API_BATCH_SIZE,
    RERANKER_API_KEY,
    RERANKER_API_TIMEOUT,
    RERANKER_INSTRUCT,
    RERANKER_MODEL,
    RERANKER_PROVIDER,
    LOCAL_RERANKER_USE_FP16,
)
from RAG.embedding_factory import create_embeddings_model
from RAG.reranker_factory import create_reranker_model

# 外部知识库路径
LOAD_DIR = pathlib.Path(FILE_PATH)

def embeddings_model():
    """返回 embedding 函数，支持本地或 API provider。"""
    return create_embeddings_model(
        embedding_provider=EMBEDDING_PROVIDER,
        embedding_model=EMBEDDING_MODEL,
        embedding_api_key=EMBEDDING_API_KEY,
        embedding_api_base_url=EMBEDDING_API_BASE_URL,
        embedding_api_dimensions=EMBEDDING_API_DIMENSIONS,
        embedding_api_batch_size=EMBEDDING_API_BATCH_SIZE,
        local_embedding_batch_size=LOCAL_EMBEDDING_BATCH_SIZE,
    )

def reranker_model():
    """返回 Reranker 模型实例，支持本地或 API provider。"""
    return create_reranker_model(
        reranker_provider=RERANKER_PROVIDER,
        reranker_model=RERANKER_MODEL,
        reranker_api_key=RERANKER_API_KEY,
        reranker_api_base_url=RERANKER_API_BASE_URL,
        reranker_api_batch_size=RERANKER_API_BATCH_SIZE,
        reranker_api_timeout=RERANKER_API_TIMEOUT,
        reranker_instruct=RERANKER_INSTRUCT,
        local_reranker_use_fp16=LOCAL_RERANKER_USE_FP16,
    )

