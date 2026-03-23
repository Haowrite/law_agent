# base.py
import os
import pathlib
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, FILE_PATH, RERANKER_MODEL

# 外部知识库路径
LOAD_DIR = pathlib.Path(FILE_PATH)

def embeddings_model():
    """返回 embedding 函数，支持 batch"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True, "batch_size": 8}
    )

def reranker_model():
    """返回 Reranker 模型实例"""
    from FlagEmbedding import FlagReranker
    # 优先使用配置文件中的模型，未配置则使用默认
    model_path = RERANKER_MODEL if RERANKER_MODEL else 'BAAI/bge-reranker-v2-m3'
    reranker = FlagReranker(model_path, use_fp16=True)
    return reranker

