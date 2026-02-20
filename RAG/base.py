# base.py
import os
import pathlib
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, FILE_PATH

# 外部知识库路径
LOAD_DIR = pathlib.Path(FILE_PATH)

def embeddings_model():
    """返回 embedding 函数，支持 batch"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True, "batch_size": 8}
    )

# m_embedding_model = embeddings_model()