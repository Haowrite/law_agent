from langchain_milvus import Milvus
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
 #将模型的输出按流的方式打印到终端
import os
import pathlib
from openai import vector_stores
from config import *
import faiss


#外部知识库路径
LOAD_DIR = pathlib.Path(FILE_PATH)

#向量数据库的存储路径，相对于main.py的路径
VECTOR_DIR = os.path.join("vector_store")

#向量数据库集合名
COLLECTION_NAME = VECTOR_COLLECTION_NAME

def embeddings_model():
    embeddings = HuggingFaceEmbeddings(model = EMBEDDING_MODEL,
                                        encode_kwargs = {"normalize_embeddings": True}
                                       )
    return embeddings


def milvus_vector_store():
    vector_store = Milvus(
        embedding_function=embeddings_model(),
        collection_name='vector_store',
        collection_description='本地知识库',
        connection_args={"uri": MILVUS_URL},
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )

    return vector_store


