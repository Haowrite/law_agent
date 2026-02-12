from RAG.vector_doc import create_vector_store
from langchain_core.tools import tool

from pydantic import BaseModel, Field
from RAG.vector_doc import VectorManager
from app_logger import timer

class vector_store_args(BaseModel): 
    query: str = Field(..., description="查询的内容")

@tool("retrieve_vector_store", description="根据输入的查询内容在法律知识库进行相关性检索, 检索出相关法律条文。知识库内容主要包括中国现行的各种法律法规等。", args_schema=vector_store_args)
def retrieve_vector_store(query: str):
    #保证本地知识库已经建立
    if VectorManager.ids is None:
        VectorManager.ids = create_vector_store()

    faiss_res = VectorManager.vector_retriever.invoke(input = query)
    bm_res = VectorManager.bm25_retriever.invoke(input = query)

    return rrf_fusion_optimized(faiss_res, bm_res)


def rrf_fusion_optimized(faiss_results: list, bm25_results: list, k: int = 60,
                         faiss_weight: float = 0.7, bm25_weight: float = 0.3,
                         max_results: int = 5):
    """
    优化的RFF融合版本，性能更好
    """
    # 创建文档ID到排名的映射
    faiss_ranks = {}
    bm25_ranks = {}

    # 记录FAISS结果的排名
    for rank, doc in enumerate(faiss_results, 1):
        doc_id = doc.metadata.get('id', str(hash(doc.page_content)))
        faiss_ranks[doc_id] = (rank, doc)

    # 记录BM25结果的排名
    for rank, doc in enumerate(bm25_results, 1):
        doc_id = doc.metadata.get('id', str(hash(doc.page_content)))
        bm25_ranks[doc_id] = (rank, doc)

    # 计算所有文档的RFF得分
    all_doc_ids = set(faiss_ranks.keys()) | set(bm25_ranks.keys())
    scored_docs = []

    for doc_id in all_doc_ids:
        # 获取文档对象（优先使用FAISS的，因为可能包含更多元数据）
        if doc_id in faiss_ranks:
            doc = faiss_ranks[doc_id][1]
        else:
            doc = bm25_ranks[doc_id][1]

        # 计算RFF得分
        faiss_rank = faiss_ranks[doc_id][0] if doc_id in faiss_ranks else float('inf')
        bm25_rank = bm25_ranks[doc_id][0] if doc_id in bm25_ranks else float('inf')

        # 如果文档在某个列表中不存在，使用一个很大的排名值
        faiss_score = faiss_weight * (1 / (k + faiss_rank)) if faiss_rank != float('inf') else 0
        bm25_score = bm25_weight * (1 / (k + bm25_rank)) if bm25_rank != float('inf') else 0

        total_score = faiss_score + bm25_score
        scored_docs.append((total_score, doc))

    # 按得分排序并返回前N个结果
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return '/n'.join([doc.page_content for score, doc in scored_docs[:max_results]])