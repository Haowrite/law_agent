# utils/retriever_process.py （新建文件，确保可被 pickle）

from RAG.vector_doc import create_vector_store, VectorManager
from RAG.base import embeddings_model
from app_logger import database_logger as logger
import json
from config import  RE_BUILD, FILE_PATH

_EMBEDDING_MODEL = None

def get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = embeddings_model()
    return _EMBEDDING_MODEL



def init_and_retrieve(query: str) -> str:
    """
    在子进程中执行：初始化向量库 + 执行检索 + 返回融合结果
    此函数必须是顶层函数，不能依赖闭包或不可序列化的对象
    """
    m_embedding_model = get_embedding_model()

    # 每个子进程首次调用时初始化
    if VectorManager.vector_store is None or VectorManager.bm25_retriever is None:
        logger.info("[子进程] 加载向量库和 BM25 索引...")
        create_vector_store(m_embedding_model, file_path=FILE_PATH, re_build=RE_BUILD)

    # 执行嵌入
    query_vector = m_embedding_model.embed_query(query)
    search_params = {"metric_type": "L2", "params": {}}
    results = VectorManager.vector_store.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=10,
        output_fields=["text", "metadata"]
    )

    faiss_results = []
    for hit in results[0]:
        metadata = hit.entity.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        doc = type('Document', (), {
            'page_content': hit.entity.get("text", ""),
            'metadata': metadata
        })()
        faiss_results.append(doc)

    bm25_results = VectorManager.bm25_retriever.invoke(query)

    return rrf_fusion_optimized(faiss_results, bm25_results)


def rrf_fusion_optimized(faiss_results, bm25_results, k=60, faiss_weight=0.7, bm25_weight=0.3, max_results=10):
    from collections import defaultdict
    faiss_ranks = {}
    bm25_ranks = {}

    for rank, doc in enumerate(faiss_results, 1):
        doc_id = doc.metadata.get('id') or str(hash(doc.page_content))
        faiss_ranks[doc_id] = (rank, doc)

    for rank, doc in enumerate(bm25_results, 1):
        doc_id = doc.metadata.get('id') or str(hash(doc.page_content))
        bm25_ranks[doc_id] = (rank, doc)

    all_doc_ids = set(faiss_ranks.keys()) | set(bm25_ranks.keys())
    scored_docs = []

    for doc_id in all_doc_ids:
        doc = faiss_ranks.get(doc_id, bm25_ranks.get(doc_id))[1]
        faiss_rank = faiss_ranks[doc_id][0] if doc_id in faiss_ranks else float('inf')
        bm25_rank = bm25_ranks[doc_id][0] if doc_id in bm25_ranks else float('inf')

        faiss_score = faiss_weight * (1 / (k + faiss_rank)) if faiss_rank != float('inf') else 0
        bm25_score = bm25_weight * (1 / (k + bm25_rank)) if bm25_rank != float('inf') else 0

        total_score = faiss_score + bm25_score
        scored_docs.append((total_score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    contents = [
        doc.page_content + f"（法律来源：{doc.metadata.get('filename', '未知来源')}{doc.metadata.get('article', '')}）"
        for _, doc in scored_docs[:max_results]
    ]
    return "\n".join(contents)