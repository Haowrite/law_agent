# utils/retrieve_process.py

from RAG.vector_doc import create_vector_store, VectorManager
from RAG.base import embeddings_model
from app_logger import database_logger as logger
import json
from config import RE_BUILD, FILE_PATH
import os
from typing import List, Any, Tuple

# 全局变量 (在每个子进程中独立存在)
_EMBEDDING_MODEL = None
_VECTORMANAGER: VectorManager = None

def get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = embeddings_model()
        # 兼容性检查：如果模型只有 embed_query，包装成 embed_documents
        if not hasattr(_EMBEDDING_MODEL, 'embed_documents'):
            single_embed = _EMBEDDING_MODEL.embed_query
            def batch_embed(texts: List[str]) -> List[List[float]]:
                return [single_embed(t) for t in texts]
            _EMBEDDING_MODEL.embed_documents = batch_embed
            logger.warning("Model wrapped with batch embed function.")
    return _EMBEDDING_MODEL

def init_vector_manager_once():
    """在子进程中初始化一次向量库"""
    global _VECTORMANAGER
    if _VECTORMANAGER is None:
        logger.info(f"[子进程{os.getpid()}] 加载向量库和 BM25 索引...")
        _VECTORMANAGER = VectorManager()
        m_model = get_embedding_model()
        create_vector_store(m_model, vector_manager=_VECTORMANAGER, file_path=FILE_PATH, re_build=RE_BUILD)
        logger.info(f"[子进程{os.getpid()}] 向量库加载完成.")
    return _VECTORMANAGER

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

def init_and_retrieve(query: str) -> str:
    """
    保留原有单条接口，供非批处理场景或测试使用。
    内部调用批量逻辑的单条版本。
    """
    results = batch_init_and_retrieve([query])
    return results[0]

def batch_init_and_retrieve(queries: List[str]) -> List[str]:
    """
    【新增】批量处理核心函数
    1. 批量 Embedding (GPU 加速)
    2. 批量 Vector Search (Milvus 加速)
    3. 循环处理 BM25 和 RRF (CPU 密集，可后续加线程池)
    """
    manager = init_vector_manager_once()
    model = get_embedding_model()
    
    if not queries:
        return []

    # 1. 批量 Embedding (关键优化点：一次 GPU 推理)
    # 假设 model.embed_documents 接受 List[str] 返回 List[List[float]]
    query_vectors = model.embed_documents(queries)

    # 2. 批量 Vector Search (关键优化点：一次网络 IO，一次索引扫描)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}} # 可根据需要调整 nprobe
    
    # data 接收 List[List[float]]，返回 List[Hits]
    milvus_results = manager.vector_store.search(
        data=query_vectors,
        anns_field="vector",
        param=search_params,
        limit=10,
        output_fields=["text", "metadata"]
    )

    final_results = []

    # 3. 后处理 (BM25 + RRF)
    # 注意：BM25Retriever.invoke 通常是单条的。
    # 如果数据量大，这里可以用 ThreadPoolExecutor 并行处理 BM25
    for i, query in enumerate(queries):
        # 解析 Milvus 结果
        hits = milvus_results[i]
        faiss_docs = []
        for hit in hits:
            metadata = hit.entity.get("metadata", {})
            if isinstance(metadata, str):
                try: metadata = json.loads(metadata)
                except: metadata = {}
            
            doc = type('Document', (), {
                'page_content': hit.entity.get("text", ""),
                'metadata': metadata
            })()
            faiss_docs.append(doc)
        
        # 执行 BM25 (单条)
        bm25_docs = manager.bm25_retriever.invoke(query)
        
        # RRF 融合
        fused_text = rrf_fusion_optimized(faiss_docs, bm25_docs)
        final_results.append(fused_text)

    return final_results