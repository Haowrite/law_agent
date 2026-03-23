# utils/retrieve_process.py

from RAG.vector_doc import create_vector_store, VectorManager
from RAG.base import embeddings_model, reranker_model  # 新增 reranker_model
from app_logger import database_logger as logger
import json
from config import RE_BUILD, FILE_PATH
import os
from typing import List, Any, Tuple

# 全局变量 (在每个子进程中独立存在)
_EMBEDDING_MODEL = None
_VECTORMANAGER: VectorManager = None
_RERANKER_MODEL = None  # 新增 Reranker 全局变量

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

def get_reranker_model():
    """获取 Reranker 模型实例（单例）"""
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        logger.info(f"[子进程{os.getpid()}] 加载 Reranker 模型...")
        _RERANKER_MODEL = reranker_model()
        logger.info(f"[子进程{os.getpid()}] Reranker 模型加载完成.")
    return _RERANKER_MODEL

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

def rrf_fusion_optimized(faiss_results, bm25_results, k=60, faiss_weight=0.7, bm25_weight=0.3, rrf_top_k=20):
    """
    RRF 融合优化版：先筛选出 rrf_top_k 条结果，供后续 Reranker 重排
    """
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

    # RRF 阶段先筛选出 top N 条（默认20条），减少后续 Reranker 计算量
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return scored_docs[:rrf_top_k]

def reranker_reorder(query: str, candidate_docs, max_results=10):
    """
    使用 Reranker 模型对候选文档重排
    :param query: 用户查询
    :param candidate_docs: RRF 筛选后的候选文档列表 [(rrf_score, doc), ...]
    :param max_results: 最终返回的文档数量
    :return: 重排后的文档列表、拼接后的文本结果
    """
    if not candidate_docs:
        return [], ""
    
    # 获取 Reranker 模型
    reranker = get_reranker_model()
    
    # 构造 [query, passage] 对
    query_passage_pairs = []
    doc_list = [doc for _, doc in candidate_docs]
    for doc in doc_list:
        query_passage_pairs.append([query, doc.page_content])
    
    # 计算 Reranker 分数（归一化到 0-1）
    scores = reranker.compute_score(query_passage_pairs, normalize=True)
    
    # 结合分数和文档排序
    scored_docs = list(zip(scores, doc_list))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # 截取最终需要的数量
    final_docs = scored_docs[:max_results]
    
    # 构造返回文本
    contents = [
        doc.page_content + f"（法律来源：{doc.metadata.get('filename', '未知来源')}{doc.metadata.get('article', '')}）"
        for _, doc in final_docs
    ]
    return final_docs, "\n".join(contents)

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
    4. Reranker 重排 (最终排序)
    """
    manager = init_vector_manager_once()
    model = get_embedding_model()
    
    if not queries:
        return []

    # 1. 批量 Embedding (关键优化点：一次 GPU 推理)
    query_vectors = model.embed_documents(queries)

    # 2. 批量 Vector Search (关键优化点：一次网络 IO，一次索引扫描)
    search_params = {"metric_type": "L2", "params": {"nprobe": 20}} # 可根据需要调整 nprobe
    
    # data 接收 List[List[float]]，返回 List[Hits]
    milvus_results = manager.vector_store.search(
        data=query_vectors,
        anns_field="vector",
        param=search_params,
        limit=20,
        output_fields=["text", "metadata"]
    )

    final_results = []

    # 3. 后处理 (BM25 + RRF + Reranker)
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
        
        # RRF 融合（筛选出候选集）
        rrf_candidates = rrf_fusion_optimized(faiss_docs, bm25_docs, rrf_top_k=10)
        
        # Reranker 重排（最终排序）
        _, fused_text = reranker_reorder(query, rrf_candidates, max_results=5)
        
        final_results.append(fused_text)

    return final_results