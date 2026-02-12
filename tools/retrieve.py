from RAG.vector_doc import create_vector_store, VectorManager
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from app_logger import timer
import logging

logger = logging.getLogger(__name__)

class vector_store_args(BaseModel): 
    query: str = Field(..., description="查询的内容")


def ensure_vector_store_initialized():
    """确保向量库和 BM25 已加载（只在首次调用时触发）"""
    if VectorManager.vector_store is None or VectorManager.bm25_retriever is None:
        logger.info("→ 首次调用，正在加载向量库和 BM25 索引...")
        # 注意：这里不传 file_path，因为我们只加载已有数据（re_build=False）
        create_vector_store(re_build=False)

@tool(
    "retrieve_vector_store",
    description="根据输入的查询内容在法律知识库进行相关性检索，检索出相关法律条文。知识库内容主要包括中国现行的各种法律法规等。",
    args_schema=vector_store_args
)
def retrieve_vector_store(query: str):
    """
    混合检索：Milvus 向量检索 + BM25 关键词检索 → RRF 融合
    """
    # ✅ 确保知识库已加载
    ensure_vector_store_initialized()

    # === 1. 向量检索（原生 Milvus）===
    from RAG.base import embeddings_model
    embedder = embeddings_model()
    query_vector = embedder.embed_query(query)

    # 执行向量搜索
    search_params = {"metric_type": "L2", "params": {}}  # FLAT 索引无需参数
    results = VectorManager.vector_store.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=10,  # 可调整
        output_fields=["text", "metadata"]
    )

    faiss_results = []
    for hit in results[0]:  # results[0] 是第一个查询的结果列表
        metadata = hit.entity.get("metadata", {})
        if isinstance(metadata, str):
            import json
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        doc = type('Document', (), {
            'page_content': hit.entity.get("text", ""),
            'metadata': metadata
        })()
        faiss_results.append(doc)

    # === 2. BM25 检索（LangChain，仍可用）===
    bm25_results = VectorManager.bm25_retriever.invoke(query)

    # === 3. 融合 ===
    fused_result = rrf_fusion_optimized(faiss_results, bm25_results)
    return fused_result


def rrf_fusion_optimized(
    faiss_results: list,
    bm25_results: list,
    k: int = 60,
    faiss_weight: float = 0.7,
    bm25_weight: float = 0.3,
    max_results: int = 5
):
    """
    优化的RFF融合版本，性能更好
    """
    from collections import defaultdict

    # 创建文档ID到 (rank, doc) 的映射
    faiss_ranks = {}
    bm25_ranks = {}

    for rank, doc in enumerate(faiss_results, 1):
        doc_id = doc.metadata.get('id')
        if not doc_id:
            doc_id = str(hash(doc.page_content))  # fallback
        faiss_ranks[doc_id] = (rank, doc)

    for rank, doc in enumerate(bm25_results, 1):
        doc_id = doc.metadata.get('id')
        if not doc_id:
            doc_id = str(hash(doc.page_content))
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

    # 返回 top-k 条文文本，用 \n\n 分隔（更清晰）
    contents = [doc.page_content for _, doc in scored_docs[:max_results]]
    return "\n\n".join(contents)