"""
子进程检索逻辑
"""

from RAG.vector_doc import create_vector_store, VectorManager
from RAG.base import embeddings_model, reranker_model
from RAG.rerank_control import rerank_or_passthrough
from app_logger import database_logger as logger
import json
from config import ENABLE_RERANKER, RE_BUILD, FILE_PATH
import os
from typing import List, Any, Tuple, Dict
from RAG.evidence import build_evidence_item

# 全局变量 (在每个子进程中独立存在)
_EMBEDDING_MODEL = None
_VECTORMANAGER = None
_RERANKER_MODEL = None


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


# ========== 文档 CRUD（在子进程中执行） ==========

def add_document_in_process(doc_abs_path: str) -> dict:
    """
    在子进程中新增文档（确保 embedding 模型已加载）
    返回: {"inserted_count": int, "message": str}
    """
    from db_crud.doc_article_crud import add_document

    manager = init_vector_manager_once()
    model = get_embedding_model()

    result = add_document(
        doc_abs_path=doc_abs_path,
        embedding_model=model,
        vector_manager=manager,
    )
    return result


def delete_document_in_process(doc_abs_path: str) -> dict:
    """
    在子进程中删除文档
    返回: {"deleted_count": int, "message": str}
    """
    from db_crud.doc_article_crud import delete_document, _rebuild_bm25
    from pymilvus import connections, Collection
    from config import MILVUS_URL, VECTOR_COLLECTION_NAME, RAG_CACHE_FILE
    from RAG.cache_sync import refresh_rag_cache_from_milvus
    from RAG.document_library import remove_file_if_exists, resolve_document_path
    from RAG.vector_doc import save_docs_to_cache

    manager = init_vector_manager_once()

    result = delete_document(doc_abs_path=doc_abs_path)
    file_deleted = False

    # 删除后重建 BM25
    if result.get("deleted_count", 0) > 0 and manager is not None:
        try:
            connections.connect(uri=MILVUS_URL)
            collection = Collection(VECTOR_COLLECTION_NAME)
            collection.load()
            _rebuild_bm25(manager, collection)
            refresh_rag_cache_from_milvus(collection, RAG_CACHE_FILE, save_docs_to_cache)
        except Exception as e:
            logger.warning(f"删除后重建 BM25 失败: {e}")

    try:
        safe_path = resolve_document_path(FILE_PATH, doc_abs_path)
        file_deleted = remove_file_if_exists(safe_path)
    except Exception as e:
        logger.warning(f"删除物理文件失败: {e}")

    result["file_deleted"] = file_deleted
    return result


# ========== 以下为检索逻辑 ==========

def _filter_docs_by_exclude_ids(docs: list, exclude_ids: set) -> list:
    """
    根据已检索条文ID集合，从初步检索结果中剔除已检索过的条文。
    条文ID格式: "{filename}::{base_article}"
    """
    if not exclude_ids:
        return docs
    filtered = []
    for doc in docs:
        filename = doc.metadata.get("filename", "")
        article = doc.metadata.get("article", "")
        base_article = article.split("_part")[0] if "_part" in article else article
        article_id = f"{filename}::{base_article}"
        if article_id not in exclude_ids:
            filtered.append(doc)
    return filtered


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

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return scored_docs[:rrf_top_k]


def reranker_reorder(query: str, candidate_docs, max_results=10):
    """
    使用 Reranker 模型对候选文档重排
    """
    return rerank_or_passthrough(
        query=query,
        candidate_docs=candidate_docs,
        max_results=max_results,
        enabled=ENABLE_RERANKER,
        get_reranker=get_reranker_model,
    )


def fetch_full_articles_from_milvus(collection, reranked_docs: List[Tuple[float, Any]]) -> Tuple[str, List[str], List[dict]]:
    """
    根据 Reranker 排序结果，从 Milvus 中检索完整条文并拼接返回。
    同时返回本次检索到的条文 ID 列表，用于后续去重。
    返回: (拼接后的条文文本, 本次检索到的条文ID列表)
    """
    if not reranked_docs:
        return "", [], []

    seen_keys = set()
    unique_article_keys = []
    for _, doc in reranked_docs:
        filename = doc.metadata.get("filename", "")
        article = doc.metadata.get("article", "")
        base_article = article.split("_part")[0] if "_part" in article else article
        key = (filename, base_article)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_article_keys.append(key)

    logger.info(f"Reranker 结果涉及 {len(unique_article_keys)} 个唯一条文，开始从 Milvus 检索完整内容...")

    full_articles = []

    for filename, base_article in unique_article_keys:
        escaped_filename = filename.replace('"', '\\"')
        escaped_article = base_article.replace('"', '\\"')

        expr = f'filename == "{escaped_filename}" and (article == "{escaped_article}" or article like "{escaped_article}_part%")'

        try:
            results = collection.query(
                expr=expr,
                output_fields=["text", "article", "start_position", "metadata"],
                limit=100
            )
        except Exception as e:
            logger.warning(f"Milvus 查询失败 (filename={filename}, article={base_article}): {e}")
            continue

        if not results:
            logger.warning(f"Milvus 未找到条文: filename={filename}, article={base_article}")
            continue

        seen_texts = set()
        unique_chunks = []
        first_metadata = {}
        for r in results:
            text = r.get("text", "")
            if text and text not in seen_texts:
                seen_texts.add(text)
                start_pos = r.get("start_position", 0)
                unique_chunks.append((start_pos, text))
                if not first_metadata:
                    first_metadata = r.get("metadata", {}) or {}

        unique_chunks.sort(key=lambda x: x[0])

        full_text = "".join([chunk_text for _, chunk_text in unique_chunks])
        full_articles.append((filename, base_article, full_text, first_metadata))
        logger.debug(f"条文拼接完成: {filename} {base_article} -> {len(unique_chunks)} 个 chunk，总长 {len(full_text)} 字符")

    # 收集本次检索到的条文ID
    new_retrieved_ids = []
    contents = []
    evidences = []
    for citation_index, (filename, article, full_text, metadata) in enumerate(full_articles, start=1):
        article_id = f"{filename}::{article}"
        new_retrieved_ids.append(article_id)
        contents.append(
            f"{full_text}"
        )
        evidence_metadata = {
            **(metadata if isinstance(metadata, dict) else {}),
            "filename": filename,
            "article": article,
        }
        evidences.append(build_evidence_item(citation_index, full_text, evidence_metadata))

    result_text = "\n\n".join(contents)
    logger.info(f"最终返回 {len(full_articles)} 条完整条文，新增 {len(new_retrieved_ids)} 个条文ID")
    return result_text, new_retrieved_ids, evidences


def init_and_retrieve(query: str, exclude_ids: set = None) -> Tuple[str, List[str], List[dict]]:
    """
    保留原有单条接口，供非批处理场景或测试使用。
    返回: (检索结果文本, 本次检索到的条文ID列表)
    """
    results = batch_init_and_retrieve([query], exclude_ids_list=[exclude_ids])
    return results[0]


def batch_init_and_retrieve(queries: List[str], exclude_ids_list: List[set] = None) -> List[Tuple[str, List[str], List[dict]]]:
    """
    批量处理核心函数
    参数:
        queries: 查询列表
        exclude_ids_list: 每个查询对应的已检索条文ID集合，用于去重。长度需与 queries 一致，为 None 时不做排除。
    返回: List[Tuple[str, List[str]]]，每个元素为 (检索结果文本, 本次新增的条文ID列表)
    """
    manager = init_vector_manager_once()
    model = get_embedding_model()
    reranker = get_reranker_model()
    if not queries:
        return []

    if exclude_ids_list is None:
        exclude_ids_list = [None] * len(queries)

    # 1. 批量 Embedding
    query_vectors = model.embed_documents(queries)

    # 2. 批量 Vector Search
    search_params = {"metric_type": "L2", "params": {"nprobe": 20}}

    milvus_results = manager.vector_store.search(
        data=query_vectors,
        anns_field="vector",
        param=search_params,
        limit=20,
        output_fields=["text", "metadata", "filename", "article", "start_position"]
    )

    final_results = []

    # 3. 后处理 (BM25 + RRF + Reranker + 完整条文检索)
    for i, query in enumerate(queries):
        hits = milvus_results[i]
        faiss_docs = []
        for hit in hits:
            metadata = hit.entity.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = {}

            if "filename" not in metadata:
                metadata["filename"] = hit.entity.get("filename", "")
            if "article" not in metadata:
                metadata["article"] = hit.entity.get("article", "")
            if "start_position" not in metadata:
                metadata["start_position"] = hit.entity.get("start_position", 0)

            doc = type('Document', (), {
                'page_content': hit.entity.get("text", ""),
                'metadata': metadata
            })()
            faiss_docs.append(doc)

        bm25_docs = manager.bm25_retriever.invoke(query)

        # 在 RRF 融合之前，根据已检索条文ID过滤掉重复条文
        current_exclude_ids = exclude_ids_list[i]
        if current_exclude_ids:
            orig_faiss_count = len(faiss_docs)
            orig_bm25_count = len(bm25_docs)
            faiss_docs = _filter_docs_by_exclude_ids(faiss_docs, current_exclude_ids)
            bm25_docs = _filter_docs_by_exclude_ids(bm25_docs, current_exclude_ids)
            logger.info(f"去重过滤: faiss {orig_faiss_count}->{len(faiss_docs)}, bm25 {orig_bm25_count}->{len(bm25_docs)}, 排除 {len(current_exclude_ids)} 个已检索条文")

        rrf_candidates = rrf_fusion_optimized(faiss_docs, bm25_docs, rrf_top_k=10)

        reranked_docs = reranker_reorder(query, rrf_candidates, max_results=6)

        fused_text, new_ids, evidences = fetch_full_articles_from_milvus(manager.vector_store, reranked_docs)

        final_results.append((fused_text, new_ids, evidences))

    return final_results
