# utils/retrieve_process.py

from RAG.vector_doc import create_vector_store, VectorManager
from RAG.base import embeddings_model, reranker_model  # 新增 reranker_model
from app_logger import database_logger as logger
import json
from config import RE_BUILD, FILE_PATH
import os
from typing import List, Any, Tuple, Dict

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
    :return: 重排后的文档列表 [(reranker_score, doc), ...]
    """
    if not candidate_docs:
        return []

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

    return final_docs


def fetch_full_articles_from_milvus(collection, reranked_docs: List[Tuple[float, Any]]) -> str:
    """
    根据 Reranker 排序结果，从 Milvus 中检索完整条文并拼接返回。

    流程：
    1. 从 reranked_docs 中提取唯一的 (filename, article) 对
    2. 对每个 (filename, article) 对，使用 Milvus 表达式检索所有对应的 chunk
    3. 按 start_position 排序并拼接成完整条文
    4. 去重后返回拼接结果

    :param collection: Milvus Collection 对象
    :param reranked_docs: Reranker 排序后的文档列表 [(score, doc), ...]
    :return: 拼接后的完整条文文本
    """
    if not reranked_docs:
        return ""

    # 1. 提取唯一的 (filename, article) 对，保持 reranker 排序顺序
    seen_keys = set()
    unique_article_keys = []  # [(filename, article), ...] 保持排序顺序
    for _, doc in reranked_docs:
        filename = doc.metadata.get("filename", "")
        article = doc.metadata.get("article", "")
        # 如果 article 含有 _part 后缀（如 "第一条_part1"），提取原始条号
        base_article = article.split("_part")[0] if "_part" in article else article
        key = (filename, base_article)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_article_keys.append(key)

    logger.info(f"Reranker 结果涉及 {len(unique_article_keys)} 个唯一条文，开始从 Milvus 检索完整内容...")

    # 2. 对每个唯一 (filename, article) 对，从 Milvus 查询所有相关 chunk
    full_articles = []  # [(filename, article, full_text), ...]

    for filename, base_article in unique_article_keys:
        # 构造 Milvus 表达式：精确匹配 filename，article 前缀匹配（兼容 _part 后缀）
        # 使用 filename 字段和 article 字段进行表达式检索
        # article 字段可能是 "第一条" 或 "第一条_part1"，需要匹配所有以 base_article 开头的
        escaped_filename = filename.replace('"', '\\"')
        escaped_article = base_article.replace('"', '\\"')

        # 精确匹配 filename，article 以 base_article 开头（使用 prefix 或 like）
        expr = f'filename == "{escaped_filename}" and (article == "{escaped_article}" or article like "{escaped_article}_part%")'

        try:
            results = collection.query(
                expr=expr,
                output_fields=["text", "article", "start_position", "metadata"],
                limit=100  # 单条法规的 chunk 数不会太多
            )
        except Exception as e:
            logger.warning(f"Milvus 查询失败 (filename={filename}, article={base_article}): {e}")
            continue

        if not results:
            logger.warning(f"Milvus 未找到条文: filename={filename}, article={base_article}")
            continue

        # 3. 去重（按 text 内容去重）并按 start_position 排序
        seen_texts = set()
        unique_chunks = []
        for r in results:
            text = r.get("text", "")
            if text and text not in seen_texts:
                seen_texts.add(text)
                start_pos = r.get("start_position", 0)
                unique_chunks.append((start_pos, text))

        # 按 start_position 升序排列
        unique_chunks.sort(key=lambda x: x[0])

        # 4. 拼接成完整条文
        full_text = "".join([chunk_text for _, chunk_text in unique_chunks])
        full_articles.append((filename, base_article, full_text))
        logger.debug(f"条文拼接完成: {filename} {base_article} -> {len(unique_chunks)} 个 chunk，总长 {len(full_text)} 字符")

    # 5. 构造最终返回文本
    contents = []
    for filename, article, full_text in full_articles:
        contents.append(
            f"{full_text}（法律来源：{filename}{article}）"
        )

    result_text = "\n".join(contents)
    logger.info(f"最终返回 {len(full_articles)} 条完整条文")
    return result_text


def init_and_retrieve(query: str) -> str:
    """
    保留原有单条接口，供非批处理场景或测试使用。
    内部调用批量逻辑的单条版本。
    """
    results = batch_init_and_retrieve([query])
    return results[0]


def batch_init_and_retrieve(queries: List[str]) -> List[str]:
    """
    批量处理核心函数：
    1. 批量 Embedding (GPU 加速)
    2. 批量 Vector Search (Milvus 加速)
    3. 循环处理 BM25 和 RRF (CPU 密集)
    4. Reranker 重排
    5. 根据 Reranker 结果从 Milvus 检索完整条文并拼接返回
    """
    manager = init_vector_manager_once()
    model = get_embedding_model()

    if not queries:
        return []

    # 1. 批量 Embedding (关键优化点：一次 GPU 推理)
    query_vectors = model.embed_documents(queries)

    # 2. 批量 Vector Search (关键优化点：一次网络 IO，一次索引扫描)
    search_params = {"metric_type": "L2", "params": {"nprobe": 20}}

    milvus_results = manager.vector_store.search(
        data=query_vectors,
        anns_field="vector",
        param=search_params,
        limit=10,
        output_fields=["text", "metadata", "filename", "article", "start_position"]
    )

    final_results = []

    # 3. 后处理 (BM25 + RRF + Reranker + 完整条文检索)
    for i, query in enumerate(queries):
        # 解析 Milvus 结果
        hits = milvus_results[i]
        faiss_docs = []
        for hit in hits:
            metadata = hit.entity.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = {}

            # 确保 metadata 中包含 filename, article, start_position
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

        # 执行 BM25 (单条)
        bm25_docs = manager.bm25_retriever.invoke(query)

        # RRF 融合（筛选出候选集）
        rrf_candidates = rrf_fusion_optimized(faiss_docs, bm25_docs, rrf_top_k=10)

        # Reranker 重排（返回排序后的文档列表）
        reranked_docs = reranker_reorder(query, rrf_candidates, max_results=5)

        # 【核心修改】从 Milvus 检索完整条文并拼接
        fused_text = fetch_full_articles_from_milvus(manager.vector_store, reranked_docs)

        final_results.append(fused_text)

    return final_results