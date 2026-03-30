"""
文档-条文 CRUD 操作（同步版本，在子进程中执行）
所有操作同时维护 MySQL doc_article 表和 Milvus 向量库
"""

import os
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session
from pymilvus import Collection, connections, utility

from app_logger import database_logger as logger
from db_crud.base import get_sync_session
from db_crud.doc_article_model import DocArticle
from RAG.retrieve_process import _VECTORMANAGER
from RAG.vector_doc import create_vector_store, VectorManager
from RAG.retrieve_process import get_embedding_model
from config import FILE_PATH, RE_BUILD

# ========== 内部工具函数 ==========
def _get_milvus_collection() -> Collection:
    """获取 Milvus Collection"""
    global _VECTORMANAGER
    if _VECTORMANAGER is None:
        logger.info(f"[子进程{os.getpid()}] 加载向量库和 BM25 索引...")
        _VECTORMANAGER = VectorManager()
        m_model = get_embedding_model()
        create_vector_store(m_model, vector_manager=_VECTORMANAGER, file_path=FILE_PATH, re_build=RE_BUILD)
        logger.info(f"[子进程{os.getpid()}] 向量库加载完成.")
    return _VECTORMANAGER.vector_store


# ========== CRUD 操作 ==========

def check_document_exists(doc_abs_path: str) -> bool:
    """检查文档是否已存在于数据库中"""
    session = get_sync_session()
    try:
        count = session.query(DocArticle).filter(
            DocArticle.doc_abs_path == doc_abs_path
        ).count()
        return count > 0
    finally:
        session.close()


def get_article_ids_by_path(doc_abs_path: str) -> List[str]:
    """根据文档路径获取所有条文ID"""
    session = get_sync_session()
    try:
        records = session.query(DocArticle.article_id).filter(
            DocArticle.doc_abs_path == doc_abs_path
        ).all()
        return [r[0] for r in records]
    finally:
        session.close()


def batch_insert_doc_articles(doc_abs_path: str, doc_id: str, article_ids: List[str]):
    """批量插入文档-条文关系记录"""
    session = get_sync_session()
    try:
        now = datetime.now()
        records = [
            DocArticle(
                article_id=aid,
                doc_id=doc_id,
                doc_abs_path=doc_abs_path,
                created_at=now,
            )
            for aid in article_ids
        ]
        session.bulk_save_objects(records)
        session.commit()
        logger.info(f"批量插入 {len(records)} 条 doc_article 记录, doc_abs_path={doc_abs_path}")
    except Exception as e:
        session.rollback()
        logger.error(f"批量插入 doc_article 失败: {e}")
        raise
    finally:
        session.close()


def delete_doc_articles_by_path(doc_abs_path: str) -> int:
    """根据文档路径删除所有关系记录，返回删除数量"""
    session = get_sync_session()
    try:
        count = session.query(DocArticle).filter(
            DocArticle.doc_abs_path == doc_abs_path
        ).delete()
        session.commit()
        logger.info(f"删除 {count} 条 doc_article 记录, doc_abs_path={doc_abs_path}")
        return count
    except Exception as e:
        session.rollback()
        logger.error(f"删除 doc_article 失败: {e}")
        raise
    finally:
        session.close()


# ========== 高级操作（同时操作 MySQL + Milvus） ==========

def delete_document(doc_abs_path: str):
    """
    删除文档：同时从 MySQL 和 Milvus 中删除
    参数：doc_abs_path - 文档的绝对路径
    """
    logger.info(f"开始删除文档: {doc_abs_path}")

    # 1. 从 MySQL 查出所有条文ID
    article_ids = get_article_ids_by_path(doc_abs_path)
    if not article_ids:
        logger.warning(f"文档不存在于数据库中: {doc_abs_path}")
        return {"deleted_count": 0, "message": "文档不存在"}

    # 2. 从 Milvus 删除对应条文向量
    try:
        collection = _get_milvus_collection()
        # Milvus 删除：使用 id in [...] 表达式
        # 分批删除，每批最多 1000 条
        batch_size = 1000
        for i in range(0, len(article_ids), batch_size):
            batch_ids = article_ids[i:i + batch_size]
            id_list_str = ", ".join([f'"{aid}"' for aid in batch_ids])
            expr = f"id in [{id_list_str}]"
            collection.delete(expr)
        collection.flush()
        logger.info(f"Milvus 删除 {len(article_ids)} 条向量, doc_abs_path={doc_abs_path}")
    except Exception as e:
        logger.error(f"Milvus 删除失败: {e}")
        raise

    # 3. 从 MySQL 删除关系记录
    deleted_count = delete_doc_articles_by_path(doc_abs_path)

    logger.info(f"文档删除完成: {doc_abs_path}, 删除条文数: {deleted_count}")
    return {"deleted_count": deleted_count, "message": "删除成功"}


def add_document(doc_abs_path: str, embedding_model, vector_manager=None):
    """
    新增文档：加载文档 -> 切分 -> 向量化 -> 写入 Milvus + MySQL
    参数：
      doc_abs_path - 文档的绝对路径
      embedding_model - Embedding 模型实例
      vector_manager - VectorManager 实例（可选，用于更新 BM25）
    返回：插入的条文数量
    """
    import torch
    from tqdm import tqdm
    from RAG.vector_doc import (
        load_documents, split_documents, _get_collection_schema,
        chinese_tokenizer, VectorManager, save_docs_to_cache
    )
    from langchain_community.retrievers import BM25Retriever
    from db_crud.base_func import count_tokens
    from config import MILVUS_URL, VECTOR_COLLECTION_NAME, RAG_CACHE_FILE

    logger.info(f"开始新增文档: {doc_abs_path}")

    # 1. 检查文档是否已存在
    if check_document_exists(doc_abs_path):
        logger.warning(f"文档已存在于数据库中，跳过: {doc_abs_path}")
        return {"inserted_count": 0, "message": "文档已存在"}

    # 2. 验证文件存在
    if not os.path.isfile(doc_abs_path):
        raise FileNotFoundError(f"文件不存在: {doc_abs_path}")

    # 3. 加载并切分文档（复用已有逻辑，临时创建目录结构）
    #    load_documents 接受目录路径，这里直接对单文件处理
    from langchain_community.document_loaders import TextLoader, Docx2txtLoader
    from RAG.vector_doc import clean_legal_text, split_by_article, compact_clean
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    ext = os.path.splitext(doc_abs_path)[1].lower()
    if ext in ('.md', '.txt'):
        loader = TextLoader(doc_abs_path, autodetect_encoding=True)
    elif ext == '.docx':
        loader = Docx2txtLoader(doc_abs_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

    raw_docs = loader.load()
    if not raw_docs:
        raise RuntimeError(f"文件内容为空: {doc_abs_path}")

    # 清洗文档
    for doc in raw_docs:
        original_len = len(doc.page_content)
        doc.page_content = clean_legal_text(doc.page_content)
        basename = os.path.basename(doc.metadata.get("source", doc_abs_path))
        filename_without_ext = os.path.splitext(basename)[0]
        doc.metadata.update({
            "source": doc_abs_path,
            "source_cleaned": True,
            "original_length": original_len,
            "cleaned_length": len(doc.page_content),
            "filename": filename_without_ext,
        })

    # 切分条文
    split_docs = split_documents(raw_docs)
    if not split_docs:
        raise RuntimeError(f"文档切分后无有效条文: {doc_abs_path}")

    # 4. 生成唯一ID
    doc_id = str(uuid.uuid4())
    for doc in split_docs:
        doc.metadata["id"] = str(uuid.uuid4())
        doc.metadata["token_num"] = count_tokens(doc.page_content)
        if "start_position" not in doc.metadata:
            doc.metadata["start_position"] = 0

    article_ids = [doc.metadata["id"] for doc in split_docs]
    texts = [doc.page_content for doc in split_docs]
    metadatas = [doc.metadata for doc in split_docs]
    filenames = [doc.metadata.get("filename", "") for doc in split_docs]
    articles = [doc.metadata.get("article", "") for doc in split_docs]
    start_positions = [doc.metadata.get("start_position", 0) for doc in split_docs]

    # 5. 向量化
    logger.info(f"正在为 {len(texts)} 条条文生成向量...")
    vectors = []
    batch_size = 128
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i:i + batch_size]
        batch_vecs = embedding_model.embed_documents(batch_texts)
        vectors.extend(batch_vecs)
        torch.cuda.empty_cache()

    # 6. 插入 Milvus
    connections.connect(uri=MILVUS_URL)
    collection = Collection(VECTOR_COLLECTION_NAME)

    batch_size_insert = 1000
    total = len(article_ids)
    for i in tqdm(range(0, total, batch_size_insert), desc="Inserting to Milvus"):
        end_idx = min(i + batch_size_insert, total)
        batch_data = [
            article_ids[i:end_idx],
            texts[i:end_idx],
            metadatas[i:end_idx],
            vectors[i:end_idx],
            filenames[i:end_idx],
            articles[i:end_idx],
            start_positions[i:end_idx],
        ]
        collection.insert(batch_data)
        del batch_data

    collection.flush()
    logger.info(f"Milvus 插入 {total} 条向量, doc_abs_path={doc_abs_path}")

    # 7. 插入 MySQL
    batch_insert_doc_articles(doc_abs_path, doc_id, article_ids)

    # 8. 更新 BM25（如果 vector_manager 存在）
    if vector_manager is not None and vector_manager.bm25_retriever is not None:
        logger.info("正在重建 BM25 索引（增量）...")
        # 重新从 Milvus 加载全部文档重建 BM25
        _rebuild_bm25(vector_manager, collection)

    logger.info(f"文档新增完成: {doc_abs_path}, 插入条文数: {total}")
    return {"inserted_count": total, "message": "新增成功"}


def _rebuild_bm25(vector_manager, collection: Collection):
    """从 Milvus 重新加载全部文档并重建 BM25"""
    import gc
    from langchain_core.documents import Document
    from langchain_community.retrievers import BM25Retriever
    from RAG.vector_doc import chinese_tokenizer

    final_docs = []
    iterator = None
    try:
        iterator = collection.query_iterator(
            expr="id != ''",
            output_fields=["text", "metadata", "id", "filename", "article", "start_position"],
            batch_size=1000
        )
        while True:
            batch = iterator.next()
            if len(batch) == 0:
                break
            for entity in batch:
                text = entity.get("text", "")
                meta = entity.get("metadata", {})
                doc = Document(page_content=text, metadata=meta)
                if "filename" not in doc.metadata:
                    doc.metadata["filename"] = entity.get("filename", "")
                if "article" not in doc.metadata:
                    doc.metadata["article"] = entity.get("article", "")
                if "start_position" not in doc.metadata:
                    doc.metadata["start_position"] = entity.get("start_position", 0)
                final_docs.append(doc)
            del batch
            gc.collect()
    finally:
        if iterator is not None:
            iterator.close()

    if final_docs:
        vector_manager.bm25_retriever = BM25Retriever.from_documents(
            final_docs, preprocess_func=chinese_tokenizer
        )
        vector_manager.bm25_retriever.k = 10
        logger.info(f"BM25 重建完成，共 {len(final_docs)} 条文档")