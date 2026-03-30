"""
- create_vector_store 内部使用 doc_article_crud 维护 MySQL 关系表
- 函数签名不变，内部逻辑优化
"""

import time
import os
import re
import uuid
import json
import gc
from typing import Optional, List

from langchain_community.document_loaders import DirectoryLoader, TextLoader, Docx2txtLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema,
    DataType, utility
)

from app_logger import database_logger as logger
from app_logger import timer
import unicodedata
from db_crud.base_func import count_tokens
import torch
from tqdm import tqdm
from config import VECTOR_COLLECTION_NAME, MILVUS_URL, EMBEDDING_DIM, RAG_CACHE_FILE
import jieba


def chinese_tokenizer(text: str) -> List[str]:
    return list(jieba.cut(text))


class VectorManager:
    vector_store: Collection = None
    bm25_retriever: Optional[BM25Retriever] = None


def _get_collection_schema() -> CollectionSchema:
    """定义 Milvus 集合 Schema（新增 filename, article, start_position 字段）"""
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="article", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="start_position", dtype=DataType.INT64),
    ]
    return CollectionSchema(fields, description="Legal articles collection")


def clean_legal_text(text: str) -> str:
    if not text or not text.strip():
        return text

    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    normalized = unicodedata.normalize('NFKC', text)

    punct_map = {
        "，": ",", "。": ".", "（": "(", "）": ")", "；": ";", "：": ":",
        "！": "!", "？": "?", "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
        "【": "[", "】": "]", "《": "<", "》": ">", "、": ",", "·": "·"
    }
    for cn, en in punct_map.items():
        normalized = normalized.replace(cn, en)

    lines = []
    for line in normalized.splitlines():
        cleaned_line = re.sub(r'[ \t]+', ' ', line).strip()
        if cleaned_line:
            lines.append(cleaned_line)

    return '\n\n'.join(lines).strip()


def split_by_article(text: str, source_path: str) -> List[Document]:
    """
    增强版：使用更灵活的正则表达式切分法律条文，兼容：
    - 紧凑格式：第一条，第一百零一条
    - 带空格格式：第 一 条， 第 一百零一 条
    - 阿拉伯数字：第1条， 第 101 条
    """
    pattern = r'(第\s*[零一二三四五六七八九十百千]+\s*条)'

    parts = re.split(f'({pattern})', text.strip())

    docs = []
    i = 0
    while i < len(parts):
        if re.fullmatch(pattern, parts[i]):
            article_num = parts[i].strip()
            content = ""
            i += 1
            while i < len(parts) and not re.fullmatch(pattern, parts[i]):
                content += parts[i]
                i += 1
            content = content.strip()
            if content:
                basename = os.path.basename(source_path)
                filename_without_ext = os.path.splitext(basename)[0]
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "filename": filename_without_ext,
                            "article": article_num,
                            "source": source_path,
                        }
                    )
                )
        else:
            i += 1
    return docs


def load_documents(source_dir: str) -> List[Document]:
    try:
        logger.info(f"开始加载文档，源目录：{source_dir}")
        logger.info(f"源目录绝对路径：{os.path.abspath(source_dir)}")

        if not os.path.isdir(source_dir):
            logger.error(f"提供的路径不是目录或不存在：{source_dir}")
            raise RuntimeError(detail=f"文档目录不存在：{source_dir}")

        all_files = []
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                all_files.append(os.path.join(root, file))
        logger.info(f"目录下找到 {len(all_files)} 个文件（所有类型）。前10个：{all_files[:10]}")

        docs = []

        text_loader = DirectoryLoader(
            path=source_dir,
            glob=["*.md", "*.txt"],
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            show_progress=True,
            recursive=True
        )
        loaded_text = text_loader.load()
        logger.info(f"TextLoader 加载了 {len(loaded_text)} 个 .md/.txt 文档")
        docs.extend(loaded_text)

        docx_loader = DirectoryLoader(
            path=source_dir,
            glob=["*.docx"],
            loader_cls=Docx2txtLoader,
            show_progress=True,
            recursive=True
        )
        loaded_docx = docx_loader.load()
        logger.info(f"Docx2txtLoader 加载了 {len(loaded_docx)} 个 .docx 文档")
        docs.extend(loaded_docx)

        for doc in docs:
            original_len = len(doc.page_content)
            doc.page_content = clean_legal_text(doc.page_content)
            basename = os.path.basename(doc.metadata["source"])
            filename_without_ext = os.path.splitext(basename)[0]
            doc.metadata.update({
                "source_cleaned": True,
                "original_length": original_len,
                "cleaned_length": len(doc.page_content),
                "filename": filename_without_ext,
            })

        logger.info(f"成功加载并清洗 {len(docs)} 个文档（含 .md / .txt / .docx）")
        return docs

    except Exception as e:
        logger.error(f"文档加载失败：{str(e)}", exc_info=True)
        raise RuntimeError(f"文档加载失败：{str(e)}")


def compact_clean(text: str) -> str:
    """
    将多行文本压缩为紧凑单行：
    - 移除多余空白行
    - 将连续换行/空格替换为单个空格
    - 保留句子间自然空格
    """
    if not text:
        return text
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_documents(documents: List[Document]) -> List[Document]:
    all_article_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        length_function=count_tokens,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""]
    )

    for doc in documents:
        article_docs = split_by_article(
            text=doc.page_content,
            source_path=doc.metadata["source"]
        )
        for art_doc in article_docs:
            art_doc.page_content = compact_clean(art_doc.page_content)

            token_num = count_tokens(art_doc.page_content)
            if token_num <= 512:
                art_doc.metadata["start_position"] = 0
                all_article_docs.append(art_doc)
            else:
                full_text = art_doc.page_content
                sub_docs = text_splitter.split_documents([art_doc])
                search_start = 0
                for i, sub_doc in enumerate(sub_docs):
                    chunk_pos = full_text.find(sub_doc.page_content, search_start)
                    if chunk_pos == -1:
                        chunk_pos = search_start
                    else:
                        search_start = chunk_pos + len(sub_doc.page_content)

                    sub_doc.metadata.update({
                        "article": art_doc.metadata["article"],
                        "filename": art_doc.metadata["filename"],
                        "source": art_doc.metadata["source"],
                        "start_position": chunk_pos,
                    })
                all_article_docs.extend(sub_docs)

    logger.info(f"条文切分完成 | 共提取 {len(all_article_docs)} 条/子条法规条文")
    return all_article_docs


def save_docs_to_cache(docs: List[Document], cache_path: str = RAG_CACHE_FILE):
    """将 Document 列表保存为 JSON 文件"""
    serializable_docs = []
    for doc in docs:
        serializable_docs.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(serializable_docs, f, ensure_ascii=False, indent=2)
    logger.info(f"文档缓存已保存至：{cache_path}")


def load_docs_from_cache(cache_path: str = RAG_CACHE_FILE) -> List[Document]:
    """从 JSON 文件加载 Document 列表"""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"缓存文件不存在：{cache_path}")
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    for item in data:
        doc = Document(
            page_content=item["page_content"],
            metadata=item["metadata"]
        )
        docs.append(doc)
    logger.info(f"从缓存加载 {len(docs)} 条文档")
    return docs


@timer('知识库向量化')
def create_vector_store(m_embedding_model, vector_manager: VectorManager, file_path: Optional[str] = None, re_build: bool = False):
    """
    构建/加载向量知识库（函数签名不变）
    重构点：re_build 时使用 doc_article_crud 维护 MySQL 关系表
    """
    from db_crud.doc_article_crud import batch_insert_doc_articles, delete_doc_articles_by_path, check_document_exists

    try:
        logger.info("=" * 50)
        logger.info(f"开始构建法规条文知识库，路径：{file_path} | re_build={re_build}")
        logger.info("=" * 50)

        # === 连接 Milvus ===
        connections.connect(uri=MILVUS_URL)
        collection_name = VECTOR_COLLECTION_NAME

        # === 获取或创建集合 ===
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
        else:
            schema = _get_collection_schema()
            collection = Collection(collection_name, schema)
            collection.create_index(
                field_name="vector",
                index_params={"index_type": "FLAT", "metric_type": "L2"}
            )

        vector_manager.vector_store = collection

        final_docs = []
        max_size = 0
        avg_size = 0

        if re_build:
            # --- 重建逻辑 ---
            docs = load_documents(file_path)
            split_docs = split_documents(docs)

            if not split_docs:
                error_msg = (
                    f"无法构建知识库：从目录 '{file_path}' 中未加载到任何有效文档。\n"
                    f"可能原因：\n"
                    f"  1. 目录路径错误。\n"
                    f"  2. 目录内无 .md, .txt, .docx 文件。\n"
                    f"  3. 文件内容为空或清洗后内容被过滤。\n"
                    f"请检查日志中 `load_documents` 函数输出的诊断信息。"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            for doc in split_docs:
                doc.metadata["id"] = str(uuid.uuid4())
                doc.metadata["token_num"] = count_tokens(doc.page_content)
                if "start_position" not in doc.metadata:
                    doc.metadata["start_position"] = 0

            ids = [doc.metadata["id"] for doc in split_docs]
            texts = [doc.page_content for doc in split_docs]
            metadatas = [doc.metadata for doc in split_docs]
            filenames = [doc.metadata.get("filename", "") for doc in split_docs]
            articles = [doc.metadata.get("article", "") for doc in split_docs]
            start_positions = [doc.metadata.get("start_position", 0) for doc in split_docs]

            # 向量化
            logger.info("正在生成向量...")
            vectors = []
            batch_size = 128
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
                batch_texts = texts[i:i + batch_size]
                batch_vecs = m_embedding_model.embed_documents(batch_texts)
                vectors.extend(batch_vecs)
                torch.cuda.empty_cache()

            # 重建集合
            logger.info("正在重建 Milvus 集合...")
            utility.drop_collection(collection_name)
            schema = _get_collection_schema()
            collection = Collection(collection_name, schema)
            collection.create_index(
                field_name="vector",
                index_params={"index_type": "FLAT", "metric_type": "L2"}
            )
            vector_manager.vector_store = collection

            logger.info("开始分批插入数据到 Milvus...")
            batch_size_insert = 1000
            total = len(ids)

            for i in tqdm(range(0, total, batch_size_insert), desc="Inserting to Milvus"):
                end_idx = min(i + batch_size_insert, total)
                batch_data = [
                    ids[i:end_idx],
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
            logger.info(f"全部 {total} 条数据已成功插入 Milvus")

            # === 同步写入 MySQL doc_article 表 ===
            # 按 source（文档路径）分组，写入关系表
            doc_path_to_ids = {}
            for doc in split_docs:
                source = doc.metadata.get("source", "")
                abs_path = os.path.abspath(source) if source else ""
                if abs_path not in doc_path_to_ids:
                    doc_path_to_ids[abs_path] = []
                doc_path_to_ids[abs_path].append(doc.metadata["id"])

            for abs_path, aid_list in doc_path_to_ids.items():
                if not abs_path:
                    continue
                # 先清除旧记录（重建场景）
                try:
                    delete_doc_articles_by_path(abs_path)
                except Exception:
                    pass
                doc_id = str(uuid.uuid4())
                batch_insert_doc_articles(abs_path, doc_id, aid_list)

            logger.info(f"MySQL doc_article 表同步完成，共 {len(doc_path_to_ids)} 个文档")

            # === 保存 split_docs 到缓存文件 ===
            save_docs_to_cache(split_docs, RAG_CACHE_FILE)

            final_docs = split_docs
            token_nums = [d.metadata['token_num'] for d in split_docs]
            max_size = max(token_nums)
            avg_size = sum(token_nums) // len(token_nums)
            logger.info(f"向量化完成 ({len(ids)} 条)")

        else:
            # --- 非重建逻辑：从 Milvus 使用 query_iterator 游标读取 ---
            logger.info("未启用重建，正在通过 query_iterator 从 Milvus 加载文档用于 BM25...")

            final_docs = []
            total_loaded = 0
            temp_token_nums = []
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

                    batch_docs = []
                    for entity in batch:
                        text = entity.get("text", "")
                        meta = entity.get("metadata", {})

                        doc = Document(page_content=text, metadata=meta)

                        if "token_num" not in meta:
                            t_num = count_tokens(text)
                            doc.metadata["token_num"] = t_num
                            temp_token_nums.append(t_num)
                        else:
                            temp_token_nums.append(meta["token_num"])

                        if "filename" not in doc.metadata:
                            doc.metadata["filename"] = entity.get("filename", "")
                        if "article" not in doc.metadata:
                            doc.metadata["article"] = entity.get("article", "")
                        if "start_position" not in doc.metadata:
                            doc.metadata["start_position"] = entity.get("start_position", 0)

                        batch_docs.append(doc)

                    final_docs.extend(batch_docs)
                    total_loaded += len(batch_docs)

                    del batch
                    del batch_docs
                    gc.collect()

                    logger.debug(f"已加载 {total_loaded} 条...")

            finally:
                if iterator is not None:
                    iterator.close()
                    logger.debug("Milvus 游标已关闭")

            if not final_docs:
                raise ValueError("Milvus 集合中未找到任何文档")

            max_size = max(temp_token_nums)
            avg_size = sum(temp_token_nums) // len(temp_token_nums)
            logger.info(f"从 Milvus 成功加载 {total_loaded} 条文档用于 BM25")

        collection.load()
        # 构建 BM25 (统一逻辑)
        logger.info("正在初始化 BM25 检索器...")
        vector_manager.bm25_retriever = BM25Retriever.from_documents(final_docs, preprocess_func=chinese_tokenizer)
        vector_manager.bm25_retriever.k = 20

        logger.info(f"最终索引：{len(final_docs)} 条 | 平均 token 长度：{avg_size} | 最大 token 长度：{max_size}")
        logger.info("=" * 50)
        if not re_build:
            logger.info("已从 Milvus 加载文档，BM25 初始化完成")

    except Exception as e:
        logger.error(f"知识库构建失败：{str(e)}", exc_info=True)
        raise RuntimeError(f"知识库构建失败：{str(e)}")
