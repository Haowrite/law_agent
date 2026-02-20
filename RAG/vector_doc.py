# vector_doc.py
import time
import os
import re
import uuid
import json
from typing import Optional, List

from langchain_community.document_loaders import DirectoryLoader, TextLoader, Docx2txtLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===== 仅使用 pymilvus（原生）=====
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema,
    DataType, utility
)
# =============================

from app_logger import database_logger as logger
from app_logger import timer
from fastapi import HTTPException
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
    """定义 Milvus 集合 Schema"""
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    return CollectionSchema(fields, description="Legal articles collection")


def clean_legal_text(text: str) -> str:
    if not text or not text.strip():
        return text

    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    normalized = unicodedata.normalize('NFKC', text)

    punct_map = {
        "，": ",", "。": ".", "（": "(", "）": ")", "；": ";", "：": ":",
        "！": "!", "？": "?", "“": '"', "”": '"', "‘": "'", "’": "'",
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
    pattern = r'(第[零一二三四五六七八九十百千]+条)'
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
        docs = []

        text_loader = DirectoryLoader(
            path=source_dir,
            glob=["**/*.md", "**/*.txt"],
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            show_progress=True,
        )
        docs.extend(text_loader.load())

        docx_loader = DirectoryLoader(
            path=source_dir,
            glob=["**/*.docx"],
            loader_cls=Docx2txtLoader,
            show_progress=True,
        )
        docs.extend(docx_loader.load())

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

        logger.info(f"✓ 成功加载并清洗 {len(docs)} 个文档（含 .md / .txt / .docx）")
        return docs

    except Exception as e:
        logger.error(f"文档加载失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文档加载失败: {str(e)}")

def compact_clean(text: str) -> str:
    """
    将多行文本压缩为紧凑单行：
    - 移除多余空白行
    - 将连续换行/空格替换为单个空格
    - 保留句子间自然空格
    """
    if not text:
        return text
    # 替换所有空白字符（包括 \n, \t, 多个空格）为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 去除首尾空格
    return text.strip()

def split_documents(documents: List[Document]) -> List[Document]:
    all_article_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=count_tokens,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""]
    )

    for doc in documents:
        article_docs = split_by_article(
            text=doc.page_content,
            source_path=doc.metadata["source"]
        )
        for art_doc in article_docs:
        # 👇 对每条 article 内容做紧凑清洗
            art_doc.page_content = compact_clean(art_doc.page_content)
            
            token_num = count_tokens(art_doc.page_content)
            if token_num <= 800:
                all_article_docs.append(art_doc)
            else:
                # 如果仍超长，先用 RecursiveCharacterTextSplitter 切分
                sub_docs = text_splitter.split_documents([art_doc])
                for i, sub_doc in enumerate(sub_docs):
                    # 👇 对每个子 chunk 也做紧凑清洗
                    sub_doc.page_content = compact_clean(sub_doc.page_content)
                    sub_doc.metadata.update({
                        "article": f"{art_doc.metadata['article']}_part{i+1}",
                        "filename": art_doc.metadata["filename"],
                        "source": art_doc.metadata["source"],
                    })
                all_article_docs.extend(sub_docs)

    logger.info(f"✓ 条文切分完成 | 共提取 {len(all_article_docs)} 条/子条法规条文")
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
    logger.info(f"✓ 文档缓存已保存至: {cache_path}")


def load_docs_from_cache(cache_path: str = RAG_CACHE_FILE) -> List[Document]:
    """从 JSON 文件加载 Document 列表"""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"缓存文件不存在: {cache_path}")
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    for item in data:
        doc = Document(
            page_content=item["page_content"],
            metadata=item["metadata"]
        )
        docs.append(doc)
    logger.info(f"✓ 从缓存加载 {len(docs)} 条文档")
    return docs


@timer('知识库向量化')
def create_vector_store(m_embedding_model, file_path: Optional[str] = None, re_build: bool = False):
    try:
        logger.info("=" * 50)
        logger.info(f"🚀 开始构建法规条文知识库，路径: {file_path} | re_build={re_build}")
        logger.info("=" * 50)

        # === 连接 Embedded Milvus ===
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

        VectorManager.vector_store = collection

        if re_build:
            docs = load_documents(file_path)
            split_docs = split_documents(docs)

            for doc in split_docs:
                doc.metadata["id"] = str(uuid.uuid4())
                doc.metadata["token_num"] = count_tokens(doc.page_content)

            ids = [doc.metadata["id"] for doc in split_docs]
            texts = [doc.page_content for doc in split_docs]
            metadatas = [doc.metadata for doc in split_docs]

            # 向量化
            logger.info("→ 正在生成向量...")
            vectors = []
            batch_size = 128
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
                batch_texts = texts[i:i + batch_size]
                batch_vecs = m_embedding_model.embed_documents(batch_texts)
                vectors.extend(batch_vecs)
                torch.cuda.empty_cache()

            # 重建集合
            logger.info("→ 正在重建 Milvus 集合...")
            utility.drop_collection(collection_name)
            schema = _get_collection_schema()
            collection = Collection(collection_name, schema)
            collection.create_index(
                field_name="vector",
                index_params={"index_type": "FLAT", "metric_type": "L2"}
            )
            VectorManager.vector_store = collection

            logger.info("→ 开始分批插入数据到 Milvus...")
            batch_size_insert = 1000
            total = len(ids)

            for i in tqdm(range(0, total, batch_size_insert), desc="Inserting to Milvus"):
                end_idx = min(i + batch_size_insert, total)
                batch_data = [
                    ids[i:end_idx],
                    texts[i:end_idx],
                    metadatas[i:end_idx],
                    vectors[i:end_idx],
                ]
                collection.insert(batch_data)
                del batch_data

            collection.flush()
            logger.info(f"✓ 全部 {total} 条数据已成功插入 Milvus")

            # === 保存 split_docs 到缓存文件 ===
            save_docs_to_cache(split_docs, RAG_CACHE_FILE)

            final_docs = split_docs
            max_size = max(d.metadata['token_num'] for d in split_docs)
            avg_size = sum(d.metadata['token_num'] for d in split_docs) // len(split_docs)
            logger.info(f"✓ 向量化完成 ({len(ids)} 条)")

        else:
            logger.info("→ 未启用重建，尝试从缓存文件加载文档用于 BM25...")
            try:
                final_docs = load_docs_from_cache(RAG_CACHE_FILE)
                if not final_docs:
                    raise ValueError("缓存文件为空")

                token_nums = [doc.metadata.get('token_num', 0) for doc in final_docs]
                max_size = max(token_nums)
                avg_size = sum(token_nums) // len(token_nums)
                logger.info(f"✓ 从缓存成功加载 {len(final_docs)} 条文档用于 BM25")

            except Exception as e:
                logger.error(f"⚠️ 从缓存加载文档失败: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail="未重建且无法从缓存加载文档，请先运行 re_build=True"
                )

        # 构建 BM25
        VectorManager.bm25_retriever = BM25Retriever.from_documents(final_docs, preprocess_func=chinese_tokenizer)
        VectorManager.bm25_retriever.k = 10

        logger.info(f"📊 最终索引: {len(final_docs)} 条 | 平均token长度: {avg_size} | 最大token长度: {max_size}")
        logger.info("=" * 50)
        if not re_build:
            logger.info("✅ 已从缓存加载文档，BM25 初始化完成")

    except Exception as e:
        logger.error(f"❌ 知识库构建失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"知识库构建失败: {str(e)}")