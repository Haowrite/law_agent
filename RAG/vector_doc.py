import time
import os
import re
import token
import uuid
from typing import Optional, List

from langchain_community.document_loaders import DirectoryLoader, TextLoader, Docx2txtLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app_logger import database_logger as logger
from app_logger import timer
from RAG.base import milvus_vector_store  # 确保导入正确
from fastapi import HTTPException
import unicodedata
from db_crud.base_func import count_tokens
import torch
from tqdm import tqdm

class VectorManager:
    vector_store: VectorStore = None
    vector_retriever: Optional[VectorStoreRetriever] = None
    bm25_retriever: Optional[BM25Retriever] = None


def clean_legal_text(text: str) -> str:
    """
    通用法律/行政法规文本清洗（适用于 .txt / .md / .docx 提取的纯文本）：
    - 移除 HTML 注释（如 <!-- ... -->）
    - 全角字符标准化（含全角空格 → 半角）
    - 中文标点 → 英文标点（提升检索一致性）
    - 压缩行内冗余空格，保留段落结构（用 \n\n 分隔）
    """
    if not text or not text.strip():
        return text

    # 1. 移除 HTML 注释（包括跨行）
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # 2. 全角转半角（NFKC 标准化）
    normalized = unicodedata.normalize('NFKC', text)

    # 3. 中文标点映射为英文标点（避开正则特殊字符）
    punct_map = {
        "，": ",", "。": ".", "（": "(", "）": ")", "；": ";", "：": ":",
        "！": "!", "？": "?", "“": '"', "”": '"', "‘": "'", "’": "'",
        "【": "[", "】": "]", "《": "<", "》": ">", "、": ",", "·": "·"
    }
    for cn, en in punct_map.items():
        normalized = normalized.replace(cn, en)

    # 4. 智能空格处理：压缩行内空格，保留非空行（段落）
    lines = []
    for line in normalized.splitlines():
        cleaned_line = re.sub(r'[ \t]+', ' ', line).strip()
        if cleaned_line:
            lines.append(cleaned_line)

    return '\n\n'.join(lines).strip()


def split_by_article(text: str, source_path: str) -> List[Document]:
    """
    将一段法规文本按“第X条”切分为多个 Document。
    支持中文数字：一、二、十、十一、一百等。
    """
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
                # 去掉文件后缀名（如 .md / .docx / .txt）
                basename = os.path.basename(source_path)
                filename_without_ext = os.path.splitext(basename)[0]

                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "filename": filename_without_ext,  # ← 无后缀
                            "article": article_num,
                            "source": source_path,
                        }
                    )
                )
        else:
            i += 1
    return docs


def load_documents(source_dir: str) -> List[Document]:
    """加载并清洗多种格式文档（.md, .txt, .docx）"""
    try:
        # 合并多个 loader：TextLoader 处理 .md/.txt，Docx2txtLoader 处理 .docx
        docs = []

        # 加载 .md 和 .txt
        text_loader = DirectoryLoader(
            path=source_dir,
            glob=["**/*.md", "**/*.txt"],
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            show_progress=True,
        )
        docs.extend(text_loader.load())

        # 加载 .docx
        docx_loader = DirectoryLoader(
            path=source_dir,
            glob=["**/*.docx"],
            loader_cls=Docx2txtLoader,
            show_progress=True,
        )
        docs.extend(docx_loader.load())

        # 清洗内容 + 更新元数据
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


def split_documents(documents: List[Document]) -> List[Document]:
    """
    按“第X条”切分每篇文档，每条作为一个独立 chunk；
    若某条超过 1000 tokens，则进一步用 RecursiveCharacterTextSplitter 切分为 ≤1000 tokens 的子块。
    """
    all_article_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=count_tokens,  # 使用你已有的 count_tokens 函数
        separators=["\n\n", "\n", "。", "；", "，", " ", ""]
    )

    for doc in documents:
        article_docs = split_by_article(
            text=doc.page_content,
            source_path=doc.metadata["source"]
        )
        for art_doc in article_docs:
            token_num = count_tokens(art_doc.page_content)
            if token_num <= 1000:
                # 不需要再切分
                all_article_docs.append(art_doc)
            else:
                # 超长，进行二次切分
                sub_docs = text_splitter.split_documents([art_doc])
                # 保留原始 article 和 filename，但添加子序号（可选）
                for i, sub_doc in enumerate(sub_docs):
                    sub_doc.metadata.update({
                        "article": f"{art_doc.metadata['article']}_part{i+1}",
                        "filename": art_doc.metadata["filename"],
                        "source": art_doc.metadata["source"],
                    })
                all_article_docs.extend(sub_docs)

    logger.info(f"✓ 条文切分完成 | 共提取 {len(all_article_docs)} 条/子条法规条文")
    return all_article_docs


@timer('知识库向量化')
def create_vector_store(file_path: Optional[str] = None, re_build: bool = False):
    try:
        logger.info("=" * 50)
        logger.info(f"🚀 开始构建法规条文知识库，路径: {file_path} | re_build={re_build}")
        logger.info("=" * 50)

        VectorManager.vector_store = milvus_vector_store()

        if re_build:
            # === 重建模式：从原始文件加载并写入向量库 ===
            docs = load_documents(file_path)
            split_docs = split_documents(docs)

            for doc in split_docs:
                doc.metadata["id"] = str(uuid.uuid4())
                doc.metadata["token_num"] = count_tokens(doc.page_content)

            ids = [doc.metadata["id"] for doc in split_docs]
            max_size = max(d.metadata['token_num'] for d in split_docs)
            avg_size = sum(d.metadata['token_num'] for d in split_docs) // len(split_docs)
            logger.info(f"→ 最长文本 {max_size} tokens")

            VectorManager.vector_store.drop()
            logger.info("→ 正在重建向量索引...")
            start_vec = time.time()
            batch_size = 50
            for i in tqdm(range(0, len(split_docs), batch_size), desc="向量化中"):
                batch = split_docs[i:i + batch_size]
                VectorManager.vector_store.add_documents(batch, ids=ids[i:i + batch_size])
                torch.cuda.empty_cache()

            vec_time = time.time() - start_vec
            logger.info(f"✓ 向量化完成 ({len(ids)} 条，耗时 {vec_time:.2f}s)")

            final_docs = split_docs  # 用于后续构建 BM25

        else:
            # === 非重建模式：直接从 Milvus 读取已有文档 ===
            logger.info("→ 未启用重建，尝试从 Milvus 加载现有文档用于 BM25...")
            try:
                
                dummy_vector = [0.0] * 1024

                # 查询最多 100,000 条
                all_docs = VectorManager.vector_store.similarity_search_by_vector(
                    embedding=dummy_vector,
                    k=100000  # 足够大的数
                )

                if not all_docs:
                    raise ValueError("从 Milvus 未检索到任何文档，请确认是否已建库")

                final_docs = all_docs
                ids = [doc.metadata.get("id", "unknown") for doc in final_docs]
                max_size = max(d.metadata['token_num'] for d in final_docs)
                avg_size = sum(d.metadata['token_num'] for d in final_docs) // len(final_docs)

                logger.info(f"✓ 从 Milvus 成功加载 {len(final_docs)} 条文档用于 BM25")

            except Exception as e:
                logger.error(f"⚠️ 从 Milvus 加载文档失败: {e}，回退到重建模式？")
                raise HTTPException(
                    status_code=500,
                    detail="未重建且无法从向量库加载文档，请先运行 re_build=True"
                )

        # === 无论是否重建，都用 final_docs 构建混合检索器 ===
        VectorManager.bm25_retriever = BM25Retriever.from_documents(final_docs)
        VectorManager.bm25_retriever.k = 10
        VectorManager.vector_retriever = VectorManager.vector_store.as_retriever(
            search_kwargs={'k': 10}
        )

        logger.info(f"📊 最终索引: {len(final_docs)} 条 | 平均token长度: {avg_size} | 最大token长度: {max_size}")
        logger.info("=" * 50)
        if not re_build:
            logger.info("✅ 已从现有向量库加载文档，BM25 初始化完成")

    except Exception as e:
        logger.error(f"❌ 知识库构建失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"知识库构建失败: {str(e)}")
                                                                                          

#分割后的 Document 元数据实例：
#     {                                                                      
#   "filename": "安全生产许可证条例",    
#   "article": "第一条",
#   "source": "/data/法规/安全生产许可证条例.md",
#   "token_num": 98,
#   "id": "a1b2c3d4-..."
# }