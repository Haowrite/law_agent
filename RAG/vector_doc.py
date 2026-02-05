import time
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.vectorstores import VectorStore
from app_logger import database_logger as logger
import unicodedata
from fastapi import HTTPException
from RAG.base import *
from typing import Optional, List
import uuid
import re
from app_logger import timer
class VectorManager:
    vector_store: VectorStore = None
    ids = None
    vector_retriever: Optional[VectorStoreRetriever] = None
    bm25_retriever: Optional[BM25Retriever] = None

def clean_markdown_text(text: str) -> str:
    """
    Markdown专用清洗（无代码块场景）：
    - 移除HTML注释（如 <!-- FORCE BREAK -->）
    - 保留标题/列表结构空格
    - 仅压缩行内冗余空格，保留段落换行
    - 标准化标点提升检索匹配率
    """
    if not text or not text.strip():
        return text
    
    # 1. 移除所有 HTML 注释（包括跨行注释）
    # 匹配 <!-- 任意内容 -->，支持跨行（re.DOTALL）
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # 2. 全角字符标准化（含全角空格→半角空格）
    normalized = unicodedata.normalize('NFKC', text)
    
    # 3. 中文标点→英文标点（避开Markdown语法符号）
    punct_map = {
        "，": ",", "。": ".", "（": "(", "）": ")", "；": ";", "：": ":",
        "！": "!", "？": "?", "“": '"', "”": '"', "‘": "'", "’": "'",
        "【": "[", "】": "]", "《": "<", "》": ">", "、": ",", "·": "·"
    }
    for cn, en in punct_map.items():
        normalized = normalized.replace(cn, en)
    
    # 4. 智能空格处理：保留结构性换行，压缩行内冗余空格
    lines = []
    for line in normalized.splitlines():
        cleaned_line = re.sub(r'[ \t]+', ' ', line).strip()
        if cleaned_line:  # 跳过空行（保留非空行间的逻辑分隔）
            lines.append(cleaned_line)
    
    # 用双换行保留段落结构（增强语义分割）
    return '\n\n'.join(lines).strip()

def load_documents(source_dir: str) -> List[Document]:
    """加载并清洗Markdown文档（无代码块优化版）"""
    try:
        loader = DirectoryLoader(
            path=source_dir,
            glob=["**/*.md"],
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            show_progress=True,
        )
        docs = loader.load()
        
        # 清洗内容 + 标记元数据
        for doc in docs:
            original_len = len(doc.page_content)
            doc.page_content = clean_markdown_text(doc.page_content)
            doc.metadata.update({
                "source_cleaned": True,
                "original_length": original_len,
                "cleaned_length": len(doc.page_content),
            })
        
        logger.info(f"✓ 成功加载并清洗 {len(docs)} 个Markdown文档")
        return docs
    except Exception as e:
        logger.error(f"文档加载失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文档加载失败: {str(e)}")

def split_documents(documents: List[Document], chunk_size: int = 700, chunk_overlap: int = 100) -> List[Document]:
    """
    两阶段Markdown感知切分（无代码块优化）：
    1. 按标题结构分割（保留层级语义）
    2. 递归切分时优先保留列表/段落完整性
    """
    # 阶段1: 按标题分割（保留标题文本在内容中）
    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
        ("####", "header_4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    
    header_splits = []
    for doc in documents:
        splits = markdown_splitter.split_text(doc.page_content)
        for split in splits:
            # 合并原始元数据（source等）
            split.metadata.update({
                k: v for k, v in doc.metadata.items() 
                if k not in split.metadata and k != "source_cleaned"
            })
            # 构建标题路径（用于检索上下文增强）
            header_path = " > ".join([
                split.metadata[h] for h in ["header_1", "header_2", "header_3", "header_4"] 
                if h in split.metadata and split.metadata[h]
            ])
            split.metadata["header_path"] = header_path or "文档根目录"
        header_splits.extend(splits)
    
    logger.debug(f"标题分割后块数: {len(header_splits)}")
    
    # 阶段2: 递归切分（优化分隔符适配Markdown结构）
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",               # 段落分隔（最高优先级）
            "\n- ", "\n* ",       # 无序列表项
            "\n1. ", "\n2. ", "\n3. ", "\n4. ", "\n5. ", "\n6. ", "\n7. ", "\n8. ", "\n9. ", "\n10. ",  # 有序列表（覆盖常见）
            "\n",                 # 换行
            "。", "！", "？", "；", # 中文句尾
            ". ", "! ", "? ",     # 英文句尾
            " ", ""               # 最小单位
        ],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    
    split_docs = text_splitter.split_documents(header_splits)
    
    # 添加唯一ID与统计元数据
    for doc in split_docs:
        doc.id = str(uuid.uuid4())
        doc.metadata["chunk_size"] = len(doc.page_content)
        # 精简元数据：移除临时字段
        doc.metadata.pop("original_length", None)
        doc.metadata.pop("cleaned_length", None)
    
    logger.info(
        f"✓ 切分完成 | 原始文档: {len(documents)} | 标题块: {len(header_splits)} | "
        f"最终chunks: {len(split_docs)} (size={chunk_size}, overlap={chunk_overlap})"
    )
    return split_docs
@timer('知识库向量化')
def create_vector_store(file_path: Optional[str] = None, re_build: bool = False):

    try:
        logger.info("="*50)
        logger.info(f"🚀 开始构建纯Markdown知识库, 知识库路径{file_path}")
        logger.info("="*50)


        VectorManager.vector_store = milvus_vector_store()

        # 1. 加载与清洗
        docs = load_documents(file_path)
        
        # 2. 智能切分（参数针对中文技术文档优化）
        split_docs = split_documents(docs, chunk_size=700, chunk_overlap=100)
        
        ids=[doc.id for doc in split_docs]

        if re_build:
            VectorManager.vector_store.drop()
            # 3. 向量化存储
            logger.info("→ 需要构建向量索引，正在构建向量索引...")
            start_vec = time.time()
            ids = VectorManager.vector_store.add_documents(
                split_docs, 
                ids=ids,
            )
            VectorManager.ids = ids
            vec_time = time.time() - start_vec
            logger.info(f"✓ 向量化完成 ({len(ids)} chunks, 耗时 {vec_time:.2f}s)")
        else:
            VectorManager.ids = ids


        # 4. 构建混合检索器（增强检索质量）
        VectorManager.bm25_retriever = BM25Retriever.from_documents(split_docs)
        VectorManager.bm25_retriever.k = 10
        # 使用MMR提升结果多样性，fetch_k增大候选集
        VectorManager.vector_retriever = VectorManager.vector_store.as_retriever(
            search_kwargs={'k': 10}
        )

        logger.info("✓ 检索器初始化完成 (BM25 k=5 | 向量 k=5 MMR)")
        logger.info("="*50)
        logger.info(f"📊 最终索引: {len(ids)} 个语义块 | 平均块大小: {sum(d.metadata['chunk_size'] for d in split_docs)//len(split_docs)} 字符")
        logger.info("="*50 + "\n")
        if not re_build:
            logger.info("⚠️由于没有重建知识库，知识库内容可能与实际文档有差异，请检查~")
        
    except Exception as e:
        logger.error(f"❌ 知识库构建失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"知识库构建失败: {str(e)}")
    

