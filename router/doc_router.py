"""
文档管理路由：新增文档、删除文档
文档 CRUD 操作提交到子进程执行（因为 Embedding 模型在子进程中，GPU 只够一个实例）
"""

import asyncio
import os
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app_logger import app_logger as logger
from config import DOCUMENT_KEY_SECRET, FILE_PATH
from db_crud.doc_article_query import (
    build_public_document_record,
    get_document_by_key,
    search_document_records_by_name,
)
from RAG.document_library import (
    document_key_for_path,
    document_library_root,
    read_document_preview,
    remove_file_if_exists,
    resolve_document_path,
    safe_document_filename,
)
from utils.agent_thread_pool import PROCESS_POOL
from RAG.retrieve_process import add_document_in_process, delete_document_in_process

router = APIRouter(prefix="/api/doc", tags=["文档管理"])


class DocRequest(BaseModel):
    doc_abs_path: str


class DocKeyRequest(BaseModel):
    doc_key: str


class DocumentPreviewResponse(BaseModel):
    doc_key: str
    filename: str
    content: str
    truncated: bool
    article_count: int
    created_at: str
    file_exists: bool


def _run_in_process(func, *args):
    return PROCESS_POOL.submit(func, *args).result()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    上传文件到服务端文档库并增量向量化。
    """
    try:
        filename = safe_document_filename(file.filename or "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    root = document_library_root(FILE_PATH)
    target_path = root / filename
    if target_path.exists():
        raise HTTPException(status_code=409, detail="同名文档已存在，请先删除旧文档")

    logger.info(f"收到上传文档请求: {filename}")
    try:
        with open(target_path, "wb") as out_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _run_in_process(add_document_in_process, str(target_path))
        )
        result["doc_key"] = document_key_for_path(str(target_path), secret=DOCUMENT_KEY_SECRET)
        result["filename"] = filename
        return result
    except Exception as e:
        remove_file_if_exists(target_path)
        logger.error(f"上传并向量化文档失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"上传并向量化文档失败: {str(e)}")
    finally:
        await file.close()


@router.get("/search")
async def search_documents(keyword: str = Query("", max_length=200), limit: int = Query(20, ge=1, le=50)):
    """
    按文档名搜索已入库文档。
    """
    loop = asyncio.get_running_loop()
    records = await loop.run_in_executor(None, lambda: search_document_records_by_name(keyword.strip(), limit))
    docs = []
    for record in records:
        doc = build_public_document_record(
            record["doc_abs_path"],
            record["article_count"],
            record["created_at"],
        )
        try:
            path = resolve_document_path(FILE_PATH, record["doc_abs_path"])
            doc["preview"] = read_document_preview(path, max_chars=240) if path.exists() else ""
        except Exception:
            doc["preview"] = ""
        docs.append(doc)
    return {"documents": docs}


@router.get("/preview", response_model=DocumentPreviewResponse)
async def preview_document(doc_key: str = Query(...), max_chars: int = Query(8000, ge=200, le=50000)):
    """
    预览文档库内文件内容。
    """
    try:
        loop = asyncio.get_running_loop()
        doc = await loop.run_in_executor(None, lambda: get_document_by_key(doc_key))
        if not doc:
            raise HTTPException(status_code=404, detail="文档不存在")

        path = resolve_document_path(FILE_PATH, doc["doc_abs_path"])
        if not path.is_file():
            raise HTTPException(status_code=404, detail="文档文件不存在")
        content = read_document_preview(path, max_chars=max_chars)
        truncated = path.suffix.lower() != ".docx" and os.path.getsize(path) > len(content.encode("utf-8"))
        return DocumentPreviewResponse(
            doc_key=doc["doc_key"],
            filename=path.name,
            content=content,
            truncated=truncated,
            article_count=doc["article_count"],
            created_at=doc["created_at"],
            file_exists=doc["file_exists"],
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"预览文档失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"预览文档失败: {str(e)}")


@router.post("/add")
async def add_document(req: DocRequest):
    """
    新增文档接口
    参数：doc_abs_path - 文档绝对路径
    在子进程中执行：加载 -> 切分 -> 向量化 -> 写入 Milvus + MySQL
    """
    if not req.doc_abs_path or not req.doc_abs_path.strip():
        raise HTTPException(status_code=400, detail="文档路径不能为空")
    try:
        req.doc_abs_path = str(resolve_document_path(FILE_PATH, req.doc_abs_path))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"收到新增文档请求: {req.doc_abs_path}")

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: PROCESS_POOL.submit(add_document_in_process, req.doc_abs_path).result()
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"新增文档失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"新增文档失败: {str(e)}")


@router.delete("/delete")
async def delete_document(req: DocKeyRequest):
    """
    删除文档接口
    参数：doc_key - 文档安全标识
    在子进程中执行：从 MySQL 查询条文ID -> 从 Milvus 删除向量 -> 从 MySQL 删除记录
    """
    if not req.doc_key or not req.doc_key.strip():
        raise HTTPException(status_code=400, detail="文档标识不能为空")

    logger.info(f"收到删除文档请求: {req.doc_key}")

    loop = asyncio.get_running_loop()
    try:
        doc = await loop.run_in_executor(None, lambda: get_document_by_key(req.doc_key.strip()))
        if not doc:
            raise HTTPException(status_code=404, detail="文档不存在")

        doc_abs_path = str(resolve_document_path(FILE_PATH, doc["doc_abs_path"]))
        result = await loop.run_in_executor(
            None,
            lambda: PROCESS_POOL.submit(delete_document_in_process, doc_abs_path).result()
        )
        result["doc_key"] = req.doc_key.strip()
        result["filename"] = doc["filename"]
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")
