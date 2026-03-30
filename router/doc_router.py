"""
文档管理路由：新增文档、删除文档
文档 CRUD 操作提交到子进程执行（因为 Embedding 模型在子进程中，GPU 只够一个实例）
"""

import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app_logger import app_logger as logger
from utils.agent_thread_pool import PROCESS_POOL
from RAG.retrieve_process import add_document_in_process, delete_document_in_process

router = APIRouter(prefix="/api/doc", tags=["文档管理"])


class DocRequest(BaseModel):
    doc_abs_path: str


@router.post("/add")
async def add_document(req: DocRequest):
    """
    新增文档接口
    参数：doc_abs_path - 文档绝对路径
    在子进程中执行：加载 -> 切分 -> 向量化 -> 写入 Milvus + MySQL
    """
    if not req.doc_abs_path or not req.doc_abs_path.strip():
        raise HTTPException(status_code=400, detail="文档路径不能为空")

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
async def delete_document(req: DocRequest):
    """
    删除文档接口
    参数：doc_abs_path - 文档绝对路径
    在子进程中执行：从 MySQL 查询条文ID -> 从 Milvus 删除向量 -> 从 MySQL 删除记录
    """
    if not req.doc_abs_path or not req.doc_abs_path.strip():
        raise HTTPException(status_code=400, detail="文档路径不能为空")

    logger.info(f"收到删除文档请求: {req.doc_abs_path}")

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: PROCESS_POOL.submit(delete_document_in_process, req.doc_abs_path).result()
        )
        return result
    except Exception as e:
        logger.error(f"删除文档失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")