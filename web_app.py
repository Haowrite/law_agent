"""
法律AI助手 - 重构后的 web_app.py
应用创建、生命周期管理、路由注册、启动入口
"""

import asyncio
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

from app_logger import app_logger as logger
from db_crud.base import wait_until_mysql_ready, async_engine
from db_crud.session_manage import m_conversation_manager, wait_until_redis_ready
from utils.agent_thread_pool import PROCESS_POOL

# 导入路由模块
from router.user_router import router as user_router
from router.chat_router import router as chat_router
from router.doc_router import router as doc_router
from db_crud.doc_article_model import DocArticle
from db_crud.session_model import ChatSession, ChatMessage, User, SummaryMessage

# ------------------------------
# Lifespan：应用生命周期管理
# ------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理：集中初始化与清理外部资源
    """
    logger.info("开始初始化应用依赖...")

    # 1. MySQL：连不上则持续报错并重试，直至成功（避免首包业务请求才失败）
    await wait_until_mysql_ready()
    # 2. Redis：同上
    await wait_until_redis_ready()
    logger.info("数据库与 Redis 均已就绪，应用进入服务阶段")

    # ==============================
    # 应用运行阶段
    # ==============================
    yield

    # 2. 关闭 Redis 连接
    try:
        if hasattr(m_conversation_manager, 'redis_client'):
            m_conversation_manager.redis_client.close()
            logger.info("CloseOperation Redis 会话管理器连接")
        else:
            logger.warning("未找到 Redis 客户端，跳过关闭")
    except Exception as e:
        logger.error(f"Redis 连接关闭异常: {e}")

    # 3. 关闭 MySQL 引擎
    try:
        await async_engine.dispose()
        logger.info("CloseOperation MySQL 引擎")
    except Exception as e:
        logger.error(f"MySQL 引擎关闭异常: {e}")

    # 4. 关闭进程池
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, PROCESS_POOL.shutdown, True)
    logger.info("CloseOperation 进程池")

    logger.info("所有外部资源已安全释放，应用退出")


# ------------------------------
# FastAPI 应用
# ------------------------------
app = FastAPI(
    title="RAG系统 API",
    description="基于 LangGraph 的RAG系统，支持会话管理",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# 静态文件 & 模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 注册路由
app.include_router(user_router)
app.include_router(chat_router)
app.include_router(doc_router)


# ------------------------------
# 基础路由（首页 & 健康检查）
# ------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health_check():
    return JSONResponse({"status": "healthy", "timestamp": time.time()})


# ------------------------------
# 启动入口
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("法律AI助手")
    logger.info("=" * 60)
    logger.info("启动服务...")
    logger.info("访问地址: http://localhost:5000")
    logger.info("API 文档: http://localhost:5000/docs")
    logger.info("按 Ctrl+C 停止服务")
    uvicorn.run("web_app:app", host="0.0.0.0", port=5000, reload=False)