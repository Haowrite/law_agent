#!/usr/bin/env python3
"""
法律AI助手
- 全链路异步（数据库 + 智能体）
- 使用 lifespan 初始化数据库
- 路由全部异步化
"""

import os

import time
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from contextlib import asynccontextmanager

from langchain_core.messages import HumanMessage
from app_logger import app_logger as logger
from db_crud.base_func import get_time
from db_crud.user_crud import create_user, authenticate_user, get_user_by_id
from arq import create_pool
from arq.connections import RedisSettings
from db_crud.arq_tasks import REDIS_SETTINGS  # 从 tasks 导入配置
from db_crud.chat_memory_crud import (
    AsyncMySQLChatHistory,
    create_chat_session,
    get_user_session_list,
    get_session_detail,
    delete_chat_session
)
from db_crud.base import init_db, async_engine  # 异步建表函数
from db_crud.session_manage import m_conversation_manager

# 智能体
from agent_service import make_graph

# ------------------------------
# 全局配置
# ------------------------------

AGENT = make_graph()

# ------------------------------
# 工具函数
# ------------------------------

def extract_ai_response(session_state: dict) -> str:
    """从智能体状态中提取AI回复"""
    try:
        return session_state["response"]
    except Exception as e:
        logger.error(f"提取AI回复失败: {e}")
        return "抱歉，处理您的请求时出现了错误。"

# ------------------------------
# Pydantic 模型
# ------------------------------

class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    session_id: str

class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    message_count: int

class SessionDetail(BaseModel):
    session_id: str
    created_at: str
    conversation_history: List[Dict]

# ------------------------------
# Lifespan：应用生命周期管理
# ------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理：集中初始化与清理外部资源
    - 启动顺序：DB 表 → arq 客户端（Redis 会话管理器是 lazy-init，无需显式启动）
    - 关闭顺序：arq 客户端 → Redis 连接 → MySQL 引擎
    """
    logger.info("🚀 开始初始化应用依赖...")

    # ------------------------------
    # 1. 初始化数据库表（如果尚未创建）
    # ------------------------------
    try:
        await init_db()
        logger.info("✅ 数据库表初始化成功")
    except Exception as e:
        logger.error(f"❌ 数据库初始化失败: {e}")
        raise

    # ------------------------------
    # 2. 创建 arq 任务提交客户端（用于 enqueue_job）
    # ------------------------------
    try:
        app.state.arq_pool = await create_pool(REDIS_SETTINGS)
        logger.info("✅ arq 任务客户端初始化成功")
    except Exception as e:
        logger.error(f"❌ arq 客户端初始化失败: {e}")
        raise

    # ==============================
    # 应用运行阶段
    # ==============================
    yield

    # ==============================
    # 关闭阶段（倒序释放资源）
    # ==============================

    # ------------------------------
    # 1. 关闭 arq 提交客户端
    # ------------------------------
    if hasattr(app.state, 'arq_pool'):
        try:
            await app.state.arq_pool.close()
            logger.info("CloseOperation arq 任务客户端")
        except Exception as e:
            logger.error(f"⚠️ arq 客户端关闭异常: {e}")

    # ------------------------------
    # 2. 关闭 Redis 连接（来自 m_conversation_manager）
    # ------------------------------
    try:
        # 你的 ConversationManager 使用的是同步 redis-py
        # 直接关闭底层连接池
        if hasattr(m_conversation_manager, 'redis_client'):
            m_conversation_manager.redis_client.close()
            logger.info("CloseOperation Redis 会话管理器连接")
        else:
            logger.warning("⚠️ 未找到 Redis 客户端，跳过关闭")
    except Exception as e:
        logger.error(f"⚠️ Redis 连接关闭异常: {e}")

    # ------------------------------
    # 3. 关闭 MySQL 引擎
    # ------------------------------
    try:
        await async_engine.dispose()
        logger.info("CloseOperation MySQL 引擎")
    except Exception as e:
        logger.error(f"⚠️ MySQL 引擎关闭异常: {e}")

    logger.info("✅ 所有外部资源已安全释放，应用退出")

# ------------------------------
# FastAPI 应用
# ------------------------------

app = FastAPI(
    title="RAG系统 API",
    description="基于 LangGraph 的RAG系统，支持会话管理",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)



app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ------------------------------
# 路由定义
# ------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health_check():
    return JSONResponse({"status": "healthy", "timestamp": time.time()})


@app.post("/api/register")
async def register(req: RegisterRequest):
    if not req.username or not req.password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")
    if len(req.username) > 50 or len(req.password) < 6:
        raise HTTPException(status_code=400, detail="用户名长度≤50，密码≥6位")
    
    user_id = await create_user(req.username, req.password)
    if user_id is None:
        raise HTTPException(status_code=409, detail="用户名已存在")
    return {"user_id": user_id, "message": "注册成功"}


@app.post("/api/login")
async def login(req: LoginRequest):
    user_id = await authenticate_user(req.username, req.password)
    if user_id is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return {"user_id": user_id, "message": "登录成功"}


@app.post("/api/new_session")
async def new_session(user_id: str = Query(...)):
    user_exists = await get_user_by_id(user_id)
    if not user_exists:
        raise HTTPException(status_code=404, detail="用户不存在")
    session_id = await create_chat_session(user_id)
    return {"session_id": session_id, "message": "新会话创建成功"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user_id: str = Query(...)):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")
    if len(req.message.strip()) > 2000:
        raise HTTPException(status_code=400, detail="消息长度不能超过2000字符")
    
    user_timestamp = get_time()
    
    # 校验会话归属
    try:
        await get_session_detail(req.session_id, user_id=user_id)
    except ValueError:
        raise HTTPException(status_code=403, detail="会话不存在或无权访问")

    session_state = await AGENT.ainvoke(
        input={
            "customer_query": req.message,
            "session_id": req.session_id,
            "user_id": user_id
        },
        config={'configurable': {'thread_id': req.session_id}}
    )

    ai_response = extract_ai_response(session_state)
    m_conversation_manager.add_message_pair(req.session_id, req.message, ai_response)
        # ✅ 异步：提交 DB 写入任务（非阻塞！）
    try:
        # 提交用户消息
        await app.state.arq_pool.enqueue_job(
            'save_chat_message',
            req.session_id,
            req.message,
            'user',
            user_timestamp
        )
        # 提交 AI 消息
        await app.state.arq_pool.enqueue_job(
            'save_chat_message',
            req.session_id,
            ai_response,
            'ai',
            get_time()  
        )
    except Exception as e:
        # 记录错误但不中断主流程（任务会进入 arq 失败队列）
        logger.warning(f"⚠️ 提交 DB 任务失败: {e}")
    return ChatResponse(response=ai_response, session_id=req.session_id)


@app.get("/api/sessions", response_model=Dict[str, List[SessionInfo]])
async def get_sessions(user_id: str = Query(...)):
    user_exists = await get_user_by_id(user_id)
    if not user_exists:
        raise HTTPException(status_code=404, detail="用户不存在")
    sessions = await get_user_session_list(user_id)
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}", response_model=Dict[str, SessionDetail])
async def get_session(session_id: str, user_id: str = Query(...)):
    user_exists = await get_user_by_id(user_id)
    if not user_exists:
        raise HTTPException(status_code=404, detail="用户不存在")
    try:
        session_data = await get_session_detail(session_id, user_id=user_id)
        return {"session": session_data}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"获取会话失败: {e}")
        raise HTTPException(status_code=500, detail="查询失败")


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, user_id: str = Query(...)):
    user_exists = await get_user_by_id(user_id)
    if not user_exists:
        raise HTTPException(status_code=404, detail="用户不存在")
    success = await delete_chat_session(session_id, user_id=user_id)
    if not success:
        raise HTTPException(status_code=404, detail="会话不存在或无权删除")
    return {"message": "会话删除成功"}

# ------------------------------
# 启动入口
# ------------------------------

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 法律AI助手")
    logger.info("=" * 60)
    logger.info("🌐 启动服务...")
    logger.info("📱 访问地址: http://localhost:5000")
    logger.info("📄 API 文档: http://localhost:5000/docs")
    logger.info("💡 按 Ctrl+C 停止服务")
    uvicorn.run("web_app:app", host="0.0.0.0", port=5000, reload=False)