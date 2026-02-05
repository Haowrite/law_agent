#!/usr/bin/env python3
"""
法律AI助手
- 全链路异步（数据库 + 智能体）
- 使用 lifespan 初始化数据库
- 路由全部异步化
"""

import os
os.environ['MODELSCOPE_CACHE'] = "F:/m_code/llm_project/Fine-tune/hf_hub"
os.environ['HF_HOME'] = "F:/m_code/llm_project/Fine-tune/hf_hub"

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

from db_crud.user_crud import create_user, authenticate_user, get_user_by_id
from db_crud.chat_memory_crud import (
    AsyncMySQLChatHistory,
    create_chat_session,
    get_user_session_list,
    get_session_detail,
    delete_chat_session
)
from db_crud.base import init_db  # 异步建表函数

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
        if "response" in session_state and session_state["response"]:
            return session_state["response"]
        if "messages" in session_state and session_state["messages"]:
            return session_state["messages"][-1].content
        return "抱歉，我无法理解您的问题。"
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
    logger.info("🚀 初始化数据库...")
    await init_db()
    yield
    logger.info("🛑 应用关闭")

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
    
    # 校验会话归属
    try:
        await get_session_detail(req.session_id, user_id=user_id)
    except ValueError:
        raise HTTPException(status_code=403, detail="会话不存在或无权访问")

    await AsyncMySQLChatHistory.add_message(req.session_id, req.message, "user")
    
    session_state = await AGENT.ainvoke(
        input={
            "messages": [HumanMessage(content=req.message)],
            "customer_query": req.message,
            "session_id": req.session_id,
            "user_id": user_id
        },
        config={'configurable': {'thread_id': req.session_id}}
    )

    ai_response = extract_ai_response(session_state)
    await AsyncMySQLChatHistory.add_message(req.session_id, ai_response, "ai")
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