"""
聊天与会话相关路由
"""

import asyncio
import time
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app_logger import app_logger as logger
from db_crud.base_func import get_time
from db_crud.user_crud import get_user_by_id
from db_crud.chat_memory_crud import (
    create_chat_session,
    get_user_session_list,
    get_session_detail,
    delete_chat_session,
    validate_session_ownership,
)
from db_crud.session_manage import m_conversation_manager
from agent_service import make_graph

AGENT = make_graph()

router = APIRouter(prefix="/api", tags=["聊天"])


# ========== Pydantic 模型 ==========

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


# ========== 工具函数 ==========

def extract_ai_response(session_state: dict) -> str:
    """从智能体状态中提取AI回复"""
    try:
        return session_state["response"]
    except Exception as e:
        logger.error(f"提取AI回复失败: {e}")
        return "抱歉，处理您的请求时出现了错误。"


# ========== 路由 ==========

@router.post("/new_session")
async def new_session(user_id: str = Query(...)):
    user_exists = await get_user_by_id(user_id)
    if not user_exists:
        raise HTTPException(status_code=404, detail="用户不存在")
    session_id = await create_chat_session(user_id)
    return {"session_id": session_id, "message": "新会话创建成功"}


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user_id: str = Query(...)):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")
    if len(req.message.strip()) > 2000:
        raise HTTPException(status_code=400, detail="消息长度不能超过2000字符")

    user_timestamp = get_time()
    is_owner = await validate_session_ownership(req.session_id, user_id)
    if not is_owner:
        raise HTTPException(status_code=403, detail="会话不存在或无权访问")

    session_state = await AGENT.ainvoke(
        input={
            "customer_query": req.message,
            "session_id": req.session_id,
            "user_id": user_id
        },
        config={'configurable': {'thread_id': req.session_id}}
    )
    ai_timestamp = get_time()
    ai_response = extract_ai_response(session_state)

    await m_conversation_manager.add_message(
        req.session_id, req.message, ai_response,
        user_time=user_timestamp, ai_time=ai_timestamp
    )
    return ChatResponse(response=ai_response, session_id=req.session_id)


@router.get("/sessions", response_model=Dict[str, List[SessionInfo]])
async def get_sessions(user_id: str = Query(...)):
    user_exists = await get_user_by_id(user_id)
    if not user_exists:
        raise HTTPException(status_code=404, detail="用户不存在")
    sessions = await get_user_session_list(user_id)
    return {"sessions": sessions}


@router.get("/sessions/{session_id}", response_model=Dict[str, SessionDetail])
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


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user_id: str = Query(...)):
    user_exists = await get_user_by_id(user_id)
    if not user_exists:
        raise HTTPException(status_code=404, detail="用户不存在")
    success = await delete_chat_session(session_id, user_id=user_id)
    if not success:
        raise HTTPException(status_code=404, detail="会话不存在或无权删除")
    return {"message": "会话删除成功"}