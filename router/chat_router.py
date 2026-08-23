"""
聊天与会话相关路由
"""

import asyncio
import json
import time
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

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
from agent_service import main_llm, make_graph
from RAG.evidence import format_evidences_for_prompt, prepare_public_citations
from RAG.evidence_verifier import verify_answer_citations
from RAG.retrieve import retrieve_vector_store
from RAG.streaming import stream_llm_text

AGENT = make_graph()

router = APIRouter(prefix="/api", tags=["聊天"])


# ========== Pydantic 模型 ==========

class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    response: str
    session_id: str
    citations: List[Dict] = Field(default_factory=list)
    verification: Dict = Field(default_factory=dict)


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


def _sse_json(payload: Dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _build_stream_answer_prompt(
    user_message: str,
    summary_context: str,
    recent_context: str,
    evidences: List[Dict],
) -> str:
    evidence_prompt = format_evidences_for_prompt(evidences) or "当前知识库未检索到可引用证据。"
    return f"""你是专业、谨慎的中文法律 AI 助手。请基于对话上下文和知识库证据回答用户问题。

对话摘要：
{summary_context or "无"}

近期对话：
{recent_context or "无"}

用户最新问题：
{user_message}

可引用证据：
{evidence_prompt}

回答要求：
1. 若涉及法律判断，每个关键结论尽量标注引用编号，例如 [1]、[2]。
2. 只能引用上方存在的证据编号，不能编造法条、案例、文件名或编号。
3. 证据不足时，明确说明“当前知识库未检索到直接依据”，不要强行下结论。
4. 给出清晰、可执行的建议；如涉及重大权益，提示咨询专业律师。
"""


async def _retrieve_stream_evidences(message: str) -> List[Dict]:
    try:
        rag_res_json = await retrieve_vector_store.ainvoke({
            "query": message,
            "exclude_ids": [],
        })
        rag_res_data = json.loads(rag_res_json)
        return rag_res_data.get("evidences", [])
    except Exception as exc:
        logger.error(f"SSE检索证据失败: {exc}")
        return []


async def _chat_stream_events(req: ChatRequest, user_id: str):
    user_timestamp = get_time()
    full_text = ""
    evidences: List[Dict] = []

    try:
        yield _sse_json({"type": "status", "stage": "retrieving", "message": "正在检索知识库"})
        summary_context, recent_context = await m_conversation_manager.get_context_for_model(req.session_id)
        evidences = await _retrieve_stream_evidences(req.message)

        yield _sse_json({"type": "status", "stage": "answering", "message": "正在生成回答"})
        prompt = _build_stream_answer_prompt(req.message, summary_context, recent_context, evidences)

        async for chunk in stream_llm_text(main_llm, [SystemMessage(content=prompt)]):
            full_text += chunk
            yield _sse_json({"type": "content", "content": chunk})

        ai_timestamp = get_time()
        citations = prepare_public_citations(evidences)
        verification = verify_answer_citations(full_text, evidences)
        await m_conversation_manager.add_message(
            req.session_id,
            req.message,
            full_text,
            user_time=user_timestamp,
            ai_time=ai_timestamp,
        )
        yield _sse_json({
            "type": "metadata",
            "session_id": req.session_id,
            "citations": citations,
            "verification": verification,
        })
    except Exception as exc:
        logger.error(f"SSE聊天失败: {exc}")
        yield _sse_json({"type": "error", "message": f"请求出错: {exc}"})
    finally:
        yield "data: [DONE]\n\n"


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
    citations = session_state.get("citations", [])
    verification = session_state.get("verification", {})

    await m_conversation_manager.add_message(
        req.session_id, req.message, ai_response,
        user_time=user_timestamp, ai_time=ai_timestamp
    )
    return ChatResponse(
        response=ai_response,
        session_id=req.session_id,
        citations=citations,
        verification=verification,
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, user_id: str = Query(...)):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")
    if len(req.message.strip()) > 2000:
        raise HTTPException(status_code=400, detail="消息长度不能超过2000字符")

    is_owner = await validate_session_ownership(req.session_id, user_id)
    if not is_owner:
        raise HTTPException(status_code=403, detail="会话不存在或无权访问")

    return StreamingResponse(
        _chat_stream_events(req, user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
    # 同步清理Redis中该会话的所有缓存key
    m_conversation_manager.delete_session(session_id)
    return {"message": "会话删除成功"}
