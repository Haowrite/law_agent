from typing import List, Dict, Optional
import uuid
from datetime import datetime

from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select, func, delete
from sqlalchemy.ext.asyncio import AsyncEngine

from db_crud.base import async_engine
from db_crud.session_model import ChatSession, ChatMessage
from db_crud.user_crud import get_user_by_id
from app_logger import database_logger as logger


class AsyncMySQLChatHistory:
    @staticmethod
    async def add_message(session_id, content: str, message_type: str) -> None:
        chat_message = ChatMessage(
            session_id=session_id,
            content=content,
            message_type=message_type,
        )
        async with AsyncSession(async_engine) as session:
            session.add(chat_message)
            await session.commit()


# ==================== 会话管理函数 ====================

async def create_chat_session(user_id: str) -> str:
    """为用户创建新会话"""
    if not await get_user_by_id(user_id):
        raise ValueError(f"用户不存在: {user_id}")

    session_id = str(uuid.uuid4())
    chat_session = ChatSession(
        session_id=session_id,
        user_id=user_id,
        timestamp=datetime.utcnow()
    )
    async with AsyncSession(async_engine) as session:
        session.add(chat_session)
        await session.commit()
    return session_id


async def get_user_session_list(user_id: str) -> List[Dict]:
    """获取用户所有会话摘要（含消息数）"""
    if not await get_user_by_id(user_id):
        return []

    async with AsyncSession(async_engine) as db:
        sessions = await db.exec(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.timestamp.desc())
        )
        sessions = sessions.all()

        result = []
        for s in sessions:
            msg_count_result = await db.exec(
                select(func.count(ChatMessage.id))
                .where(ChatMessage.session_id == s.session_id)
            )
            msg_count = msg_count_result.first() or 0

            result.append({
                "session_id": s.session_id,
                "created_at": s.timestamp,  # 直接转时间戳
                "message_count": msg_count
            })
        return result


async def delete_all_sessions_by_user(user_id: str) -> None:
    """删除用户所有会话及消息"""
    async with AsyncSession(async_engine) as session:
        session_ids_result = await session.exec(
            select(ChatSession.session_id).where(ChatSession.user_id == user_id)
        )
        session_ids = session_ids_result.all()

        if session_ids:
            await session.exec(
                delete(ChatMessage).where(ChatMessage.session_id.in_(session_ids))
            )
            await session.exec(
                delete(ChatSession).where(ChatSession.user_id == user_id)
            )
            await session.commit()


async def get_session_detail(session_id: str, user_id: Optional[str] = None) -> Dict:
    """获取会话详情（含完整对话）"""
    async with AsyncSession(async_engine) as db:
        query = select(ChatSession).where(ChatSession.session_id == session_id)
        if user_id:
            query = query.where(ChatSession.user_id == user_id)

        chat_session = (await db.exec(query)).first()
        if not chat_session:
            raise ValueError(f"会话不存在或无权访问: {session_id}")

        messages = await db.exec(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.timestamp.asc())
        )
        messages = messages.all()

        conversation_history = []
        for msg in messages:
            if msg.message_type not in ["user", "ai"]:
                continue
            conversation_history.append({
                "is_user": msg.message_type == "user",
                "content": msg.content,
                "timestamp": msg.timestamp,
                "role": msg.message_type
            })

        return {
            "session_id": session_id,
            "user_id": chat_session.user_id,
            "created_at": chat_session.timestamp,
            "conversation_history": conversation_history
        }


async def delete_chat_session(session_id: str, user_id: Optional[str] = None) -> bool:
    """删除指定会话（可选校验用户）"""
    async with AsyncSession(async_engine) as session:
        query = select(ChatSession).where(ChatSession.session_id == session_id)
        if user_id:
            query = query.where(ChatSession.user_id == user_id)

        chat_session = (await session.exec(query)).first()
        if not chat_session:
            return False

        await session.exec(delete(ChatMessage).where(ChatMessage.session_id == session_id))
        await session.exec(delete(ChatSession).where(ChatSession.session_id == session_id))
        await session.commit()
        return True