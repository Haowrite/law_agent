from typing import List, Dict, Optional
import uuid
from datetime import datetime

from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select, func, delete
from sqlalchemy.ext.asyncio import AsyncEngine

from db_crud.base import async_engine
from db_crud.session_model import ChatSession, ChatMessage, SummaryMessage
from db_crud.user_crud import get_user_by_id
from db_crud.base_func import count_tokens, get_time
from app_logger import database_logger as logger

class AsyncMySQLChatHistory:
    @staticmethod
    async def add_message(session_id, content: str, message_type: str, time_stamp: str) -> int:
        chat_message = ChatMessage(
            session_id=session_id,
            content=content,
            message_type=message_type,
            use_token=count_tokens(content),
            timestamp=time_stamp,
            is_summarized=False
        )
        
        # 确保 async_engine 是正确配置的 AsyncEngine
        async with AsyncSession(async_engine) as session:
            session.add(chat_message)
            
            # 关键点 1: 显式刷新，确保 ID 生成
            # 这一步会强制 SQLAlchemy 去数据库获取生成的 ID
            await session.flush() 
            
            # 关键点 2: 在 Session 关闭前获取 ID
            # 此时 ID 肯定已经有了
            msg_id = chat_message.id
            
            # 提交事务
            await session.commit()
            
        return msg_id

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
    """获取用户所有会话（含消息数）"""
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
                "created_at": s.timestamp,
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
                delete(SummaryMessage).where(SummaryMessage.session_id.in_(session_ids))
            )
            await session.exec(
                delete(ChatMessage).where(ChatMessage.session_id.in_(session_ids))
            )
            await session.exec(
                delete(ChatSession).where(ChatSession.user_id == user_id)
            )
            await session.commit()

async def get_session_detail(session_id: str, user_id: Optional[str] = None) -> Dict:
    """获取会话详情（含完整对话） - 供前端展示使用"""
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
                "role": msg.message_type,
                "use_token": msg.use_token,
                "is_summarized": msg.is_summarized
            })

        return {
            "session_id": session_id,
            "user_id": chat_session.user_id,
            "created_at": chat_session.timestamp,
            "conversation_history": conversation_history
        }

async def delete_chat_session(session_id: str, user_id: Optional[str] = None) -> bool:
    """删除指定会话"""
    async with AsyncSession(async_engine) as session:
        query = select(ChatSession).where(ChatSession.session_id == session_id)
        if user_id:
            query = query.where(ChatSession.user_id == user_id)

        chat_session = (await session.exec(query)).first()
        if not chat_session:
            return False
        
        await session.exec(delete(SummaryMessage).where(SummaryMessage.session_id == session_id))
        await session.exec(delete(ChatMessage).where(ChatMessage.session_id == session_id))
        await session.exec(delete(ChatSession).where(ChatSession.session_id == session_id))
        await session.commit()
        return True
    

async def validate_session_ownership(session_id: str, user_id: str) -> bool:
    """
    会话归属校验。
    仅检查该会话是否存在且属于指定用户，不加载任何消息。
    """
    async with AsyncSession(async_engine) as db:
        result = await db.exec(
            select(ChatSession.session_id)  # 只查询session_id，不select *
            .where(ChatSession.session_id == session_id, ChatSession.user_id == user_id)
        )
        return result.first() is not None

# ==================== 底层原子操作函数 ====================

async def get_all_messages_for_load(session_id: str):
    """加载会话所有消息"""
    async with AsyncSession(async_engine) as db:
        result = await db.exec(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.timestamp.asc())
        )
        return result.all()

async def get_all_summaries_for_load(session_id: str):
    """加载会话所有摘要"""
    async with AsyncSession(async_engine) as db:
        result = await db.exec(
            select(SummaryMessage)
            .where(SummaryMessage.session_id == session_id)
            .order_by(SummaryMessage.timestamp.asc())
        )
        return result.all()

async def db_add_summary_and_update_messages(session_id: str, summary_content: str, token_count: int, message_ids: List[str]):
    """
    事务操作：新增摘要 + 更新消息状态。
    保证原子性，要么全成功，要么全失败。
    """
    async with AsyncSession(async_engine) as session:
        try:
            # 1. 新增摘要
            summary = SummaryMessage(
                session_id=session_id,
                summary_content=summary_content,
                token_count=token_count,
                timestamp=get_time()
            )
            session.add(summary)
            await session.flush() # 获取生成的 summary_id
            
            # 2. 批量更新消息状态
            # SQLModel/SQLAlchemy bulk update
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.id.in_(message_ids))
            )
            messages = (await session.exec(stmt)).all()
            for msg in messages:
                msg.is_summarized = True
            
            await session.commit()
            return summary.summary_id # 返回ID用于Redis关联
        except Exception as e:
            await session.rollback()
            logger.error(f"数据库事务失败(新增摘要): {e}")
            raise e

async def db_delete_summaries(summary_ids: List[str]):
    """批量删除摘要"""
    if not summary_ids:
        return
    async with AsyncSession(async_engine) as session:
        await session.exec(
            delete(SummaryMessage).where(SummaryMessage.summary_id.in_(summary_ids))
        )
        await session.commit()

async def db_force_mark_summarized(message_ids: List[str]):
    """兜底策略：强制将消息标记为已摘要"""
    if not message_ids:
        return
    async with AsyncSession(async_engine) as session:
        messages = (await session.exec(
            select(ChatMessage).where(ChatMessage.id.in_(message_ids))
        )).all()
        for msg in messages:
            msg.is_summarized = True
        session.add_all(messages)
        await session.commit()