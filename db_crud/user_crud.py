# user_crud.py
import bcrypt
from typing import Optional
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from db_crud.base import async_engine  # 直接用原名，避免混淆
from db_crud.session_model import User


def hash_password(password: str) -> str:
    """对密码进行 bcrypt 哈希（纯 CPU 操作，无需 async）"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证明文密码与哈希是否匹配（纯 CPU 操作，无需 async）"""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


async def create_user(username: str, password: str) -> Optional[str]:
    """
    创建新用户，返回 user_id；若用户名已存在则返回 None
    """
    async with AsyncSession(async_engine) as session:
        # 检查用户名是否已存在
        existing = await session.exec(select(User).where(User.username == username))
        if existing.first():
            return None

        user = User(
            username=username,
            password_hash=hash_password(password)
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user.user_id


async def get_user_by_username(username: str) -> Optional[User]:
    """根据用户名获取用户（用于登录）"""
    async with AsyncSession(async_engine) as session:
        result = await session.exec(select(User).where(User.username == username))
        return result.first()


async def authenticate_user(username: str, password: str) -> Optional[str]:
    """
    验证用户名和密码，成功返回 user_id，失败返回 None
    """
    user = await get_user_by_username(username)
    if not user:
        return None
    if verify_password(password, user.password_hash):
        return user.user_id
    return None


async def get_user_by_id(user_id: str) -> bool:
    """根据 user_id 判断用户是否存在（FastAPI 路由中常用）"""
    async with AsyncSession(async_engine) as session:
        result = await session.exec(select(User).where(User.user_id == user_id))
        return result.first() is not None


async def delete_user(user_id: str) -> bool:
    """
    删除用户及其所有会话和消息（级联删除需手动处理）
    注意：此操作危险，建议仅用于测试或明确需求
    """
    from db_crud.chat_memory_crud import delete_all_sessions_by_user  # 避免循环导入

    async with AsyncSession(async_engine) as session:
        result = await session.exec(select(User).where(User.user_id == user_id))
        user = result.first()
        if not user:
            return False

        # 先删除该用户的所有会话和消息（确保 chat_memory_crud 也是异步的！）
        await delete_all_sessions_by_user(user_id)

        # 再删除用户
        await session.delete(user)
        await session.commit()
        return True