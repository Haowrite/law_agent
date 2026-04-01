# user_crud.py
import bcrypt
import random
import smtplib
import redis
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.utils import formataddr
from typing import Optional
from sqlmodel import select, or_
from sqlmodel.ext.asyncio.session import AsyncSession
from db_crud.base import async_engine
from db_crud.session_model import User
from config import (
    REDIS_HOST, REDIS_PORT,
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_SENDER_NAME,
    VERIFY_CODE_EXPIRE,
)

# Redis 客户端（用于验证码缓存）
_redis_client = redis.Redis(host=REDIS_HOST, port=int(REDIS_PORT), db=0, decode_responses=True)

# ======================== 密码工具 ========================

def hash_password(password: str) -> str:
    """对密码进行 bcrypt 哈希（纯 CPU 操作，无需 async）"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证明文密码与哈希是否匹配（纯 CPU 操作，无需 async）"""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


# ======================== 验证码：生成、缓存、发送 ========================

def generate_verify_code(length: int = 6) -> str:
    """生成指定位数的纯数字验证码"""
    return ''.join([str(random.randint(0, 9)) for _ in range(length)])


def store_verify_code(email: str, code: str) -> None:
    """将验证码存入 Redis，带过期时间"""
    key = f"verify_code:{email}"
    _redis_client.setex(key, VERIFY_CODE_EXPIRE, code)


def check_verify_code(email: str, code: str) -> bool:
    """校验验证码是否匹配，校验成功后删除"""
    key = f"verify_code:{email}"
    cached = _redis_client.get(key)
    if cached and cached == code:
        _redis_client.delete(key)
        return True
    return False


def send_verify_email(to_email: str, code: str) -> bool:
    """
    通过 QQ 邮箱 SMTP 发送验证码邮件
    返回 True 表示发送成功，False 表示失败
    """
    subject = "AI 法律助手 — 邮箱验证码"
    html_body = f"""
    <div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:24px;
                background:#f9f9f9;border-radius:12px;border:1px solid #e0e0e0">
      <h2 style="color:#333;margin-bottom:8px">邮箱验证码</h2>
      <p style="color:#555;font-size:14px;line-height:1.6">
        您正在注册 <b>AI 法律助手</b>，验证码如下：
      </p>
      <div style="font-size:32px;font-weight:700;letter-spacing:6px;text-align:center;
                  color:#4c8df6;padding:16px 0">{code}</div>
      <p style="color:#888;font-size:12px">
        验证码 {VERIFY_CODE_EXPIRE // 60} 分钟内有效，请勿泄露给他人。
      </p>
    </div>
    """
    msg = MIMEMultipart("alternative")
    msg["From"] = formataddr((str(Header(SMTP_SENDER_NAME, 'utf-8')), SMTP_USER))
    msg["To"] = to_email
    msg["Subject"] = Header(subject, 'utf-8')
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, [to_email], msg.as_string())
        return True
    except Exception as e:
        print(f"[SMTP ERROR] 发送验证码到 {to_email} 失败: {e}")
        return False


# ======================== 用户 CRUD ========================

async def create_user(username: str, password: str, email: Optional[str] = None) -> Optional[str]:
    """
    创建新用户，返回 user_id；若用户名或邮箱已存在则返回 None
    """
    async with AsyncSession(async_engine) as session:
        # 检查用户名是否已存在
        existing = await session.exec(select(User).where(User.username == username))
        if existing.first():
            return None

        # 检查邮箱是否已存在
        if email:
            existing_email = await session.exec(select(User).where(User.email == email))
            if existing_email.first():
                return None

        user = User(
            username=username,
            email=email,
            password_hash=hash_password(password)
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user.user_id


async def check_email_exists(email: str) -> bool:
    """检查邮箱是否已被注册"""
    async with AsyncSession(async_engine) as session:
        result = await session.exec(select(User).where(User.email == email))
        return result.first() is not None


async def get_user_by_username(username: str) -> Optional[User]:
    """根据用户名获取用户（用于登录）"""
    async with AsyncSession(async_engine) as session:
        result = await session.exec(select(User).where(User.username == username))
        return result.first()


async def get_user_by_email(email: str) -> Optional[User]:
    """根据邮箱获取用户（用于登录）"""
    async with AsyncSession(async_engine) as session:
        result = await session.exec(select(User).where(User.email == email))
        return result.first()


async def authenticate_user(account: str, password: str) -> Optional[str]:
    """
    验证用户名或邮箱 + 密码，成功返回 user_id，失败返回 None
    自动判断 account 是邮箱还是用户名
    """
    if '@' in account:
        user = await get_user_by_email(account)
    else:
        user = await get_user_by_username(account)

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