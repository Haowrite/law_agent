import asyncio

from config import DATABASE_URL
from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app_logger import database_logger as logger

# ========== 异步引擎（FastAPI 主进程使用） ==========

def get_async_engine():
    try:
        engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True, pool_size=20, max_overflow=30, pool_recycle=3600)
        logger.info(f"异步数据库引擎创建成功: {DATABASE_URL}")
        return engine
    except Exception as e:
        logger.error(f"创建异步引擎失败: {e}")
        raise

async_engine = get_async_engine()


# ========== 同步引擎（子进程使用，文档CRUD等） ==========
# 使用进程级单例：每个子进程只创建一次 engine 和 sessionmaker

_sync_engine = None
_SyncSessionLocal = None


def _make_sync_url(async_url: str) -> str:
    """将 async 数据库 URL 转换为 sync URL（如 mysql+aiomysql -> mysql+pymysql）"""
    return (
        async_url
        .replace("mysql+aiomysql", "mysql+pymysql")
        .replace("mysql+asyncmy", "mysql+pymysql")
        .replace("sqlite+aiosqlite", "sqlite")
    )


def get_sync_engine():
    """获取同步引擎（进程级单例，每个进程只创建一次）"""
    global _sync_engine
    if _sync_engine is None:
        try:
            sync_url = _make_sync_url(DATABASE_URL)
            _sync_engine = create_engine(
                sync_url, echo=False, pool_pre_ping=True,
                pool_size=10, max_overflow=20, pool_recycle=3600
            )
            logger.info(f"同步数据库引擎创建成功: {sync_url}")
        except Exception as e:
            logger.error(f"创建同步引擎失败: {e}")
            raise
    return _sync_engine


def get_sync_session() -> Session:
    """获取同步 Session（供子进程中使用，复用进程级单例 engine）"""
    global _SyncSessionLocal
    if _SyncSessionLocal is None:
        engine = get_sync_engine()
        _SyncSessionLocal = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    return _SyncSessionLocal()


# 异步初始化数据库表（应在应用启动时调用）
async def init_db():
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        logger.info("数据库表初始化完成")
    except Exception as e:
        logger.error(f"数据库表初始化失败: {e}")
        raise


async def wait_until_mysql_ready(retry_interval: float = 3.0):
    """
    在应用启动阶段阻塞直至 MySQL 可连并完成表初始化。
    连接失败时持续打错误日志并周期性重试，避免拖到业务请求时才暴露。
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            await init_db()
            logger.info(f"MySQL 已就绪（第 {attempt} 次尝试成功）")
            return
        except Exception as e:
            logger.error(
                f"MySQL 连接或初始化失败（第 {attempt} 次），{retry_interval}s 后重试: {e}"
            )
            await asyncio.sleep(retry_interval)
