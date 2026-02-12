# base.py
from config import DATABASE_URL
from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import create_async_engine
from app_logger import database_logger as logger

# 创建异步引擎（不立即建表）
def get_async_engine():
    try:
        engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True, pool_size=20, max_overflow=20, pool_recycle=3600)
        logger.info(f"异步数据库引擎创建成功: {DATABASE_URL}")
        return engine
    except Exception as e:
        logger.error(f"创建异步引擎失败: {e}")
        raise

async_engine = get_async_engine()


# 异步初始化数据库表（应在应用启动时调用）
async def init_db():
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        logger.info("数据库表初始化完成")
    except Exception as e:
        logger.error(f"数据库表初始化失败: {e}")
        raise