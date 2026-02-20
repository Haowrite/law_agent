# tasks.py
from arq.connections import RedisSettings
from db_crud.chat_memory_crud import AsyncMySQLChatHistory

REDIS_SETTINGS = RedisSettings(
    host='localhost',
    port=6379,
    database=1,
    password=None
)

async def save_chat_message(ctx, session_id: str, content: str, role: str, timestamp: str):
    try:
        await AsyncMySQLChatHistory.add_message(
            session_id=session_id,
            content=content,
            message_type=role,
            time_stamp=timestamp
        )
        return f"Saved {role} message for session {session_id}"
    except Exception as e:
        print(f"❌ Failed to save message to DB: {e}")
        raise

# ⬇️ 关键：arq worker 需要这个才能启动！
class WorkerSettings:
    redis_settings = REDIS_SETTINGS
    functions = [save_chat_message]
    max_jobs = 20
    job_timeout = 30