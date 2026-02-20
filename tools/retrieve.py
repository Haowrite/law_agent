from langchain_core.tools import tool
from pydantic import BaseModel, Field

import asyncio
from utils.agent_thread_pool import PROCESS_POOL
from utils.retrieve_process import init_and_retrieve

class vector_store_args(BaseModel): 
    query: str = Field(..., description="查询的内容")


@tool(
    "retrieve_vector_store",
    description="根据输入的查询内容在法律知识库进行相关性检索...",
    args_schema=vector_store_args,
)
async def retrieve_vector_store(query: str) -> str:
    loop = asyncio.get_running_loop()
    # 调用子进程中的 init_and_retrieve
    result = await loop.run_in_executor(PROCESS_POOL, init_and_retrieve, query)
    return result