# retrieve.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import asyncio
import uuid
import time
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# 假设 PROCESS_POOL 在你的 agent_thread_pool 中已定义
# 如果它是 ProcessPoolExecutor，则可以直接传列表过去
from utils.agent_thread_pool import PROCESS_POOL 
from utils.retrieve_process import batch_init_and_retrieve

# ================= 配置 =================
BATCH_SIZE = 32           # 攒够多少个请求处理一次
BATCH_TIMEOUT = 0.05      # 最大等待时间 (秒)，即 50ms
# =======================================

# 全局队列和 Pending 映射
_request_queue = asyncio.Queue()
_pending_futures: Dict[str, asyncio.Future] = {}
_processor_task = None

class vector_store_args(BaseModel): 
    query: str = Field(..., description="查询的内容")

async def _batch_processor_loop():
    """
    后台协程：不断从队列取请求，攒批，然后提交到进程池执行
    """
    buffer: List[Dict[str, Any]] = []
    last_flush_time = time.time()

    while True:
        try:
            # 尝试获取一个请求
            # 使用 wait_for 设置超时，以便定期检查是否要 flush
            try:
                item = await asyncio.wait_for(_request_queue.get(), timeout=BATCH_TIMEOUT)
                buffer.append(item)
                last_flush_time = time.time() # 重置计时器
            except asyncio.TimeoutError:
                pass # 超时，检查是否需要 flush

            # 判断是否触发 Flush 条件：数量达标 OR 时间达标且有数据
            should_flush = False
            if len(buffer) >= BATCH_SIZE:
                should_flush = True
            elif len(buffer) > 0 and (time.time() - last_flush_time) >= BATCH_TIMEOUT:
                should_flush = True

            if should_flush and buffer:
                # 执行批量处理
                current_batch = buffer
                buffer = []
                
                # 提取 ID 和 Query
                request_ids = [item['id'] for item in current_batch]
                queries = [item['query'] for item in current_batch]
                
                logger_info = f"Processing batch: {len(queries)} queries"
                # 可以在这里打印日志，如果需要的话
                
                # 【关键】提交到进程池执行批量函数
                # PROCESS_POOL.submit 返回一个 concurrent.futures.Future
                # 我们需要将其转换为 asyncio.Future 或者在一个线程中等待它
                loop = asyncio.get_running_loop()
                
                try:
                    # 在默认线程池中运行阻塞的进程池调用，避免阻塞事件循环
                    results = await loop.run_in_executor(
                        None, 
                        lambda: PROCESS_POOL.submit(batch_init_and_retrieve, queries).result()
                    )
                    
                    # 分发结果
                    for i, req_id in enumerate(request_ids):
                        if req_id in _pending_futures:
                            future = _pending_futures.pop(req_id)
                            if not future.done():
                                future.set_result(results[i])
                        else:
                            # 可能超时被清理了
                            pass
                            
                except Exception as e:
                    # 出错时通知所有该批次的请求
                    for req_id in request_ids:
                        if req_id in _pending_futures:
                            future = _pending_futures.pop(req_id)
                            if not future.done():
                                future.set_exception(e)
                                
        except Exception as e:
            # 防止循环意外退出
            print(f"Batch processor error: {e}")
            await asyncio.sleep(1)

def _ensure_processor_started():
    """确保后台处理任务已启动"""
    global _processor_task
    if _processor_task is None or _processor_task.done():
        _processor_task = asyncio.create_task(_batch_processor_loop())

@tool(
    "retrieve_vector_store",
    description="根据输入的查询内容在法律知识库进行相关性检索...",
    args_schema=vector_store_args,
)
async def retrieve_vector_store(query: str) -> str:
    _ensure_processor_started()
    
    request_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    
    # 创建 Future 用于等待结果
    future = loop.create_future()
    _pending_futures[request_id] = future
    
    # 将请求放入队列
    await _request_queue.put({"id": request_id, "query": query})
    
    try:
        # 等待结果 (设置总超时，防止死锁)
        result = await asyncio.wait_for(future, timeout=60.0)
        return result
    except asyncio.TimeoutError:
        # 清理 pending
        if request_id in _pending_futures:
            del _pending_futures[request_id]
        raise TimeoutError("RAG retrieval timeout")
    except Exception as e:
        if request_id in _pending_futures:
            del _pending_futures[request_id]
        raise e