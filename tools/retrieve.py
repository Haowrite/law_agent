# retrieve.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import asyncio
import uuid
import time
import torch # 需要引入 torch 来清理缓存
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from utils.agent_thread_pool import PROCESS_POOL 
from utils.retrieve_process import batch_init_and_retrieve

# ================= 配置 =================
INITIAL_BATCH_SIZE = 32   # 初始尝试的批量大小
MIN_BATCH_SIZE = 1        # 最小降级到 1 (串行)
BATCH_TIMEOUT = 0.05      # 最大等待时间 (秒)
# =======================================

_request_queue = asyncio.Queue()
_pending_futures: Dict[str, asyncio.Future] = {}
_processor_task = None

class vector_store_args(BaseModel): 
    query: str = Field(..., description="查询的内容")

def _run_batch_with_fallback(queries: List[str]) -> List[str]:
    """
    【核心优化】带显存保护机制的批量执行函数
    如果显存不足，自动将批次减半重试，直到成功或降至最小批次
    """
    current_batch_size = INITIAL_BATCH_SIZE
    work_queue = list(queries) # 待处理的查询列表
    final_results = [None] * len(queries) # 预分配结果列表，保持顺序
    
    # 我们需要记录原始索引，以便将结果放回正确的位置
    # 但为了简化，这里采用一种策略：
    # 如果整体批次失败，我们将其拆分为更小的块递归处理
    
    def process_chunk(chunk_queries: List[str], start_index: int):
        """尝试处理一个切片，如果 OOM 则递归拆分"""
        if not chunk_queries:
            return
            
        try:
            # 尝试执行
            # 注意：这里需要 modify retrieve_process.py 让它支持处理切片并返回对应结果
            # 或者我们在这里循环调用单条 (效率低但保底)
            # 为了利用现有代码，我们假设 batch_init_and_retrieve 能处理任意长度列表
            
            results = batch_init_and_retrieve(chunk_queries)
            
            # 填入结果
            for i, res in enumerate(results):
                final_results[start_index + i] = res
                
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "cuDNN error" in str(e):
                # 显存不足警告
                print(f"⚠️  Detect OOM! Batch size {len(chunk_queries)} too large. Splitting...")
                
                # 强制清理 PyTorch 缓存 (关键步骤)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 递归拆分：将当前批次一分为二
                mid = len(chunk_queries) // 2
                if mid == 0:
                    # 如果已经是一个一个了还 OOM，那是真的没办法了，抛出异常
                    raise e
                
                # 处理前半部分
                process_chunk(chunk_queries[:mid], start_index)
                # 处理后半部分
                process_chunk(chunk_queries[mid:], start_index + mid)
            else:
                # 其他错误直接抛出
                raise e

    # 开始处理整个列表
    process_chunk(work_queue, 0)
    
    return final_results

async def _batch_processor_loop():
    buffer: List[Dict[str, Any]] = []
    last_flush_time = time.time()

    while True:
        try:
            try:
                item = await asyncio.wait_for(_request_queue.get(), timeout=BATCH_TIMEOUT)
                buffer.append(item)
                last_flush_time = time.time()
            except asyncio.TimeoutError:
                pass

            should_flush = False
            if len(buffer) >= INITIAL_BATCH_SIZE:
                should_flush = True
            elif len(buffer) > 0 and (time.time() - last_flush_time) >= BATCH_TIMEOUT:
                should_flush = True

            if should_flush and buffer:
                current_batch = buffer
                buffer = []
                
                request_ids = [item['id'] for item in current_batch]
                queries = [item['query'] for item in current_batch]
                
                loop = asyncio.get_running_loop()
                
                try:
                    # 使用带 fallback 的包装函数
                    results = await loop.run_in_executor(
                        None, 
                        lambda: PROCESS_POOL.submit(_run_batch_with_fallback, queries).result()
                    )
                    
                    for i, req_id in enumerate(request_ids):
                        if req_id in _pending_futures:
                            future = _pending_futures.pop(req_id)
                            if not future.done():
                                future.set_result(results[i])
                            
                except Exception as e:
                    print(f"❌ Batch processing failed completely: {e}")
                    for req_id in request_ids:
                        if req_id in _pending_futures:
                            future = _pending_futures.pop(req_id)
                            if not future.done():
                                future.set_exception(e)
                                
        except Exception as e:
            print(f"Batch processor loop error: {e}")
            await asyncio.sleep(1)

def _ensure_processor_started():
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
    
    future = loop.create_future()
    _pending_futures[request_id] = future
    
    await _request_queue.put({"id": request_id, "query": query})
    
    try:
        result = await asyncio.wait_for(future, timeout=60.0)
        return result
    except asyncio.TimeoutError:
        if request_id in _pending_futures:
            del _pending_futures[request_id]
        raise TimeoutError("RAG retrieval timeout")
    except Exception as e:
        if request_id in _pending_futures:
            del _pending_futures[request_id]
        raise e