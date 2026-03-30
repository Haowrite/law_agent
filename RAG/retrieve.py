from os import wait

from langchain_core.tools import tool
from pydantic import BaseModel, Field
import asyncio
import uuid
import time
import torch # 需要引入 torch 来清理缓存
import json as _json
from typing import Dict, List, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from utils.agent_thread_pool import PROCESS_POOL 
from RAG.retrieve_process import batch_init_and_retrieve

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
    exclude_ids: list = Field(default_factory=list, description="已检索过的条文ID列表，用于去重")

def _run_batch_with_fallback(queries: List[str], exclude_ids_list: List[Optional[set]] = None) -> List[Any]:
    """
    【核心优化】带显存保护机制的批量执行函数
    如果显存不足，自动将批次减半重试，直到成功或降至最小批次
    """
    current_batch_size = INITIAL_BATCH_SIZE
    work_queue = list(queries) # 待处理的查询列表
    final_results = [None] * len(queries) # 预分配结果列表，保持顺序

    if exclude_ids_list is None:
        exclude_ids_list = [None] * len(queries)

    def process_chunk(chunk_queries: List[str], chunk_exclude_ids: List[Optional[set]], start_index: int):
        """尝试处理一个切片，如果 OOM 则递归拆分"""
        if not chunk_queries:
            return

        try:
            results = batch_init_and_retrieve(chunk_queries, exclude_ids_list=chunk_exclude_ids)

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
                    raise e

                # 处理前半部分
                process_chunk(chunk_queries[:mid], chunk_exclude_ids[:mid], start_index)
                # 处理后半部分
                process_chunk(chunk_queries[mid:], chunk_exclude_ids[mid:], start_index + mid)
            else:
                raise e

    # 开始处理整个列表
    process_chunk(work_queue, list(exclude_ids_list), 0)
    
    return final_results

async def _batch_processor_loop():
    buffer: List[Dict[str, Any]] = []
    last_flush_time = time.time()

    while True:
        try:
            try:
                wait_time = BATCH_TIMEOUT
                if len(buffer) == 0:
                    wait_time = None  # 如果缓冲区为空，等待直到有新请求
                else:
                    wait_time = max(5, BATCH_TIMEOUT - (time.time() - last_flush_time))
                item = await asyncio.wait_for(_request_queue.get(), timeout=wait_time)
                if len(buffer) == 0:
                    last_flush_time = time.time()  # 新批次开始，重置计时 
                buffer.append(item)
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
                exclude_ids_list = [item.get('exclude_ids') for item in current_batch]

                loop = asyncio.get_running_loop()

                try:
                    # 使用带 fallback 的包装函数
                    results = await loop.run_in_executor(
                        None,
                        lambda: PROCESS_POOL.submit(_run_batch_with_fallback, queries, exclude_ids_list).result()
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
    description="根据输入的查询内容在法律知识库进行相关性检索，支持传入已检索条文ID进行去重...",
    args_schema=vector_store_args,
)
async def retrieve_vector_store(query: str, exclude_ids: list = None) -> str:
    """
    返回 JSON 字符串，格式: {"text": "检索结果文本", "retrieved_ids": ["id1", "id2", ...]}
    """
    _ensure_processor_started()

    # 将 exclude_ids 列表转为 set 以便快速查找
    exclude_ids_set = set(exclude_ids) if exclude_ids else None

    request_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()

    future = loop.create_future()
    _pending_futures[request_id] = future

    await _request_queue.put({
        "id": request_id,
        "query": query,
        "exclude_ids": exclude_ids_set,
    })

    try:
        result = await asyncio.wait_for(future, timeout=60.0)
        # result 是 (text, new_ids) 的元组
        text, new_ids = result
        return _json.dumps({"text": text, "retrieved_ids": new_ids}, ensure_ascii=False)
    except asyncio.TimeoutError:
        if request_id in _pending_futures:
            del _pending_futures[request_id]
        raise TimeoutError("RAG retrieval timeout")
    except Exception as e:
        if request_id in _pending_futures:
            del _pending_futures[request_id]
        raise e