import redis
import json
import time
import asyncio
from typing import List, Dict, Tuple, Optional
from db_crud.base_func import count_tokens
from app_logger import database_logger as logger
from config import SUMMARY_MODEL, TEMPERATURE, REDIS_HOST, REDIS_PORT
from model.get_model import get_llm
from app_logger import timer
from db_crud.chat_memory_crud import AsyncMySQLChatHistory, db_add_summary_and_update_messages, db_delete_summaries, db_force_mark_summarized, get_all_messages_for_load, get_all_summaries_for_load

# ======================上下文摘要智能体============================
class summary_memory_mananger:
    def __init__(self, summary_model):
        self.model_path = SUMMARY_MODEL
        self.model = summary_model
        self.system_prompt = """# 角色
你是一个专业的法律对话摘要助手，专门将多轮法律咨询对话提炼为高度精简的“问题-解决”摘要。

# 任务
请严格遵循以下原则处理输入的对话历史：
1.  **极致聚焦**：仅提取和呈现**用户的核心法律问题**与**助手提供的核心解决方案**，包括关键的法律依据、定性结论或具体行动建议。略去次要细节、背景寒暄和重复信息。
2.  **结构化输出**：强制将信息压缩并组织到以下两个固定部分中，确保逻辑直接、语言极度精炼。

# 输出要求
- **输出格式**：必须严格按照以下两段式结构输出，无需任何额外标题：
  **【用户咨询问题】**
  （用一句高度概括的话，点明用户遭遇的核心法律问题或争议点）

  **【解决方案摘要】**
  （用一段话，清晰列举助手提供的关键法律意见、核心建议或行动步骤。可包含关键法律点、结论和后续操作。）

- **强制要求**：
  - 内容必须完全基于对话原文，严禁任何增补、推断或评论。
  - 语言必须客观、书面、精炼，直接陈述事实与建议。
  - 除上述两部分外，禁止输出任何其他文字、符号或解释。
  
现在请对以下多轮对话内容进行摘要：
{conversation}
"""
    @timer("摘要对话")
    def summary_conversation(self, conversation: str):
        message = [{'role': 'system', 'content': self.system_prompt.format(conversation=conversation)}]
        return self.model.invoke(message).content

# =================================Redis上下文管理器=================================

class ConversationManager:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0,
                 token_threshold=26000, m=0.5, compress_batch_size=10, ttl_seconds=86400):
        
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
        self.token_threshold = token_threshold
        self.compress_batch_size = compress_batch_size
        self.ttl_seconds = ttl_seconds
        
        # 阈值划分
        self.max_unsummarized_tokens = int(token_threshold * m)
        self.max_summary_tokens = int(token_threshold * (1 - m))

        # Keys
        self.key_unsummarized = "session:{}:unsummarized"  # List
        self.key_summarized = "session:{}:summarized"      # List (前端展示用)
        self.key_summary = "session:{}:summary"            # List
        self.key_meta = "session:{}:meta"                  # Hash

    def _get_meta(self, session_id: str) -> Dict:
        key = self.key_meta.format(session_id)
        data = self.redis_client.hgetall(key)
        return {
            "total": int(data.get("total", 0)),
            "unsum": int(data.get("unsum", 0)),
            "sum": int(data.get("sum", 0))
        }

    def _set_meta(self, session_id: str, total: int, unsum: int, sum_val: int):
        key = self.key_meta.format(session_id)
        self.redis_client.hset(key, mapping={
            "total": str(total), 
            "unsum": str(unsum), 
            "sum": str(sum_val)
        })
        self.redis_client.expire(key, self.ttl_seconds)

    def _incr_meta(self, session_id: str, field: str, amount: int):
        key = self.key_meta.format(session_id)
        val = self.redis_client.hincrby(key, field, amount)
        self.redis_client.expire(key, self.ttl_seconds)
        return val

    async def add_message(self, session_id: str, user_message: str, agent_message: str, user_time, ai_time) -> bool:
        """添加消息：先落库，再刷缓存"""
        try:
            # 1. 落库
            u_id = await AsyncMySQLChatHistory.add_message(
                session_id, user_message, "user", user_time
            )
            a_id = await AsyncMySQLChatHistory.add_message(
                session_id, agent_message, "ai", ai_time
            )

            # 2. 计算Token
            u_tokens = count_tokens(user_message)
            a_tokens = count_tokens(agent_message)
            total_new_tokens = u_tokens + a_tokens

            # 3. 更新Redis
            key_un = self.key_unsummarized.format(session_id)
            
            u_msg = {"id": u_id, "role": "user", "content": user_message, "tokens": u_tokens}
            a_msg = {"id": a_id, "role": "agent", "content": agent_message, "tokens": a_tokens}

            pipe = self.redis_client.pipeline()
            pipe.rpush(key_un, json.dumps(u_msg))
            pipe.rpush(key_un, json.dumps(a_msg))
            pipe.expire(key_un, self.ttl_seconds)
            pipe.execute()

            new_unsum_tokens = self._incr_meta(session_id, "unsum", total_new_tokens)
            self._incr_meta(session_id, "total", total_new_tokens)

            # 4. 检查阈值
            if new_unsum_tokens > self.max_unsummarized_tokens:
                # 异步触发压缩，不阻塞当前响应
                await asyncio.create_task(self._check_and_compress(session_id))

            return True
        except Exception as e:
            logger.error(f"添加消息失败: {e}")
            return False

    async def _check_and_compress(self, session_id: str):
        """检查并执行压缩，保证数据库与缓存一致性"""
        key_un = self.key_unsummarized.format(session_id)
        key_summ = self.key_summary.format(session_id)
        key_sumed = self.key_summarized.format(session_id)

        while True:
            current_unsum = self._get_meta(session_id)['unsum']
            if current_unsum <= self.max_unsummarized_tokens:
                break
            
            # 1. 提取待压缩消息
            count = self.compress_batch_size * 2
            msgs_json = self.redis_client.lrange(key_un, 0, count - 1)
            if len(msgs_json) < 2: break

            msgs = [json.loads(m) for m in msgs_json]
            batch_tokens = sum(m['tokens'] for m in msgs)
            msg_ids = [m['id'] for m in msgs]

            # 2. 生成摘要
            conv_text = "\n".join([f"{m['role']}:{m['content']}" for m in msgs])
            try:
                summary_content = memory_summarier.summary_conversation(conv_text)
                summary_tokens = count_tokens(summary_content)
            except Exception as e:
                logger.error(f"摘要生成失败: {e}")
                break # 避免无限重试

            # 3. 更新数据库 (关键一致性步骤)
            # 如果此处失败，抛出异常，Redis不会被修改，数据处于"未压缩但超限"状态，下次加载或重试时可修复
            try:
                db_summary_id = await db_add_summary_and_update_messages(
                    session_id, summary_content, summary_tokens, msg_ids
                )
            except Exception as e:
                logger.error(f"数据库更新失败，中止压缩流程: {e}")
                break

            # 4. 更新Redis (数据库成功后执行)
            pipe = self.redis_client.pipeline()
            # 移除未摘要
            for _ in msgs: pipe.lpop(key_un)
            
            # 加入已摘要列表 (备份)
            for m in msgs: pipe.rpush(key_sumed, json.dumps(m))
            pipe.expire(key_sumed, self.ttl_seconds)

            # 加入摘要列表
            summary_obj = {
                "db_id": db_summary_id,
                "content": summary_content,
                "tokens": summary_tokens,
                "timestamp": time.time()
            }
            pipe.rpush(key_summ, json.dumps(summary_obj))
            pipe.expire(key_summ, self.ttl_seconds)
            pipe.execute()

            # 5. 更新计数
            self._incr_meta(session_id, "unsum", -batch_tokens)
            new_sum_tokens = self._incr_meta(session_id, "sum", summary_tokens)

            logger.info(f"压缩完成，新摘要Token: {summary_tokens}")

            # 6. 检查摘要是否超限
            if new_sum_tokens > self.max_summary_tokens:
                await self._prune_old_summaries(session_id)

    async def _prune_old_summaries(self, session_id: str):
        """删除旧摘要：先删数据库，再删缓存"""
        key_summ = self.key_summary.format(session_id)
        
        while True:
            current_sum = self._get_meta(session_id)['sum']
            if current_sum <= self.max_summary_tokens:
                break
            
            # 1. 查看最旧的摘要 (不立即弹出)
            oldest_json = self.redis_client.lindex(key_summ, 0)
            if not oldest_json: break
            
            oldest = json.loads(oldest_json)
            db_id = oldest.get('db_id')
            removed_tokens = oldest['tokens']

            # 2. 删除数据库
            if db_id:
                try:
                    await db_delete_summaries([db_id])
                except Exception as e:
                    logger.error(f"数据库删除摘要失败 {db_id}: {e}")
                    break # 中止，待下次修复

            # 3. 删除缓存
            self.redis_client.lpop(key_summ)
            
            # 4. 更新计数
            self._incr_meta(session_id, "sum", -removed_tokens)
            logger.info(f"删除旧摘要 {db_id} 完成")

    async def ensure_session_loaded(self, session_id: str) -> bool:
        """加载会话：包含严格的兜底修复逻辑"""
        key_un = self.key_unsummarized.format(session_id)
        
        if self.redis_client.exists(key_un):
            return True

        logger.info(f"加载会话 {session_id} 并执行一致性检查...")
        
        # 1. 从数据库加载全量数据
        all_msgs = await get_all_messages_for_load(session_id)
        all_summs = await get_all_summaries_for_load(session_id)

        # 2. 处理消息：分离 & 兜底修复
        # 策略：倒序遍历，填满 unsummarized 阈值，剩余的归入 summarized
        unsum_list = []
        sum_list = [] # 已摘要的消息（原文）
        current_unsum_tokens = 0
        
        # 倒序处理（最新的在前）
        reversed_msgs = list(reversed(all_msgs))
        unsum_ids_need_fix = [] # 需要修复状态的消息ID

        for msg in reversed_msgs:
            msg_data = {
                "id": msg.id, "role": msg.message_type, 
                "content": msg.content, "tokens": msg.use_token
            }
            
            # 如果未填满阈值，且数据库状态为未摘要，则放入未摘要列表
            if current_unsum_tokens < self.max_unsummarized_tokens and not msg.is_summarized:
                unsum_list.append(msg_data)
                current_unsum_tokens += msg.use_token
            else:
                # 否则放入已摘要列表
                sum_list.append(msg_data)
                
                # 兜底修复：如果该消息在数据库中状态为 False，说明上次异步更新失败
                if not msg.is_summarized:
                    unsum_ids_need_fix.append(msg.id)

        # 执行数据库修复：强制标记为已摘要
        if unsum_ids_need_fix:
            logger.warning(f"兜底修复：检测到 {len(unsum_ids_need_fix)} 条消息状态不一致，正在修正...")
            await asyncio.create_task(db_force_mark_summarized(unsum_ids_need_fix))

        # 恢复正序
        unsum_list.reverse()
        sum_list.reverse()

        # 3. 处理摘要：分离 & 兜底修复
        keep_summaries = []
        delete_summary_ids = []
        current_sum_tokens = 0
        
        # 倒序遍历（最新的在前）
        for summ in reversed(all_summs):
            if current_sum_tokens < self.max_summary_tokens:
                keep_summaries.append({
                    "db_id": summ.summary_id,
                    "content": summ.summary_content,
                    "tokens": summ.token_count,
                    "timestamp": summ.timestamp
                })
                current_sum_tokens += summ.token_count
            else:
                # 超出阈值，需要删除
                delete_summary_ids.append(summ.summary_id)

        # 执行数据库修复：删除多余摘要
        if delete_summary_ids:
            logger.warning(f"兜底修复：检测到 {len(delete_summary_ids)} 条摘要超出阈值，正在删除...")
            await asyncio.create_task(db_delete_summaries(delete_summary_ids))

        # 4. 写入 Redis
        pipe = self.redis_client.pipeline()
        
        # 写入未摘要
        pipe.delete(key_un)
        for m in unsum_list: pipe.rpush(key_un, json.dumps(m))
        pipe.expire(key_un, self.ttl_seconds)

        # 写入已摘要消息
        key_sumed = self.key_summarized.format(session_id)
        pipe.delete(key_sumed)
        for m in sum_list: pipe.rpush(key_sumed, json.dumps(m))
        pipe.expire(key_sumed, self.ttl_seconds)

        # 写入摘要
        key_summ = self.key_summary.format(session_id)
        pipe.delete(key_summ)
        for s in reversed(keep_summaries): pipe.rpush(key_summ, json.dumps(s)) # 恢复时间正序
        pipe.expire(key_summ, self.ttl_seconds)

        # 写入Meta
        total_tokens = current_unsum_tokens + current_sum_tokens
        pipe.hset(self.key_meta.format(session_id), mapping={
            "total": str(total_tokens),
            "unsum": str(current_unsum_tokens),
            "sum": str(current_sum_tokens)
        })
        pipe.expire(self.key_meta.format(session_id), self.ttl_seconds)
        
        pipe.execute()
        return True

    def delete_session(self, session_id: str):
        """删除会话在Redis中的所有相关key"""
        keys = [
            self.key_unsummarized.format(session_id),
            self.key_summarized.format(session_id),
            self.key_summary.format(session_id),
            self.key_meta.format(session_id),
        ]
        deleted = self.redis_client.delete(*keys)
        logger.info(f"已删除会话 {session_id} 的 Redis 缓存，共清理 {deleted} 个key")
        return deleted

    async def get_context_for_model(self, session_id: str) -> Tuple[str, str]:
        await self.ensure_session_loaded(session_id)
        
        summaries = self.redis_client.lrange(self.key_summary.format(session_id), 0, -1)
        summ_content = "\n".join([json.loads(s)['content'] for s in summaries])

        messages = self.redis_client.lrange(self.key_unsummarized.format(session_id), 0, -1)
        unsum_content = "\n".join([f"{json.loads(m)['role']}:{json.loads(m)['content']}" for m in messages])

        return summ_content, unsum_content
    
    async def get_lists_for_frontend(self, session_id: str) -> Dict:
        await self.ensure_session_loaded(session_id)
        
        unsum = [json.loads(m) for m in self.redis_client.lrange(self.key_unsummarized.format(session_id), 0, -1)]
        sumd = [json.loads(m) for m in self.redis_client.lrange(self.key_summarized.format(session_id), 0, -1)]
        
        return {"unsummarized": unsum, "summarized": sumd}

# 初始化
summary_llm = get_llm(SUMMARY_MODEL, TEMPERATURE)
memory_summarier = summary_memory_mananger(summary_llm)
m_conversation_manager = ConversationManager(redis_host=REDIS_HOST, redis_port=REDIS_PORT)