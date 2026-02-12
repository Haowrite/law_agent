import redis
import json
import time
from typing import List, Dict, Tuple, Optional
from db_crud.base_func import count_tokens
from app_logger import database_logger as logger # 新增日志模块
from config import SUMMARY_MODEL, TEMPERATURE, REDIS_HOST, REDIS_PORT
from model.get_model import get_llm
from app_logger import timer
from db_crud.chat_memory_crud import get_session_detail

# ======================上下文摘要智能体============================
class summary_memory_mananger:
    def __init__(self, summary_model):
        self.model_path = SUMMARY_MODEL
        self.model = summary_model
        self.system_prompt = """# 角色
你是一个专业的法律对话摘要助手，专门将多轮法律咨询对话提炼为简洁、准确、信息完整的摘要。

# 任务
请严格遵循以下原则处理输入的对话历史：

1. **全面覆盖关键信息**：必须包含用户的核心法律问题、涉及的主体（如个人、公司、行政机关等）、具体行为或事件、以及助手提供的关键法律依据或建议。
2. **保持逻辑连贯**：摘要应形成一段语义连贯的叙述，避免碎片化罗列。
3. **内容精简**：在不丢失法律事实和关键细节的前提下，尽可能压缩冗余表述，去除寒暄、重复或无关内容。

# 输出要求
- 仅输出摘要文本，不要添加标题、前缀、解释或额外说明。
- 使用客观、中立、书面化的语言。
- 不得引入对话中未提及的信息或主观推断。

# 输入示例
user: 你好，我在一家公司工作了3年，最近公司以“业务调整”为由把我辞退了，但没给赔偿，这合法吗？
assistant: 根据《劳动合同法》第40条和第46条，若非因员工过错而解除劳动合同，用人单位应支付经济补偿。您工作满3年，可主张3个月工资的经济补偿。建议保留解除通知、工资流水等证据。
user: 工资是每月8000元，他们说我是自愿离职，但我没签任何文件。
assistant: 若公司无法证明您自愿离职，则可能构成违法解除。您可向当地劳动仲裁委员会申请仲裁，主张2N赔偿（即6个月工资，共48000元）。

# 输出示例
用户咨询被公司以“业务调整”为由辞退且未获赔偿是否合法。其工作3年、月薪8000元，公司称其自愿离职但无签字文件。助手指出若非员工过错解除合同，公司应支付经济补偿；若属违法解除，可主张6个月工资（48000元）的赔偿，并建议申请劳动仲裁。

现在请对以下多轮对话内容进行摘要：
{conversation}
"""
    @timer("摘要对话")
    def summary_conversation(self, conversation: str):
        message = [{'role': 'system', 'content': self.system_prompt.format(conversation=conversation)}]
        
        return self.model.invoke(message).content

    @timer("摘要摘要")
    def summary_summary(self, summary: str):

        message = [{'role': 'system', 'content': '你是一个摘要助手，需要对用户输入的文本进行摘要，摘要要尽可能保留重要信息，直接返回摘要后的内容，不要添加其他无关内容。'},
                   {'role': 'user', 'content': f"请对以下内容进行摘要：\n{summary}"}]
        return self.model.invoke(message).content


#=================================Redis上下文管理器=================================

class ConversationManager:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0,
                 token_threshold=26000, m=0.5, compress_batch_size=5, ttl_seconds=86400):
        """
        初始化会话管理器（移除 k 参数，改用纯 token 阈值控制）

        Args:
            redis_host: Redis主机地址
            redis_port: Redis端口
            redis_db: Redis数据库
            token_threshold: 总 token 上限
            m: 完整对话占总 token 的比例（0 < m < 1）
            compress_batch_size: 每次压缩的消息轮数（每轮 = user + agent）
            ttl_seconds: 会话缓存过期时间（秒），默认1天
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        self.token_threshold = token_threshold
        self.m = m
        self.compress_batch_size = compress_batch_size
        self.ttl_seconds = ttl_seconds

        # 计算阈值
        self.max_uncompressed_tokens = int(self.token_threshold * self.m)      # a1
        self.max_summary_tokens = int(self.token_threshold * (1 - self.m))    # a2

        # 键名模板
        self.session_key = "session:{}"
        self.messages_key = "session:{}:messages"
        self.summary_key = "session:{}:summary"
        self.token_count_key = "session:{}:tokens"
        self.uncompressed_tokens_key = "session:{}:uncompressed_tokens"

    def _set_key_with_ttl(self, key: str, value: str = None, is_list: bool = False):
        """统一设置带 TTL 的键"""
        if is_list:
            self.redis_client.expire(key, self.ttl_seconds)
        else:
            if value is not None:
                self.redis_client.set(key, value)
            self.redis_client.expire(key, self.ttl_seconds)

    def _get_message_string(self, messages: List[Dict]) -> str:
        lines = []
        for msg in messages:
            lines.append(f"{msg['role']}:{msg['content']}")
        return "\n".join(lines)

    def _mock_summarize_method1(self, conversation: str) -> Tuple[str, int]:
        summary = memory_summarier.summary_conversation(conversation)
        estimated_tokens = count_tokens(summary)
        return summary, estimated_tokens

    def _calculate_tokens(self, text: str) -> int:
        return count_tokens(text)

    def add_message_pair(self, session_id: str, user_message: str,
                        agent_message: str, user_tokens: int = None,
                        agent_tokens: int = None) -> bool:
        try:
            if user_tokens is None:
                user_tokens = self._calculate_tokens(user_message)
            if agent_tokens is None:
                agent_tokens = self._calculate_tokens(agent_message)

            user_msg = {"role": "user", "content": user_message, "tokens": user_tokens}
            agent_msg = {"role": "agent", "content": agent_message, "tokens": agent_tokens}

            messages_key = self.messages_key.format(session_id)
            self.redis_client.rpush(messages_key, json.dumps(user_msg))
            self.redis_client.rpush(messages_key, json.dumps(agent_msg))
            self._set_key_with_ttl(messages_key, is_list=True)

            self._update_token_count(session_id, user_tokens + agent_tokens)
            self._update_uncompressed_token_count(session_id, user_tokens + agent_tokens)

            self._check_and_compress(session_id)

            logger.info(f"会话 {session_id} 成功添加一对消息", extra={
                "session_id": session_id,
                "user_tokens": user_tokens,
                "agent_tokens": agent_tokens
            })
            return True

        except Exception as e:
            logger.error(f"会话 {session_id} 添加消息失败: {e}", extra={"session_id": session_id})
            return False

    def _update_token_count(self, session_id: str, tokens_to_add: int) -> int:
        token_key = self.token_count_key.format(session_id)
        if self.redis_client.exists(token_key):
            new_tokens = self.redis_client.incrby(token_key, tokens_to_add)
        else:
            new_tokens = tokens_to_add
            self.redis_client.set(token_key, new_tokens)
        self._set_key_with_ttl(token_key)
        return new_tokens

    def _update_uncompressed_token_count(self, session_id: str, tokens_to_add: int) -> int:
        key = self.uncompressed_tokens_key.format(session_id)
        if self.redis_client.exists(key):
            new_tokens = self.redis_client.incrby(key, tokens_to_add)
        else:
            new_tokens = tokens_to_add
            self.redis_client.set(key, new_tokens)
        self._set_key_with_ttl(key)
        return new_tokens

    def _get_token_count(self, session_id: str) -> int:
        count = self.redis_client.get(self.token_count_key.format(session_id))
        return int(count) if count else 0

    def _get_uncompressed_token_count(self, session_id: str) -> int:
        count = self.redis_client.get(self.uncompressed_tokens_key.format(session_id))
        return int(count) if count else 0

    def _get_summary_total_tokens(self, session_id: str) -> int:
        summary_key = self.summary_key.format(session_id)
        summaries = self.redis_client.lrange(summary_key, 0, -1)
        total = 0
        for s in summaries:
            obj = json.loads(s)
            total += obj.get('tokens', 0)
        return total

    def _check_and_compress(self, session_id: str):
        # 第一步：检查完整对话是否超 a1 → 压缩最早消息
        uncompressed_tokens = self._get_uncompressed_token_count(session_id)
        while uncompressed_tokens > self.max_uncompressed_tokens:
            logger.info(f"会话 {session_id} 完整对话超限 ({uncompressed_tokens} > {self.max_uncompressed_tokens})，执行压缩", extra={
                "session_id": session_id,
                "uncompressed_tokens": uncompressed_tokens,
                "max_uncompressed": self.max_uncompressed_tokens
            })
            if not self._compress_earliest_messages(session_id):
                break
            uncompressed_tokens = self._get_uncompressed_token_count(session_id)

        # 第二步：检查摘要是否超 a2 → FIFO 删除最旧摘要
        summary_tokens = self._get_summary_total_tokens(session_id)
        while summary_tokens > self.max_summary_tokens:
            logger.info(f"会话 {session_id} 摘要超限 ({summary_tokens} > {self.max_summary_tokens})，删除最旧摘要", extra={
                "session_id": session_id,
                "summary_tokens": summary_tokens,
                "max_summary": self.max_summary_tokens
            })
            if not self._remove_oldest_summary(session_id):
                break
            summary_tokens = self._get_summary_total_tokens(session_id)

    def _compress_earliest_messages(self, session_id: str) -> bool:
        try:
            messages_key = self.messages_key.format(session_id)
            summary_key = self.summary_key.format(session_id)

            num_to_fetch = min(self.compress_batch_size * 2, self.redis_client.llen(messages_key))
            if num_to_fetch < 2:
                return False

            messages_json = self.redis_client.lrange(messages_key, 0, num_to_fetch - 1)
            if not messages_json or len(messages_json) < 2:
                return False

            messages = [json.loads(m) for m in messages_json]
            if len(messages) % 2 != 0:
                messages = messages[:-1]
                if len(messages) < 2:
                    return False

            compress_text = self._get_message_string(messages)
            summary, summary_tokens = self._mock_summarize_method1(compress_text)

            original_tokens = sum(msg['tokens'] for msg in messages)
            tokens_saved = original_tokens - summary_tokens

            summary_msg = {
                "role": "system",
                "content": summary,
                "tokens": summary_tokens,
                "original_tokens": original_tokens,
                "compressed_count": len(messages),
                "timestamp": time.time()
            }

            self.redis_client.rpush(summary_key, json.dumps(summary_msg))
            self._set_key_with_ttl(summary_key, is_list=True)

            for _ in range(len(messages)):
                self.redis_client.lpop(messages_key)
            self._set_key_with_ttl(messages_key, is_list=True)

            self._update_token_count(session_id, -tokens_saved)
            self._update_uncompressed_token_count(session_id, -original_tokens)

            logger.info(f"会话 {session_id} 压缩成功", extra={
                "session_id": session_id,
                "original_tokens": original_tokens,
                "summary_tokens": summary_tokens,
                "saved": tokens_saved
            })
            return True

        except Exception as e:
            logger.error(f"会话 {session_id} 压缩失败: {e}", extra={"session_id": session_id})
            return False

    def _remove_oldest_summary(self, session_id: str) -> bool:
        """FIFO 删除最旧的摘要条目"""
        try:
            summary_key = self.summary_key.format(session_id)
            oldest = self.redis_client.lpop(summary_key)
            if oldest is None:
                return False

            obj = json.loads(oldest)
            removed_tokens = obj.get('tokens', 0)
            self._update_token_count(session_id, -removed_tokens)

            logger.info(f"会话 {session_id} 删除最旧摘要", extra={
                "session_id": session_id,
                "removed_tokens": removed_tokens
            })
            return True
        except Exception as e:
            logger.error(f"会话 {session_id} 删除摘要失败: {e}", extra={"session_id": session_id})
            return False

    async def get_conversation(self, session_id: str, max_tokens: int = None) -> Tuple[str, str]:
        try:
            await self.ensure_session_loaded(session_id)
            messages_key = self.messages_key.format(session_id)
            summary_key = self.summary_key.format(session_id)

            summaries = self.redis_client.lrange(summary_key, 0, -1)
            summary_content = "\n".join(
                f"system:{json.loads(s)['content']}" for s in summaries
            ) if summaries else ""

            messages = self.redis_client.lrange(messages_key, 0, -1)
            uncompressed_content = "\n".join(
                f"{json.loads(m)['role']}:{json.loads(m)['content']}" for m in messages
            ) if messages else ""

            return summary_content, uncompressed_content

        except Exception as e:
            logger.error(f"会话 {session_id} 获取对话内容失败: {e}", extra={"session_id": session_id})
            return "", ""

    def delete_session(self, session_id: str) -> bool:
        try:
            keys = [
                self.messages_key.format(session_id),
                self.summary_key.format(session_id),
                self.token_count_key.format(session_id),
                self.uncompressed_tokens_key.format(session_id)
            ]
            self.redis_client.delete(*keys)
            logger.info(f"会话 {session_id} 已成功删除", extra={"session_id": session_id})
            return True
        except Exception as e:
            logger.error(f"会话 {session_id} 删除失败: {e}", extra={"session_id": session_id})
            return False

    async def ensure_session_loaded(self, session_id: str) -> bool:
        messages_key = self.messages_key.format(session_id)
        summary_key = self.summary_key.format(session_id)

        # 如果 Redis 中已有数据，无需重新加载
        if self.redis_client.exists(messages_key) and self.redis_client.llen(messages_key) > 0:
            return True

        try:
            session_data = await get_session_detail(session_id)
        except ValueError:
            logger.warning(f"会话 {session_id} 在数据库中不存在", extra={"session_id": session_id})
            # 初始化空会话
            self._set_key_with_ttl(messages_key, is_list=True)
            self._set_key_with_ttl(self.token_count_key.format(session_id), "0")
            self._set_key_with_ttl(self.uncompressed_tokens_key.format(session_id), "0")
            self.redis_client.delete(summary_key)
            return True

        messages = session_data["conversation_history"]
        if not messages:
            self._set_key_with_ttl(messages_key, is_list=True)
            self._set_key_with_ttl(self.token_count_key.format(session_id), "0")
            self._set_key_with_ttl(self.uncompressed_tokens_key.format(session_id), "0")
            self.redis_client.delete(summary_key)
            return True

        # 构造有效消息对（user/ai 交替）
        paired_messages = []
        i = 0
        while i < len(messages):
            if messages[i]["role"] == "user":
                user_msg = messages[i]
                agent_msg = messages[i + 1] if i + 1 < len(messages) and messages[i + 1]["role"] == "ai" else None
                paired_messages.append((user_msg, agent_msg))
                i += 2 if agent_msg else 1
            else:
                i += 1

        if not paired_messages:
            self._set_key_with_ttl(messages_key, is_list=True)
            self._set_key_with_ttl(self.token_count_key.format(session_id), "0")
            self._set_key_with_ttl(self.uncompressed_tokens_key.format(session_id), "0")
            self.redis_client.delete(summary_key)
            return True

        # 计算每轮 token
        message_pairs_with_tokens = []
        for user_msg, agent_msg in paired_messages:
            u_content = user_msg["content"]
            a_content = agent_msg["content"] if agent_msg else ""
            u_tokens = self._calculate_tokens(u_content)
            a_tokens = self._calculate_tokens(a_content) if agent_msg else 0
            message_pairs_with_tokens.append((u_content, a_content, u_tokens, a_tokens))

        total_rounds = len(message_pairs_with_tokens)

        # Step 1: 优先加载最新的完整对话（不超过 a1）
        selected_rounds = []
        accumulated_uncompressed = 0
        start_index = total_rounds  # 从末尾开始往前找

        for i in range(total_rounds - 1, -1, -1):
            u_content, a_content, u_tok, a_tok = message_pairs_with_tokens[i]
            if accumulated_uncompressed + u_tok + a_tok <= self.max_uncompressed_tokens:
                selected_rounds.append((u_content, a_content, u_tok, a_tok))
                accumulated_uncompressed += u_tok + a_tok
                start_index = i
            else:
                break

        selected_rounds.reverse()  # 从旧到新

        # Step 2: 处理被截断的旧消息（索引 0 到 start_index - 1）
        old_messages_to_summarize = message_pairs_with_tokens[:start_index]
        summaries_to_add = []

        if old_messages_to_summarize:
            # 按 compress_batch_size 分组（每组是连续的轮次）
            groups = []
            for i in range(0, len(old_messages_to_summarize), self.compress_batch_size):
                group = old_messages_to_summarize[i:i + self.compress_batch_size]
                groups.append(group)

            # 从旧到新依次摘要（因为 Redis 的 summary 是 FIFO，先压入的更旧）
            current_summary_tokens = 0
            for group in groups:
                if current_summary_tokens >= self.max_summary_tokens:
                    break

                # 构造对话字符串
                lines = []
                original_tokens = 0
                for u_cont, a_cont, u_tok, a_tok in group:
                    lines.append(f"user:{u_cont}")
                    if a_cont:
                        lines.append(f"agent:{a_cont}")
                    original_tokens += u_tok + a_tok

                conversation_text = "\n".join(lines)
                try:
                    summary_text = memory_summarier.summary_conversation(conversation_text)
                    summary_tokens = self._calculate_tokens(summary_text)

                    # 检查是否还能容纳这个摘要
                    if current_summary_tokens + summary_tokens > self.max_summary_tokens:
                        # 如果单个摘要就超限，仍可考虑截断或跳过？这里选择跳过以保安全
                        logger.warning(
                            f"会话 {session_id} 摘要单组超限（{summary_tokens} > 剩余 {self.max_summary_tokens - current_summary_tokens}），跳过",
                            extra={"session_id": session_id}
                        )
                        continue

                    summaries_to_add.append({
                        "role": "system",
                        "content": summary_text,
                        "tokens": summary_tokens,
                        "original_tokens": original_tokens,
                        "compressed_count": len(group) * 2,  # user+agent per round
                        "timestamp": time.time()
                    })
                    current_summary_tokens += summary_tokens

                except Exception as e:
                    logger.error(f"会话 {session_id} 加载时摘要失败: {e}", extra={"session_id": session_id})
                    continue

        # Step 3: 写入 Redis
        pipe = self.redis_client.pipeline()

        # 清空旧数据
        pipe.delete(messages_key)
        pipe.delete(summary_key)

        # 写入完整对话（uncompressed）
        for u_content, a_content, u_tok, a_tok in selected_rounds:
            user_msg = {"role": "user", "content": u_content, "tokens": u_tok}
            agent_msg = {"role": "agent", "content": a_content, "tokens": a_tok}
            pipe.rpush(messages_key, json.dumps(user_msg))
            pipe.rpush(messages_key, json.dumps(agent_msg))

        # 写入摘要（按时间顺序：旧摘要先入）
        total_summary_tokens = 0
        for summary_obj in summaries_to_add:
            pipe.rpush(summary_key, json.dumps(summary_obj))
            total_summary_tokens += summary_obj["tokens"]

        # 更新 token 计数
        total_tokens = accumulated_uncompressed + total_summary_tokens
        pipe.set(self.token_count_key.format(session_id), total_tokens)
        pipe.set(self.uncompressed_tokens_key.format(session_id), accumulated_uncompressed)

        # 设置 TTL
        for key in [messages_key, summary_key,
                    self.token_count_key.format(session_id),
                    self.uncompressed_tokens_key.format(session_id)]:
            pipe.expire(key, self.ttl_seconds)

        pipe.execute()

        logger.info(
            f"会话 {session_id} 从数据库加载完成",
            extra={
                "session_id": session_id,
                "loaded_full_rounds": len(selected_rounds),
                "summarized_groups": len(summaries_to_add),
                "uncompressed_tokens": accumulated_uncompressed,
                "summary_tokens": total_summary_tokens,
                "total_tokens": total_tokens,
                "max_uncompressed": self.max_uncompressed_tokens,
                "max_summary": self.max_summary_tokens
            }
        )
        return True
    
summary_llm = get_llm(SUMMARY_MODEL, TEMPERATURE)
memory_summarier = summary_memory_mananger(summary_llm)
m_conversation_manager = ConversationManager(redis_host=REDIS_HOST, redis_port=REDIS_PORT)