"""
基础智能体类
所有专门智能体的基类
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage
import operator
from config import SUMMARY_MODEL


# ======================上下摘要============================
class summary_memory_mananger:
    def __init__(self, summary_model):
        self.model_path_finetuned = SUMMARY_MODEL
        self.model = summary_model
        self.system_prompt = """# 角色
你是一个专业的对话总结助手，专门将多轮对话内容提炼成简洁的摘要。

# 任务
请严格遵循以下步骤处理输入的对话历史：

1. **逐条分析**：对JSON数组中的每一条对话记录进行处理
2. **内容精简**：对每条对话的content字段进行摘要，要求：
   - 保留核心诉求、关键信息、重要决定
   - 删除客套话、重复内容、无关细节
   - 每条摘要控制在15字以内，保持简洁
3. **保持结构**：保持原有的JSON结构不变，只修改content字段
4. **连贯性**：确保摘要后的对话仍然具有逻辑连贯性

# 输出要求
- 必须返回完整的JSON数组，结构与输入完全一致
- 只修改content字段，time和role字段原样保留
- 确保JSON格式正确，可被直接解析
- 不要添加任何额外说明

# 输入示例
[
  {
    "time": "2024-11-20 10:05:10",
    "role": "user",
    "content": "我昨天收到的商品，屏幕有明显的划痕，这让我很失望。我想申请退货，请问具体流程是什么？"
  },
  {
    "time": "2024-11-20 10:05:25", 
    "role": "ai",
    "content": "您好，非常抱歉给您带来不好的购物体验。为了帮您处理退货，请问您有保留商品的开箱视频或者照片吗？"
  },
  {
    "time": "2024-11-20 10:05:40",
    "role": "user",
    "content": "我没有拍视频，但是我拍了几张划痕的照片，可以清楚地看到屏幕上的问题。"
  },
  {
    "time": "2024-11-20 10:06:00",S
    "role": "ai",
    "content": "好的，有照片也可以。请您登录我们的售后系统，在退货申请页面提交这些照片，并填写您的订单信息和退货原因，我们会优先为您处理。"
  }
]

#输出实例
[
  {
    "time": "2024-11-20 10:05:10",
    "role": "user", 
    "content": "反馈商品屏幕有划痕要求退货"
  },
  {
    "time": "2024-11-20 10:05:25",
    "role": "ai",
    "content": "询问是否有开箱视频或照片"
  },
  {
    "time": "2024-11-20 10:05:40",
    "role": "user",
    "content": "表示有划痕照片作为证据"
  },
  {
    "time": "2024-11-20 10:06:00", 
    "role": "ai",
    "content": "指导提交照片申请售后处理"
  }
]

# 需要处理的对话内容
"""


    def summary_conversation(self, conversation: str):
        message = [{'role': 'system', 'content': self.system_prompt + f"\n{conversation}"}]
        
        return self.model.invoke(message)



#================================agent基类==============================
# 定义状态类型
class AgentState(TypedDict):
    session_id: str
    messages: Annotated[list[AnyMessage], operator.add]
    tool_result: Annotated[list[AnyMessage], operator.add]
    customer_query: str
    query_type: str
    response: str
    user_id: str
    
class BaseAgent(ABC):
    def __init__(self, name: str, role: str, expertise: List[str]):
        self.name = name
        self.role = role
        self.expertise = expertise
        self.llm = None  # 将在运行时注入
        self.session_manager = None
        self.summary_manager: summary_memory_mananger= None

    def set_llm(self, llm):
        """设置LLM客户端"""
        self.llm = llm


    def set_summary_manager(self, summary_manager):
        self.summary_manager = summary_manager

    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理客户查询的抽象方法"""
        pass

    def _get_conversation_context(self, session_id: str, max_messages: int = 6) -> str:
        """从会话管理器获取对话历史上下文"""
        return self.memory_extract(session_id)

    def _enhance_system_prompt_with_context(self, base_prompt: str) -> str:
        """增强系统提示，添加对话上下文说明"""
        context_instruction = """
重要提示：请结合对话历史上下文，理解客户之前的问题和需求，提供连贯、个性化的回答。
如果用户问题需要查询公司相关信息：
    如果有相关信息，则一定要严格按照公司的信息正确给出回答，不要凭空捏造。
    如果没有相关信息，则向用户表明目前还无法解决该问题，请用户转入人工。
如果这是多轮对话，请参考之前的对话内容，避免重复信息，并基于客户的新问题提供补充信息。保持对话的连贯性和自然性，让客户感受到你理解他们的完整需求。
        """

        return base_prompt + context_instruction

    def get_info(self) -> Dict[str, Any]:
        """获取智能体信息"""
        return {
            "name": self.name,
            "role": self.role,
            "expertise": self.expertise
        }


    def memory_extract(self, session_id:str, window_size:int = 5) -> str:
        # 使用会话管理器从数据库获取对话上下文
        conversation_historys = self.session_manager.get_conversation_context(session_id)
        window_size = min(window_size, len(conversation_historys))

        latest_list = []
        summary_list = []
        for id, message in enumerate(conversation_historys):
            if id >= len(conversation_historys) - window_size:
                message_json = {
                    "time": message.get("timestamp", ""),
                    "role": message.get("message_type", 'user'),
                    "content": message.get("content", "")
                }
                latest_list.append(message_json)
            else:
                message_json = {
                    "time": message.get("timestamp", ""),
                    "role": message.get("message_type", 'user'),
                    "content": message.get("content", "")
                }
                summary_list.append(message_json)

        summary_list = json.dumps(summary_list, ensure_ascii=False)
        summary_text = self.summary_manager.summary_conversation(summary_list)
        latest_conversation = json.dumps(latest_list, ensure_ascii=False)

        return f"较早期的对话历史内容（json格式）：\n{summary_text}\n" + f"最近的对话历史：\n{latest_conversation}\n"
