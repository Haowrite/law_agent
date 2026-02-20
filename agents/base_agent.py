"""
基础智能体类
所有专门智能体的基类
"""

from abc import ABC
from re import A
from typing import List, Any, Annotated
from typing_extensions import TypedDict
from langchain_core.language_models.chat_models import BaseChatModel
import operator
from app_logger import timer
from db_crud.session_manage import m_conversation_manager
from pydantic import BaseModel, Field


#================================agent基类==============================

class general_agent_output_structure(BaseModel):  
    enable_answer: bool = Field(description="表示当前的信息是否可以回答用户问题。", default=False)
    response: str = Field(description="回答给用户的内容，不能回答时为空字符串。", default="")  
    search_query: str = Field(description="提供给检索工具使用的查询问题关键字。", default="")

# 定义状态类型
class AgentState(TypedDict):
    session_id: str = None
    ai_actions: Annotated[List[Any], operator.add] = []
    rag_result: Annotated[list[Any], operator.add] = []
    rag_cnt: Annotated[int, operator.add] = 0
    customer_query: str = None
    response: str = None
    user_id: str = None
    run_process: Annotated[List[tuple], operator.add] 
    
class BaseAgent(ABC):
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.llm: BaseChatModel = None

    def set_llm(self, llm: BaseChatModel):
        """设置LLM客户端"""
        self.llm = llm.with_structured_output(general_agent_output_structure)


    async def get_conversation_context(self, session_id:str) -> str:
        # 使用会话管理器从数据库获取对话上下文
        return await m_conversation_manager.get_conversation(session_id)
        
    
