"""
综合客服智能体
统一处理所有类型的客户咨询
"""
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from .base_agent import BaseAgent
from agents.base_agent import AgentState
from app_logger import app_logger as logger

class GeneralAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="法律AI助手",
            role="处理所有类型客户咨询",
            expertise=["法律咨询"]
        )

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理各种类型的客户查询"""
        # 调用LLM
        try:
            response = await self.llm.ainvoke(state["messages"])
            
        except Exception as e:
            logger.info(f"综合客服调用LLM时出错: {e}")
            response = AIMessage("抱歉，处理您的咨询时遇到系统错误，请稍后重试。")
        
        response.usage_metadata['agent_name'] = self.name
        return {"messages": [response]}