"""
集中式客服系统
使用单一智能体处理用户询问并执行RAG检索
"""
from json import tool
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph

from agents.base_agent import AgentState
from app_logger import app_logger as logger, timer
from RAG.vector_doc import create_vector_store
# 导入配置
from config import *
# 导入智能体和工具
from agents import GeneralAgent
from tools.retrieve import retrieve_vector_store
from tools.base import tool_dict
from model.get_model import get_llm
from config import MODEL, SUMMARY_MODEL, TEMPERATURE, FILE_PATH, RE_BUILD
from agents.base_agent import summary_memory_mananger


# 建立本地知识库
create_vector_store(file_path=FILE_PATH, re_build=RE_BUILD)

main_llm = get_llm(MODEL, TEMPERATURE)
summary_llm = get_llm(SUMMARY_MODEL, TEMPERATURE)
memory_summary = summary_memory_mananger(summary_llm)

# 初始化智能体
def initialize_agents():
    """初始化智能体"""
    agents = {
        "general_agent": GeneralAgent(),
    }

    # 为每个智能体设置LLM和会话管理器
    for name, agent in agents.items():
        agent.set_llm(main_llm)  # 延迟获取LLM
        agent.set_summary_manager(memory_summary)

    return agents


@timer("分类节点")
def router_edge_node(state: AgentState):
    ai_message = state["messages"][-1]
    if isinstance(ai_message, AIMessage):
        if ai_message.tool_calls is not None and len(ai_message.tool_calls) > 0:
            return 'tool_call_node'
    return 'final_response_node'

# RAG检索节点
@timer("工具调用节点")
async def tool_call_node(state: AgentState):
    """执行RAG检索并将结果存储到state中"""
    ai_message = state['messages'][-1]

    tool_message_list = []
    for tool_call in ai_message.tool_calls:
        tool_res = await tool_dict[tool_call['name']].ainvoke(tool_call['args'])
        tool_message_list.append(tool_res)

    return {'tool_result': tool_message_list}


# 最终响应节点
@timer("最终响应节点")
def final_response_node(state: AgentState):
    """最终响应节点，保存会话信息"""
    # 添加AI消息到数据库
    try:
        state['response'] = state['messages'][-1].content
    except Exception as e:
        logger.info(f"Error adding message to session: {e}")

    return {}



# 构件助手
def make_graph():
    """构建LangGraph工作流图 - 集中式版本"""

    workflow = StateGraph(AgentState)
    agents_list = initialize_agents()

    # 添加节点
    workflow.add_node("general_agent", agents_list["general_agent"].process)
    workflow.add_node("final_response", final_response_node)
    workflow.add_node("tool_call_node", tool_call_node)

    
    # 设置入口点和边
    workflow.set_entry_point("general_agent")
    workflow.add_conditional_edges("general_agent", router_edge_node)
    workflow.add_edge("tool_call_node", "general_agent")
    workflow.set_finish_point("final_response")

    # 编译工作流
    app = workflow.compile()
    logger.info("✅ 集中式客服系统工作流图构建完成")
    return app


# 创建默认工作流实例
if __name__ == "__main__":
    app = make_graph()
    resp = app.invoke({
        "messages": [HumanMessage(content="我喜欢摄影请帮我推荐适合我的产品")], 
        "customer_query": "我喜欢摄影帮我推荐一个产品",
        "session_id": "test_session"
    })

    print("🚀 集中式客服系统启动成功！")