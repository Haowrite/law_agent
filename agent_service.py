"""
法律AI助手
使用单一智能体处理用户询问并执行RAG检索
"""
from re import A
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph

from agents.base_agent import AgentState
from app_logger import llm_logger as logger, timer
from RAG.vector_doc import create_vector_store
# 导入配置
from config import *
# 导入智能体和工具
from agents import GeneralAgent
from tools.base import tool_dict
from model.get_model import get_llm
from config import MODEL,  TEMPERATURE, FILE_PATH, RE_BUILD


# 建立本地知识库
create_vector_store(file_path=FILE_PATH, re_build=RE_BUILD)

main_llm = get_llm(MODEL, TEMPERATURE)

# 初始化智能体
def initialize_agents():
    """初始化智能体"""
    agents = {
        "general_agent": GeneralAgent(),
    }

    # 为每个智能体设置LLM和会话管理器
    for name, agent in agents.items():
        agent.set_llm(main_llm)  
    return agents


@timer("分类节点")
def router_edge_node(state: AgentState):
    ai_action = state['ai_actions'][-1]
    if not ai_action.enable_answer and ai_action.search_query != "":
        return 'tool_call_node'
    elif ai_action.enable_answer and ai_action.response != "":
        return 'final_response_node'
    else:
        state['ai_actions'][-1].response = "无法解决您的问题。建议您详细描述具体法律场景、涉及的主体或相关法条，以便我们更好地帮助您。"
        return 'final_response_node'



@timer("RAG检索节点")
# RAG检索节点
def tool_call_node(state: AgentState):
    """执行RAG检索并将结果存储到state中，目前只有检索工具，如果后期添加多个工具，修改agent的返回json格式，system_prompt添加详细的工具调用的json结构"""
    ai_action = state['ai_actions'][-1]
    
    logger.info(f"🔍 执行RAG检索... 用户问题：{state['customer_query']} |  检索问题关键字: {ai_action.search_query}")

    rag_res = tool_dict['retrieve_vector_store'].invoke(ai_action.search_query)

    logger.info(f"🔍 检索到的内容：{rag_res}")
    return {'rag_result': [rag_res], 'rag_cnt': 1}


# 最终响应节点
@timer("最终响应节点")
def final_response_node(state: AgentState):
    """最终响应节点，保存会话信息"""
    # 添加AI消息到数据库
    try:
        state['response'] = state['ai_actions'][-1].response
    except Exception as e:
        logger.info(f"Error adding message to session: {e}")
    
    logger.info(f"💬 最终响应：{state['response']}")
    return {'response': state['ai_actions'][-1].response}



# 构件助手
def make_graph():
    """构建LangGraph工作流图 - 集中式版本"""

    workflow = StateGraph(AgentState)
    agents_list = initialize_agents()

    # 添加节点
    workflow.add_node("general_agent", agents_list["general_agent"].process)
    workflow.add_node("final_response_node", final_response_node)
    workflow.add_node("tool_call_node", tool_call_node)

    
    # 设置入口点和边
    workflow.set_entry_point("general_agent")
    workflow.add_conditional_edges("general_agent", router_edge_node)
    workflow.add_edge("tool_call_node", "general_agent")
    workflow.set_finish_point("final_response_node")

    # 编译工作流
    app = workflow.compile()
    logger.info("✅ 法律AI助手构建完成")
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