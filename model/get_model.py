from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from app_logger import llm_logger as logger
from modelscope import  AutoTokenizer
from config import API_KEY

# def get_model(model_name:str = "Qwen/Qwen3-4B", temperature: float = 0):
#     logger.info(f"加载模型: {model_name}")
    
#     vllm_model = vllm.VLLM(
#         model=model_name,
#         trust_remote_code = True,
#         temperature = temperature,
#     )
    
#     return vllm_model


def get_model(model_name:str = "Qwen/Qwen3-4B", temperature: float = 0):
    llm: BaseChatModel = ChatOpenAI(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model=model_name, 
        extra_body={
                "enable_thinking": False,  # 关键：非流式调用必须设置为 false
                "result_format": "message"  # 确保返回格式正确
            }
    )

    return llm


def get_llm(model_name = "Qwen/Qwen3-4B",  temperature: float = 0.1):
    logger.info(f"加载模型: {model_name}")
    llm = get_model(model_name, temperature)
    return llm


def get_tokenizer(model_name = "Qwen/Qwen3-0.6B"):
    logger.info(f"加载分词器: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer
