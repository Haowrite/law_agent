from  datetime import datetime
import uuid
from model.get_model import get_tokenizer
def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_id():
    return str(uuid.uuid4())


tokenizer = get_tokenizer()

def count_tokens(text):
    """计算文本的token数量"""
    tokens = tokenizer.encode(text)
    return len(tokens)
