import os

from numpy import dtype
os.environ["MODELSCOPE_CACHE"] = "/home/RAG_agent/model/models"
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download

snapshot_download('Qwen/Qwen3-Embedding-0.6B', cache_dir="/home/RAG_agent/model/models")