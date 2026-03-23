from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('BAAI/bge-large-zh-v1.5', cache_dir='/home/RAG_agent/model/models/bge')