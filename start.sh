#!/bin/bash
# 启动Python脚本并设置HF环境变量

# 设置Hugging Face缓存路径
export HF_HOME="/home/RAG_agent/model/models"
export MODELSCOPE_CACHE="/home/RAG_agent/model/models"

# 启动Web应用
python3 web_app.py
