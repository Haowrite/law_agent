#!/bin/bash
# start_dev.sh

export HF_HOME="/home/RAG_agent/model/models"
export MODELSCOPE_CACHE="/home/RAG_agent/model/models"
export NO_PROXY="dashscope.aliyuncs.com,.aliyuncs.com,${NO_PROXY}"
export no_proxy="dashscope.aliyuncs.com,.aliyuncs.com,${no_proxy}"

echo "🌐 启动 Web 服务..."
python3 web_app.py
