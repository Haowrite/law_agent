#!/bin/bash
# start_dev.sh

export HF_HOME="/home/RAG_agent/model/models"
export MODELSCOPE_CACHE="/home/RAG_agent/model/models"

echo "🌐 启动 Web 服务..."
python3 web_app.py

