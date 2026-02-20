#!/bin/bash
# start_dev.sh

export HF_HOME="/home/RAG_agent/model/models"
export MODELSCOPE_CACHE="/home/RAG_agent/model/models"

echo "🚀 启动 arq worker..."
arq db_crud.arq_tasks.WorkerSettings &

ARQ_PID=$!

echo "🌐 启动 Web 服务..."
python3 web_app.py

# 可选：Web 退出时自动 kill arq
trap "kill $ARQ_PID" EXIT