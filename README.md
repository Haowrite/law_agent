# 法律AI助手 (Legal AI Assistant)

基于 LangGraph 的智能法律咨询 RAG 系统，支持会话管理、上下文压缩、向量检索等功能。

## 项目简介

本项目是一个法律领域的智能问答系统，采用 RAG (Retrieval-Augmented Generation) 架构，结合大语言模型与向量知识库，为用户提供专业的法律咨询服务。

### 核心特性

- **智能对话**: 基于 LangGraph 的多轮对话管理，支持上下文理解
- **RAG 检索**: 结合向量检索与 BM25 混合检索，提高召回准确率
- **上下文压缩**: 自动摘要历史对话，突破上下文长度限制
- **会话管理**: Redis + MySQL 双层存储，支持会话持久化与快速加载
- **用户系统**: 支持邮箱注册、验证码登录
- **文档管理**: 支持法律文档的上传与向量化

## 技术栈

| 类别 | 技术 |
|------|------|
| 后端框架 | FastAPI, Uvicorn |
| AI 框架 | LangChain, LangGraph |
| 大语言模型 | Qwen3.5-plus (阿里云百炼) |
| 嵌入模型 | Qwen3-Embedding-0.6B |
| 重排模型 | BGE-Reranker-v2-m3 |
| 向量数据库 | Milvus (Lite) |
| 关系数据库 | MySQL (异步 AsyncMy) |
| 缓存 | Redis |
| 分词 | Jieba |

## 项目结构

```
RAG_agent/
├── web_app.py              # FastAPI 应用入口
├── agent_service.py        # LangGraph 工作流定义
├── config.py               # 配置管理
├── start.sh                # 启动脚本
├── requirements.txt        # 依赖列表
│
├── agents/                 # 智能体模块
│   ├── __init__.py
│   ├── base_agent.py       # 基础智能体类
│   └── general_agent.py    # 通用对话智能体
│
├── RAG/                    # 检索增强生成模块
│   ├── __init__.py
│   ├── retrieve.py         # 向量检索工具 (批量+显存保护)
│   ├── retrieve_process.py # 检索处理逻辑
│   └── vector_doc.py       # 向量库管理 (Milvus)
│
├── db_crud/                # 数据库操作层
│   ├── __init__.py
│   ├── base.py             # 数据库连接与初始化
│   ├── base_func.py        # 基础工具函数
│   ├── session_manage.py   # 会话管理 (Redis + 摘要压缩)
│   ├── session_model.py    # 数据模型定义 (SQLModel)
│   ├── chat_memory_crud.py # 聊天记录 CRUD
│   └── doc_article_model.py# 文档管理模型
│
├── model/                  # 模型加载模块
│   ├── __init__.py
│   ├── get_model.py        # LLM 获取接口
│   └── download.py         # 模型下载工具
│
├── router/                 # API 路由
│   ├── user_router.py      # 用户相关 API
│   ├── chat_router.py      # 聊天相关 API
│   └── doc_router.py       # 文档管理 API
│
├── utils/                  # 工具模块
│   └── agent_thread_pool.py# 进程池管理
│
├── static/                 # 静态资源
├── templates/              # Jinja2 模板
├── files/                  # 法律文档存储
├── RAG_DB/                 # Milvus 数据库文件
└── log/                    # 日志目录
```

## 架构设计

### 1. 对话工作流 (LangGraph)

```
┌─────────────────┐
│  general_agent  │ ← 用户输入 + RAG 结果
└────────┬────────┘
         │ <──────────────────────────| 
         ▼                            |
┌─────────────────┐                   |
│ router_edge_node│ → 路由决策        |
└────────┬────────┘                   |
         │                            |
    ┌────┴────┐                       |
    ▼         ▼                       |
┌───────┐  ┌─────────────────┐        |
│ RAG   │  │ final_response  │        |
│检索   │  │     _node       │        |    
└───┬───┘  └─────────────────┘        |
    │                                 |
    └────────────────────────────────── 返回 general_agent
```

### 2. 会话管理架构

```
┌─────────────────────────────────────────────────────┐
│                  ConversationManager                 │
├─────────────────────────────────────────────────────┤
│  Redis 缓存层                                        │
│  ├── session:{id}:unsummarized  (未摘要消息)         │
│  ├── session:{id}:summarized    (已摘要消息备份)      │
│  ├── session:{id}:summary       (摘要列表)           │
│  └── session:{id}:meta          (Token 计数)        │
├─────────────────────────────────────────────────────┤
│  MySQL 持久层                                        │
│  ├── chat_sessions     (会话表)                      │
│  ├── chat_messages     (消息表)                      │
│  └── summary_messages  (摘要表)                      │
└─────────────────────────────────────────────────────┘
```

### 3. 上下文压缩策略

- **阈值配置**: 未摘要区 70%，摘要区 30%
- **触发条件**: 未摘要区 Token 超过阈值时自动触发压缩
- **压缩流程**: 
  1. 提取最早的 N 条消息
  2. 调用摘要模型生成压缩摘要
  3. 更新 MySQL (标记消息已摘要 + 保存摘要)
  4. 更新 Redis (移动消息 + 添加摘要)
- **兜底机制**: 服务重启时自动检测并修复数据不一致

## 快速开始

### 1. 环境要求

- Python 3.10+
- MySQL 8.0+
- Redis 6.0+
- CUDA 11.0+ (GPU 推理)

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env_temp` 为 `.env` 并修改配置：

```bash
cp .env_temp .env
```

关键配置项：

```env
# 大语言模型
MAIN_MODEL=qwen3.5-plus
API_KEY=your_api_key

# 嵌入模型路径
EMBEDDING_MODEL=/path/to/bge-large-zh-v1.5
TOKENIZER_MODEL=/path/to/bge-large-zh-v1.5

# 重排模型路径
RERANKER_MODEL=/path/to/bge-reranker-v2-m3
RERANKER_TOKENIZER=/path/to/bge-reranker-v2-m3

# 数据库
DATABASE_URL=mysql+asyncmy://user:password@localhost:3306/agent_project
REDIS_HOST=localhost
REDIS_PORT=6379

# 向量库
VECTOR_COLLECTION_NAME=legal_knowledge
RE_BUILD=False  # 首次运行设为 True
```

### 4. 下载模型

```bash
# 从 ModelScope 下载嵌入模型
python -c "from modelscope import snapshot_snapshot; snapshot_snapshot('Qwen/Qwen3-Embedding-0.6B')"

# 下载重排模型
python -c "from modelscope import snapshot_snapshot; 'BAAI/bge-reranker-v2-m3'"
```

### 5. 初始化数据库

```sql
CREATE DATABASE agent_project CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

表结构会在首次启动时自动创建。

### 6. 启动服务

```bash
bash start.sh
# 或
python web_app.py
```

服务启动后：
- Web 界面: http://localhost:5000
- API 文档: http://localhost:5000/docs
- ReDoc: http://localhost:5000/redoc

## API 接口

### 用户相关

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/register` | 用户注册 |
| POST | `/api/login` | 用户登录 |
| POST | `/api/send-code` | 发送验证码 |

### 会话相关

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/chat` | 发送消息 (流式响应) |
| GET | `/api/sessions` | 获取会话列表 |
| DELETE | `/api/session/{id}` | 删除会话 |
| GET | `/api/history/{id}` | 获取历史消息 |

### 文档相关

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/upload` | 上传文档 |
| GET | `/api/documents` | 获取文档列表 |

## 核心模块说明

### RAG 检索模块 (`RAG/retrieve.py`)

- **批量处理**: 使用进程池 + 动态批处理优化吞吐量
- **显存保护**: 自动检测 OOM 并降级批次大小
- **去重机制**: 支持传入已检索 ID 避免重复召回

### 会话管理 (`db_crud/session_manage.py`)

- **双层存储**: Redis 热数据 + MySQL 持久化
- **自动压缩**: Token 超阈值自动触发摘要
- **一致性保证**: 数据库优先写入，失败时中止缓存更新

### 向量管理 (`RAG/vector_doc.py`)

- **混合检索**: Milvus 向量检索 + BM25 关键词检索
- **重排序**: BGE-Reranker 二次排序提升精度
- **增量更新**: 支持文档增量添加

## 开发指南

### 日志配置

日志级别通过 `LOG_LEVEL` 环境变量控制，日志文件存储在 `./log/` 目录。

### 添加新的智能体

1. 继承 `BaseAgent` 类
2. 实现 `process` 方法
3. 在 `agent_service.py` 中注册节点

### 自定义检索策略

修改 `RAG/retrieve_process.py` 中的检索逻辑，支持：
- 调整召回数量
- 修改相似度阈值
- 自定义重排策略

## 常见问题

### Q: 首次启动报错 "Collection not found"

A: 将 `.env` 中 `RE_BUILD=True`，首次会创建向量库。

### Q: 显存不足 (CUDA OOM)

A: 系统会自动降级批次大小，如仍不足可减小 `INITIAL_BATCH_SIZE`。

### Q: Redis 连接失败

A: 检查 Redis 服务是否启动，端口是否正确。



https://github.com/user-attachments/assets/5f311744-f9d9-4051-b09e-af26eb79abaf

