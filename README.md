# 法律AI助手 (Legal AI Assistant)

基于 LangGraph、FastAPI 和 RAG 构建的法律垂直领域 AI 助手，支持法律法规知识库问答、引用可验证回答、深度研究报告、知识库动态维护、多轮会话记忆和 SSE 真流式输出。

## 项目简介

本项目面向法律咨询、法务辅助检索和企业内部制度问答等场景。系统使用法律法规文档构建本地知识库，将大模型的生成能力与 BM25、Milvus 向量检索、Reranker 重排序和证据校验结合起来，使回答不仅能生成自然语言结论，还能追溯到具体文档、条文和原文摘录。

项目重点解决三个问题：

1. 用户问题口语化，法律条文表述专业，普通关键词搜索很难稳定命中。
2. 法律回答需要可追溯依据，不能只依赖模型生成一个“看起来合理”的结论。
3. 真实系统需要知识库更新、会话持久化、并发处理和流式交互，而不只是一次模型 API 调用。

## 项目背景与应用价值

法律问答和普通闲聊不同，答案需要同时满足“能回答”和“有依据”。例如用户问“公司突然把我调去外地合法吗”，系统不能只输出“可能违法”，还需要说明可能涉及劳动合同变更、协商一致、工作地点和薪资变化等判断要点，并给出对应条文依据。

本项目的应用价值：

- 面向普通用户：快速了解法律问题可能涉及的依据、处理路径和风险点。
- 面向律师 / 法务：辅助检索法规材料，先生成结构化研究初稿，减少人工查条文和整理材料的时间。
- 面向企业内部：接入合同模板、规章制度、合规文件后，支持内部制度问答。
- 面向知识库维护：支持上传、搜索、预览、删除和重新向量化文档，使法律知识库可以持续更新。

本项目不定位为替代律师，而是把“检索依据、整理信息、生成初稿、提示风险”这些环节自动化。

## 为什么不直接问大模型

直接向通用大模型提问，在法律场景里存在明显限制：

- 知识时效不可控：模型参数中的知识有时间边界，无法保证覆盖最新法规、司法解释、公司制度或用户上传材料。
- 引用来源不可控：模型可能给出“根据相关法律”的表述，但无法保证引用来自项目知识库中真实存在的条文。
- 幻觉风险更高：法律场景中，编造法条编号、扩大适用条件、忽略程序期限都可能导致错误判断。
- 私有资料无法直接使用：合同、内部规章、企业制度、用户材料不会天然存在于通用模型里。
- 缺少业务系统能力：真实应用还需要用户系统、会话管理、知识库维护、权限控制、并发调度和前端交互。

因此本项目采用 RAG 架构：先从可控知识库中检索证据，再让大模型基于证据生成回答，最后通过证据校验器检查引用有效性和缺引用风险。

## 核心特性

- **LangGraph 问答编排**：将用户问题、RAG 检索、证据回答和最终响应组织为可维护的状态机流程。
- **混合检索 RAG**：结合 BM25 关键词召回、Milvus 向量召回、RRF 融合排序和 Reranker 精排，提高法律条文召回质量。
- **引用可验证回答**：RAG 返回结构化 evidence，回答中使用 `[1]`、`[2]` 标注依据，前端展示文档名、条文编号、摘录和内容哈希。
- **证据校验器**：回答完成后检查引用编号是否存在、法律判断句是否缺少引用，并输出 passed / warning / failed 状态。
- **深度研究模式**：将复杂法律问题拆解为法律依据、适用条件、救济路径、风险例外等子问题，多轮检索后调用 LLM 生成研究报告。
- **SSE 真流式输出**：后端直接消费 LLM `astream`，通过 SSE 推送 `status`、`content`、`metadata` 和 `[DONE]` 事件。
- **上下文记忆管理**：Redis 缓存近期上下文，MySQL 持久化历史消息和摘要，Token 超阈值后自动压缩早期对话。
- **知识库管理**：支持文档上传、自动切分向量化、按文档名检索、预览、删除，并同步清理 MySQL、Milvus 和 RAG 缓存。

## 技术栈

| 类别 | 技术 |
|------|------|
| 后端框架 | FastAPI, Uvicorn |
| AI 框架 | LangChain, LangGraph |
| 大语言模型 | Qwen / DashScope OpenAI 兼容接口 |
| 嵌入模型 | DashScope text-embedding API 或本地 embedding 模型 |
| 重排模型 | DashScope qwen-rerank API 或本地 reranker 模型 |
| 向量数据库 | Milvus (Lite) |
| 关系数据库 | MySQL (SQLAlchemy AsyncMy) |
| 缓存 | Redis |
| 分词 | Jieba |
| 流式响应 | SSE, FastAPI StreamingResponse |

## 技术选型说明

- **为什么用 LangGraph**：法律问答链路不是单次 LLM 调用，而是包含“判断是否检索、调用 RAG、基于证据回答、校验引用”等多个步骤。LangGraph 适合用状态机管理这类流程。
- **为什么用 Milvus + BM25**：向量检索擅长语义相似，BM25 擅长精确关键词。法律条文既有语义匹配需求，也有“仲裁时效、协商一致、劳动报酬”等关键词匹配需求，因此采用混合检索。
- **为什么用 Reranker**：召回阶段更关注“不要漏”，Reranker 用来在候选条文里重新判断问题和条文的相关性，提高最终提供给模型的证据质量。
- **为什么用 MySQL**：用户、会话、消息、摘要、文档元数据和条文映射都是结构化业务数据，需要可靠持久化。
- **为什么用 Redis**：多轮对话上下文读取频繁，Redis List 适合缓存近期消息、摘要和 token 计数，减少每轮都查 MySQL 的成本。
- **为什么用 SSE**：LLM 回答是服务端向客户端单向持续输出，SSE 比 WebSocket 更轻量，适合 token 流式返回，并且便于保留普通 `/api/chat` 兜底接口。

## 项目结构

```text
RAG_agent/
├── web_app.py              # FastAPI 应用入口
├── agent_service.py        # LangGraph 工作流定义
├── config.py               # 配置管理
├── start.sh                # 启动脚本
├── requirements.txt        # 依赖列表
│
├── agents/                 # 智能体模块
│   ├── base_agent.py       # 基础智能体类
│   └── general_agent.py    # 通用对话智能体
│
├── RAG/                    # 检索增强生成模块
│   ├── retrieve.py         # RAG 检索工具
│   ├── retrieve_process.py # 混合检索、重排与子进程处理
│   ├── vector_doc.py       # 文档切分与向量库管理
│   ├── evidence.py         # 引用证据结构化与公开字段
│   ├── evidence_verifier.py# 引用有效性与缺引用校验
│   ├── research_service.py # 深度研究任务编排
│   ├── streaming.py        # LLM astream 流式输出工具
│   ├── embedding_factory.py# 本地/API embedding 切换
│   └── reranker_factory.py # 本地/API reranker 切换
│
├── db_crud/                # 数据库操作层
│   ├── base.py             # 数据库连接与初始化
│   ├── session_manage.py   # 会话管理 (Redis + 摘要压缩)
│   ├── chat_memory_crud.py # 聊天记录 CRUD
│   ├── doc_article_crud.py # 文档条文写入与清理
│   └── doc_article_query.py# 文档公开查询
│
├── router/                 # API 路由
│   ├── user_router.py      # 用户相关 API
│   ├── chat_router.py      # 聊天与 SSE API
│   ├── doc_router.py       # 文档管理 API
│   └── research_router.py  # 深度研究 API
│
├── utils/                  # 工具模块
├── templates/              # 前端模板
├── docs/                   # 设计文档、实施计划和简历项目说明
├── tests/                  # 单元测试
├── files/                  # 法律文档存储
├── RAG_DB/                 # Milvus / RAG 缓存目录
└── log/                    # 日志目录
```

## 架构设计

### 对话工作流

```text
用户输入
  -> general_agent 判断是否需要检索
  -> tool_call_node 调用 RAG 检索
  -> general_agent 基于证据生成回答
  -> final_response_node 生成 citations 和 verification
  -> 前端展示回答、证据和校验状态
```

### 会话管理架构

```text
ConversationManager
├── Redis 缓存层
│   ├── session:{id}:unsummarized  未摘要消息
│   ├── session:{id}:summarized    已摘要消息备份
│   ├── session:{id}:summary       摘要列表
│   └── session:{id}:meta          token 计数
└── MySQL 持久层
    ├── chatsession                会话表
    ├── chatmessage                消息表
    └── summarymessage             摘要表
```

### 上下文压缩策略

1. 用户和 AI 消息先写入 MySQL，再刷入 Redis。
2. Redis 维护未摘要消息、摘要列表和 token 计数。
3. 未摘要 token 超阈值后，取最早一批消息生成摘要。
4. 摘要写入 MySQL 后，再更新 Redis 列表。
5. 模型调用时组合“历史摘要 + 近期原文消息”。

## 数据结构与性能优化

### MySQL 表结构

`user`：用户账号表。

| 字段 | 类型 | 含义 |
|------|------|------|
| `user_id` | `VARCHAR` 主键 | 用户唯一 ID。 |
| `username` | `VARCHAR(50)` 唯一索引 | 登录用户名。 |
| `email` | `VARCHAR(120)` 唯一索引，可空 | 邮箱和验证码场景使用。 |
| `password_hash` | `VARCHAR(128)` | bcrypt 密码哈希。 |
| `created_at` | `VARCHAR` | 用户创建时间。 |

`chatsession`：聊天会话表。

| 字段 | 类型 | 含义 |
|------|------|------|
| `session_id` | `VARCHAR` 主键 | 会话唯一 ID。 |
| `user_id` | `VARCHAR` 索引 | 会话所属用户，用于权限校验和会话列表。 |
| `timestamp` | `VARCHAR` | 会话创建时间。 |

`chatmessage`：聊天消息表。

| 字段 | 类型 | 含义 |
|------|------|------|
| `id` | `VARCHAR` 主键 | 消息唯一 ID。 |
| `session_id` | `VARCHAR` 索引 | 所属会话 ID。 |
| `content` | `TEXT` | 用户或 AI 的消息正文。 |
| `timestamp` | `VARCHAR` | 消息创建时间。 |
| `message_type` | `VARCHAR(10)` | 消息角色，例如 `user`、`ai`。 |
| `use_token` | `INTEGER` | 消息 token 估算值。 |
| `is_summarized` | `BOOLEAN` | 是否已被摘要压缩。 |

`summarymessage`：历史摘要表。

| 字段 | 类型 | 含义 |
|------|------|------|
| `summary_id` | `VARCHAR` 主键 | 摘要唯一 ID。 |
| `session_id` | `VARCHAR` 索引 | 摘要所属会话 ID。 |
| `summary_content` | `TEXT` | 历史对话摘要内容。 |
| `timestamp` | `VARCHAR` | 摘要创建时间。 |
| `token_count` | `INTEGER` | 摘要 token 数。 |

`doc_article`：文档与条文切片映射表。

| 字段 | 类型 | 含义 |
|------|------|------|
| `article_id` | `VARCHAR(64)` 主键 | 条文切片 ID，对应 Milvus 向量主键。 |
| `doc_id` | `VARCHAR(64)` | 文档 ID，用于归组同一文档的切片。 |
| `doc_abs_path` | `VARCHAR(512)` 索引 | 服务端文档路径，仅后端使用，不返回前端。 |
| `created_at` | `DATETIME` | 条文切片入库时间。 |

### Redis 与 Milvus 结构

Redis 会话缓存：

| Key | 类型 | 含义 |
|-----|------|------|
| `session:{id}:unsummarized` | `List` | 近期未摘要消息，模型上下文直接使用。 |
| `session:{id}:summarized` | `List` | 已摘要消息备份，供前端展示历史。 |
| `session:{id}:summary` | `List` | 历史摘要列表。 |
| `session:{id}:meta` | `Hash` | `total`、`unsum`、`sum` 三类 token 计数。 |

Milvus collection：

| 字段 | 类型 | 含义 |
|------|------|------|
| `id` | `VARCHAR` 主键 | 向量 ID，与 `doc_article.article_id` 对应。 |
| `text` | `VARCHAR` | 条文切片文本。 |
| `metadata` | `JSON` | 来源、条文号等扩展信息。 |
| `vector` | `FLOAT_VECTOR` | embedding 向量。 |
| `filename` | `VARCHAR` | 文档名。 |
| `article` | `VARCHAR` | 条文编号。 |
| `start_position` | `INT64` | 切片在原文中的起始位置。 |

### 已落地性能优化

- 会话列表接口将“先查会话、再逐个统计消息数”的 N+1 查询改为一次 `LEFT JOIN + GROUP BY` 聚合查询，减少数据库往返。
- 历史消息摘要状态从 ORM 对象逐条加载修改，改为批量 `UPDATE chatmessage SET is_summarized = true WHERE id IN (...)`，减少对象加载和内存开销。
- Redis 缓存近期上下文和摘要 token 计数，降低每轮聊天读取 MySQL 的频率。
- Token 阈值滑动窗口控制上下文长度，超阈值后对早期对话做摘要压缩。
- 文档新增、删除、重建时同步刷新 MySQL、Milvus 和 RAG/BM25 缓存，避免旧向量或旧切片残留。
- SSE 真流式输出让用户先看到模型生成内容，再接收引用证据和校验结果。

### 后续可做优化

- 为 `chatmessage(session_id, timestamp)` 和 `summarymessage(session_id, timestamp)` 增加联合索引，加速上下文恢复和会话详情加载。
- 将文档元数据从 `doc_article` 中拆成独立文档表，维护 `filename`、`doc_key`、`article_count`，提升文档管理查询效率。
- 文档名搜索从 `%keyword% LIKE` 升级为 MySQL FULLTEXT 或 Elasticsearch。
- 深度研究任务状态从内存 job store 升级为 Redis / MySQL 持久任务表，支持服务重启恢复。
- Milvus 数据规模扩大后，将 `FLAT` 索引升级为 `HNSW` 或 `IVF_FLAT`，通过压测平衡召回率和延迟。

## 快速开始

### 1. 环境要求

- Python 3.10+
- MySQL 8.0+
- Redis 6.0+
- Conda 环境：建议使用 `agent`

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

# 嵌入模型：api 使用 EMBEDDING_API_MODEL，local 使用 LOCAL_EMBEDDING_MODEL
EMBEDDING_PROVIDER=api
LOCAL_EMBEDDING_MODEL=/path/to/bge-large-zh-v1.5
EMBEDDING_API_MODEL=text-embedding-v4
EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_API_DIMENSIONS=1024
EMBEDDING_API_BATCH_SIZE=10
LOCAL_EMBEDDING_BATCH_SIZE=8
EMBEDDING_DIM=1024

# tokenizer 仅本地摘要/分词逻辑需要
TOKENIZER_MODEL=/path/to/tokenizer

# 如果要切回本地 embedding：
# EMBEDDING_PROVIDER=local
# LOCAL_EMBEDDING_MODEL=/path/to/bge-large-zh-v1.5
# LOCAL_EMBEDDING_BATCH_SIZE=8
# EMBEDDING_DIM=1024

# 重排模型：api 使用 RERANKER_API_MODEL，local 使用 LOCAL_RERANKER_MODEL
ENABLE_RERANKER=True
RERANKER_PROVIDER=api
LOCAL_RERANKER_MODEL=/path/to/bge-reranker-v2-m3
RERANKER_API_MODEL=qwen3-rerank
RERANKER_API_KEY=your_reranker_api_key
RERANKER_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-api/v1
RERANKER_API_BATCH_SIZE=500
RERANKER_API_TIMEOUT=60
LOCAL_RERANKER_USE_FP16=True

# 如果要切回本地 reranker：
# ENABLE_RERANKER=True
# RERANKER_PROVIDER=local
# LOCAL_RERANKER_MODEL=/path/to/bge-reranker-v2-m3

# 数据库
DATABASE_URL=mysql+asyncmy://user:password@localhost:3306/agent_project
REDIS_HOST=localhost
REDIS_PORT=6379

# 向量库
VECTOR_COLLECTION_NAME=legal_knowledge
RE_BUILD=False
```

如果使用不支持 `dimensions` 参数的 embedding 接口，可将 `EMBEDDING_API_DIMENSIONS` 留空或设为 `none`，并把 `EMBEDDING_DIM` 设置为该模型实际返回维度。

### 4. 初始化数据库

```sql
CREATE DATABASE agent_project CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

表结构会在首次启动时自动创建。

### 5. 启动服务

```bash
bash start.sh
# 或
python web_app.py
```

服务启动后：

- Web 界面：http://localhost:5000
- API 文档：http://localhost:5000/docs
- ReDoc：http://localhost:5000/redoc

## API 接口

### 用户相关

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/register` | 用户注册 |
| POST | `/api/login` | 用户登录 |
| POST | `/api/send_code` | 发送邮箱验证码 |

### 会话相关

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/new_session` | 创建新会话 |
| POST | `/api/chat` | 普通非流式问答，返回回答、引用和校验结果 |
| POST | `/api/chat/stream` | SSE 真流式问答 |
| GET | `/api/sessions` | 获取会话列表 |
| GET | `/api/sessions/{session_id}` | 获取会话详情 |
| DELETE | `/api/sessions/{session_id}` | 删除会话 |

### 文档相关

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/doc/upload` | 上传文档并增量向量化 |
| GET | `/api/doc/search` | 按文档名搜索知识库文档 |
| GET | `/api/doc/preview` | 通过 doc_key 预览文档内容 |
| POST | `/api/doc/add` | 添加服务端已有文档到知识库 |
| DELETE | `/api/doc/delete` | 删除文档并同步清理 MySQL、Milvus 和缓存 |

### 深度研究

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/research/start` | 创建深度研究任务 |
| GET | `/api/research/status/{job_id}` | 查询研究进度 |
| GET | `/api/research/result/{job_id}` | 获取研究报告、引用和校验结果 |

## 核心模块说明

### RAG 检索模块 (`RAG/retrieve.py`)

- **混合召回**：BM25 + Milvus 向量检索双路召回。
- **融合排序**：使用 RRF 融合稀疏检索和向量检索结果。
- **重排序**：支持 API 或本地 Reranker 二次排序。
- **证据输出**：返回结构化 evidence，供引用回答和证据校验使用。

### 会话管理 (`db_crud/session_manage.py`)

- **双层存储**：Redis 热数据 + MySQL 持久化。
- **自动压缩**：Token 超阈值自动触发摘要。
- **一致性保证**：数据库优先写入，失败时中止缓存更新。

### 向量管理 (`RAG/vector_doc.py`)

- **混合检索**：Milvus 向量检索 + BM25 关键词检索。
- **动态更新**：支持上传、删除和重新向量化文档。
- **一致性清理**：重建或删除时同步处理 MySQL、Milvus 和 RAG 缓存。

### 引用与证据校验 (`RAG/evidence.py`, `RAG/evidence_verifier.py`)

- **引用编号**：将检索结果编号为 `[1]`、`[2]`。
- **内容哈希**：为证据摘录生成短指纹，用于去重和证据标识。
- **引用校验**：检查回答是否引用了不存在的证据编号，以及法律判断是否缺少引用。

### 深度研究 (`RAG/research_service.py`)

- **问题拆解**：将复杂法律问题拆成依据、条件、救济、风险等子问题。
- **多轮检索**：对每个子问题分别检索并合并证据。
- **报告生成**：调用 LLM 基于证据生成 Markdown 研究报告。

### SSE 流式输出 (`router/chat_router.py`, `RAG/streaming.py`)

- **真实 token 流**：后端消费 LLM `astream`，逐块发送 `content` 事件。
- **元数据分离**：回答完成后再发送 citations 和 verification。
- **前端兼容**：流式失败时保留普通 `/api/chat` 兜底链路。

## 开发指南

### 日志配置

日志级别通过 `LOG_LEVEL` 环境变量控制，日志文件存储在 `./log/` 目录。

### 自定义检索策略

修改 `RAG/retrieve_process.py` 中的检索逻辑，支持：

- 调整召回数量。
- 修改相似度阈值。
- 自定义重排策略。

## 常见问题

### Q: 首次启动报错 "Collection not found"

A: 将 `.env` 中 `RE_BUILD=True`，首次启动会创建向量库。

### Q: Redis 连接失败

A: 检查 Redis 服务是否启动，端口是否正确；也可以查看启动脚本中的 Redis 等待逻辑。

### Q: embedding API 报 dimensions 参数错误

A: 将 `EMBEDDING_API_DIMENSIONS` 留空或设为 `none`，并确认 `EMBEDDING_DIM` 与模型真实返回维度一致。
