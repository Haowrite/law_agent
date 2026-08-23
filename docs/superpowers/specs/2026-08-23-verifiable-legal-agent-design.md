# 法律 Agent 引用可验证回答、证据校验器与深度研究模式设计文档

## 1. 背景和目标

当前系统已经有法律 RAG 能力：用户提问后，Agent 会通过知识库检索法律文档，再基于检索内容回答。现有问题是：

- 回答里虽然可能写了“根据某法某条”，但用户不能方便验证这句话来自哪个切片、哪份文档。
- 检索结果只作为大段文本传给模型，前端没有结构化证据卡片。
- 没有回答后的证据校验，模型可能漏标引用，或者某些结论没有引用。
- 深度研究类问题只能走普通聊天，缺少“拆问题、多轮检索、汇总报告”的工作流。
- 当前 UI 偏暗，按钮小，功能入口不够醒目。

本次目标按顺序交付三层能力：

1. 引用可验证回答：回答中出现 `[1]`、`[2]` 这种引用编号，用户可以看到证据来源、条文、摘录和校验信息。
2. 证据校验器：回答生成后检查引用是否存在、关键法律结论是否带引用、引用编号是否有效。
3. 深度研究模式：把复杂问题拆成多个研究子问题，逐项检索和校验，最后生成研究报告。

同时优化前端 UI，让主要操作入口更大、更亮、更有层次，用户一眼能看到“发送”“深度研究”“知识库文档”“证据”等入口。

## 2. 当前项目技术栈

后端：

- FastAPI：`web_app.py` 注册路由。
- LangGraph：`agent_service.py` 组织 Agent 节点、检索节点和最终响应节点。
- LangChain：LLM、工具调用、文档对象、BM25。
- Milvus：向量库，存储切片文本、metadata、向量。
- MySQL / SQLModel / SQLAlchemy：存储用户、会话、文档与条文关系。
- Redis：缓存会话上下文。
- Conda 环境：主要运行环境为 `agent`。

RAG：

- `RAG/retrieve.py`：对外工具 `retrieve_vector_store`，返回 JSON 字符串。
- `RAG/retrieve_process.py`：子进程内执行向量检索、BM25、RRF、Reranker、完整条文拼接。
- `RAG/vector_doc.py`：加载、清洗、切分、写入 Milvus。
- `RAG/document_library.py`：文档库路径安全、文档 key、预览。

前端：

- 单文件模板：`templates/index.html`。
- 原生 HTML/CSS/JavaScript。
- `marked` 渲染 Markdown。
- 当前聊天前端优先尝试 `/api/chat/stream`，失败后回退 `/api/chat`。

## 3. 总体架构

本次设计新增一个统一的“证据模型”，让普通问答、证据校验器和深度研究模式都复用同一种数据结构。

### 3.1 数据流

普通聊天：

```text
用户问题
  -> GeneralAgent 判断是否需要检索
  -> retrieve_vector_store
  -> Milvus + BM25 + Reranker
  -> 返回结构化 evidence 列表
  -> Agent 根据带编号的证据生成回答
  -> Evidence Verifier 检查回答
  -> 返回 response + citations + verification
  -> 前端渲染回答和证据卡片
```

普通聊天真流式：

```text
用户问题
  -> /api/chat/stream 鉴权
  -> 检索知识库证据
  -> 拼接对话上下文 + 编号证据
  -> LLM astream 逐块生成回答
  -> SSE content 事件实时返回
  -> 完整回答入库
  -> SSE metadata 事件返回 citations + verification
```

深度研究：

```text
用户问题
  -> 创建 research job
  -> 问题拆解为 3-5 个子问题
  -> 每个子问题执行 RAG 检索
  -> 收集每个子问题的小结和证据
  -> 调用 LLM 生成深度研究报告
  -> Evidence Verifier 校验报告
  -> 前端通过 job_id 查询进度和结果
```

### 3.2 为什么不直接让模型自由引用

法律 Agent 不能只相信模型“自己说引用了”。因此引用编号由系统从 RAG evidence 列表生成，模型只能引用 `[1]`、`[2]` 这些系统给出的编号。回答结束后，校验器会再检查：

- 回答里引用的编号是否存在。
- 法律判断句是否带引用。
- 引用证据是否为空。

## 4. 证据数据结构

公开给前端的证据对象不包含服务端绝对路径。

```json
{
  "citation_id": 1,
  "doc_key": "不可逆文档标识",
  "chunk_id": "Milvus中的切片ID或稳定hash",
  "filename": "民法典",
  "article": "第五百零九条",
  "source_label": "民法典 / 第五百零九条",
  "excerpt": "当事人应当按照约定全面履行自己的义务……",
  "content_hash": "证据内容hash前16位",
  "score": 0.87
}
```

字段说明：

- `citation_id`：回答中的编号，例如 `[1]`。
- `doc_key`：安全文档标识，用 HMAC 生成，不能反推出服务端路径。
- `chunk_id`：切片标识，方便后续查原文。
- `filename`：显示给用户的文档名。
- `article`：条文或全文片段编号。
- `source_label`：前端显示标题。
- `excerpt`：可验证摘录，控制长度，避免整篇泄露。
- `content_hash`：用户可用来确认证据内容没有被 UI 随意改动。
- `score`：可选，代表检索或重排分数。

## 5. 阶段一：引用可验证回答

### 5.1 后端设计

新增模块：

- `RAG/evidence.py`

职责：

- 从 Milvus 查询结果构造 evidence。
- 给 evidence 编号。
- 格式化给模型看的证据上下文。
- 格式化给前端看的公开证据对象。

修改：

- `RAG/retrieve_process.py`
  - `fetch_full_articles_from_milvus` 不再只返回纯文本，还返回 evidence 列表。
  - `batch_init_and_retrieve` 返回 `(text, retrieved_ids, evidences)`。
- `RAG/retrieve.py`
  - `retrieve_vector_store` JSON 增加 `evidences`。
- `agent_service.py`
  - `tool_call_node` 保存 `rag_evidences`。
  - `final_response_node` 把 `citations` 和 `verification` 放进 state。
- `router/chat_router.py`
  - `ChatResponse` 增加 `citations` 和 `verification` 字段。

### 5.2 Prompt 设计

给模型的知识库内容从普通文本改成：

```text
[证据1]
来源：民法典 / 第五百零九条
原文摘录：当事人应当按照约定全面履行自己的义务……

[证据2]
来源：劳动合同法 / 第三十条
原文摘录：用人单位应当按照劳动合同约定和国家规定……
```

回答要求：

- 每个法律结论句尽量带 `[1]`、`[2]`。
- 不能引用不存在的编号。
- 不能把证据原文大段照抄。
- 如果证据不足，明确说“当前知识库未检索到直接依据”。

### 5.3 前端设计

聊天气泡底部显示证据条：

```text
引用依据  3条
[1] 民法典 / 第五百零九条
[2] 劳动合同法 / 第三十条
```

点击证据后，右侧打开证据面板：

- 来源标题
- 条文/片段编号
- 原文摘录
- 内容哈希
- 文件状态

### 5.4 真 SSE 流式输出

前端优先调用 `POST /api/chat/stream`。后端必须使用 LLM 的 `astream` 能力逐块返回正文，而不是先调用普通 `/api/chat` 拿到完整答案后再按字符切片。

SSE 事件约定：

```text
data: {"type":"status","stage":"retrieving","message":"正在检索知识库"}

data: {"type":"content","content":"根据"}

data: {"type":"metadata","citations":[...],"verification":{...}}

data: [DONE]
```

设计要点：

- `content` 事件只承载 LLM 实时生成的文本片段。
- `metadata` 在回答生成完成并执行证据校验后发送。
- 前端收到 `metadata` 后渲染引用卡片和校验状态。
- 如果流式接口失败，前端可以回退 `/api/chat`，但回退路径只作为兼容兜底。

## 6. 阶段二：证据校验器

### 6.1 校验器定位

证据校验器不是替代律师判断，它是回答质量守门员。第一版采用确定性规则，不额外调用模型，原因是：

- 速度快。
- 可测试。
- 不增加 API 成本。
- 先解决“有没有引用、引用是否存在”的基础问题。

### 6.2 校验规则

新增模块：

- `RAG/evidence_verifier.py`

输入：

```python
answer: str
evidences: list[dict]
```

输出：

```json
{
  "status": "passed | warning | failed",
  "claims_checked": 5,
  "cited_claims": 4,
  "missing_citation_count": 1,
  "invalid_citations": [],
  "warnings": ["以下句子可能缺少引用：……"]
}
```

规则：

- 找出回答中的 `[1]`、`[2]`。
- 检查引用编号是否在 evidence 列表中。
- 将回答按句号、问号、分号、换行切句。
- 包含“应当、可以、不得、责任、违法、赔偿、起诉、仲裁、期限、根据”等法律判断词的句子视为需要引用。
- 需要引用但没有 `[n]` 的句子进入 warning。

### 6.3 前端展示

回答底部显示校验状态：

- 绿色：证据校验通过。
- 黄色：部分结论缺少引用。
- 红色：引用编号无效或没有证据。

点击状态可展开校验详情。

## 7. 阶段三：深度研究模式

### 7.1 为什么要异步

深度研究会做多轮检索和报告生成，耗时可能明显长于普通问答。同步聊天接口容易超时，也无法展示进度。因此设计为异步 job。

### 7.2 API 设计

新增路由：

- `POST /api/research/start`
- `GET /api/research/status/{job_id}`
- `GET /api/research/result/{job_id}`

第一版 job 存在内存中。后续如果需要服务重启后恢复，可以迁移到 MySQL。

启动请求：

```json
{
  "question": "公司单方调岗是否合法？",
  "session_id": "当前会话ID"
}
```

状态响应：

```json
{
  "job_id": "uuid",
  "status": "running",
  "progress": 60,
  "stage": "正在交叉验证"
}
```

结果响应：

```json
{
  "job_id": "uuid",
  "status": "completed",
  "report": "研究报告正文",
  "citations": [],
  "verification": {}
}
```

### 7.3 研究流程

第一版采用规则拆解，不额外调用 LLM 拆问题：

1. 核心法律依据是什么？
2. 构成条件或适用条件是什么？
3. 当事人可以采取哪些救济路径？
4. 有哪些风险和注意事项？

每个子问题调用 RAG 检索后，系统把“原问题、子问题检索小结、编号证据列表”交给 LLM 生成报告。LLM 报告要求：

- 使用 Markdown 输出。
- 包含问题概述、核心结论、法律依据、适用条件/判断要点、风险与例外、行动建议、证据清单。
- 每个法律判断尽量带 `[n]` 引用。
- 只能引用系统提供的证据编号。
- 证据不足时明确降低结论确定性。

LLM 调用失败时，才使用模板报告兜底：

```text
# 深度研究报告

## 一、问题概述
## 二、核心结论
## 三、法律依据
## 四、条件与风险
## 五、行动建议
## 六、证据清单
```

### 7.4 前端设计

输入框附近新增醒目的“深度研究”按钮。点击后：

- 如果输入框为空，提示用户输入问题。
- 创建研究任务。
- 面板显示进度条和当前阶段。
- 完成后把研究报告作为 AI 消息插入聊天区。
- 报告下方显示证据卡片和校验状态。

## 8. UI 优化设计

### 8.1 当前问题

- 页面整体过暗，主操作按钮不够突出。
- 按钮尺寸偏小，可点击区域不明显。
- 文档库、深度研究、发送等入口视觉优先级接近，用户难以判断下一步。

### 8.2 优化原则

- 主按钮更亮、更大：发送和深度研究使用高对比色。
- 次按钮清晰但不抢主按钮。
- 证据和研究入口使用明确文字，不只依赖图标。
- 保持专业法律产品气质，避免花哨的营销风。

### 8.3 具体改动

- 增大输入栏按钮高度到 44px 左右。
- 发送按钮保持图标，但颜色更亮。
- 新增“深度研究”文字按钮。
- “文档库”按钮改成图标 + 文本。
- 聊天气泡最大宽度稍增，证据卡片有浅色边框和状态色。
- 背景从纯深色改成更有层次的深灰，提升可读性。

## 9. 安全与隐私

- 前端永远不展示服务端绝对路径。
- 文档仍通过 `doc_key` 操作。
- 证据对象只显示 `filename`、`article`、`excerpt`、`content_hash`。
- 摘录长度受控，避免一次性暴露整篇内部文档。
- 深度研究 job 只绑定当前会话，不做跨用户共享。

## 10. 测试策略

后端单元测试：

- evidence 构造：能生成 citation id、source label、hash。
- verifier：能识别缺引用、无效引用、通过状态。
- retrieve 工具：JSON 包含 `text`、`retrieved_ids`、`evidences`。
- research service：能创建 job、推进阶段、返回报告结构。

前端静态验证：

- `templates/index.html` 内联 JS 能通过 Node 语法检查。
- 搜索模板中不残留 `doc_abs_path`。
- 新按钮和证据面板函数存在。

集成验证：

- `conda run -n agent python -m py_compile ...`
- `conda run -n agent python -m pytest ...`
- 路由导入检查：`from web_app import app` 或导入相关 router。

## 11. 交付顺序

必须严格按以下顺序：

1. 引用可验证回答。
2. 证据校验器。
3. 深度研究模式。
4. UI 优化。
5. 统一验证。

每一阶段都要先写测试，再实现，最后跑验证。
