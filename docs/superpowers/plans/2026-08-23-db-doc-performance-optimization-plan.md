# Database Documentation And Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete database structure documentation, update README, implement two low-risk database performance optimizations, run tests, and regenerate the source zip.

**Architecture:** Keep the existing SQLModel and SQLAlchemy data layer. Optimize query shape without changing API contracts or database schema, then document the real MySQL, Redis, and Milvus storage design.

**Tech Stack:** FastAPI, SQLModel, SQLAlchemy, MySQL, Redis, Milvus, pytest.

## Global Constraints

- Keep the resume project description as six core work items; only improve the descriptions inside those six items.
- Do not expose server absolute paths in user-facing documentation examples.
- Do not add destructive database migrations in this change.
- Use TDD for behavior changes.

---

### Task 1: Optimize Chat Session List Query

**Files:**
- Modify: `db_crud/chat_memory_crud.py`
- Create: `tests/test_chat_memory_crud_optimizations.py`

**Interfaces:**
- Produces: `_build_user_session_list_stmt(user_id: str)`
- Consumes: `ChatSession`, `ChatMessage`, SQLAlchemy `func.count`

- [ ] Write a failing test proving the session list query uses a joined aggregate.
- [ ] Run the focused test and verify it fails because the helper is missing.
- [ ] Implement `_build_user_session_list_stmt` with `LEFT OUTER JOIN`, `COUNT`, `GROUP BY`, and descending timestamp ordering.
- [ ] Update `get_user_session_list` to execute the aggregate statement once and preserve the returned response shape.
- [ ] Run the focused test and verify it passes.

### Task 2: Optimize Summary Message Updates

**Files:**
- Modify: `db_crud/chat_memory_crud.py`
- Modify: `tests/test_chat_memory_crud_optimizations.py`

**Interfaces:**
- Produces: `_build_mark_messages_summarized_stmt(message_ids: List[str])`
- Consumes: `ChatMessage`, SQLAlchemy `update`

- [ ] Write a failing test proving summarized message state is expressed as one SQL `UPDATE`.
- [ ] Run the focused test and verify it fails because the helper is missing.
- [ ] Implement `_build_mark_messages_summarized_stmt`.
- [ ] Update `db_add_summary_and_update_messages` and `db_force_mark_summarized` to use the bulk update statement.
- [ ] Run the focused test and verify it passes.

### Task 3: Document Data Structures And Optimization Design

**Files:**
- Modify: `docs/project/law_agent_resume_technical_description.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: SQLModel models in `db_crud/session_model.py` and `db_crud/doc_article_model.py`.
- Consumes: Redis key definitions in `db_crud/session_manage.py`.
- Consumes: Milvus schema in `RAG/vector_doc.py`.

- [ ] Add database table structure sections with every field's meaning.
- [ ] Add Redis key and Milvus collection structure descriptions.
- [ ] Add implemented performance optimizations and future optimization designs.
- [ ] Keep the resume section to the existing six core work items and only improve item descriptions.

### Task 4: Verification And Packaging

**Files:**
- Output: `/tmp/law_agent_source_20260823.zip`

- [ ] Run focused tests.
- [ ] Run full test suite.
- [ ] Regenerate source zip from `HEAD` plus current working tree state after changes are committed or staged into archive logic.
- [ ] Verify the zip excludes `.git`, `.vscode`, model files, logs, and backup files.
