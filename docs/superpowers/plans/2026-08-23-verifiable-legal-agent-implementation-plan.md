# Verifiable Legal Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build citation-verifiable answers, an evidence verifier, deep research mode, and a clearer premium UI for the existing legal RAG assistant.

**Architecture:** Extend the current RAG output from plain text to structured evidence, feed numbered evidence into the Agent prompt, verify generated answers after the LLM responds, and expose citations/verification to the frontend. Deep research is implemented as an asynchronous in-memory job service that reuses the same retrieval and verification primitives, then calls the LLM to synthesize the final report. Chat streaming uses a real SSE endpoint backed by the LLM `astream` API; it must not slice a completed answer to imitate streaming.

**Tech Stack:** FastAPI, LangGraph, LangChain, Milvus, SQLModel/MySQL, Redis, vanilla HTML/CSS/JavaScript, pytest, Node syntax check.

## Global Constraints

- Do not expose server absolute paths to the frontend.
- Keep document operations based on `doc_key`.
- Implement in this order: citation-verifiable answers, evidence verifier, deep research mode, UI optimization.
- Preserve the design document in `docs/superpowers/specs/2026-08-23-verifiable-legal-agent-design.md`.
- Use TDD for production behavior changes.
- Keep changes scoped to the legal Agent, RAG, routers, and `templates/index.html`.

---

## File Structure

- Create `RAG/evidence.py`: evidence construction, numbering, excerpt hashing, prompt formatting.
- Create `RAG/evidence_verifier.py`: deterministic citation verification.
- Create `RAG/research_service.py`: in-memory research job orchestration.
- Create `RAG/streaming.py`: small helper for consuming LLM `astream` chunks.
- Create `router/research_router.py`: research job API.
- Modify `RAG/retrieve_process.py`: return evidence with retrieval results.
- Modify `RAG/retrieve.py`: include evidence in tool JSON.
- Modify `agents/base_agent.py`: add state fields for evidence and verification.
- Modify `agents/general_agent.py`: instruct model to cite numbered evidence.
- Modify `agent_service.py`: carry evidence through graph, run verifier in final node.
- Modify `router/chat_router.py`: return citations and verification in `/api/chat`, and add `/api/chat/stream` for true SSE.
- Modify `web_app.py`: include research router.
- Modify `templates/index.html`: show citations, verifier status, deep research controls, stronger UI.
- Add tests under `tests/` for evidence, verifier, retrieval JSON shape, research service.

---

## Task 1: Citation-Verifiable Answers

**Files:**
- Create: `RAG/evidence.py`
- Modify: `RAG/retrieve_process.py`
- Modify: `RAG/retrieve.py`
- Modify: `agents/base_agent.py`
- Modify: `agents/general_agent.py`
- Modify: `agent_service.py`
- Modify: `router/chat_router.py`
- Test: `tests/test_evidence.py`
- Test: `tests/test_retrieve_tool_contract.py`

**Interfaces:**
- Produces: `build_evidence_item(...) -> dict`
- Produces: `format_evidences_for_prompt(evidences: list[dict]) -> str`
- Produces: `prepare_public_citations(evidences: list[dict]) -> list[dict]`
- Changes: retrieval JSON includes `evidences`.
- Changes: chat response includes `citations`.

**Steps:**

- [ ] Write failing tests in `tests/test_evidence.py`:
  - `build_evidence_item` creates `citation_id`, `source_label`, `excerpt`, `content_hash`.
  - public citation does not include `doc_abs_path`.
  - prompt formatting includes `[证据1]`.

- [ ] Run:
  - `conda run -n agent python -m pytest tests/test_evidence.py -q`
  - Expected: FAIL because `RAG.evidence` does not exist.

- [ ] Implement `RAG/evidence.py` with:
  - `content_hash(text: str) -> str`
  - `build_evidence_item(...) -> dict`
  - `format_evidences_for_prompt(evidences) -> str`
  - `prepare_public_citations(evidences) -> list[dict]`

- [ ] Run evidence tests and make them pass.

- [ ] Modify retrieval:
  - `fetch_full_articles_from_milvus` returns `(text, ids, evidences)`.
  - `batch_init_and_retrieve` returns `(text, ids, evidences)`.
  - `retrieve_vector_store` JSON includes `evidences`.

- [ ] Add `rag_evidences` to `AgentState`.

- [ ] Modify `tool_call_node` to append evidences.

- [ ] Modify `GeneralAgent` prompt to show formatted evidence and require `[1]` style citations.

- [ ] Modify `/api/chat` response model to return `citations`.

- [ ] Run:
  - `conda run -n agent python -m pytest tests/test_evidence.py tests/test_retrieve_tool_contract.py -q`
  - `conda run -n agent python -m py_compile RAG/evidence.py RAG/retrieve.py RAG/retrieve_process.py agent_service.py router/chat_router.py`

---

## Task 2: Evidence Verifier

**Files:**
- Create: `RAG/evidence_verifier.py`
- Modify: `agent_service.py`
- Modify: `router/chat_router.py`
- Test: `tests/test_evidence_verifier.py`

**Interfaces:**
- Produces: `verify_answer_citations(answer: str, evidences: list[dict]) -> dict`
- Changes: chat response includes `verification`.

**Steps:**

- [ ] Write failing tests:
  - answer with valid `[1]` passes.
  - answer with `[3]` when only one evidence exists warns or fails.
  - legal judgment sentence without citation emits warning.

- [ ] Run:
  - `conda run -n agent python -m pytest tests/test_evidence_verifier.py -q`
  - Expected: FAIL because module does not exist.

- [ ] Implement verifier:
  - Extract citations with regex `\[(\d+)\]`.
  - Split answer into claim-like sentences.
  - Detect legal keywords.
  - Return `status`, `claims_checked`, `cited_claims`, `missing_citation_count`, `invalid_citations`, `warnings`.

- [ ] Modify `final_response_node` to verify `state["response"]` against `state["rag_evidences"]`.

- [ ] Modify `/api/chat` response model to include verification.

- [ ] Run:
  - `conda run -n agent python -m pytest tests/test_evidence_verifier.py -q`
  - `conda run -n agent python -m py_compile RAG/evidence_verifier.py agent_service.py router/chat_router.py`

---

## Task 3: Deep Research Mode

**Files:**
- Create: `RAG/research_service.py`
- Create: `router/research_router.py`
- Modify: `web_app.py`
- Test: `tests/test_research_service.py`

**Interfaces:**
- Produces: `start_research_job(question: str, session_id: str) -> dict`
- Produces: `get_research_job(job_id: str) -> dict | None`
- Produces: `run_research_job(job_id: str) -> None`
- Produces: `generate_research_report_with_llm(question, sections, evidences) -> str`
- API: `POST /api/research/start`
- API: `GET /api/research/status/{job_id}`
- API: `GET /api/research/result/{job_id}`

**Steps:**

- [ ] Write failing tests:
  - creating a job returns `job_id`, `status=pending`.
  - rule-based subquestions include legal basis, conditions, remedies, risks.
  - completed job result has `report`, `citations`, `verification`.

- [ ] Run:
  - `conda run -n agent python -m pytest tests/test_research_service.py -q`
  - Expected: FAIL because service does not exist.

- [ ] Implement in-memory job store in `RAG/research_service.py`.

- [ ] Implement rule-based subquestion generation.

- [ ] Implement LLM report generation:
  - prompt includes original question, subquestion retrieval summaries, and formatted numbered evidence.
  - report must cite only provided evidence ids.
  - fallback to template report only if LLM generation fails.

- [ ] Implement FastAPI research router and include it in `web_app.py`.

- [ ] Run:
  - `conda run -n agent python -m pytest tests/test_research_service.py -q`
  - `conda run -n agent python -m py_compile RAG/research_service.py router/research_router.py web_app.py`

---

## Task 3.5: True SSE Chat Streaming

**Files:**
- Create: `RAG/streaming.py`
- Modify: `router/chat_router.py`
- Modify: `templates/index.html`
- Test: `tests/test_streaming.py`

**Interfaces:**
- Produces: `stream_llm_text(llm, messages) -> AsyncIterator[str]`
- API: `POST /api/chat/stream`
- SSE events: `status`, `content`, `metadata`, `[DONE]`

**Steps:**

- [ ] Write failing test proving `stream_llm_text` consumes an LLM `astream` iterator and yields chunks as they arrive.

- [ ] Implement `RAG/streaming.py`.

- [ ] Add `/api/chat/stream`:
  - validate message length and session ownership.
  - retrieve structured evidence.
  - build prompt from conversation context and numbered evidence.
  - call `main_llm.astream` through `stream_llm_text`.
  - emit `content` events for chunks.
  - after completion, save the full conversation and emit `metadata` with citations and verification.

- [ ] Update frontend SSE parser:
  - append only `content` chunks to the visible answer.
  - store `metadata` for citation and verification rendering.
  - ignore `status` events in answer text.

- [ ] Run:
  - `conda run -n agent python -m pytest tests/test_streaming.py -q`
  - `conda run -n agent python -m py_compile RAG/streaming.py router/chat_router.py`

---

## Task 4: UI Optimization

**Files:**
- Modify: `templates/index.html`

**Interfaces:**
- Consumes: `/api/chat` response fields `response`, `citations`, `verification`.
- Consumes: `/api/chat/stream` SSE `content` and `metadata` events.
- Consumes: research endpoints.
- Produces: evidence panel UI, verification badge UI, deep research button and progress UI.

**Steps:**

- [ ] Increase primary button size and contrast.

- [ ] Change document library button from icon-only to icon plus text.

- [ ] Add “深度研究” button near the input area.

- [ ] Add citation rendering:
  - response bubble stores `citations`.
  - evidence strip appears below assistant answer.
  - clicking a citation opens evidence panel.

- [ ] Add verification badge:
  - green passed, yellow warning, red failed.
  - click or hover shows warning list.

- [ ] Add research progress modal/panel:
  - shows job stage and progress.
  - inserts final report into chat when complete.

- [ ] Run:
  - Node inline script syntax check for `templates/index.html`.
  - `rg -n "doc_abs_path" templates/index.html` must return no frontend path usage.

---

## Task 5: Final Verification

**Files:**
- All touched files.

**Steps:**

- [ ] Run unit tests:
  - `conda run -n agent python -m pytest tests/test_evidence.py tests/test_evidence_verifier.py tests/test_research_service.py tests/test_document_library.py tests/test_vector_doc_split.py tests/test_rag_cache_refresh.py tests/test_doc_article_query_public.py -q`

- [ ] Run py_compile:
  - `conda run -n agent python -m py_compile RAG/evidence.py RAG/evidence_verifier.py RAG/research_service.py RAG/retrieve.py RAG/retrieve_process.py agent_service.py agents/base_agent.py agents/general_agent.py router/chat_router.py router/research_router.py web_app.py`

- [ ] Run route import:
  - `conda run -n agent python -c "from web_app import app; print(len(app.routes))"`

- [ ] Run frontend JS syntax check:
  - `node -e "const fs=require('fs'); const html=fs.readFileSync('templates/index.html','utf8'); const start=html.indexOf('<script>'); const end=html.indexOf('</script>', start); const js=html.slice(start+'<script>'.length,end); new Function(js); console.log('js syntax ok');"`

- [ ] Confirm docs remain:
  - `test -f docs/superpowers/specs/2026-08-23-verifiable-legal-agent-design.md`
  - `test -f docs/superpowers/plans/2026-08-23-verifiable-legal-agent-implementation-plan.md`
