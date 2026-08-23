import asyncio


def test_start_research_job_creates_pending_job():
    from RAG.research_service import start_research_job, get_research_job

    job = start_research_job("公司单方调岗是否合法？", "session-1")
    saved = get_research_job(job["job_id"])

    assert saved["status"] == "pending"
    assert saved["question"] == "公司单方调岗是否合法？"
    assert saved["progress"] == 0


def test_build_research_subquestions_contains_core_legal_angles():
    from RAG.research_service import build_research_subquestions

    questions = build_research_subquestions("公司单方调岗是否合法？")

    assert len(questions) == 4
    assert "法律依据" in questions[0]
    assert "适用条件" in questions[1]
    assert "救济路径" in questions[2]
    assert "风险" in questions[3]


def test_run_research_job_completes_with_report(monkeypatch):
    import RAG.research_service as service
    generated = []

    async def fake_retrieve(query, exclude_ids=None):
        return service.json.dumps({
            "text": "用人单位应当按照约定履行义务。[1]",
            "retrieved_ids": [query],
            "evidences": [
                {
                    "citation_id": 1,
                    "doc_key": "safe-key",
                    "chunk_id": "chunk-1",
                    "filename": "劳动合同法",
                    "article": "第三十条",
                    "source_label": "劳动合同法 / 第三十条",
                    "excerpt": "用人单位应当按照劳动合同约定支付劳动报酬。",
                    "content_hash": "abc123",
                    "score": None,
                }
            ],
        })

    monkeypatch.setattr(service, "retrieve_for_research", fake_retrieve)

    async def fake_generate_report(question, sections, evidences):
        generated.append({
            "question": question,
            "sections": sections,
            "evidences": evidences,
        })
        return "# LLM深度研究报告\n\n用人单位调整岗位需要有事实和法律依据。[1]"

    monkeypatch.setattr(service, "generate_research_report_with_llm", fake_generate_report)

    job = service.start_research_job("公司单方调岗是否合法？", "session-1")
    asyncio.run(service.run_research_job(job["job_id"]))
    saved = service.get_research_job(job["job_id"])

    assert saved["status"] == "completed"
    assert "LLM深度研究报告" in saved["report"]
    assert generated
    assert saved["citations"]
    assert saved["verification"]["status"] in {"passed", "warning", "failed"}
