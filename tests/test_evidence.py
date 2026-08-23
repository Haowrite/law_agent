def test_build_evidence_item_creates_public_traceable_fields():
    from RAG.evidence import build_evidence_item

    evidence = build_evidence_item(
        citation_id=1,
        text="当事人应当按照约定全面履行自己的义务。" * 20,
        metadata={
            "id": "chunk-1",
            "filename": "民法典",
            "article": "第五百零九条",
            "source": "/home/RAG_agent/files/private/民法典.md",
        },
        score=0.87,
    )

    assert evidence["citation_id"] == 1
    assert evidence["chunk_id"] == "chunk-1"
    assert evidence["filename"] == "民法典"
    assert evidence["article"] == "第五百零九条"
    assert evidence["source_label"] == "民法典 / 第五百零九条"
    assert len(evidence["content_hash"]) == 16
    assert len(evidence["excerpt"]) <= 260
    assert "doc_abs_path" not in evidence
    assert "/home/RAG_agent" not in str(evidence)


def test_format_evidences_for_prompt_uses_numbered_evidence():
    from RAG.evidence import build_evidence_item, format_evidences_for_prompt

    evidence = build_evidence_item(
        citation_id=1,
        text="用人单位应当按照劳动合同约定支付劳动报酬。",
        metadata={"filename": "劳动合同法", "article": "第三十条"},
    )

    prompt = format_evidences_for_prompt([evidence])

    assert "[证据1]" in prompt
    assert "来源：劳动合同法 / 第三十条" in prompt
    assert "原文摘录：用人单位应当按照劳动合同约定支付劳动报酬。" in prompt


def test_prepare_public_citations_keeps_safe_fields_only():
    from RAG.evidence import build_evidence_item, prepare_public_citations

    evidence = build_evidence_item(
        citation_id=1,
        text="证据内容",
        metadata={
            "filename": "示例文档",
            "article": "全文片段1",
            "source": "/secret/path/example.md",
        },
    )

    public = prepare_public_citations([evidence])

    assert public == [
        {
            "citation_id": 1,
            "doc_key": evidence["doc_key"],
            "chunk_id": evidence["chunk_id"],
            "filename": "示例文档",
            "article": "全文片段1",
            "source_label": "示例文档 / 全文片段1",
            "excerpt": "证据内容",
            "content_hash": evidence["content_hash"],
            "score": None,
        }
    ]
    assert "/secret/path" not in str(public)
