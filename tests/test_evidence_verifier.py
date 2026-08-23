def _evidence(citation_id=1):
    return {
        "citation_id": citation_id,
        "source_label": "民法典 / 第五百零九条",
        "excerpt": "当事人应当按照约定全面履行自己的义务。",
    }


def test_verify_answer_citations_passes_when_legal_claim_has_valid_citation():
    from RAG.evidence_verifier import verify_answer_citations

    result = verify_answer_citations(
        "用人单位应当按照约定履行义务。[1]",
        [_evidence(1)],
    )

    assert result["status"] == "passed"
    assert result["invalid_citations"] == []
    assert result["missing_citation_count"] == 0


def test_verify_answer_citations_flags_invalid_citation_number():
    from RAG.evidence_verifier import verify_answer_citations

    result = verify_answer_citations(
        "用人单位应当按照约定履行义务。[3]",
        [_evidence(1)],
    )

    assert result["status"] == "failed"
    assert result["invalid_citations"] == [3]


def test_verify_answer_citations_warns_when_legal_claim_lacks_citation():
    from RAG.evidence_verifier import verify_answer_citations

    result = verify_answer_citations(
        "用人单位应当按照约定履行义务。建议先协商。",
        [_evidence(1)],
    )

    assert result["status"] == "warning"
    assert result["missing_citation_count"] == 1
    assert "缺少引用" in result["warnings"][0]
