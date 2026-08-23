import re
from typing import Any, Dict, List


LEGAL_KEYWORDS = (
    "根据",
    "应当",
    "可以",
    "不得",
    "禁止",
    "责任",
    "违法",
    "赔偿",
    "起诉",
    "仲裁",
    "期限",
    "义务",
    "权利",
    "风险",
)


def _extract_citations(text: str) -> List[int]:
    return [int(match) for match in re.findall(r"\[(\d+)\]", text or "")]


def _split_sentences(answer: str) -> List[str]:
    pieces = re.split(r"(?<=[。！？；;.!?])|\n+", answer or "")
    return [piece.strip() for piece in pieces if piece.strip()]


def _needs_citation(sentence: str) -> bool:
    return any(keyword in sentence for keyword in LEGAL_KEYWORDS)


def verify_answer_citations(answer: str, evidences: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_ids = {int(evidence.get("citation_id")) for evidence in evidences if evidence.get("citation_id") is not None}
    used_ids = _extract_citations(answer)
    invalid_citations = sorted({citation_id for citation_id in used_ids if citation_id not in valid_ids})

    claim_sentences = [sentence for sentence in _split_sentences(answer) if _needs_citation(sentence)]
    missing = []
    for sentence in claim_sentences:
        if _extract_citations(sentence):
            continue
        index = (answer or "").find(sentence)
        trailing = (answer or "")[index + len(sentence): index + len(sentence) + 8] if index >= 0 else ""
        if _extract_citations(trailing):
            continue
        missing.append(sentence)

    warnings = []
    for sentence in missing:
        warnings.append(f"以下法律判断可能缺少引用：{sentence}")
    for citation_id in invalid_citations:
        warnings.append(f"回答引用了不存在的证据编号：[{citation_id}]")

    if invalid_citations:
        status = "failed"
    elif missing:
        status = "warning"
    else:
        status = "passed"

    return {
        "status": status,
        "claims_checked": len(claim_sentences),
        "cited_claims": len(claim_sentences) - len(missing),
        "missing_citation_count": len(missing),
        "invalid_citations": invalid_citations,
        "warnings": warnings,
    }
