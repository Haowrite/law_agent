import hashlib
from typing import Any, Dict, List, Optional

from config import DOCUMENT_KEY_SECRET
from RAG.document_library import document_key_for_path


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _short_excerpt(text: str, max_chars: int = 240) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."


def build_evidence_item(
    citation_id: int,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    score: Optional[float] = None,
) -> Dict[str, Any]:
    metadata = metadata or {}
    filename = metadata.get("filename") or "未知文档"
    article = metadata.get("article") or "全文片段"
    source = metadata.get("source") or f"{filename}/{article}"
    chunk_id = metadata.get("id") or content_hash(f"{filename}:{article}:{text}")

    return {
        "citation_id": citation_id,
        "doc_key": document_key_for_path(source, secret=DOCUMENT_KEY_SECRET),
        "chunk_id": chunk_id,
        "filename": filename,
        "article": article,
        "source_label": f"{filename} / {article}",
        "excerpt": _short_excerpt(text),
        "content_hash": content_hash(text),
        "score": score,
    }


def format_evidences_for_prompt(evidences: List[Dict[str, Any]]) -> str:
    if not evidences:
        return ""

    blocks = []
    for evidence in evidences:
        blocks.append(
            "\n".join([
                f"[证据{evidence['citation_id']}]",
                f"来源：{evidence['source_label']}",
                f"原文摘录：{evidence['excerpt']}",
            ])
        )
    return "\n\n".join(blocks)


def prepare_public_citations(evidences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    public_fields = [
        "citation_id",
        "doc_key",
        "chunk_id",
        "filename",
        "article",
        "source_label",
        "excerpt",
        "content_hash",
        "score",
    ]
    return [
        {field: evidence.get(field) for field in public_fields}
        for evidence in evidences
    ]
