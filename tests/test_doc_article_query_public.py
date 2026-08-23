from datetime import datetime


def test_public_document_record_omits_server_path():
    from db_crud.doc_article_query import build_public_document_record

    record = build_public_document_record(
        "/srv/private/docs/劳动合同.md",
        article_count=12,
        created_at=datetime(2026, 8, 22, 12, 0, 0),
        secret="test-secret",
    )

    assert record["doc_key"]
    assert record["filename"] == "劳动合同.md"
    assert record["article_count"] == 12
    assert record["created_at"] == "2026-08-22T12:00:00"
    assert "doc_abs_path" not in record
    assert "/srv/private" not in str(record)
