from pathlib import Path


def test_safe_document_filename_keeps_extension_and_removes_path_parts():
    from RAG.document_library import safe_document_filename

    assert safe_document_filename("../民法典.md") == "民法典.md"
    assert safe_document_filename("合同/../..//劳动 合同.txt") == "劳动_合同.txt"


def test_safe_document_filename_rejects_unsupported_extensions():
    from RAG.document_library import safe_document_filename

    try:
        safe_document_filename("payload.py")
    except ValueError as exc:
        assert "不支持的文件格式" in str(exc)
    else:
        raise AssertionError("Expected unsupported extension to be rejected.")


def test_resolve_document_path_requires_document_root(tmp_path):
    from RAG.document_library import resolve_document_path

    root = tmp_path / "library"
    root.mkdir()
    doc = root / "a.md"
    doc.write_text("content", encoding="utf-8")

    assert resolve_document_path(str(root), str(doc)) == doc.resolve()

    try:
        resolve_document_path(str(root), str(tmp_path / "outside.md"))
    except ValueError as exc:
        assert "文档路径必须位于文档库目录内" in str(exc)
    else:
        raise AssertionError("Expected outside path to be rejected.")


def test_read_document_preview_limits_text(tmp_path):
    from RAG.document_library import read_document_preview

    doc = tmp_path / "a.md"
    doc.write_text("abcdef", encoding="utf-8")

    assert read_document_preview(Path(doc), max_chars=3) == "abc"


def test_document_key_is_stable_and_does_not_expose_path():
    from RAG.document_library import document_key_for_path

    path = "/srv/private/docs/劳动合同.md"

    first = document_key_for_path(path, secret="test-secret")
    second = document_key_for_path(path, secret="test-secret")

    assert first == second
    assert len(first) == 32
    assert "/srv/private" not in first
    assert "劳动合同" not in first
