def test_rebuild_sync_clears_all_existing_doc_article_rows_before_insert():
    from RAG.doc_article_sync import sync_rebuild_doc_articles

    calls = []

    def delete_all():
        calls.append(("delete_all",))
        return 12

    def insert(doc_abs_path, doc_id, article_ids):
        calls.append(("insert", doc_abs_path, doc_id, article_ids))

    ids = iter(["doc-1", "doc-2"])

    inserted = sync_rebuild_doc_articles(
        {
            "/laws/a.md": ["a-1", "a-2"],
            "/laws/b.md": ["b-1"],
        },
        delete_all_doc_articles=delete_all,
        batch_insert_doc_articles=insert,
        doc_id_factory=lambda: next(ids),
    )

    assert inserted == 2
    assert calls == [
        ("delete_all",),
        ("insert", "/laws/a.md", "doc-1", ["a-1", "a-2"]),
        ("insert", "/laws/b.md", "doc-2", ["b-1"]),
    ]


def test_rebuild_sync_skips_empty_paths_after_clearing_old_rows():
    from RAG.doc_article_sync import sync_rebuild_doc_articles

    calls = []

    inserted = sync_rebuild_doc_articles(
        {
            "": ["ignored"],
            "/laws/a.md": ["a-1"],
        },
        delete_all_doc_articles=lambda: calls.append(("delete_all",)),
        batch_insert_doc_articles=lambda *args: calls.append(("insert",) + args),
        doc_id_factory=lambda: "doc-1",
    )

    assert inserted == 1
    assert calls == [
        ("delete_all",),
        ("insert", "/laws/a.md", "doc-1", ["a-1"]),
    ]
