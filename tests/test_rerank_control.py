class _Doc:
    def __init__(self, text):
        self.page_content = text


class _Reranker:
    def __init__(self):
        self.calls = []

    def compute_score(self, pairs, normalize=True):
        self.calls.append({"pairs": pairs, "normalize": normalize})
        return [0.1, 0.9]


def test_disabled_reranker_returns_existing_candidate_order_without_loading_model():
    from RAG.rerank_control import rerank_or_passthrough

    calls = []
    candidate_docs = [(0.7, _Doc("第一篇")), (0.6, _Doc("第二篇"))]

    result = rerank_or_passthrough(
        query="查询",
        candidate_docs=candidate_docs,
        max_results=1,
        enabled=False,
        get_reranker=lambda: calls.append("loaded"),
    )

    assert result == [candidate_docs[0]]
    assert calls == []


def test_enabled_reranker_scores_and_sorts_documents():
    from RAG.rerank_control import rerank_or_passthrough

    reranker = _Reranker()
    candidate_docs = [(0.7, _Doc("第一篇")), (0.6, _Doc("第二篇"))]

    result = rerank_or_passthrough(
        query="查询",
        candidate_docs=candidate_docs,
        max_results=2,
        enabled=True,
        get_reranker=lambda: reranker,
    )

    assert [score for score, _ in result] == [0.9, 0.1]
    assert [doc.page_content for _, doc in result] == ["第二篇", "第一篇"]
    assert reranker.calls == [
        {
            "pairs": [["查询", "第一篇"], ["查询", "第二篇"]],
            "normalize": True,
        }
    ]
