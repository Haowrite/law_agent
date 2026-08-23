class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Session:
    def __init__(self):
        self.calls = []

    def post(self, url, headers, json, timeout):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return _Response(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]
            }
        )


def test_compute_score_calls_qwen3_rerank_api_and_restores_input_order():
    from RAG.api_reranker import DashScopeReranker

    session = _Session()
    reranker = DashScopeReranker(
        api_key="test-key",
        base_url="https://example.cn-beijing.maas.aliyuncs.com/compatible-api/v1",
        model="qwen3-rerank",
        batch_size=10,
        timeout=30,
        client=session,
    )

    scores = reranker.compute_score(
        [
            ["什么是文本排序模型", "量子计算是计算科学的前沿领域"],
            ["什么是文本排序模型", "文本排序模型用于对候选文档排序"],
        ],
        normalize=True,
    )

    assert scores == [0.2, 0.9]
    assert session.calls == [
        {
            "url": "https://example.cn-beijing.maas.aliyuncs.com/compatible-api/v1/reranks",
            "headers": {
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
            "json": {
                "model": "qwen3-rerank",
                "query": "什么是文本排序模型",
                "documents": [
                    "量子计算是计算科学的前沿领域",
                    "文本排序模型用于对候选文档排序",
                ],
                "top_n": 2,
            },
            "timeout": 30,
        }
    ]


def test_compute_score_groups_different_queries():
    from RAG.api_reranker import DashScopeReranker

    class MultiQuerySession(_Session):
        def post(self, url, headers, json, timeout):
            self.calls.append({"query": json["query"], "documents": json["documents"]})
            return _Response(
                {
                    "results": [
                        {
                            "index": 0,
                            "relevance_score": 0.8 if json["query"] == "问题A" else 0.3,
                        }
                    ]
                }
            )

    session = MultiQuerySession()
    reranker = DashScopeReranker(
        api_key="test-key",
        base_url="https://example.cn-beijing.maas.aliyuncs.com/compatible-api/v1",
        model="qwen3-rerank",
        client=session,
    )

    scores = reranker.compute_score([["问题A", "文档A"], ["问题B", "文档B"]])

    assert scores == [0.8, 0.3]
    assert session.calls == [
        {"query": "问题A", "documents": ["文档A"]},
        {"query": "问题B", "documents": ["文档B"]},
    ]


def test_requires_reranker_api_key_when_no_client_is_injected():
    from RAG.api_reranker import DashScopeReranker

    try:
        DashScopeReranker(
            api_key="",
            base_url="https://example.cn-beijing.maas.aliyuncs.com/compatible-api/v1",
            model="qwen3-rerank",
        )
    except ValueError as exc:
        assert "RERANKER_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected ValueError when RERANKER_API_KEY is empty.")
