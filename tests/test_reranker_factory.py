def test_reranker_factory_uses_api_provider():
    from RAG.reranker_factory import create_reranker_model

    calls = []

    result = create_reranker_model(
        reranker_provider="api",
        reranker_model="qwen3-rerank",
        reranker_api_key="test-key",
        reranker_api_base_url="https://dashscope.aliyuncs.com/compatible-api/v1",
        reranker_api_batch_size=500,
        reranker_api_timeout=60,
        reranker_instruct="instruction",
        local_reranker_use_fp16=True,
        api_factory=lambda **kwargs: calls.append(kwargs) or "api-reranker",
        local_factory=lambda **kwargs: "local-reranker",
    )

    assert result == "api-reranker"
    assert calls == [
        {
            "api_key": "test-key",
            "base_url": "https://dashscope.aliyuncs.com/compatible-api/v1",
            "model": "qwen3-rerank",
            "batch_size": 500,
            "timeout": 60,
            "instruct": "instruction",
        }
    ]


def test_reranker_factory_uses_local_provider():
    from RAG.reranker_factory import create_reranker_model

    calls = []

    result = create_reranker_model(
        reranker_provider="local",
        reranker_model="/models/bge-reranker-v2-m3",
        reranker_api_key="unused",
        reranker_api_base_url="unused",
        reranker_api_batch_size=500,
        reranker_api_timeout=60,
        reranker_instruct="",
        local_reranker_use_fp16=False,
        api_factory=lambda **kwargs: "api-reranker",
        local_factory=lambda **kwargs: calls.append(kwargs) or "local-reranker",
    )

    assert result == "local-reranker"
    assert calls == [
        {
            "model_name_or_path": "/models/bge-reranker-v2-m3",
            "use_fp16": False,
        }
    ]


def test_reranker_factory_rejects_unknown_provider():
    from RAG.reranker_factory import create_reranker_model

    try:
        create_reranker_model(
            reranker_provider="remote",
            reranker_model="qwen3-rerank",
            reranker_api_key="test-key",
            reranker_api_base_url="https://dashscope.aliyuncs.com/compatible-api/v1",
            reranker_api_batch_size=500,
            reranker_api_timeout=60,
            reranker_instruct="",
            local_reranker_use_fp16=True,
        )
    except ValueError as exc:
        assert "RERANKER_PROVIDER" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown RERANKER_PROVIDER.")
