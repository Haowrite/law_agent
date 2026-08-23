def test_embedding_factory_uses_api_provider():
    from RAG.embedding_factory import create_embeddings_model

    calls = []

    result = create_embeddings_model(
        embedding_provider="api",
        embedding_model="text-embedding-v4",
        embedding_api_key="test-key",
        embedding_api_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        embedding_api_dimensions=1024,
        embedding_api_batch_size=10,
        local_embedding_batch_size=8,
        api_factory=lambda **kwargs: calls.append(kwargs) or "api-model",
        local_factory=lambda **kwargs: "local-model",
    )

    assert result == "api-model"
    assert calls == [
        {
            "api_key": "test-key",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "text-embedding-v4",
            "dimensions": 1024,
            "batch_size": 10,
        }
    ]


def test_embedding_factory_uses_local_provider():
    from RAG.embedding_factory import create_embeddings_model

    calls = []

    result = create_embeddings_model(
        embedding_provider="local",
        embedding_model="/models/bge-large-zh-v1.5",
        embedding_api_key="unused",
        embedding_api_base_url="unused",
        embedding_api_dimensions=1024,
        embedding_api_batch_size=10,
        local_embedding_batch_size=4,
        api_factory=lambda **kwargs: "api-model",
        local_factory=lambda **kwargs: calls.append(kwargs) or "local-model",
    )

    assert result == "local-model"
    assert calls == [
        {
            "model_name": "/models/bge-large-zh-v1.5",
            "encode_kwargs": {
                "normalize_embeddings": True,
                "batch_size": 4,
            },
        }
    ]


def test_embedding_factory_rejects_unknown_provider():
    from RAG.embedding_factory import create_embeddings_model

    try:
        create_embeddings_model(
            embedding_provider="remote",
            embedding_model="text-embedding-v4",
            embedding_api_key="test-key",
            embedding_api_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            embedding_api_dimensions=1024,
            embedding_api_batch_size=10,
            local_embedding_batch_size=8,
        )
    except ValueError as exc:
        assert "EMBEDDING_PROVIDER" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown EMBEDDING_PROVIDER.")
