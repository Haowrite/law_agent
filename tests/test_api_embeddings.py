class _EmbeddingItem:
    def __init__(self, embedding, index):
        self.embedding = embedding
        self.index = index


class _EmbeddingResponse:
    def __init__(self, data):
        self.data = data


class _EmbeddingsEndpoint:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        data = [
            _EmbeddingItem([float(index), float(index + 1)], index)
            for index, _ in enumerate(kwargs["input"])
        ]
        return _EmbeddingResponse(data)


class _Client:
    def __init__(self):
        self.embeddings = _EmbeddingsEndpoint()


class _BadResponse:
    text = '{"code":"InvalidParameter","message":"input rows exceed limit"}'

    def raise_for_status(self):
        import requests

        raise requests.HTTPError("400 Client Error: Bad Request")


class _BadSession:
    def post(self, url, headers, json, timeout):
        return _BadResponse()


def test_embed_documents_calls_openai_compatible_embedding_api_with_dimensions():
    from RAG.api_embeddings import OpenAICompatibleEmbeddings

    client = _Client()
    embeddings = OpenAICompatibleEmbeddings(
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="text-embedding-v4",
        dimensions=2,
        batch_size=2,
        client=client,
    )

    result = embeddings.embed_documents(["第一条", "第二条", "第三条"])

    assert result == [[0.0, 1.0], [1.0, 2.0], [0.0, 1.0]]
    assert client.embeddings.calls == [
        {
            "model": "text-embedding-v4",
            "input": ["第一条", "第二条"],
            "dimensions": 2,
        },
        {
            "model": "text-embedding-v4",
            "input": ["第三条"],
            "dimensions": 2,
        },
    ]


def test_embed_query_returns_first_embedding():
    from RAG.api_embeddings import OpenAICompatibleEmbeddings

    client = _Client()
    embeddings = OpenAICompatibleEmbeddings(
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="text-embedding-v4",
        dimensions=None,
        batch_size=10,
        client=client,
    )

    assert embeddings.embed_query("查询文本") == [0.0, 1.0]
    assert client.embeddings.calls == [
        {
            "model": "text-embedding-v4",
            "input": ["查询文本"],
        }
    ]


def test_requires_api_key_when_no_client_is_injected():
    from RAG.api_embeddings import OpenAICompatibleEmbeddings

    try:
        OpenAICompatibleEmbeddings(
            api_key="",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="text-embedding-v4",
            dimensions=1024,
            batch_size=10,
        )
    except ValueError as exc:
        assert "EMBEDDING_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected ValueError when EMBEDDING_API_KEY is empty.")


def test_http_error_includes_embedding_response_body():
    from RAG.api_embeddings import OpenAICompatibleEmbeddings

    embeddings = OpenAICompatibleEmbeddings(
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="text-embedding-v4",
        dimensions=1024,
        batch_size=10,
        client=_BadSession(),
    )

    try:
        embeddings.embed_documents(["第一条"])
    except RuntimeError as exc:
        message = str(exc)
        assert "Embedding API request failed" in message
        assert "input rows exceed limit" in message
    else:
        raise AssertionError("Expected RuntimeError with response body.")
