from typing import Any, List, Optional

import requests


class OpenAICompatibleEmbeddings:
    """Embedding adapter for OpenAI-compatible HTTP APIs."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        dimensions: Optional[int] = None,
        batch_size: int = 32,
        client: Optional[Any] = None,
    ):
        if not client and not api_key:
            raise ValueError("EMBEDDING_API_KEY is required for API embeddings.")
        if not model:
            raise ValueError("EMBEDDING_MODEL is required for API embeddings.")
        if batch_size <= 0:
            raise ValueError("EMBEDDING_API_BATCH_SIZE must be greater than 0.")

        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = client or requests.Session()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            if not batch:
                continue

            request_body = {
                "model": self.model,
                "input": batch,
            }
            if self.dimensions:
                request_body["dimensions"] = self.dimensions

            data = self._create_embeddings(request_body)
            ordered_data = sorted(data, key=lambda item: item["index"])
            vectors.extend([list(item["embedding"]) for item in ordered_data])

        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def _create_embeddings(self, request_body: dict) -> List[dict]:
        if hasattr(self.client, "embeddings"):
            response = self.client.embeddings.create(**request_body)
            return [
                {"index": item.index, "embedding": item.embedding}
                for item in response.data
            ]

        response = self.client.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=request_body,
            timeout=60,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = response.text[:2000] if getattr(response, "text", None) else ""
            raise RuntimeError(f"Embedding API request failed: {exc}; response={body}") from exc
        payload = response.json()
        return payload["data"]
