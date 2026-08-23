from collections import OrderedDict
from typing import List, Optional, Sequence

import requests


class DashScopeReranker:
    """Reranker adapter with a FlagReranker-compatible compute_score API."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        batch_size: int = 500,
        timeout: int = 60,
        instruct: Optional[str] = None,
        client=None,
    ):
        if not client and not api_key:
            raise ValueError("RERANKER_API_KEY is required for API reranker.")
        if not model:
            raise ValueError("RERANKER_MODEL is required for API reranker.")
        if batch_size <= 0:
            raise ValueError("RERANKER_API_BATCH_SIZE must be greater than 0.")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.instruct = instruct
        self.client = client or requests.Session()

    def compute_score(
        self,
        sentence_pairs: Sequence[Sequence[str]],
        normalize: bool = True,
    ) -> List[float]:
        if not sentence_pairs:
            return []

        scores = [0.0] * len(sentence_pairs)
        grouped_pairs = self._group_by_query(sentence_pairs)

        for query, indexed_documents in grouped_pairs.items():
            for start in range(0, len(indexed_documents), self.batch_size):
                batch = indexed_documents[start:start + self.batch_size]
                original_indexes = [index for index, _ in batch]
                documents = [document for _, document in batch]
                response_scores = self._rerank(query, documents)

                for relative_index, score in response_scores.items():
                    scores[original_indexes[relative_index]] = score

        return scores

    def _group_by_query(self, sentence_pairs: Sequence[Sequence[str]]):
        grouped_pairs = OrderedDict()
        for index, pair in enumerate(sentence_pairs):
            if len(pair) != 2:
                raise ValueError("Each reranker input item must be [query, document].")
            query, document = pair
            grouped_pairs.setdefault(query, []).append((index, document))
        return grouped_pairs

    def _rerank(self, query: str, documents: List[str]) -> dict:
        request_body = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }
        if self.instruct:
            request_body["instruct"] = self.instruct

        response = self.client.post(
            f"{self.base_url}/reranks",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=request_body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        return {
            item["index"]: float(item["relevance_score"])
            for item in payload.get("results", [])
        }
