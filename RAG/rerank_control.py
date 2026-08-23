from typing import Callable


def rerank_or_passthrough(query, candidate_docs, max_results: int, enabled: bool, get_reranker: Callable):
    if not candidate_docs:
        return []

    if not enabled:
        return candidate_docs[:max_results]

    reranker = get_reranker()
    doc_list = [doc for _, doc in candidate_docs]
    query_passage_pairs = [[query, doc.page_content] for doc in doc_list]
    scores = reranker.compute_score(query_passage_pairs, normalize=True)

    scored_docs = list(zip(scores, doc_list))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return scored_docs[:max_results]
