from typing import Callable

from RAG.api_reranker import DashScopeReranker


def _default_local_factory(**kwargs):
    from FlagEmbedding import FlagReranker

    return FlagReranker(**kwargs)


def create_reranker_model(
    reranker_provider: str,
    reranker_model: str,
    reranker_api_key: str,
    reranker_api_base_url: str,
    reranker_api_batch_size: int,
    reranker_api_timeout: int,
    reranker_instruct: str,
    local_reranker_use_fp16: bool,
    api_factory: Callable = DashScopeReranker,
    local_factory: Callable = _default_local_factory,
):
    provider = (reranker_provider or "api").lower()

    if provider == "api":
        return api_factory(
            api_key=reranker_api_key,
            base_url=reranker_api_base_url,
            model=reranker_model,
            batch_size=reranker_api_batch_size,
            timeout=reranker_api_timeout,
            instruct=reranker_instruct or None,
        )

    if provider == "local":
        return local_factory(
            model_name_or_path=reranker_model,
            use_fp16=local_reranker_use_fp16,
        )

    raise ValueError("RERANKER_PROVIDER must be either 'api' or 'local'.")
