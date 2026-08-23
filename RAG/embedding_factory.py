from typing import Callable, Optional

from RAG.api_embeddings import OpenAICompatibleEmbeddings


def _default_local_factory(**kwargs):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(**kwargs)


def create_embeddings_model(
    embedding_provider: str,
    embedding_model: str,
    embedding_api_key: str,
    embedding_api_base_url: str,
    embedding_api_dimensions: Optional[int],
    embedding_api_batch_size: int,
    local_embedding_batch_size: int,
    api_factory: Callable = OpenAICompatibleEmbeddings,
    local_factory: Callable = _default_local_factory,
):
    provider = (embedding_provider or "api").lower()

    if provider == "api":
        return api_factory(
            api_key=embedding_api_key,
            base_url=embedding_api_base_url,
            model=embedding_model,
            dimensions=embedding_api_dimensions,
            batch_size=embedding_api_batch_size,
        )

    if provider == "local":
        return local_factory(
            model_name=embedding_model,
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": local_embedding_batch_size,
            },
        )

    raise ValueError("EMBEDDING_PROVIDER must be either 'api' or 'local'.")
