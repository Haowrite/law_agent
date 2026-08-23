from typing import Any, AsyncIterator, Iterable


def _chunk_content(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        return chunk.get("content") or chunk.get("text") or ""
    return getattr(chunk, "content", "") or ""


async def stream_llm_text(llm: Any, messages: Iterable[Any]) -> AsyncIterator[str]:
    async for chunk in llm.astream(messages):
        content = _chunk_content(chunk)
        if content:
            yield content
