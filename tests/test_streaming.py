import asyncio


class FakeChunk:
    def __init__(self, content):
        self.content = content


class FakeStreamingLLM:
    async def astream(self, messages):
        yield FakeChunk("第")
        yield FakeChunk("一段")
        yield FakeChunk("")
        yield {"content": "。"}


def test_stream_llm_text_yields_real_llm_chunks():
    from RAG.streaming import stream_llm_text

    async def collect():
        return [chunk async for chunk in stream_llm_text(FakeStreamingLLM(), ["prompt"])]

    assert asyncio.run(collect()) == ["第", "一段", "。"]
