import json


def test_retrieve_tool_payload_includes_evidences(monkeypatch):
    import RAG.retrieve as retrieve

    def fake_ensure_processor_started():
        return None

    async def fake_wait_for(future, timeout):
        return (
            "检索文本",
            ["民法典::第五百零九条"],
            [
                {
                    "citation_id": 1,
                    "filename": "民法典",
                    "article": "第五百零九条",
                    "source_label": "民法典 / 第五百零九条",
                    "excerpt": "当事人应当按照约定全面履行自己的义务。",
                    "content_hash": "abc123",
                    "doc_key": "safe-key",
                    "chunk_id": "chunk-1",
                    "score": None,
                }
            ],
        )

    class FakeQueue:
        async def put(self, item):
            self.item = item

    class FakeLoop:
        def create_future(self):
            return object()

    monkeypatch.setattr(retrieve, "_ensure_processor_started", fake_ensure_processor_started)
    monkeypatch.setattr(retrieve, "_request_queue", FakeQueue())
    monkeypatch.setattr(retrieve.asyncio, "get_running_loop", lambda: FakeLoop())
    monkeypatch.setattr(retrieve.asyncio, "wait_for", fake_wait_for)

    payload = retrieve.asyncio.run(
        retrieve.retrieve_vector_store.coroutine("合同履行义务", exclude_ids=[])
    )
    data = json.loads(payload)

    assert data["text"] == "检索文本"
    assert data["retrieved_ids"] == ["民法典::第五百零九条"]
    assert data["evidences"][0]["citation_id"] == 1
