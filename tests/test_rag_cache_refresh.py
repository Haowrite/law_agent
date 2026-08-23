class _Collection:
    def __init__(self, batches):
        self.batches = batches

    def query_iterator(self, expr, output_fields, batch_size):
        return _Iterator(self.batches)


class _Iterator:
    def __init__(self, batches):
        self.batches = list(batches)
        self.closed = False

    def next(self):
        if self.batches:
            return self.batches.pop(0)
        return []

    def close(self):
        self.closed = True


def test_load_docs_from_milvus_collection_builds_documents():
    from RAG.cache_sync import load_docs_from_milvus_collection

    docs = load_docs_from_milvus_collection(
        _Collection(
            [
                [
                    {
                        "text": "第一条 内容",
                        "metadata": {"id": "1", "token_num": 3},
                        "filename": "民法典",
                        "article": "第一条",
                        "start_position": 0,
                    }
                ]
            ]
        )
    )

    assert len(docs) == 1
    assert docs[0].page_content == "第一条 内容"
    assert docs[0].metadata["filename"] == "民法典"
