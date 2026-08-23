import gc
from typing import List

from langchain_core.documents import Document


def load_docs_from_milvus_collection(collection, batch_size: int = 1000) -> List[Document]:
    final_docs = []
    iterator = None

    try:
        iterator = collection.query_iterator(
            expr="id != ''",
            output_fields=["text", "metadata", "id", "filename", "article", "start_position"],
            batch_size=batch_size,
        )

        while True:
            batch = iterator.next()
            if len(batch) == 0:
                break

            for entity in batch:
                text = entity.get("text", "")
                meta = entity.get("metadata", {}) or {}
                doc = Document(page_content=text, metadata=meta)

                if "filename" not in doc.metadata:
                    doc.metadata["filename"] = entity.get("filename", "")
                if "article" not in doc.metadata:
                    doc.metadata["article"] = entity.get("article", "")
                if "start_position" not in doc.metadata:
                    doc.metadata["start_position"] = entity.get("start_position", 0)

                final_docs.append(doc)

            del batch
            gc.collect()
    finally:
        if iterator is not None:
            iterator.close()

    return final_docs


def refresh_rag_cache_from_milvus(collection, cache_path: str, save_docs_to_cache):
    docs = load_docs_from_milvus_collection(collection)
    save_docs_to_cache(docs, cache_path)
    return len(docs)
