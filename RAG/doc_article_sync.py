import uuid
from typing import Callable, Dict, List


def sync_rebuild_doc_articles(
    doc_path_to_ids: Dict[str, List[str]],
    delete_all_doc_articles: Callable[[], int],
    batch_insert_doc_articles: Callable[[str, str, List[str]], None],
    doc_id_factory: Callable[[], object] = uuid.uuid4,
) -> int:
    delete_all_doc_articles()

    inserted_docs = 0
    for abs_path, aid_list in doc_path_to_ids.items():
        if not abs_path:
            continue
        batch_insert_doc_articles(abs_path, str(doc_id_factory()), aid_list)
        inserted_docs += 1

    return inserted_docs
