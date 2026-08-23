import os
import hmac
from typing import List, Optional

from sqlalchemy import func

from config import DOCUMENT_KEY_SECRET
from db_crud.base import get_sync_session
from db_crud.doc_article_model import DocArticle
from RAG.document_library import document_key_for_path


def build_public_document_record(
    doc_abs_path: str,
    article_count: int,
    created_at,
    secret: str = DOCUMENT_KEY_SECRET,
) -> dict:
    return {
        "doc_key": document_key_for_path(doc_abs_path, secret=secret),
        "filename": os.path.basename(doc_abs_path),
        "article_count": int(article_count or 0),
        "created_at": created_at.isoformat() if created_at else "",
        "file_exists": os.path.isfile(doc_abs_path),
    }


def search_document_records_by_name(keyword: str, limit: int = 20) -> List[dict]:
    session = get_sync_session()
    try:
        query = session.query(
            DocArticle.doc_abs_path,
            func.count(DocArticle.article_id).label("article_count"),
            func.min(DocArticle.created_at).label("created_at"),
        ).group_by(DocArticle.doc_abs_path)

        if keyword:
            query = query.filter(DocArticle.doc_abs_path.like(f"%{keyword}%"))

        rows = query.order_by(func.min(DocArticle.created_at).desc()).limit(limit).all()
        return [
            {
                "doc_abs_path": row.doc_abs_path,
                "article_count": int(row.article_count or 0),
                "created_at": row.created_at,
            }
            for row in rows
        ]
    finally:
        session.close()


def search_documents_by_name(keyword: str, limit: int = 20) -> List[dict]:
    records = search_document_records_by_name(keyword, limit)
    return [
        build_public_document_record(
            record["doc_abs_path"],
            record["article_count"],
            record["created_at"],
        )
        for record in records
    ]


def get_document_by_key(doc_key: str) -> Optional[dict]:
    session = get_sync_session()
    try:
        rows = session.query(
            DocArticle.doc_abs_path,
            func.count(DocArticle.article_id).label("article_count"),
            func.min(DocArticle.created_at).label("created_at"),
        ).group_by(DocArticle.doc_abs_path).all()

        for row in rows:
            candidate_key = document_key_for_path(row.doc_abs_path, secret=DOCUMENT_KEY_SECRET)
            if hmac.compare_digest(candidate_key, doc_key):
                return {
                    "doc_abs_path": row.doc_abs_path,
                    **build_public_document_record(row.doc_abs_path, row.article_count, row.created_at),
                }
        return None
    finally:
        session.close()
