"""
文档-条文关系表模型
字段：
  - doc_id:       文档ID
  - doc_abs_path: 文档绝对路径（建立索引）
  - article_id:   条文ID（主键，索引）
  - created_at:   条文创建时间
"""

from sqlmodel import SQLModel, Field
from sqlalchemy import Column, String, DateTime, Index
from datetime import datetime


class DocArticle(SQLModel, table=True):
    __tablename__ = "doc_article"

    article_id: str = Field(
        sa_column=Column(String(64), primary_key=True, index=True, comment="条文ID（Milvus中的向量ID）")
    )
    doc_id: str = Field(
        sa_column=Column(String(64), nullable=False, comment="文档ID")
    )
    doc_abs_path: str = Field(
        sa_column=Column(String(512), nullable=False, index=True, comment="文档绝对路径")
    )
    created_at: datetime = Field(
        sa_column=Column(DateTime, default=datetime.now, comment="条文创建时间")
    )
