# session_model.py
from sqlmodel import SQLModel, Field
import sqlalchemy as sa
from db_crud.base_func import get_time, get_id

class User(SQLModel, table=True):
    user_id: str = Field(default_factory=get_id, primary_key=True)
    username: str = Field(sa_column=sa.Column(sa.String(50), unique=True, index=True, nullable=False))
    password_hash: str = Field(sa_column=sa.Column(sa.String(128), nullable=False))  # bcrypt hash
    created_at: str = Field(default_factory=get_time)

class ChatSession(SQLModel, table=True):
    session_id: str = Field(default_factory=get_id, primary_key=True)
    user_id: str = Field(index=True)  # 关联 User.user_id
    timestamp: str = Field(description="会话创建时间", default_factory=get_time)

class ChatMessage(SQLModel, table=True):
    id: str = Field(default_factory=get_id, primary_key=True)
    session_id: str = Field(index=True)
    content: str = Field(sa_column=sa.Column(sa.Text, nullable=False))
    timestamp: str = Field(default_factory=get_time)
    message_type: str = Field(max_length=10)
    use_token: int = Field(default=0)