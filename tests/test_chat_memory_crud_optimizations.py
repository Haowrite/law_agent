from sqlalchemy.sql.dml import Update


def test_user_session_list_query_uses_single_joined_aggregate():
    from db_crud.chat_memory_crud import _build_user_session_list_stmt

    stmt = _build_user_session_list_stmt("user-1")
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True})).lower()

    assert "left outer join" in sql
    assert "count(" in sql
    assert "group by" in sql
    assert "order by" in sql
    assert "user-1" in sql


def test_mark_messages_summarized_uses_bulk_update_statement():
    from db_crud.chat_memory_crud import _build_mark_messages_summarized_stmt

    stmt = _build_mark_messages_summarized_stmt(["m1", "m2"])
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True})).lower()

    assert isinstance(stmt, Update)
    assert "update" in sql
    assert "is_summarized" in sql
    assert "where" in sql
    assert "m1" in sql
    assert "m2" in sql
