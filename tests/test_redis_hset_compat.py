def test_hset_compat_writes_fields_without_mapping_keyword():
    from db_crud.session_manage import _hset_compat

    calls = []

    class FakeRedisOrPipeline:
        def hset(self, *args, **kwargs):
            assert "mapping" not in kwargs
            calls.append(args)

    _hset_compat(
        FakeRedisOrPipeline(),
        "session:test:meta",
        {"total": "3", "unsum": "2", "sum": "1"},
    )

    assert calls == [
        ("session:test:meta", "total", "3"),
        ("session:test:meta", "unsum", "2"),
        ("session:test:meta", "sum", "1"),
    ]
