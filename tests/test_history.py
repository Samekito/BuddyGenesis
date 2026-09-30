import sqlite3

import pytest

from chainlit.element import ElementDict
from chainlit.step import StepDict

from buddy.history import SCHEMA_DIR, create_data_layer, ensure_schema, normalize_database_url

EXPECTED_TABLES = {"users", "threads", "steps", "elements", "feedbacks"}


def test_neon_style_url_gets_asyncpg_driver_and_ssl():
    url, needs_ssl = normalize_database_url("postgresql://u:p@ep-x.neon.tech/neondb?sslmode=require&channel_binding=require")

    assert url == "postgresql+asyncpg://u:p@ep-x.neon.tech/neondb"
    assert needs_ssl


def test_postgres_url_without_sslmode_does_not_force_ssl():
    url, needs_ssl = normalize_database_url("postgres://u:p@localhost:5432/buddy")

    assert url == "postgresql+asyncpg://u:p@localhost:5432/buddy"
    assert not needs_ssl


def test_sqlite_url_is_left_alone():
    assert normalize_database_url("sqlite+aiosqlite:///data/x.db") == ("sqlite+aiosqlite:///data/x.db", False)


def test_plain_sqlite_url_gets_async_driver():
    assert normalize_database_url("sqlite:///data/x.db") == ("sqlite+aiosqlite:///data/x.db", False)


def test_unsupported_database_scheme_is_rejected_clearly():
    with pytest.raises(ValueError, match="mysql"):
        normalize_database_url("mysql://u:p@host/db")


def test_data_layer_keeps_message_text_out_of_error_logs(tmp_path):
    data_layer = create_data_layer(f"sqlite+aiosqlite:///{(tmp_path / 'h.db').as_posix()}")

    assert data_layer.engine.sync_engine.hide_parameters


async def test_ensure_schema_creates_missing_folder_and_all_tables(tmp_path):
    database_file = tmp_path / "nested" / "history.db"

    await ensure_schema(f"sqlite+aiosqlite:///{database_file.as_posix()}")

    tables = {row[0] for row in sqlite3.connect(database_file).execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert EXPECTED_TABLES <= tables


async def test_ensure_schema_is_idempotent(tmp_path):
    url = f"sqlite+aiosqlite:///{(tmp_path / 'history.db').as_posix()}"

    await ensure_schema(url)
    await ensure_schema(url)


def _schema_columns(table: str) -> set[str]:
    connection = sqlite3.connect(":memory:")
    connection.executescript((SCHEMA_DIR / "sqlite.sql").read_text())
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def test_schema_has_a_column_for_every_step_field_chainlit_writes():
    # The data layer swallows "no such column" errors, so a Chainlit upgrade adding a field would
    # silently stop saving history. This fails loudly instead. `feedback` lives in its own table.
    assert set(StepDict.__annotations__) - {"feedback"} <= _schema_columns("steps")


def test_schema_has_a_column_for_every_element_field_chainlit_writes():
    assert set(ElementDict.__annotations__) <= _schema_columns("elements")
