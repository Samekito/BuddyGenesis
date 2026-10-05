import sqlite3
import ssl

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from chainlit.element import ElementDict
from chainlit.step import StepDict

from buddy import history
from buddy.history import (
    SCHEMA_DIR,
    connect_args,
    create_data_layer,
    database_reachable,
    migrate,
    normalize_database_url,
)

EXPECTED_TABLES = {"users", "threads", "steps", "elements", "feedbacks", "accounts", "email_tokens", "schema_migrations"}
ALL_MIGRATIONS = [1, 2]


def _sqlite_engine(database_file):
    return create_async_engine(f"sqlite+aiosqlite:///{database_file.as_posix()}")


def test_neon_style_url_gets_asyncpg_driver_and_ssl():
    url, needs_ssl = normalize_database_url("postgresql://u:p@ep-x.neon.tech/neondb?sslmode=require&channel_binding=require")

    assert url == "postgresql+asyncpg://u:p@ep-x.neon.tech/neondb"
    assert needs_ssl


def test_postgres_url_without_sslmode_still_uses_tls():
    url, needs_tls = normalize_database_url("postgres://u:p@db.example.com:5432/buddy")

    assert url == "postgresql+asyncpg://u:p@db.example.com:5432/buddy"
    assert needs_tls


def test_postgres_tls_can_be_switched_off_explicitly_for_local_servers():
    assert not normalize_database_url("postgres://u:p@localhost/buddy?sslmode=disable")[1]


def test_tls_connections_verify_the_server_certificate_and_hostname():
    context = connect_args(needs_tls=True)["ssl"]

    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname


def test_non_tls_connections_get_no_ssl_argument():
    assert connect_args(needs_tls=False) == {}


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


async def test_data_layer_creates_the_missing_database_folder(tmp_path):
    database_file = tmp_path / "nested" / "history.db"

    data_layer = create_data_layer(f"sqlite+aiosqlite:///{database_file.as_posix()}")
    await migrate(data_layer.engine)
    await data_layer.engine.dispose()

    assert database_file.exists()


async def test_migrate_creates_all_tables_and_records_each_migration(tmp_path):
    engine = _sqlite_engine(tmp_path / "history.db")

    applied = await migrate(engine)
    await engine.dispose()

    tables = {row[0] for row in sqlite3.connect(tmp_path / "history.db").execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert EXPECTED_TABLES <= tables
    assert applied == ALL_MIGRATIONS


async def test_migrate_applies_nothing_the_second_time(tmp_path):
    engine = _sqlite_engine(tmp_path / "history.db")

    await migrate(engine)
    applied_again = await migrate(engine)
    await engine.dispose()

    assert applied_again == []


async def test_migrate_adopts_a_database_created_before_migrations_existed(tmp_path):
    # Databases in use today were built by the old unversioned schema file; 001 must apply cleanly on top.
    sqlite3.connect(tmp_path / "history.db").executescript((SCHEMA_DIR / "sqlite" / "001_initial.sql").read_text())
    engine = _sqlite_engine(tmp_path / "history.db")

    applied = await migrate(engine)
    await engine.dispose()

    assert applied == ALL_MIGRATIONS


async def test_new_migration_files_are_applied_in_number_order(tmp_path, monkeypatch):
    schema_dir = tmp_path / "schema"
    (schema_dir / "sqlite").mkdir(parents=True)
    (schema_dir / "sqlite" / "002_second.sql").write_text("CREATE TABLE second (id INTEGER REFERENCES first(id))")
    (schema_dir / "sqlite" / "001_first.sql").write_text("CREATE TABLE first (id INTEGER PRIMARY KEY)")
    monkeypatch.setattr(history, "SCHEMA_DIR", schema_dir)
    engine = _sqlite_engine(tmp_path / "history.db")

    applied = await migrate(engine)
    await engine.dispose()

    assert applied == [1, 2]


async def test_badly_named_migration_file_is_refused(tmp_path, monkeypatch):
    (tmp_path / "sqlite").mkdir()
    (tmp_path / "sqlite" / "add_column.sql").write_text("SELECT 1")
    monkeypatch.setattr(history, "SCHEMA_DIR", tmp_path)
    engine = _sqlite_engine(tmp_path / "history.db")

    with pytest.raises(ValueError, match="NNN_description"):
        await migrate(engine)
    await engine.dispose()


async def test_reachable_database_passes_the_ping(tmp_path):
    engine = _sqlite_engine(tmp_path / "history.db")

    assert await database_reachable(engine)
    await engine.dispose()


async def test_unreachable_database_fails_the_ping(tmp_path):
    engine = _sqlite_engine(tmp_path / "missing-folder" / "history.db")

    assert not await database_reachable(engine)
    await engine.dispose()


def _schema_columns(table: str) -> set[str]:
    connection = sqlite3.connect(":memory:")
    for migration_file in sorted((SCHEMA_DIR / "sqlite").glob("*.sql")):
        connection.executescript(migration_file.read_text())
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def test_schema_has_a_column_for_every_step_field_chainlit_writes():
    # The data layer swallows "no such column" errors, so a Chainlit upgrade adding a field would
    # silently stop saving history. This fails loudly instead. `feedback` lives in its own table.
    assert set(StepDict.__annotations__) - {"feedback"} <= _schema_columns("steps")


def test_schema_has_a_column_for_every_element_field_chainlit_writes():
    assert set(ElementDict.__annotations__) <= _schema_columns("elements")


async def test_data_layer_recovers_when_the_database_closed_an_idle_connection(tmp_path):
    # Neon closes idle connections when it suspends; the pool must not hand a dead one back out.
    data_layer = create_data_layer(f"sqlite+aiosqlite:///{(tmp_path / 'h.db').as_posix()}")
    async with data_layer.engine.connect() as connection:
        raw = await connection.get_raw_connection()
        await raw.driver_connection.close()

    assert await database_reachable(data_layer.engine)
    await data_layer.engine.dispose()
