"""Chat history persistence via Chainlit's SQLAlchemyDataLayer (replaces the old Literal AI service).

Consumed by app.py. DATABASE_URL selects SQLite (local dev) or Postgres (hosted, e.g. free Neon).
"""

import json
import sqlite3
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

SCHEMA_DIR = Path(__file__).parent / "schema"


def normalize_database_url(url: str) -> tuple[str, bool]:
    """Returns (async SQLAlchemy URL, whether SSL is required).

    Hosted Postgres providers hand out `postgresql://...?sslmode=require`; asyncpg needs the
    `+asyncpg` driver prefix and rejects `sslmode`, so SSL is passed separately instead.
    Note: Chainlit's ssl_require skips certificate verification, so `verify-full` is downgraded
    to encrypted-but-unverified — acceptable for Neon, which is reached over the public internet anyway.
    """
    parts = urlsplit(url)
    if parts.scheme in ("postgres", "postgresql", "postgresql+asyncpg"):
        needs_ssl = "sslmode=require" in parts.query or "sslmode=verify" in parts.query
        return urlunsplit(("postgresql+asyncpg", parts.netloc, parts.path, "", "")), needs_ssl
    if parts.scheme in ("sqlite", "sqlite+aiosqlite"):
        return "sqlite+aiosqlite" + url[len(parts.scheme):], False
    raise ValueError(f"Unsupported DATABASE_URL scheme '{parts.scheme}' — use sqlite or postgresql.")


def is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def create_data_layer(database_url: str) -> SQLAlchemyDataLayer:
    url, needs_ssl = normalize_database_url(database_url)
    if is_sqlite(url):
        _prepare_sqlite(url)
    data_layer = SQLAlchemyDataLayer(conninfo=url, ssl_require=needs_ssl)
    # Chainlit logs failed writes with the SQL error text, which by default includes the bound
    # parameters — i.e. full user messages and answers. Keep them out of the logs.
    data_layer.engine.sync_engine.hide_parameters = True
    return data_layer


async def ensure_schema(database_url: str) -> None:
    """Creates the history tables if missing. Every statement is IF NOT EXISTS, so reruns are no-ops."""
    url, needs_ssl = normalize_database_url(database_url)
    if is_sqlite(url):
        # Runs at app startup, before Chainlit first builds the data layer, so the folder may not exist yet.
        _prepare_sqlite(url)
    schema_file = SCHEMA_DIR / ("sqlite.sql" if is_sqlite(url) else "postgres.sql")
    statements = [s.strip() for s in schema_file.read_text().split(";") if s.strip()]
    connect_args = {"ssl": "require"} if needs_ssl else {}
    engine = create_async_engine(url, connect_args=connect_args, hide_parameters=True)
    try:
        async with engine.begin() as connection:
            for statement in statements:
                await connection.execute(text(statement))
    finally:
        await engine.dispose()


def _prepare_sqlite(url: str) -> None:
    database_path = url.split(":///", 1)[-1]
    if database_path and database_path != ":memory:":
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    # Chainlit binds thread/step tags as Python lists (Postgres arrays). SQLite cannot bind lists,
    # and the data layer swallows the error, silently dropping the thread — so store them as JSON.
    sqlite3.register_adapter(list, json.dumps)
    sqlite3.register_adapter(dict, json.dumps)
