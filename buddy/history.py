"""Chat history persistence via Chainlit's SQLAlchemyDataLayer, and the database schema migrations.

Consumed by app.py. DATABASE_URL selects SQLite (local dev) or Postgres (hosted, e.g. free Neon).
The data layer's engine is the app's only database engine; buddy/accounts.py shares it.
"""

import asyncio
import json
import logging
import re
import sqlite3
import ssl
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

SCHEMA_DIR = Path(__file__).parent / "schema"
# buddy/schema/<dialect>/NNN_description.sql, applied in NNN order and recorded once applied.
MIGRATION_FILE = re.compile(r"^(\d+)_\w+\.sql$")
MIGRATIONS_TABLE = 'CREATE TABLE IF NOT EXISTS schema_migrations ("version" INTEGER PRIMARY KEY, "appliedAt" TEXT NOT NULL)'
# Long enough for a free Neon database to wake from suspend.
PING_TIMEOUT_SECONDS = 10
# What a database failure can look like: asyncpg raises plain OSErrors (refused connection,
# DNS, timeout, TLS certificate) while connecting, which SQLAlchemy does not wrap.
DATABASE_ERRORS = (SQLAlchemyError, OSError)

logger = logging.getLogger(__name__)


def normalize_database_url(url: str) -> tuple[str, bool]:
    """Returns (async SQLAlchemy URL, whether to use TLS).

    Hosted Postgres providers hand out `postgresql://...?sslmode=require`; asyncpg needs the
    `+asyncpg` driver prefix and rejects `sslmode`, so TLS is configured separately instead.
    Postgres always uses TLS unless the URL says `sslmode=disable` (e.g. a local dev server).
    """
    parts = urlsplit(url)
    if parts.scheme in ("postgres", "postgresql", "postgresql+asyncpg"):
        needs_tls = "sslmode=disable" not in parts.query
        return urlunsplit(("postgresql+asyncpg", parts.netloc, parts.path, "", "")), needs_tls
    if parts.scheme in ("sqlite", "sqlite+aiosqlite"):
        return "sqlite+aiosqlite" + url[len(parts.scheme):], False
    raise ValueError(f"Unsupported DATABASE_URL scheme '{parts.scheme}' — use sqlite or postgresql.")


def is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def connect_args(needs_tls: bool) -> dict:
    # A default context verifies the server's certificate and hostname. Chainlit's own
    # ssl_require switch skips both, which would leave the connection open to interception.
    return {"ssl": ssl.create_default_context()} if needs_tls else {}


def create_data_layer(database_url: str) -> SQLAlchemyDataLayer:
    url, needs_tls = normalize_database_url(database_url)
    if is_sqlite(url):
        _prepare_sqlite(url)
    data_layer = SQLAlchemyDataLayer(conninfo=url, connect_args=connect_args(needs_tls))
    # Chainlit builds its engine without options, so swap in one with the two this app needs
    # (Chainlit itself only uses `engine` and `async_session`; no connection is open yet):
    # - pool_pre_ping: Neon closes idle connections when it suspends, and handing such a dead
    #   connection back out failed sign-ins, resets and history writes with "connection is closed".
    # - hide_parameters: Chainlit logs failed writes with the SQL error text, which by default
    #   includes the bound parameters, i.e. full messages, answers and password hashes.
    data_layer.engine = create_async_engine(
        url, connect_args=connect_args(needs_tls), pool_pre_ping=True, hide_parameters=True
    )
    data_layer.async_session = sessionmaker(bind=data_layer.engine, expire_on_commit=False, class_=AsyncSession)
    return data_layer


async def migrate(engine: AsyncEngine) -> list[int]:
    """Applies the migrations not yet recorded in schema_migrations; returns the versions it applied.

    Each migration commits together with its record. Write migrations so rerunning one is
    harmless (IF NOT EXISTS etc.): SQLite commits DDL as it goes, so a failure can leave one half-applied.
    """
    async with engine.begin() as connection:
        await connection.execute(text(MIGRATIONS_TABLE))
        applied = set((await connection.execute(text('SELECT "version" FROM schema_migrations'))).scalars())
    newly_applied = []
    for version, migration_file in _migration_files("sqlite" if engine.dialect.name == "sqlite" else "postgres"):
        if version in applied:
            continue
        async with engine.begin() as connection:
            for statement in _statements(migration_file):
                await connection.execute(text(statement))
            await connection.execute(
                text('INSERT INTO schema_migrations ("version", "appliedAt") VALUES (:version, :applied_at)'),
                {"version": version, "applied_at": datetime.now(timezone.utc).isoformat()},
            )
        newly_applied.append(version)
    return newly_applied


async def database_reachable(engine: AsyncEngine) -> bool:
    try:
        async with asyncio.timeout(PING_TIMEOUT_SECONDS):
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
    except DATABASE_ERRORS as error:
        # The type is enough to act on; the message can hold the host name.
        logger.warning("Database ping failed: %s", type(error).__name__, extra={"event": "database_unreachable"})
        return False
    return True


def _migration_files(dialect: str) -> list[tuple[int, Path]]:
    files = []
    for path in (SCHEMA_DIR / dialect).glob("*.sql"):
        match = MIGRATION_FILE.match(path.name)
        if match is None:
            raise ValueError(f"Migration file '{path.name}' must be named NNN_description.sql.")
        files.append((int(match.group(1)), path))
    return sorted(files)


def _statements(migration_file: Path) -> list[str]:
    # Plain ";" splitting: migrations must not put a ";" inside a string literal.
    return [statement.strip() for statement in migration_file.read_text(encoding="utf-8").split(";") if statement.strip()]


def _prepare_sqlite(url: str) -> None:
    database_path = url.split(":///", 1)[-1]
    if database_path and database_path != ":memory:":
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    # Chainlit binds thread/step tags as Python lists (Postgres arrays). SQLite cannot bind lists,
    # and the data layer swallows the error, silently dropping the thread — so store them as JSON.
    sqlite3.register_adapter(list, json.dumps)
    sqlite3.register_adapter(dict, json.dumps)
