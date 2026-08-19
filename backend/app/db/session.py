"""Database connection management.

The database is treated as *optional infrastructure*. If Postgres is not
running, the agent still works -- you lose task history, the audit trail and
long-term memory, but the browser still drives and the task still completes.
That choice matters for a tool people run on their own machine: forgetting to
start Docker should degrade the experience, not block it.

`Database.available` is the flag everything else checks.

Driver note
-----------
The driver is asyncpg rather than psycopg, and that is a platform requirement
rather than a preference. On Windows, Playwright needs the default
`ProactorEventLoop` for subprocess support, while psycopg's async mode refuses
to run on it and demands a `SelectorEventLoop`. Since the browser and the
database share one event loop in this process, only asyncpg satisfies both.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings, get_settings
from app.db.models import Base

logger = logging.getLogger(__name__)


class Database:
    """Owns the async engine and hands out sessions."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None
        self._available = False

    @property
    def available(self) -> bool:
        """False when we could not reach Postgres at startup."""
        return self._available

    async def connect(self) -> bool:
        """Connect, enable pgvector and create tables. Never raises.

        Returns True when the database is usable.
        """
        try:
            self._engine = create_async_engine(
                self._settings.database_url,
                # Long-running agent tasks can leave a connection idle for
                # minutes; recycling avoids Postgres closing it underneath us.
                pool_pre_ping=True,
                pool_recycle=1800,
                echo=False,
            )
            # asyncpg is strictly typed and has no built-in `vector` codec, so
            # one is registered on every new connection. Without this, writing
            # an embedding fails with an opaque parameter-type error.
            _register_vector_codec(self._engine)

            async with self._engine.begin() as connection:
                # Idempotent; the compose file also does this on first boot,
                # but a hand-rolled Postgres will not have.
                await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await connection.run_sync(Base.metadata.create_all)

            self._sessionmaker = async_sessionmaker(
                self._engine, expire_on_commit=False, class_=AsyncSession
            )
            self._available = True
            logger.info("Database connected: %s", _redact(self._settings.database_url))
        except Exception as exc:  # noqa: BLE001 - degradation is the point
            self._available = False
            logger.warning(
                "Database unavailable (%s). Task history and long-term memory are "
                "disabled for this session; run `docker compose up -d` to enable them.",
                exc,
            )
        return self._available

    async def disconnect(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._sessionmaker = None
            self._available = False

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession | None]:
        """Yield a session, or None when the database is unavailable.

        Callers write ``async with db.session() as s: if s is None: return``,
        which keeps the degraded path explicit at every call site rather than
        hiding it behind a silent no-op object.
        """
        if not self._available or self._sessionmaker is None:
            yield None
            return

        async with self._sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise


def _register_vector_codec(engine: AsyncEngine) -> None:
    """Teach asyncpg to pass pgvector values through as text.

    Two libraries have to agree here, and they do not by default.
    `pgvector.sqlalchemy.VECTOR` is a *text* type: it serialises a list to
    ``'[1,2,3]'`` on the way out and parses that string back on the way in.
    `pgvector.asyncpg.register_vector`, meanwhile, installs a *binary* codec
    that expects a list and rejects a string outright::

        invalid input for query argument $1: '[0.06, -0.004, ...'
        (expected list or ndarray)

    So rather than the binary codec, we register an identity codec in text
    format. asyncpg then ships SQLAlchemy's string straight to Postgres, which
    parses it natively, and hands the text back for SQLAlchemy to decode. Each
    library keeps doing its own job and neither has to know about the other.

    Hooked on the sync engine because that is where SQLAlchemy emits `connect`;
    `await_` bridges back into the async driver from the sync callback.
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _on_connect(dbapi_connection, _record):  # type: ignore[no-untyped-def]
        async def register(connection) -> None:  # type: ignore[no-untyped-def]
            await connection.set_type_codec(
                "vector",
                schema="public",
                encoder=str,
                decoder=str,
                format="text",
            )

        dbapi_connection.await_(register(dbapi_connection.driver_connection))


def _redact(url: str) -> str:
    """Strip the password out of a connection URL before logging it."""
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    credentials, host = rest.rsplit("@", 1)
    user = credentials.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"


@lru_cache
def get_database() -> Database:
    """Process-wide database singleton."""
    return Database(get_settings())
