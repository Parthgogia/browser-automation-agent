"""Long-term memory against a real Postgres + pgvector.

Skipped automatically when the database is not up, so `pytest` stays green on a
machine that has never run `docker compose up`. When it *is* up, these are the
tests that catch the driver-level problems no mock can reproduce -- the pgvector
codec, the vector round-trip, and the cosine ordering.
"""

from __future__ import annotations

import uuid

import pytest

from app.config import Settings, get_settings
from app.db.models import Memory
from app.db.session import Database
from app.llm.mock import MockProvider
from app.memory.store import MemoryStore


@pytest.fixture
async def database() -> Database:
    """A connected Database, or a skip if Postgres is not running."""
    db = Database(get_settings())
    if not await db.connect():
        pytest.skip("Postgres is not available (run: docker compose up -d)")
    yield db
    await db.disconnect()


@pytest.fixture
async def store(database: Database) -> MemoryStore:
    memory = MemoryStore(database, MockProvider(Settings()))
    yield memory
    # Leave the table as we found it, so repeated runs stay deterministic.
    async with database.session() as session:
        if session is not None:
            from sqlalchemy import delete

            await session.execute(delete(Memory).where(Memory.category == "pytest"))


async def test_a_memory_survives_a_round_trip(store: MemoryStore) -> None:
    """The whole point: an embedding written as a vector comes back as one.

    This is the test that would have caught the asyncpg/pgvector codec
    mismatch, where writing succeeded but every read failed with
    "invalid input for query argument".
    """
    fact = f"The user prefers Dell laptops. [{uuid.uuid4().hex}]"
    stored = await store.add(fact, category="pytest")
    assert stored is not None

    hits = await store.search(fact, category="pytest")
    assert [hit.content for hit in hits] == [fact]
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-6)


async def test_results_are_ordered_by_similarity(store: MemoryStore) -> None:
    """Cosine distance must rank the exact match above an unrelated fact."""
    marker = uuid.uuid4().hex
    target = f"The user ships to Pune by default. [{marker}]"
    await store.add(target, category="pytest")
    await store.add(f"Croma had the cheapest monitor last time. [{marker}]", category="pytest")

    # -1.0, not 0.0: cosine similarity is bounded by [-1, 1], and two
    # unrelated vectors land near zero on either side of it.
    hits = await store.search(target, category="pytest", min_similarity=-1.0, limit=5)

    assert len(hits) >= 2
    assert hits[0].content == target
    assert hits[0].similarity > hits[-1].similarity


async def test_near_duplicates_are_not_stored_twice(store: MemoryStore) -> None:
    """Otherwise every repeat of a task piles up another copy of the same fact."""
    fact = f"The user likes mechanical keyboards. [{uuid.uuid4().hex}]"

    first = await store.add(fact, category="pytest")
    second = await store.add(fact, category="pytest")

    assert first is not None and second is not None
    assert first.id == second.id


async def test_unrelated_queries_return_nothing(store: MemoryStore) -> None:
    """The similarity floor exists so the planner is not fed noise."""
    await store.add(f"The user prefers Dell. [{uuid.uuid4().hex}]", category="pytest")

    hits = await store.search(
        "completely unrelated query about tropical fish", category="pytest"
    )
    assert hits == []


async def test_recall_renders_nothing_when_there_is_nothing(store: MemoryStore) -> None:
    """An empty section must be omitted, not shown as an empty heading."""
    assert await store.recall_for_goal("a goal with no relevant memories") == ""
