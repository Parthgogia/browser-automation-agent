"""Semantic memory over pgvector.

What goes in here is narrow on purpose: durable facts about *the user*, not a
transcript of everything the agent has ever seen. Stuffing page contents into
memory produces a store that retrieves plausible-looking noise on every query
and quietly degrades the planner. Preferences, addresses, and hard-won facts
about specific sites are the things worth keeping.

Retrieval uses cosine distance. The embeddings from both providers are already
L2-normalised, so cosine and inner product would rank identically; cosine is
used because it makes the threshold interpretable across providers -- 1.0 is
identical and unrelated text sits near 0, though it can dip slightly negative,
so the floor is a tuned constant rather than zero.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select, update

from app.db.models import Memory
from app.db.session import Database
from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)

#: Below this cosine similarity a "match" is noise. Similarity here is
#: ``1 - cosine_distance`` and so ranges over [-1, 1]; unrelated text lands near
#: zero and can be slightly negative, which is why 0.0 is not a neutral floor.
#: Tuned to be permissive -- a missing memory is invisible, a wrong one actively
#: misleads the planner, but too high a bar makes the feature never fire at all.
_MIN_SIMILARITY = 0.35

#: Near-identical text is treated as the same memory rather than stored twice.
_DEDUPE_SIMILARITY = 0.95


@dataclass(slots=True)
class MemoryHit:
    """One retrieved memory and how well it matched."""

    id: str
    content: str
    category: str
    similarity: float
    created_at: datetime | None = None


class MemoryStore:
    """Embeds, stores and retrieves durable facts."""

    def __init__(self, database: Database, provider: LLMProvider) -> None:
        self._db = database
        self._llm = provider

    @property
    def available(self) -> bool:
        return self._db.available

    # --------------------------------------------------------------- write --

    async def add(
        self, content: str, *, category: str = "preference", task_id: str | None = None
    ) -> MemoryHit | None:
        """Store `content`, unless something near-identical is already stored.

        Returns the stored memory, or the existing near-duplicate.
        """
        content = content.strip()
        if not content:
            return None

        existing = await self.search(content, limit=1, min_similarity=_DEDUPE_SIMILARITY)
        if existing:
            logger.debug("Skipping duplicate memory: %s", content[:60])
            return existing[0]

        vectors = await self._llm.embed([content])
        if not vectors:
            return None

        async with self._db.session() as session:
            if session is None:
                return None
            memory = Memory(
                content=content,
                category=category,
                embedding=vectors[0],
                task_id=task_id,
            )
            session.add(memory)
            await session.flush()
            return MemoryHit(
                id=memory.id,
                content=memory.content,
                category=memory.category,
                similarity=1.0,
                created_at=memory.created_at,
            )

    # ---------------------------------------------------------------- read --

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        min_similarity: float = _MIN_SIMILARITY,
        category: str | None = None,
    ) -> list[MemoryHit]:
        """Return the memories most similar to `query`."""
        if not self._db.available or not query.strip():
            return []

        vectors = await self._llm.embed([query])
        if not vectors:
            return []
        query_vector = vectors[0]

        async with self._db.session() as session:
            if session is None:
                return []

            # pgvector exposes cosine *distance* (0 = identical), so similarity
            # is 1 - distance and ordering ascending by distance is correct.
            distance = Memory.embedding.cosine_distance(query_vector).label("distance")
            statement = select(Memory, distance).order_by(distance).limit(limit)
            if category:
                statement = statement.where(Memory.category == category)

            rows = (await session.execute(statement)).all()

            hits = [
                MemoryHit(
                    id=memory.id,
                    content=memory.content,
                    category=memory.category,
                    similarity=1.0 - float(dist),
                    created_at=memory.created_at,
                )
                for memory, dist in rows
                if 1.0 - float(dist) >= min_similarity
            ]

            if hits:
                # Cheap usage signal, for pruning stale memories later.
                await session.execute(
                    update(Memory)
                    .where(Memory.id.in_([hit.id for hit in hits]))
                    .values(hits=Memory.hits + 1)
                )
            return hits

    async def recall_for_goal(self, goal: str, *, limit: int = 5) -> str:
        """Memories relevant to `goal`, formatted for the planner prompt.

        Returns an empty string when nothing is relevant, so the caller can
        drop the section entirely rather than showing an empty heading.
        """
        hits = await self.search(goal, limit=limit)
        if not hits:
            return ""
        lines = [f"- {hit.content}" for hit in hits]
        return "What you know about this user from previous tasks:\n" + "\n".join(lines)
