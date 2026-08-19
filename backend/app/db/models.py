"""SQLAlchemy models: the durable record of everything the agent did.

Four tables, each answering a different question:

``tasks``       -- what was asked, and how did it end?
``task_events`` -- the full narration, replayable into the UI after a refresh.
``tool_calls``  -- the audit trail: every effect the agent had, with its risk
                   rating and whether a human approved it.
``memories``    -- what the agent learned about the user, as embeddings.

The separation between `task_events` and `tool_calls` is deliberate. Events are
a presentation log and are allowed to be lossy and verbose; tool calls are the
compliance record and stay minimal and precise.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for every model."""


class TaskStatus(StrEnum):
    """Lifecycle of a single task run."""

    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    #: Paused at an approval gate, waiting for a human decision.
    AWAITING_APPROVAL = "awaiting_approval"
    #: Paused because the agent asked the user a question.
    AWAITING_INPUT = "awaiting_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}


class Task(Base):
    """One natural-language request and its outcome."""

    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    goal: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default=TaskStatus.PENDING, index=True)
    #: Which persistent browser profile this run used.
    profile: Mapped[str] = mapped_column(String(64), default="default")

    #: The plan as a list of step descriptions, revised in place as the agent
    #: replans. Stored as JSON because it is read whole and never queried into.
    plan: Mapped[list[str]] = mapped_column(JSON, default=list)
    steps_taken: Mapped[int] = mapped_column(Integer, default=0)

    #: What the agent told the user at the end.
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    success: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: Set when the run ended by asking the user something.
    pending_question: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    events: Mapped[list[TaskEvent]] = relationship(
        back_populates="task", cascade="all, delete-orphan", lazy="selectin"
    )
    tool_calls: Mapped[list[ToolCallRecord]] = relationship(
        back_populates="task", cascade="all, delete-orphan", lazy="selectin"
    )


class TaskEvent(Base):
    """One entry in a task's narration stream."""

    __tablename__ = "task_events"
    # Events are always read as "everything for this task, in order".
    __table_args__ = (Index("ix_task_events_task_created", "task_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    task_id: Mapped[str] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), index=True
    )
    type: Mapped[str] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(Text, default="")
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    task: Mapped[Task] = relationship(back_populates="events")


class ToolCallRecord(Base):
    """Audit record for one executed (or refused) tool call."""

    __tablename__ = "tool_calls"
    __table_args__ = (Index("ix_tool_calls_task_step", "task_id", "step"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    task_id: Mapped[str] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), index=True
    )
    step: Mapped[int] = mapped_column(Integer)
    tool_name: Mapped[str] = mapped_column(String(64), index=True)
    arguments: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    ok: Mapped[bool] = mapped_column(Boolean, default=True)
    #: The tool's model-facing message, truncated for storage.
    result: Mapped[str] = mapped_column(Text, default="")
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)

    #: Risk rating assigned by the safety policy, as its integer value.
    risk: Mapped[int] = mapped_column(Integer, default=0)
    #: Null when no approval was needed; true/false once a human decided.
    approved: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    #: URL the browser was on when the call ran -- the single most useful
    #: column when reconstructing what the agent was actually looking at.
    page_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    screenshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    task: Mapped[Task] = relationship(back_populates="tool_calls")


class Memory(Base):
    """A durable fact about the user, stored with its embedding.

    The `embedding` column is an unconstrained pgvector column rather than a
    fixed width. That lets the embedding model change (768-dim
    text-embedding-004, 3072-dim gemini-embedding-001, the mock provider's
    768-dim hashes) without a migration. The cost is that an ANN index cannot
    be built until the width is pinned; at the volumes personal memory reaches
    -- thousands of rows, not millions -- an exact scan is comfortably fast.
    See `docs/ARCHITECTURE.md` for when to revisit that.
    """

    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    content: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(32), default="preference", index=True)
    embedding: Mapped[list[float]] = mapped_column(Vector())
    #: Which run produced this memory, for provenance.
    task_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    #: How many times recall has surfaced this memory -- a cheap relevance
    #: signal for future pruning.
    hits: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
