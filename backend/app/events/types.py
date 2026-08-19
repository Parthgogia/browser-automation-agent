"""The event vocabulary shared by the agent, the WebSocket layer and the UI.

Everything the user sees while a task runs arrives as one of these events. The
agent never talks to a WebSocket directly; it publishes events and the API
layer forwards them. That indirection is what lets the same agent run from the
CLI, from a test, or behind the HTTP server without changing a line.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EventType(StrEnum):
    """Every kind of event a task run can emit.

    Kept deliberately small and stable -- the frontend switches on these
    strings, so adding is cheap but renaming is a breaking change.
    """

    # Lifecycle
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"

    # Planning and reasoning
    PLAN_CREATED = "plan_created"
    PLAN_REVISED = "plan_revised"
    THOUGHT = "thought"

    # Acting
    STEP_STARTED = "step_started"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    # Perception
    OBSERVATION = "observation"
    SCREENSHOT = "screenshot"

    # Human-in-the-loop
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_RESOLVED = "approval_resolved"

    # Diagnostics
    ERROR = "error"
    LOG = "log"


class AgentEvent(BaseModel):
    """A single timestamped occurrence within one task run.

    `data` is intentionally untyped at this level: each event type carries a
    different payload, and forcing a discriminated union here would make the
    producer side noisy for very little safety. The shape of each payload is
    documented in `docs/ARCHITECTURE.md`.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str
    type: EventType
    #: Human-readable one-liner. The UI renders this when it has no special
    #: handling for the event type, so it should always be meaningful alone.
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    #: Index of the agent step this event belongs to, when applicable.
    step: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
