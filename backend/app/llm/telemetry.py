"""Task-scoped telemetry for actual LLM inference and embedding requests."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

_recorder: ContextVar[Callable[[str], None] | None] = ContextVar(
    "llm_request_recorder", default=None
)


@contextmanager
def capture_provider_requests(recorder: Callable[[str], None]) -> Iterator[None]:
    """Capture provider HTTP inference requests within this async task context."""
    token: Token[Callable[[str], None] | None] = _recorder.set(recorder)
    try:
        yield
    finally:
        _recorder.reset(token)


def record_provider_request(provider: str) -> None:
    """Record one outbound model or embedding request when a task is active."""
    recorder = _recorder.get()
    if recorder is not None:
        recorder(provider)
