"""In-process publish/subscribe bus for task events.

Design notes
------------
The bus is intentionally the simplest thing that works for a single-process
deployment: a dict of task_id -> set of subscriber queues, plus a bounded
replay buffer so a client that connects mid-run still sees what it missed.

Two properties matter and are worth stating explicitly:

* **Publishing never blocks and never fails.** A slow or dead WebSocket must
  not be able to stall the agent. If a subscriber's queue is full we drop the
  oldest event for that subscriber only.
* **Replay is bounded.** We keep the last `REPLAY_LIMIT` events per task so a
  browser refresh can rebuild the timeline without hitting the database.

Scaling past one process means swapping this class for a Redis pub/sub or
Postgres LISTEN/NOTIFY implementation; the interface is small on purpose so
that substitution touches only this file.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from app.events.types import AgentEvent

logger = logging.getLogger(__name__)

#: Events retained per task for late subscribers.
REPLAY_LIMIT = 500

#: Per-subscriber queue depth before we start shedding the oldest events.
SUBSCRIBER_QUEUE_SIZE = 256


class Subscription:
    """A live feed of one task's events.

    Deliberately *not* an async generator. A consumer that wants a heartbeat
    has to bound its wait, and the obvious way to do that --
    ``asyncio.wait_for(agen.__anext__(), timeout)`` -- is a trap: on timeout
    `wait_for` cancels the pending ``__anext__``, which throws `CancelledError`
    into the generator frame and *terminates the generator*. Every subsequent
    call then raises `StopAsyncIteration`, so the stream dies silently at the
    first idle period.

    Exposing :meth:`next` with its own timeout keeps the cancellation on
    ``asyncio.Queue.get``, which is documented to be cancellation-safe: a
    cancelled getter re-wakes another waiter rather than dropping an item.
    """

    def __init__(self, queue: asyncio.Queue[AgentEvent], backlog: list[AgentEvent]) -> None:
        self._queue = queue
        self._backlog = deque(backlog)

    async def next(self, timeout: float | None = None) -> AgentEvent | None:
        """The next event, or None if `timeout` elapsed without one.

        Buffered history is drained first, so a late subscriber catches up
        before it starts waiting on anything.
        """
        if self._backlog:
            return self._backlog.popleft()
        if timeout is None:
            return await self._queue.get()
        try:
            return await asyncio.wait_for(self._queue.get(), timeout)
        except TimeoutError:
            return None

    # `async for` support, for consumers that never need a timeout.
    def __aiter__(self) -> Subscription:
        return self

    async def __anext__(self) -> AgentEvent:
        event = await self.next()
        assert event is not None  # only reachable with timeout=None
        return event


class EventBus:
    """Fan-out of :class:`AgentEvent` objects, keyed by task id."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[AgentEvent]]] = defaultdict(set)
        self._replay: dict[str, deque[AgentEvent]] = defaultdict(
            lambda: deque(maxlen=REPLAY_LIMIT)
        )

    # ------------------------------------------------------------ publish ---

    def publish(self, event: AgentEvent) -> None:
        """Broadcast `event` to every subscriber of its task.

        Synchronous and non-blocking by design so that agent code can emit
        events from anywhere without awaiting.
        """
        self._replay[event.task_id].append(event)

        for queue in self._subscribers[event.task_id]:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Shed the oldest event for this subscriber rather than the
                # newest: a stalled client is most interested in catching up
                # to the present.
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):  # pragma: no cover
                    logger.warning("Dropping event %s for a saturated subscriber", event.type)

    # ---------------------------------------------------------- subscribe ---

    @asynccontextmanager
    async def subscribe(
        self, task_id: str, *, replay: bool = True
    ) -> AsyncIterator[Subscription]:
        """Yield a :class:`Subscription` to `task_id`.

        Used as a context manager so the subscriber is always torn down::

            async with bus.subscribe(task_id) as events:
                async for event in events:          # simple consumers
                    ...

            async with bus.subscribe(task_id) as events:
                event = await events.next(timeout=25)   # consumers needing a heartbeat

        If `replay` is true the subscription first re-emits the buffered
        history, which is what lets a page refresh rebuild the full timeline.
        """
        queue: asyncio.Queue[AgentEvent] = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        self._subscribers[task_id].add(queue)

        backlog = list(self._replay[task_id]) if replay else []

        try:
            yield Subscription(queue, backlog)
        finally:
            self._subscribers[task_id].discard(queue)
            if not self._subscribers[task_id]:
                self._subscribers.pop(task_id, None)

    # ------------------------------------------------------------- replay ---

    def history(self, task_id: str) -> list[AgentEvent]:
        """Buffered events for `task_id`, oldest first."""
        return list(self._replay[task_id])

    def forget(self, task_id: str) -> None:
        """Drop the replay buffer for a finished task."""
        self._replay.pop(task_id, None)


@lru_cache
def get_event_bus() -> EventBus:
    """Process-wide bus singleton."""
    return EventBus()
