"""WebSocket endpoint: the live narration channel.

One socket per task. On connect the client is sent the whole backlog for that
task and then every subsequent event as it happens, so a page refresh mid-run
rebuilds the timeline exactly. The socket is read-only in the outbound
direction; approvals go back over REST, because they need a durable, retryable
request rather than a fire-and-forget frame.
"""

from __future__ import annotations

import contextlib
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stream"])

#: Sent when no event has occurred for this long, to keep intermediaries from
#: dropping an idle socket during a slow page load.
_HEARTBEAT_SECONDS = 25.0


@router.websocket("/ws/tasks/{task_id}")
async def stream_task(websocket: WebSocket, task_id: str) -> None:
    """Stream one task's events until the client disconnects."""
    await websocket.accept()
    bus = websocket.app.state.events

    try:
        async with bus.subscribe(task_id) as events:
            while True:
                event = await events.next(timeout=_HEARTBEAT_SECONDS)
                if event is None:
                    # Nothing happened for a while. Keep the socket warm; the
                    # agent may simply be waiting on a slow page.
                    await websocket.send_json({"type": "heartbeat", "task_id": task_id})
                    continue

                await websocket.send_json(event.model_dump(mode="json"))
    except WebSocketDisconnect:
        logger.debug("Client disconnected from task %s", task_id)
    except Exception:  # noqa: BLE001 - never let a socket error escape
        logger.exception("Error while streaming task %s", task_id)
        with contextlib.suppress(Exception):
            await websocket.close(code=1011)
