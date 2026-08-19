"""FastAPI application: composition root and lifecycle.

This is the one place where the concrete pieces are wired together. Every other
module takes its collaborators as arguments, which is what makes them testable;
here we finally decide *which* database, *which* LLM, *which* browser registry.

Startup is ordered so the process always comes up, even partially:

1. Settings and directories -- cannot fail.
2. Database -- may fail; the app continues without history or memory.
3. LLM provider -- may fail if a key is missing; the error is explicit.
4. Tools, safety, runtime, runner -- pure construction.

Shutdown reverses it, and crucially cancels in-flight runs so no orphaned
Chromium process is left behind.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.agent.runner import AgentRunner
from app.agent.runtime import AgentRuntime
from app.api import rest_router, ws_router
from app.browser.registry import get_session_registry
from app.config import get_settings
from app.db.repository import TaskRepository
from app.db.session import get_database
from app.events import get_event_bus
from app.llm.factory import get_llm_provider
from app.memory.store import MemoryStore
from app.safety.policy import SafetyPolicy
from app.tools.registry import build_registry

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    """Set up log formatting once, at startup."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%H:%M:%S",
    )
    # These are chatty at INFO and drown out the agent's own narration.
    for noisy in ("httpx", "httpcore", "urllib3", "sqlalchemy.engine.Engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build every service on startup and tear it down on shutdown."""
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_directories()

    database = get_database()
    await database.connect()

    llm = get_llm_provider()
    repository = TaskRepository(database)
    memory = MemoryStore(database, llm) if database.available else None
    sessions = get_session_registry()

    runtime = AgentRuntime(
        settings=settings,
        llm=llm,
        tools=build_registry(),
        sessions=sessions,
        safety=SafetyPolicy(settings),
        events=get_event_bus(),
        repository=repository,
        memory=memory,
    )

    # Exposed on app.state so routes can reach them without global lookups.
    app.state.settings = settings
    app.state.database = database
    app.state.llm = llm
    app.state.repository = repository
    app.state.memory = memory
    app.state.sessions = sessions
    app.state.tools = runtime.tools
    app.state.events = runtime.events
    app.state.runtime = runtime
    app.state.runner = AgentRunner(runtime)

    logger.info(
        "Ready on http://%s:%s  (llm=%s, database=%s, browser=%s)",
        settings.host,
        settings.port,
        llm.name,
        "up" if database.available else "unavailable",
        "headless" if settings.browser_headless else "headed",
    )

    try:
        yield
    finally:
        # Order matters: stop the agents first so nothing tries to use a
        # browser or a connection that is already being torn down.
        await app.state.runner.shutdown()
        await sessions.release_all()
        await database.disconnect()
        logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()
    settings.ensure_directories()

    app = FastAPI(
        title="Browser Automation Agent",
        version=__version__,
        summary="An LLM agent that operates a real web browser to complete tasks.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(rest_router)
    app.include_router(ws_router)

    # Screenshots are served straight off disk rather than base64-encoded into
    # events: the browser caches them, and the event stream stays small enough
    # to keep the timeline responsive.
    app.mount(
        "/screenshots",
        StaticFiles(directory=settings.screenshot_dir),
        name="screenshots",
    )

    return app


app = create_app()


def main() -> None:
    """Entry point for `python -m app.main`."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
