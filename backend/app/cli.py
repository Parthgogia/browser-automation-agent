"""Command-line interface: run a task without the web stack.

    agent "find the cheapest RTX 5070 laptop under 120000 INR"

Useful for three things the UI is bad at: quick iteration on prompts, running
the agent from a script, and debugging without a browser tab of your own in the
way. It uses exactly the same graph, runtime and tools as the server -- the only
difference is that approvals are answered at the terminal instead of in a
dialog.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.agent.runtime import AgentRuntime
from app.agent.state import initial_state
from app.browser.registry import get_session_registry
from app.config import get_settings
from app.db.repository import TaskRepository
from app.db.session import get_database
from app.events import EventType, get_event_bus
from app.llm.factory import get_llm_provider
from app.main import configure_logging
from app.memory.store import MemoryStore
from app.safety.policy import SafetyPolicy
from app.tools.registry import build_registry

#: Event types worth printing as the run proceeds. The rest are noise at the
#: terminal (screenshots, raw observations).
_INTERESTING = {
    EventType.PLAN_CREATED,
    EventType.PLAN_REVISED,
    EventType.THOUGHT,
    EventType.TOOL_CALL,
    EventType.TOOL_RESULT,
    EventType.APPROVAL_REQUIRED,
    EventType.ERROR,
    EventType.TASK_COMPLETED,
    EventType.TASK_FAILED,
}

_PREFIX = {
    EventType.PLAN_CREATED: "PLAN ",
    EventType.PLAN_REVISED: "REPLAN",
    EventType.THOUGHT: "THINK",
    EventType.TOOL_CALL: "  ->",
    EventType.TOOL_RESULT: "  <-",
    EventType.APPROVAL_REQUIRED: "PAUSE",
    EventType.ERROR: "ERROR",
    EventType.TASK_COMPLETED: "DONE ",
    EventType.TASK_FAILED: "FAIL ",
}


def build_runtime() -> tuple[AgentRuntime, object]:
    """Assemble the same runtime the server uses. Returns (runtime, database)."""
    settings = get_settings()
    settings.ensure_directories()

    database = get_database()
    llm = get_llm_provider()

    runtime = AgentRuntime(
        settings=settings,
        llm=llm,
        tools=build_registry(),
        sessions=get_session_registry(),
        safety=SafetyPolicy(settings),
        events=get_event_bus(),
        repository=TaskRepository(database),
        memory=MemoryStore(database, llm),
    )
    return runtime, database


async def run_task(goal: str, profile: str | None, *, quiet: bool) -> int:
    """Execute one task to completion. Returns a process exit code."""
    from app.agent.graph import build_graph

    settings = get_settings()
    configure_logging("WARNING" if quiet else settings.log_level)

    runtime, database = build_runtime()
    await database.connect()

    if runtime.memory is not None and not runtime.memory.available:
        runtime.memory = None

    task_id = "cli-" + str(abs(hash(goal)) % 10**8)
    profile = profile or settings.browser_default_profile
    await runtime.repository.create_task(task_id, goal, profile)

    graph = build_graph(runtime)
    config = {
        "configurable": {"thread_id": task_id},
        "recursion_limit": settings.agent_max_steps * 8 + 50,
    }

    printer = asyncio.create_task(_print_events(runtime, task_id))
    exit_code = 1

    try:
        payload: object = initial_state(task_id, goal, profile)
        while True:
            final = await graph.ainvoke(payload, config=config)  # type: ignore[arg-type]

            interrupts = final.get("__interrupt__")
            if not interrupts:
                break

            # The graph paused on an approval gate. Ask, then resume.
            from langgraph.types import Command

            approved = _ask_at_terminal(interrupts)
            payload = Command(resume={"approved": approved})

        print("\n" + "=" * 72)
        print(final.get("result") or "The agent produced no result.")
        print("=" * 72)
        exit_code = 0 if final.get("success") else 1
    finally:
        printer.cancel()
        await runtime.sessions.release(task_id)
        await database.disconnect()

    return exit_code


async def _print_events(runtime: AgentRuntime, task_id: str) -> None:
    """Mirror the event stream to stdout while the task runs."""
    async with runtime.events.subscribe(task_id, replay=False) as events:
        async for event in events:
            if event.type not in _INTERESTING:
                continue
            prefix = _PREFIX.get(event.type, "     ")
            message = event.message.strip().replace("\n", "\n       ")
            print(f"{prefix} {message}", flush=True)

            if event.type is EventType.PLAN_CREATED and event.data.get("plan"):
                for number, step in enumerate(event.data["plan"], start=1):
                    print(f"       {number}. {step}", flush=True)


def _ask_at_terminal(interrupts: object) -> bool:
    """Prompt for approval on stdin.

    Defaults to *no* on anything other than an explicit yes, including EOF --
    a non-interactive shell should never auto-approve a purchase.
    """
    payload = {}
    if isinstance(interrupts, (list, tuple)) and interrupts:
        payload = getattr(interrupts[0], "value", {}) or {}

    print("\n" + "-" * 72)
    print("APPROVAL NEEDED")
    print(f"  Action: {payload.get('tool')}({payload.get('arguments')})")
    print(f"  Page:   {payload.get('page_url', 'unknown')}")
    print(f"  Why:    {payload.get('reason', 'This action may be irreversible.')}")
    print("-" * 72)

    try:
        answer = input("Approve? [y/N] ").strip().lower()
    except EOFError:
        answer = "n"
    return answer in {"y", "yes"}


def main() -> None:
    """Console-script entry point."""
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Run a browser task described in plain language.",
    )
    parser.add_argument("goal", help="What the agent should do")
    parser.add_argument(
        "--profile",
        help="Persistent browser profile to use (keeps cookies and logins)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Only print the final result"
    )
    args = parser.parse_args()

    try:
        sys.exit(asyncio.run(run_task(args.goal, args.profile, quiet=args.quiet)))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
