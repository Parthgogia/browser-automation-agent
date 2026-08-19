"use client";

/**
 * Shows what is actually running.
 *
 * A degraded setup -- the mock LLM, no database -- produces confusing results
 * rather than errors, so it is stated plainly here instead of leaving the user
 * to wonder why the agent seems oddly literal-minded.
 */

import type { Health, TaskDetail } from "@/lib/types";

const STATUS_COLOURS: Record<string, string> = {
  running: "text-[var(--color-accent)]",
  planning: "text-[var(--color-accent)]",
  awaiting_approval: "text-[var(--color-warn)]",
  awaiting_input: "text-[var(--color-warn)]",
  completed: "text-[var(--color-ok)]",
  failed: "text-[var(--color-danger)]",
  cancelled: "text-[var(--color-danger)]",
};

interface Props {
  health: Health | null;
  task: TaskDetail | null;
  connected: boolean;
  onCancel: () => void;
}

export function StatusBar({ health, task, connected, onCancel }: Props) {
  return (
    <div className="flex items-center gap-3 border-b border-[var(--color-border)] px-4 py-2 text-xs">
      <span className="font-semibold tracking-tight">Browser Agent</span>

      {task && (
        <span className={STATUS_COLOURS[task.status] ?? "text-[var(--color-ink-dim)]"}>
          {task.status.replace(/_/g, " ")}
          {task.steps_taken > 0 && ` - step ${task.steps_taken}`}
        </span>
      )}

      <span className="ml-auto flex items-center gap-3 text-[var(--color-ink-dim)]">
        {health && (
          <>
            <span title="Configured language model">
              {health.llm_provider === "mock" ? (
                <span className="text-[var(--color-warn)]">
                  mock LLM - set GEMINI_API_KEY for real reasoning
                </span>
              ) : (
                `${health.llm_provider}/${health.llm_model}`
              )}
            </span>
            {!health.database && (
              <span className="text-[var(--color-warn)]" title="Run: docker compose up -d">
                no database
              </span>
            )}
          </>
        )}

        <span
          className={connected ? "text-[var(--color-ok)]" : "text-[var(--color-ink-dim)]"}
          title={connected ? "Live event stream connected" : "Not streaming"}
        >
          {connected ? "live" : "idle"}
        </span>

        {task?.running && (
          <button
            type="button"
            onClick={onCancel}
            className="rounded border border-[var(--color-border)] px-2 py-0.5 transition hover:border-[var(--color-danger)] hover:text-[var(--color-danger)]"
          >
            Stop
          </button>
        )}
      </span>
    </div>
  );
}
