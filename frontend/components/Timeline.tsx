"use client";

/**
 * The execution timeline: what the agent thought, tried, and got back.
 *
 * This is the part of the UI that matters most. A browser agent is opaque by
 * nature -- pages flash past faster than anyone can follow -- so the timeline
 * is how a person builds enough of a mental model to trust it, or to see
 * exactly where it went wrong. Every event is therefore shown with its own
 * shape rather than flattened into a log of grey text.
 */

import { useEffect, useRef } from "react";
import type { AgentEvent, EventType } from "@/lib/types";

/** Events that would only add noise to a human-readable timeline. */
const HIDDEN: ReadonlySet<EventType> = new Set(["screenshot", "observation", "log"]);

const STYLES: Partial<Record<EventType, { label: string; className: string }>> = {
  task_started: { label: "Task", className: "text-[var(--color-accent)]" },
  plan_created: { label: "Plan", className: "text-[var(--color-accent)]" },
  plan_revised: { label: "Replan", className: "text-[var(--color-warn)]" },
  thought: { label: "Thinking", className: "text-[var(--color-ink-dim)]" },
  tool_call: { label: "Action", className: "text-[var(--color-ink)]" },
  tool_result: { label: "Result", className: "text-[var(--color-ink-dim)]" },
  approval_required: { label: "Approval", className: "text-[var(--color-warn)]" },
  approval_resolved: { label: "Approval", className: "text-[var(--color-ink-dim)]" },
  task_completed: { label: "Done", className: "text-[var(--color-ok)]" },
  task_failed: { label: "Failed", className: "text-[var(--color-danger)]" },
  task_cancelled: { label: "Cancelled", className: "text-[var(--color-danger)]" },
  error: { label: "Error", className: "text-[var(--color-danger)]" },
};

export function Timeline({ events }: { events: AgentEvent[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  const visible = events.filter((event) => !HIDDEN.has(event.type));

  // Follow the agent as it works. Only when new events arrive, so a user who
  // has scrolled up to read something is not yanked back down.
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [visible.length]);

  if (visible.length === 0) {
    return (
      <p className="p-6 text-sm text-[var(--color-ink-dim)]">
        Describe a task above and the agent will start working. Everything it
        does appears here, step by step.
      </p>
    );
  }

  return (
    <ol className="flex flex-col gap-3 p-4">
      {visible.map((event) => (
        <TimelineEntry key={event.id} event={event} />
      ))}
      <div ref={endRef} />
    </ol>
  );
}

function TimelineEntry({ event }: { event: AgentEvent }) {
  const style = STYLES[event.type] ?? {
    label: event.type,
    className: "text-[var(--color-ink-dim)]",
  };

  const failed = event.type === "tool_result" && event.data.ok === false;

  return (
    <li className="flex gap-3 text-sm">
      <div className="w-16 shrink-0 pt-0.5 text-right">
        <span className={`text-[11px] font-medium uppercase tracking-wide ${style.className}`}>
          {style.label}
        </span>
        {event.step !== null && (
          <div className="text-[10px] text-[var(--color-ink-dim)]">step {event.step}</div>
        )}
      </div>

      <div className="min-w-0 flex-1">
        <p
          className={`whitespace-pre-wrap break-words ${
            failed ? "text-[var(--color-danger)]" : ""
          } ${event.type === "tool_call" ? "font-mono text-[13px]" : ""}`}
        >
          {event.message}
        </p>

        {/* A plan is a list, so render it as one rather than as a paragraph. */}
        {Array.isArray(event.data.plan) && (event.data.plan as string[]).length > 0 && (
          <ol className="mt-1.5 list-decimal space-y-0.5 pl-5 text-[13px] text-[var(--color-ink-dim)]">
            {(event.data.plan as string[]).map((step, index) => (
              <li key={index}>{step}</li>
            ))}
          </ol>
        )}

        {typeof event.data.advice === "string" && event.data.advice && (
          <p className="mt-1 text-[13px] text-[var(--color-warn)]">
            Next: {event.data.advice}
          </p>
        )}
      </div>
    </li>
  );
}
