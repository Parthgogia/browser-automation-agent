"use client";

/**
 * The whole UI: a two-pane view of one running task.
 *
 * Left, what the agent is thinking and doing. Right, what it is looking at,
 * with the approval gate rendered over the page it applies to. That split is
 * the entire design idea -- reasoning and perception side by side, so a person
 * can check one against the other.
 */

import { useCallback, useEffect, useState } from "react";

import { BrowserPreview } from "@/components/BrowserPreview";
import { StatusBar } from "@/components/StatusBar";
import { TaskComposer } from "@/components/TaskComposer";
import { Timeline } from "@/components/Timeline";
import { api } from "@/lib/api";
import { useTaskStream } from "@/lib/useTaskStream";
import type { Health, TaskDetail } from "@/lib/types";

/** How often to re-read the task row while it is running. */
const POLL_MS = 2000;

export default function Page() {
  const [taskId, setTaskId] = useState<string | null>(null);
  const [task, setTask] = useState<TaskDetail | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  const stream = useTaskStream(taskId);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  // The event stream carries narration; the task row carries status and the
  // final summary. Poll it while the run is live, then stop.
  useEffect(() => {
    if (!taskId) return;

    let cancelled = false;
    const refresh = () =>
      api
        .getTask(taskId)
        .then((detail) => !cancelled && setTask(detail))
        .catch(() => undefined);

    refresh();
    const timer = setInterval(() => {
      if (task && !task.running && !task.awaiting_approval) return;
      refresh();
    }, POLL_MS);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [taskId, task?.running, task?.awaiting_approval]);

  const startTask = useCallback(async (goal: string) => {
    setError(null);
    try {
      const created = await api.createTask(goal);
      setTask(created);
      setTaskId(created.id);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not start the task");
    }
  }, []);

  const decideApproval = useCallback(
    async (approved: boolean) => {
      if (!taskId) return;
      // Close the dialog immediately; the agent resumes asynchronously.
      stream.clearApproval();
      try {
        setTask(await api.resolveApproval(taskId, approved));
      } catch (exc) {
        setError(exc instanceof Error ? exc.message : "Could not send the decision");
      }
    },
    [taskId, stream],
  );

  const cancel = useCallback(async () => {
    if (!taskId) return;
    try {
      setTask(await api.cancelTask(taskId));
    } catch {
      // A task that already finished cannot be cancelled; nothing to report.
    }
  }, [taskId]);

  const busy = Boolean(task?.running || task?.awaiting_approval);

  return (
    <main className="flex h-screen flex-col">
      <StatusBar
        health={health}
        task={task}
        connected={stream.connected}
        onCancel={cancel}
      />

      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[minmax(0,5fr)_minmax(0,4fr)]">
        {/* Reasoning */}
        <section className="flex min-h-0 flex-col border-r border-[var(--color-border)]">
          <TaskComposer onSubmit={startTask} disabled={busy} />

          {error && (
            <p className="border-b border-[var(--color-danger)]/40 bg-[var(--color-danger)]/10 px-4 py-2 text-sm text-[var(--color-danger)]">
              {error}
            </p>
          )}

          <div className="min-h-0 flex-1 overflow-y-auto">
            <Timeline events={stream.events} />
          </div>

          {task?.result_summary && !task.running && (
            <ResultPanel task={task} />
          )}
        </section>

        {/* Perception */}
        <BrowserPreview
          screenshot={stream.screenshot}
          pageUrl={stream.pageUrl}
          approval={stream.approval}
          onApprove={decideApproval}
        />
      </div>
    </main>
  );
}

/** The agent's final answer, pinned below the timeline so it is not scrolled past. */
function ResultPanel({ task }: { task: TaskDetail }) {
  const question = task.status === "awaiting_input";
  const failed = task.success === false;

  const accent = question
    ? "var(--color-warn)"
    : failed
      ? "var(--color-danger)"
      : "var(--color-ok)";

  return (
    <div
      className="border-t p-4"
      style={{ borderColor: accent, background: "var(--color-surface)" }}
    >
      <h2
        className="text-[11px] font-medium uppercase tracking-wide"
        style={{ color: accent }}
      >
        {question ? "The agent has a question" : failed ? "Could not finish" : "Result"}
      </h2>
      <p className="mt-1.5 whitespace-pre-wrap text-sm">
        {task.pending_question ?? task.result_summary}
      </p>
    </div>
  );
}
