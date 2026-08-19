"use client";

/** Where a task starts: one text box and a few worked examples. */

import { useState } from "react";

const EXAMPLES = [
  "Find the cheapest RTX 5070 laptop under Rs 1.2L",
  "Compare MacBook Air prices on Amazon and Flipkart",
  "What are the top 3 news stories on Hacker News right now?",
];

interface Props {
  onSubmit: (goal: string) => void;
  disabled: boolean;
}

export function TaskComposer({ onSubmit, disabled }: Props) {
  const [goal, setGoal] = useState("");

  const submit = () => {
    const trimmed = goal.trim();
    if (trimmed.length >= 3 && !disabled) {
      onSubmit(trimmed);
      setGoal("");
    }
  };

  return (
    <div className="border-b border-[var(--color-border)] p-4">
      <div className="flex gap-2">
        <textarea
          value={goal}
          onChange={(event) => setGoal(event.target.value)}
          onKeyDown={(event) => {
            // Enter submits; Shift+Enter adds a line, as in a chat box.
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
          rows={2}
          disabled={disabled}
          placeholder="Describe what you want done..."
          className="flex-1 resize-none rounded border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm outline-none placeholder:text-[var(--color-ink-dim)] focus:border-[var(--color-accent)] disabled:opacity-50"
        />
        <button
          type="button"
          onClick={submit}
          disabled={disabled || goal.trim().length < 3}
          className="self-stretch rounded bg-[var(--color-accent)] px-4 text-sm font-medium text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Run
        </button>
      </div>

      <div className="mt-2 flex flex-wrap gap-1.5">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            disabled={disabled}
            onClick={() => setGoal(example)}
            className="rounded-full border border-[var(--color-border)] px-2.5 py-1 text-[11px] text-[var(--color-ink-dim)] transition hover:border-[var(--color-accent)] hover:text-[var(--color-ink)] disabled:opacity-40"
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  );
}
