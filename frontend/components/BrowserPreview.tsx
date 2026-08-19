"use client";

/**
 * Live view of what the agent is looking at.
 *
 * The screenshots carry the same numbered overlays the model is given, so this
 * pane is not just reassurance -- it lets a person read the agent's own
 * addressing scheme off the picture and understand why it clicked [12] instead
 * of [13].
 */

import type { ApprovalRequest } from "@/lib/types";

interface Props {
  screenshot: string | null;
  pageUrl: string | null;
  approval: ApprovalRequest | null;
  onApprove: (approved: boolean) => void;
}

export function BrowserPreview({ screenshot, pageUrl, approval, onApprove }: Props) {
  return (
    <section className="flex h-full min-h-0 flex-col">
      <header className="flex items-center gap-2 border-b border-[var(--color-border)] px-4 py-2.5">
        <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--color-ink-dim)]">
          Browser
        </span>
        <span
          className="truncate font-mono text-xs text-[var(--color-ink-dim)]"
          title={pageUrl ?? undefined}
        >
          {pageUrl ?? "about:blank"}
        </span>
      </header>

      <div className="relative min-h-0 flex-1 overflow-auto bg-black/30 p-3">
        {screenshot ? (
          // Not next/image: these are runtime-generated files served by the
          // backend, so there is nothing for the optimiser to do.
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={screenshot}
            alt="Current browser page"
            className="w-full rounded border border-[var(--color-border)]"
          />
        ) : (
          <p className="p-8 text-center text-sm text-[var(--color-ink-dim)]">
            The page appears here once the agent opens the browser.
          </p>
        )}

        {approval && <ApprovalOverlay request={approval} onDecide={onApprove} />}
      </div>
    </section>
  );
}

/**
 * The approval gate, rendered over the screenshot on purpose: the decision is
 * about *this page*, and showing it anywhere else would ask the user to
 * approve an action they cannot see.
 */
function ApprovalOverlay({
  request,
  onDecide,
}: {
  request: ApprovalRequest;
  onDecide: (approved: boolean) => void;
}) {
  return (
    <div className="absolute inset-0 flex items-end justify-center bg-black/60 p-4 backdrop-blur-[2px]">
      <div className="w-full max-w-lg rounded-lg border border-[var(--color-warn)] bg-[var(--color-surface)] p-4 shadow-2xl">
        <h2 className="text-sm font-semibold text-[var(--color-warn)]">
          The agent needs your approval
        </h2>

        <p className="mt-2 text-sm text-[var(--color-ink)]">{request.reason}</p>

        <dl className="mt-3 space-y-1 rounded border border-[var(--color-border)] bg-[var(--color-canvas)] p-2.5 font-mono text-xs">
          <div className="flex gap-2">
            <dt className="text-[var(--color-ink-dim)]">action</dt>
            <dd className="break-all">
              {request.tool}({JSON.stringify(request.arguments)})
            </dd>
          </div>
          <div className="flex gap-2">
            <dt className="text-[var(--color-ink-dim)]">page</dt>
            <dd className="break-all text-[var(--color-ink-dim)]">{request.page_url}</dd>
          </div>
        </dl>

        <div className="mt-3 flex gap-2">
          <button
            type="button"
            onClick={() => onDecide(true)}
            className="flex-1 rounded bg-[var(--color-ok)] px-3 py-2 text-sm font-medium text-black transition hover:opacity-90"
          >
            Approve
          </button>
          <button
            type="button"
            onClick={() => onDecide(false)}
            className="flex-1 rounded border border-[var(--color-border)] px-3 py-2 text-sm font-medium transition hover:bg-[var(--color-surface-2)]"
          >
            Reject
          </button>
        </div>

        <p className="mt-2 text-[11px] text-[var(--color-ink-dim)]">
          Rejecting does not end the task -- the agent is told no and looks for
          another way.
        </p>
      </div>
    </div>
  );
}
