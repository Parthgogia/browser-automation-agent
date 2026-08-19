"use client";

/**
 * Subscribes to a task's live event stream.
 *
 * The backend replays a task's whole backlog on connect, so this hook does not
 * need to fetch history separately -- opening the socket is enough to rebuild
 * the timeline after a page refresh, which is the behaviour that makes the UI
 * feel reliable rather than fragile.
 */

import { useEffect, useRef, useState } from "react";
import type { AgentEvent, ApprovalRequest } from "./types";

const WS_BASE =
  process.env.NEXT_PUBLIC_WS_URL ?? "ws://127.0.0.1:8000";

export interface TaskStream {
  events: AgentEvent[];
  /** Latest screenshot URL, for the live browser preview. */
  screenshot: string | null;
  /** Page the browser is currently on. */
  pageUrl: string | null;
  /** Set while the agent is blocked waiting for a decision. */
  approval: ApprovalRequest | null;
  /** Cleared when the user answers, so the dialog closes immediately. */
  clearApproval: () => void;
  connected: boolean;
}

export function useTaskStream(taskId: string | null): TaskStream {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [screenshot, setScreenshot] = useState<string | null>(null);
  const [pageUrl, setPageUrl] = useState<string | null>(null);
  const [approval, setApproval] = useState<ApprovalRequest | null>(null);
  const [connected, setConnected] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!taskId) return;

    // A new task means a new timeline; drop everything from the previous one.
    setEvents([]);
    setScreenshot(null);
    setPageUrl(null);
    setApproval(null);

    const socket = new WebSocket(`${WS_BASE}/ws/tasks/${taskId}`);
    socketRef.current = socket;

    socket.onopen = () => setConnected(true);
    socket.onclose = () => setConnected(false);
    socket.onerror = () => setConnected(false);

    socket.onmessage = (message) => {
      const payload = JSON.parse(message.data);
      if (payload.type === "heartbeat") return;

      const event = payload as AgentEvent;
      setEvents((previous) => [...previous, event]);

      switch (event.type) {
        case "screenshot":
          setScreenshot((event.data.url as string) ?? null);
          setPageUrl((event.data.page_url as string) ?? null);
          break;
        case "observation":
          setPageUrl((event.data.url as string) ?? null);
          break;
        case "approval_required":
          setApproval({
            step: event.step ?? 0,
            tool: event.data.tool as string,
            arguments: (event.data.arguments as Record<string, unknown>) ?? {},
            reason: event.message,
            page_url: (event.data.page_url as string) ?? "",
          });
          break;
        case "approval_resolved":
          setApproval(null);
          break;
      }
    };

    return () => socket.close();
  }, [taskId]);

  return {
    events,
    screenshot,
    pageUrl,
    approval,
    clearApproval: () => setApproval(null),
    connected,
  };
}
