/**
 * Thin REST client.
 *
 * Every path is relative: `next.config.ts` proxies `/api` and `/screenshots`
 * to the FastAPI backend, so the browser only ever talks to its own origin and
 * there is no CORS or backend URL to configure in the client.
 */

import type { AgentEvent, Health, TaskDetail, TaskSummary } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });

  if (!response.ok) {
    // Surface FastAPI's `detail` when there is one; it is written for humans.
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body?.detail ?? detail;
    } catch {
      // Non-JSON error body; the status text will have to do.
    }
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  createTask: (goal: string, profile?: string) =>
    request<TaskDetail>("/api/tasks", {
      method: "POST",
      body: JSON.stringify({ goal, profile }),
    }),

  getTask: (taskId: string) => request<TaskDetail>(`/api/tasks/${taskId}`),

  listTasks: (limit = 25) => request<TaskSummary[]>(`/api/tasks?limit=${limit}`),

  getEvents: (taskId: string) => request<AgentEvent[]>(`/api/tasks/${taskId}/events`),

  /** Answer a pending approval. `approved: false` tells the agent to back off. */
  resolveApproval: (taskId: string, approved: boolean) =>
    request<TaskDetail>(`/api/tasks/${taskId}/approval`, {
      method: "POST",
      body: JSON.stringify({ approved }),
    }),

  cancelTask: (taskId: string) =>
    request<TaskDetail>(`/api/tasks/${taskId}/cancel`, { method: "POST" }),

  health: () => request<Health>("/api/health"),
};
