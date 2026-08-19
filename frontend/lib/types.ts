/**
 * Types mirroring the backend's Pydantic schemas.
 *
 * Kept hand-written rather than generated: the surface is small, and having it
 * in one readable file makes the client/server contract easy to audit. If the
 * API grows, generate these from `/openapi.json` instead.
 */

/** Must stay in sync with `app.events.types.EventType`. */
export type EventType =
  | "task_started"
  | "task_completed"
  | "task_failed"
  | "task_cancelled"
  | "plan_created"
  | "plan_revised"
  | "thought"
  | "step_started"
  | "tool_call"
  | "tool_result"
  | "observation"
  | "screenshot"
  | "approval_required"
  | "approval_resolved"
  | "error"
  | "log";

export interface AgentEvent {
  id: string;
  task_id: string;
  type: EventType;
  message: string;
  data: Record<string, unknown>;
  step: number | null;
  created_at: string;
}

export type TaskStatus =
  | "pending"
  | "planning"
  | "running"
  | "awaiting_approval"
  | "awaiting_input"
  | "completed"
  | "failed"
  | "cancelled";

export interface TaskSummary {
  id: string;
  goal: string;
  status: TaskStatus;
  profile: string;
  success: boolean | null;
  steps_taken: number;
  created_at: string;
  finished_at: string | null;
}

export interface TaskDetail extends TaskSummary {
  plan: string[];
  result_summary: string | null;
  error: string | null;
  pending_question: string | null;
  awaiting_approval: boolean;
  running: boolean;
}

/** Payload of an `approval_required` event. */
export interface ApprovalRequest {
  step: number;
  tool: string;
  arguments: Record<string, unknown>;
  reason: string;
  page_url: string;
}

export interface Health {
  status: string;
  llm_provider: string;
  llm_model: string;
  database: boolean;
  memory: boolean;
  browser_headless: boolean;
  vision_mode: string;
  active_sessions: number;
}
