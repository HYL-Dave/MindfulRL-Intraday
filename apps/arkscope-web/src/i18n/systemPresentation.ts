import type { TFunction } from "i18next";

import { ApiError, type ApiStatus } from "../api";

const STATUS_ROUTE = "/status" as const;
const SAFE_CODE = /^[A-Za-z][A-Za-z0-9_.-]{0,63}$/u;

export interface SidecarFailureOutcome {
  kind: "status_request_failed";
  status: number | null;
  code: string | null;
  route: typeof STATUS_ROUTE;
}

export type SystemStatusState =
  | { kind: "loading" }
  | { kind: "ready"; status: ApiStatus }
  | { kind: "error"; outcome: SidecarFailureOutcome };

export type SystemStatusPresentation =
  | { kind: "loading"; message: string }
  | { kind: "ready"; message: string }
  | { kind: "error"; title: string; retryLabel: string; diagnostics: string[] };

function safeHttpStatus(value: unknown): number | null {
  return typeof value === "number"
    && Number.isInteger(value)
    && value >= 400
    && value <= 599
    ? value
    : null;
}

function safeCode(value: unknown): string | null {
  return typeof value === "string" && SAFE_CODE.test(value) ? value : null;
}

export function captureSidecarFailure(error: unknown): SidecarFailureOutcome {
  let status: number | null = null;
  let code: string | null = null;
  if (error instanceof ApiError) {
    try {
      status = safeHttpStatus(error.status);
      code = safeCode(error.code);
    } catch {
      status = null;
      code = null;
    }
  }
  return {
    kind: "status_request_failed",
    status,
    code,
    route: STATUS_ROUTE,
  };
}

function reviewedDiagnostics(outcome: SidecarFailureOutcome): string[] {
  const diagnostics: string[] = [];
  const status = safeHttpStatus(outcome.status);
  const code = safeCode(outcome.code);
  if (status !== null) diagnostics.push(`HTTP ${status}`);
  if (code !== null) diagnostics.push(code);
  if (outcome.route === STATUS_ROUTE) diagnostics.push(STATUS_ROUTE);
  return diagnostics;
}

export function presentSystemStatus(
  state: SystemStatusState,
  developerMode: boolean,
  t: TFunction<"system">,
): SystemStatusPresentation {
  switch (state.kind) {
    case "loading":
      return { kind: "loading", message: t(($) => $.sidecar.loading) };
    case "ready":
      return { kind: "ready", message: t(($) => $.sidecar.ready) };
    case "error":
      return {
        kind: "error",
        title: t(($) => $.sidecar.failure),
        retryLabel: t(($) => $.sidecar.retry),
        diagnostics: developerMode ? reviewedDiagnostics(state.outcome) : [],
      };
    default: {
      const exhaustive: never = state;
      return exhaustive;
    }
  }
}
