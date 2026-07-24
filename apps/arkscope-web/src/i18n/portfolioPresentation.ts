import type { TFunction } from "i18next";

import type {
  PortfolioActivityItem,
  PortfolioActivityObjective,
  PortfolioActivitySource,
  PortfolioActivityState,
  PortfolioCaptureRun,
  PortfolioCaptureRunState,
  PortfolioCoverageGapActivityItem,
  PortfolioIntentLabel,
  PortfolioManualActivityItem,
  PortfolioUnmatchedActivityItem,
} from "../api";

export type PortfolioT = TFunction<"portfolio">;

export const PORTFOLIO_OPERATIONS = [
  "holdings_load",
  "holding_create",
  "holding_update",
  "holding_close",
  "overview_load",
  "overview_toggle_aggregate",
  "activity_load",
  "activity_save_annotation",
  "activity_clear_annotation",
  "capture_load_status",
  "capture_save_schedule",
  "capture_start",
  "capture_apply",
] as const;

export type PortfolioOperation = (typeof PORTFOLIO_OPERATIONS)[number];

export const PORTFOLIO_CLOSED_IDS = {
  views: ["holdings", "activity", "account_details", "sync_records"],
  activityIntents: ["profit_take", "stop_loss", "rebalance", "thesis_broken", "cash_need", "other"],
  activitySources: ["broker", "manual", "system"],
  activityStates: ["realized_gain", "realized_loss", "realized_flat", "outcome_unknown", "unmatched", "manual_adjustment", "coverage_gap", "history_start"],
  activityKinds: ["order", "execution", "unmatched", "manual_adjustment", "coverage_gap", "history_start"],
  objectiveSides: ["buy", "sell", "mixed", "unknown"],
  objectiveOutcomes: ["gain", "loss", "flat", "unknown"],
  positionDirections: ["increase", "reduce", "unknown"],
  closeScopes: ["none", "partial", "complete", "unknown"],
  positionContexts: ["complete", "unknown"],
  grossNotionalKinds: ["deterministic_arithmetic"],
  executionCoverages: ["complete", "incomplete", "gap"],
  coverageReasons: ["execution_leg_incomplete", "broker_day_gap"],
  manualActions: ["create", "update", "close"],
  captureRunStates: ["running", "succeeded", "partial", "failed", "blocked", "interrupted"],
  captureTriggers: ["startup", "scheduled", "manual"],
  captureLegStates: ["not_attempted", "complete", "partial", "failed"],
} as const satisfies {
  views: readonly PortfolioView[];
  activityIntents: readonly PortfolioIntentLabel[];
  activitySources: readonly PortfolioActivitySource[];
  activityStates: readonly PortfolioActivityState[];
  activityKinds: readonly PortfolioActivityItem["kind"][];
  objectiveSides: readonly PortfolioActivityObjective["side"][];
  objectiveOutcomes: readonly PortfolioActivityObjective["realized_outcome"][];
  positionDirections: readonly PortfolioActivityObjective["position_direction"][];
  closeScopes: readonly PortfolioActivityObjective["close_scope"][];
  positionContexts: readonly PortfolioActivityObjective["position_context"][];
  grossNotionalKinds: readonly PortfolioActivityObjective["gross_notional_kind"][];
  executionCoverages: readonly PortfolioExecutionCoverage[];
  coverageReasons: readonly PortfolioCoverageReason[];
  manualActions: readonly PortfolioManualActivityItem["action"][];
  captureRunStates: readonly PortfolioCaptureRunState[];
  captureTriggers: readonly PortfolioCaptureRun["trigger"][];
  captureLegStates: readonly PortfolioCaptureLegState[];
};

export type PortfolioView = "holdings" | "activity" | "account_details" | "sync_records";
export type PortfolioExecutionCoverage = PortfolioUnmatchedActivityItem["execution_coverage"];
export type PortfolioCoverageReason = PortfolioCoverageGapActivityItem["reason_code"];
export type PortfolioCaptureLegState = "not_attempted" | "complete" | "partial" | "failed";

export interface PortfolioErrorState {
  readonly operation: PortfolioOperation;
  readonly category: "http" | "network" | "unknown";
  readonly status: number | null;
  readonly code: string | null;
  readonly routeTemplate: string | null;
}

export interface PortfolioErrorPresentation {
  title: string;
  diagnostics: Array<{ label: string; value: string }>;
}

const STABLE_CODE = /^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$/;
const SAFE_PATH = /^[\x21-\x7e]+$/;

function ownValue(value: unknown, field: PropertyKey): unknown {
  if ((typeof value !== "object" || value === null) && typeof value !== "function") {
    return undefined;
  }
  try {
    const descriptor = Object.getOwnPropertyDescriptor(value, field);
    return descriptor && "value" in descriptor ? descriptor.value : undefined;
  } catch {
    return undefined;
  }
}

function safeStatus(value: unknown): number | null {
  return typeof value === "number"
    && Number.isInteger(value)
    && value >= 100
    && value <= 599
    ? value
    : null;
}

function safeCode(value: unknown): string | null {
  return typeof value === "string"
    && value.length <= 64
    && STABLE_CODE.test(value)
    ? value
    : null;
}

function pathname(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const end = [value.indexOf("?"), value.indexOf("#")]
    .filter((index) => index >= 0)
    .reduce((left, right) => Math.min(left, right), value.length);
  const result = value.slice(0, end);
  return result.startsWith("/")
    && !result.startsWith("//")
    && result.length <= 160
    && SAFE_PATH.test(result)
    ? result
    : null;
}

function routeTemplate(operation: PortfolioOperation, path: string | null): string | null {
  if (path === null) return null;
  switch (operation) {
    case "holdings_load":
      return path === "/portfolio" ? "/portfolio" : null;
    case "holding_create":
      return path === "/portfolio/positions" ? "/portfolio/positions" : null;
    case "holding_update":
    case "holding_close":
      return /^\/portfolio\/positions\/[0-9]+$/u.test(path)
        ? "/portfolio/positions/{position_id}"
        : null;
    case "overview_load":
      return path === "/portfolio/overview" ? "/portfolio/overview" : null;
    case "overview_toggle_aggregate":
      return /^\/portfolio\/accounts\/[0-9]+$/u.test(path)
        ? "/portfolio/accounts/{account_id}"
        : null;
    case "activity_load":
      return path === "/portfolio/activity" ? "/portfolio/activity" : null;
    case "activity_save_annotation":
    case "activity_clear_annotation":
      return /^\/portfolio\/activity\/annotations\/[A-Za-z0-9%._~:-]+$/u.test(path)
        ? "/portfolio/activity/annotations/{activity_id}"
        : null;
    case "capture_load_status":
      return path === "/portfolio/capture" ? "/portfolio/capture" : null;
    case "capture_save_schedule":
      return path === "/portfolio/capture/settings" ? "/portfolio/capture/settings" : null;
    case "capture_start":
      return path === "/portfolio/capture/runs" ? "/portfolio/capture/runs" : null;
    case "capture_apply":
      return /^\/portfolio\/capture\/runs\/[0-9]+\/apply$/u.test(path)
        ? "/portfolio/capture/runs/{run_id}/apply"
        : null;
  }
}

function reviewedRouteTemplate(operation: PortfolioOperation, value: unknown): string | null {
  if (typeof value !== "string") return null;
  switch (operation) {
    case "holdings_load":
      return value === "/portfolio" ? value : null;
    case "holding_create":
      return value === "/portfolio/positions" ? value : null;
    case "holding_update":
    case "holding_close":
      return value === "/portfolio/positions/{position_id}" ? value : null;
    case "overview_load":
      return value === "/portfolio/overview" ? value : null;
    case "overview_toggle_aggregate":
      return value === "/portfolio/accounts/{account_id}" ? value : null;
    case "activity_load":
      return value === "/portfolio/activity" ? value : null;
    case "activity_save_annotation":
    case "activity_clear_annotation":
      return value === "/portfolio/activity/annotations/{activity_id}" ? value : null;
    case "capture_load_status":
      return value === "/portfolio/capture" ? value : null;
    case "capture_save_schedule":
      return value === "/portfolio/capture/settings" ? value : null;
    case "capture_start":
      return value === "/portfolio/capture/runs" ? value : null;
    case "capture_apply":
      return value === "/portfolio/capture/runs/{run_id}/apply" ? value : null;
  }
}

export function capturePortfolioError(
  operation: PortfolioOperation,
  error: unknown,
): PortfolioErrorState {
  const status = safeStatus(ownValue(error, "status"));
  const code = safeCode(ownValue(error, "code"));
  const route = routeTemplate(operation, pathname(ownValue(error, "path")));
  let isError = false;
  try {
    isError = error instanceof Error;
  } catch {
    isError = false;
  }
  const state: PortfolioErrorState = Object.freeze({
    operation,
    category: status !== null ? "http" : isError ? "network" : "unknown",
    status,
    code,
    routeTemplate: route,
  });
  return state;
}

function operationTitle(operation: PortfolioOperation, t: PortfolioT): string {
  switch (operation) {
    case "holdings_load": return t(($) => $.holdings.operations.holdingsLoad);
    case "holding_create": return t(($) => $.holdings.operations.holdingCreate);
    case "holding_update": return t(($) => $.holdings.operations.holdingUpdate);
    case "holding_close": return t(($) => $.holdings.operations.holdingClose);
    case "overview_load": return t(($) => $.accountOverview.operations.overviewLoad);
    case "overview_toggle_aggregate": return t(($) => $.accountOverview.operations.overviewToggleAggregate);
    case "activity_load": return t(($) => $.activity.operations.activityLoad);
    case "activity_save_annotation": return t(($) => $.activity.operations.activitySaveAnnotation);
    case "activity_clear_annotation": return t(($) => $.activity.operations.activityClearAnnotation);
    case "capture_load_status": return t(($) => $.capture.operations.captureLoadStatus);
    case "capture_save_schedule": return t(($) => $.capture.operations.captureSaveSchedule);
    case "capture_start": return t(($) => $.capture.operations.captureStart);
    case "capture_apply": return t(($) => $.capture.operations.captureApply);
  }
}

export function presentPortfolioError(
  state: PortfolioErrorState,
  t: PortfolioT,
  developerMode = false,
): PortfolioErrorPresentation {
  const diagnostics: Array<{ label: string; value: string }> = [];
  const status = developerMode ? safeStatus(ownValue(state, "status")) : null;
  const code = developerMode ? safeCode(ownValue(state, "code")) : null;
  const route = developerMode
    ? reviewedRouteTemplate(state.operation, ownValue(state, "routeTemplate"))
    : null;
  if (status !== null) {
    diagnostics.push({
      label: t(($) => $.capture.surface.runsStateHeader),
      value: String(status),
    });
  }
  if (code !== null) {
    diagnostics.push({ label: t(($) => $.capture.diagnostics.code), value: code });
  }
  if (route !== null) {
    diagnostics.push({ label: t(($) => $.capture.diagnostics.route), value: route });
  }
  return { title: operationTitle(state.operation, t), diagnostics };
}

export function portfolioValidationLabel(
  state: "ticker_quantity_required" | "quantity_nonzero" | "avg_cost_number" | "capture_interval",
  t: PortfolioT,
): string {
  switch (state) {
    case "ticker_quantity_required": return t(($) => $.holdings.validation.tickerQuantityRequired);
    case "quantity_nonzero": return t(($) => $.holdings.validation.quantityNonzero);
    case "avg_cost_number": return t(($) => $.holdings.validation.avgCostNumber);
    case "capture_interval": return t(($) => $.capture.validation.interval);
  }
}

export function portfolioEmptyStateLabel(
  state: "holdings" | "activity" | "capture_runs" | "accounts" | "account_snapshot",
  t: PortfolioT,
): string {
  switch (state) {
    case "holdings": return t(($) => $.holdings.empty.holdings);
    case "activity": return t(($) => $.activity.empty.activity);
    case "capture_runs": return t(($) => $.capture.empty.runs);
    case "accounts": return t(($) => $.accountOverview.empty.accounts);
    case "account_snapshot": return t(($) => $.accountOverview.empty.accountSnapshot);
  }
}

export function portfolioActivityFieldLabel(field: string, t: PortfolioT): string {
  switch (field) {
    case "quantity": return t(($) => $.activity.fields.quantity);
    case "avg_cost": return t(($) => $.activity.fields.avgCost);
    case "currency": return t(($) => $.activity.fields.currency);
    case "notes": return t(($) => $.activity.fields.notes);
    case "thesis": return t(($) => $.activity.fields.thesis);
    case "tags": return t(($) => $.activity.fields.tags);
    case "market_value": return t(($) => $.activity.fields.marketValue);
    case "unrealized_pnl": return t(($) => $.activity.fields.unrealizedPnl);
    default: return t(($) => $.activity.unknown.field);
  }
}

export function portfolioStableIdLabel(id: string, t: PortfolioT): string {
  return t(($) => $.activity.unknown.stableId, { id });
}

export function portfolioViewLabel(id: PortfolioView, t: PortfolioT): string {
  switch (id) {
    case "holdings": return t(($) => $.holdings.surface.viewHoldings);
    case "activity": return t(($) => $.holdings.surface.viewActivity);
    case "account_details": return t(($) => $.holdings.surface.viewAccountDetails);
    case "sync_records": return t(($) => $.holdings.surface.viewSyncRecords);
  }
}

export function portfolioActivityIntentLabel(id: PortfolioIntentLabel, t: PortfolioT): string {
  switch (id) {
    case "profit_take": return t(($) => $.activity.surface.intentProfitTake);
    case "stop_loss": return t(($) => $.activity.surface.intentStopLoss);
    case "rebalance": return t(($) => $.activity.surface.intentRebalance);
    case "thesis_broken": return t(($) => $.activity.surface.intentThesisBroken);
    case "cash_need": return t(($) => $.activity.surface.intentCashNeed);
    case "other": return t(($) => $.activity.surface.intentOther);
  }
}

export function portfolioActivitySourceLabel(id: PortfolioActivitySource, t: PortfolioT): string {
  switch (id) {
    case "broker": return t(($) => $.activity.surface.sourceBroker);
    case "manual": return t(($) => $.activity.surface.sourceManual);
    case "system": return t(($) => $.activity.surface.sourceSystem);
  }
}

export function portfolioActivityStateLabel(id: PortfolioActivityState, t: PortfolioT): string {
  switch (id) {
    case "realized_gain": return t(($) => $.activity.surface.outcomeGain);
    case "realized_loss": return t(($) => $.activity.surface.stateRealizedLoss);
    case "realized_flat": return t(($) => $.activity.surface.stateRealizedFlat);
    case "outcome_unknown": return t(($) => $.activity.surface.stateOutcomeUnknown);
    case "unmatched": return t(($) => $.activity.surface.stateUnmatched);
    case "manual_adjustment": return t(($) => $.activity.surface.stateManualAdjustment);
    case "coverage_gap": return t(($) => $.activity.surface.stateCoverageGap);
    case "history_start": return t(($) => $.activity.surface.stateHistoryStart);
  }
}

export function portfolioActivityKindLabel(id: PortfolioActivityItem["kind"], t: PortfolioT): string {
  switch (id) {
    case "order": return t(($) => $.activity.surface.eventOrder);
    case "execution": return t(($) => $.activity.surface.eventExecution);
    case "unmatched": return t(($) => $.activity.surface.eventUnmatched);
    case "manual_adjustment": return t(($) => $.activity.surface.stateManualAdjustment);
    case "coverage_gap": return t(($) => $.activity.surface.stateCoverageGap);
    case "history_start": return t(($) => $.activity.surface.eventHistoryStart);
  }
}

export function portfolioObjectiveSideLabel(id: PortfolioActivityObjective["side"], t: PortfolioT): string {
  switch (id) {
    case "buy": return t(($) => $.activity.surface.sideBuy);
    case "sell": return t(($) => $.activity.surface.sideSell);
    case "mixed": return t(($) => $.activity.surface.sideMixed);
    case "unknown": return t(($) => $.activity.surface.sideUnknown);
  }
}

export function portfolioObjectiveOutcomeLabel(
  id: PortfolioActivityObjective["realized_outcome"],
  t: PortfolioT,
): string {
  switch (id) {
    case "gain": return t(($) => $.activity.surface.outcomeGain);
    case "loss": return t(($) => $.activity.surface.outcomeLoss);
    case "flat": return t(($) => $.activity.surface.outcomeFlat);
    case "unknown": return t(($) => $.activity.surface.outcomeUnknown);
  }
}

export function portfolioPositionDirectionLabel(
  id: PortfolioActivityObjective["position_direction"],
  t: PortfolioT,
): string {
  switch (id) {
    case "increase": return t(($) => $.activity.surface.positionDirectionIncrease);
    case "reduce": return t(($) => $.activity.surface.positionDirectionReduce);
    case "unknown": return t(($) => $.activity.surface.unknown);
  }
}

export function portfolioCloseScopeLabel(id: PortfolioActivityObjective["close_scope"], t: PortfolioT): string {
  switch (id) {
    case "none": return t(($) => $.activity.surface.closeScopeNone);
    case "partial": return t(($) => $.activity.surface.closeScopePartial);
    case "complete": return t(($) => $.activity.surface.closeScopeComplete);
    case "unknown": return t(($) => $.activity.surface.unknown);
  }
}

export function portfolioPositionContextLabel(
  id: PortfolioActivityObjective["position_context"],
  t: PortfolioT,
): string {
  switch (id) {
    case "complete": return t(($) => $.activity.surface.positionContextComplete);
    case "unknown": return t(($) => $.activity.surface.unknown);
  }
}

export function portfolioGrossNotionalKindLabel(
  id: PortfolioActivityObjective["gross_notional_kind"],
  t: PortfolioT,
): string {
  switch (id) {
    case "deterministic_arithmetic": return t(($) => $.activity.surface.grossNotionalKind);
  }
}

export function portfolioExecutionCoverageLabel(id: PortfolioExecutionCoverage, t: PortfolioT): string {
  switch (id) {
    case "complete": return t(($) => $.activity.surface.executionCoverageComplete);
    case "incomplete": return t(($) => $.activity.surface.objectiveCoverageIncomplete);
    case "gap": return t(($) => $.activity.surface.stateCoverageGap);
  }
}

export function portfolioCoverageReasonLabel(id: PortfolioCoverageReason, t: PortfolioT): string {
  switch (id) {
    case "execution_leg_incomplete": return t(($) => $.activity.surface.eventExecutionIncomplete);
    case "broker_day_gap": return t(($) => $.activity.surface.eventBrokerGap);
  }
}

export function portfolioManualActionLabel(id: PortfolioManualActivityItem["action"], t: PortfolioT): string {
  switch (id) {
    case "create": return t(($) => $.activity.surface.manualCreate);
    case "update": return t(($) => $.activity.surface.manualUpdate);
    case "close": return t(($) => $.activity.surface.manualClose);
  }
}

export function portfolioCaptureRunStateLabel(id: PortfolioCaptureRunState, t: PortfolioT): string {
  switch (id) {
    case "running": return t(($) => $.capture.surface.runRunning);
    case "succeeded": return t(($) => $.capture.surface.runSucceeded);
    case "partial": return t(($) => $.capture.surface.runPartial);
    case "failed": return t(($) => $.capture.surface.runFailed);
    case "blocked": return t(($) => $.capture.surface.runBlocked);
    case "interrupted": return t(($) => $.capture.surface.runInterrupted);
  }
}

export function portfolioCaptureRunDetailLabel(id: PortfolioCaptureRunState, t: PortfolioT): string {
  switch (id) {
    case "running": return t(($) => $.capture.surface.detailInformation);
    case "succeeded": return t(($) => $.capture.surface.detailInformation);
    case "partial": return t(($) => $.capture.outcomes.capturePartial);
    case "failed": return t(($) => $.capture.surface.detailFailed);
    case "blocked": return t(($) => $.capture.outcomes.captureBlocked);
    case "interrupted": return t(($) => $.capture.surface.detailInterrupted);
  }
}

export function portfolioCaptureTriggerLabel(id: PortfolioCaptureRun["trigger"], t: PortfolioT): string {
  switch (id) {
    case "startup": return t(($) => $.capture.surface.triggerStartup);
    case "scheduled": return t(($) => $.capture.surface.triggerScheduled);
    case "manual": return t(($) => $.capture.surface.triggerManual);
  }
}

export function portfolioCaptureLegStateLabel(id: PortfolioCaptureLegState, t: PortfolioT): string {
  switch (id) {
    case "not_attempted": return t(($) => $.capture.surface.legNotAttempted);
    case "complete": return t(($) => $.capture.surface.legComplete);
    case "partial": return t(($) => $.capture.surface.runPartial);
    case "failed": return t(($) => $.capture.surface.runFailed);
  }
}

export function portfolioCountCopy(
  kind: "holdings" | "activity" | "review_changes" | "recent_fields",
  count: number,
  t: PortfolioT,
): string {
  switch (kind) {
    case "holdings": return count === 1
      ? t(($) => $.holdings.count.one, { count })
      : t(($) => $.holdings.count.other, { count });
    case "activity": return count === 1
      ? t(($) => $.activity.count.one, { count })
      : t(($) => $.activity.count.other, { count });
    case "review_changes": return count === 1
      ? t(($) => $.capture.reviewCount.one, { count })
      : t(($) => $.capture.reviewCount.other, { count });
    case "recent_fields": return count === 1
      ? t(($) => $.recentActivity.fieldCount.one, { count })
      : t(($) => $.recentActivity.fieldCount.other, { count });
  }
}

export function portfolioOutcomeLabel(
  outcome: "schedule_saved" | "capture_applied" | "capture_partial" | "capture_blocked",
  t: PortfolioT,
): string {
  switch (outcome) {
    case "schedule_saved": return t(($) => $.capture.outcomes.scheduleSaved);
    case "capture_applied": return t(($) => $.capture.outcomes.captureApplied);
    case "capture_partial": return t(($) => $.capture.outcomes.capturePartial);
    case "capture_blocked": return t(($) => $.capture.outcomes.captureBlocked);
  }
}

export function preservePortfolioFacts<T extends {
  source: string;
  userValue: string;
  stableId: string;
  measuredValue: number;
}>(facts: T): T {
  return facts;
}
