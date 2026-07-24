import { readFileSync } from "node:fs";

import { createInstance, type TFunction } from "i18next";
import { describe, expect, it } from "vitest";

import { initializeI18n } from "./resources";

type Locale = "zh-Hant" | "en";
type PortfolioT = TFunction<"portfolio">;

const OPERATIONS = [
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

type Operation = (typeof OPERATIONS)[number];

const CLOSED_IDS = {
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
} as const;

type ClosedIds = typeof CLOSED_IDS;

interface PortfolioPresentationModule {
  PORTFOLIO_OPERATIONS: readonly Operation[];
  PORTFOLIO_CLOSED_IDS: ClosedIds;
  capturePortfolioError: (operation: Operation, error: unknown) => {
    operation: Operation;
    category: "http" | "network" | "unknown";
    status: number | null;
    code: string | null;
    routeTemplate: string | null;
  };
  presentPortfolioError: (
    state: ReturnType<PortfolioPresentationModule["capturePortfolioError"]>,
    t: PortfolioT,
    developerMode?: boolean,
  ) => {
    title: string;
    diagnostics: Array<{ label: string; value: string }>;
  };
  portfolioValidationLabel: (
    state: "ticker_quantity_required" | "quantity_nonzero" | "avg_cost_number" | "capture_interval",
    t: PortfolioT,
  ) => string;
  portfolioEmptyStateLabel: (
    state: "holdings" | "activity" | "capture_runs" | "accounts" | "account_snapshot",
    t: PortfolioT,
  ) => string;
  portfolioActivityFieldLabel: (field: string, t: PortfolioT) => string;
  portfolioStableIdLabel: (id: string, t: PortfolioT) => string;
  portfolioViewLabel: (id: ClosedIds["views"][number], t: PortfolioT) => string;
  portfolioActivityIntentLabel: (id: ClosedIds["activityIntents"][number], t: PortfolioT) => string;
  portfolioActivitySourceLabel: (id: ClosedIds["activitySources"][number], t: PortfolioT) => string;
  portfolioActivityStateLabel: (id: ClosedIds["activityStates"][number], t: PortfolioT) => string;
  portfolioActivityKindLabel: (id: ClosedIds["activityKinds"][number], t: PortfolioT) => string;
  portfolioObjectiveSideLabel: (id: ClosedIds["objectiveSides"][number], t: PortfolioT) => string;
  portfolioObjectiveOutcomeLabel: (id: ClosedIds["objectiveOutcomes"][number], t: PortfolioT) => string;
  portfolioPositionDirectionLabel: (id: ClosedIds["positionDirections"][number], t: PortfolioT) => string;
  portfolioCloseScopeLabel: (id: ClosedIds["closeScopes"][number], t: PortfolioT) => string;
  portfolioPositionContextLabel: (id: ClosedIds["positionContexts"][number], t: PortfolioT) => string;
  portfolioGrossNotionalKindLabel: (id: ClosedIds["grossNotionalKinds"][number], t: PortfolioT) => string;
  portfolioExecutionCoverageLabel: (id: ClosedIds["executionCoverages"][number], t: PortfolioT) => string;
  portfolioCoverageReasonLabel: (id: ClosedIds["coverageReasons"][number], t: PortfolioT) => string;
  portfolioManualActionLabel: (id: ClosedIds["manualActions"][number], t: PortfolioT) => string;
  portfolioCaptureRunStateLabel: (id: ClosedIds["captureRunStates"][number], t: PortfolioT) => string;
  portfolioCaptureRunDetailLabel: (id: ClosedIds["captureRunStates"][number], t: PortfolioT) => string;
  portfolioCaptureTriggerLabel: (id: ClosedIds["captureTriggers"][number], t: PortfolioT) => string;
  portfolioCaptureLegStateLabel: (id: ClosedIds["captureLegStates"][number], t: PortfolioT) => string;
  portfolioCountCopy: (
    kind: "holdings" | "activity" | "review_changes" | "recent_fields",
    count: number,
    t: PortfolioT,
  ) => string;
  portfolioOutcomeLabel: (
    outcome: "schedule_saved" | "capture_applied" | "capture_partial" | "capture_blocked",
    t: PortfolioT,
  ) => string;
  preservePortfolioFacts: <T extends {
    source: string;
    userValue: string;
    stableId: string;
    measuredValue: number;
  }>(facts: T) => T;
}

function closedLabels(presentation: PortfolioPresentationModule, t: PortfolioT): string[] {
  return [
    ...CLOSED_IDS.views.map((id) => presentation.portfolioViewLabel(id, t)),
    ...CLOSED_IDS.activityIntents.map((id) => presentation.portfolioActivityIntentLabel(id, t)),
    ...CLOSED_IDS.activitySources.map((id) => presentation.portfolioActivitySourceLabel(id, t)),
    ...CLOSED_IDS.activityStates.map((id) => presentation.portfolioActivityStateLabel(id, t)),
    ...CLOSED_IDS.activityKinds.map((id) => presentation.portfolioActivityKindLabel(id, t)),
    ...CLOSED_IDS.objectiveSides.map((id) => presentation.portfolioObjectiveSideLabel(id, t)),
    ...CLOSED_IDS.objectiveOutcomes.map((id) => presentation.portfolioObjectiveOutcomeLabel(id, t)),
    ...CLOSED_IDS.positionDirections.map((id) => presentation.portfolioPositionDirectionLabel(id, t)),
    ...CLOSED_IDS.closeScopes.map((id) => presentation.portfolioCloseScopeLabel(id, t)),
    ...CLOSED_IDS.positionContexts.map((id) => presentation.portfolioPositionContextLabel(id, t)),
    ...CLOSED_IDS.grossNotionalKinds.map((id) => presentation.portfolioGrossNotionalKindLabel(id, t)),
    ...CLOSED_IDS.executionCoverages.map((id) => presentation.portfolioExecutionCoverageLabel(id, t)),
    ...CLOSED_IDS.coverageReasons.map((id) => presentation.portfolioCoverageReasonLabel(id, t)),
    ...CLOSED_IDS.manualActions.map((id) => presentation.portfolioManualActionLabel(id, t)),
    ...CLOSED_IDS.captureRunStates.map((id) => presentation.portfolioCaptureRunStateLabel(id, t)),
    ...CLOSED_IDS.captureTriggers.map((id) => presentation.portfolioCaptureTriggerLabel(id, t)),
    ...CLOSED_IDS.captureLegStates.map((id) => presentation.portfolioCaptureLegStateLabel(id, t)),
  ];
}

const modulePath = "./portfolioPresentation";

function loadPresentation(): Promise<PortfolioPresentationModule> {
  return import(/* @vite-ignore */ modulePath) as Promise<PortfolioPresentationModule>;
}

function portfolioT(locale: Locale): PortfolioT {
  const instance = createInstance();
  initializeI18n(instance, locale);
  return instance.getFixedT(locale, "portfolio");
}

function flattenStrings(value: unknown): string[] {
  if (typeof value === "string") return [value];
  if (Array.isArray(value)) return value.flatMap(flattenStrings);
  if (value && typeof value === "object") {
    return Object.values(value as Record<string, unknown>).flatMap(flattenStrings);
  }
  return [];
}

describe("portfolio presentation", () => {
  it("maps every Portfolio operation outcome in both locales", async () => {
    const presentation = await loadPresentation();
    const zh = portfolioT("zh-Hant");
    const en = portfolioT("en");
    expect(presentation.PORTFOLIO_OPERATIONS).toEqual(OPERATIONS);
    expect(presentation.PORTFOLIO_CLOSED_IDS).toEqual(CLOSED_IDS);
    const expectedTitles: Record<Operation, { "zh-Hant": string; en: string }> = {
      holdings_load: { "zh-Hant": "持倉載入失敗", en: "Could not load holdings" },
      holding_create: { "zh-Hant": "新增持倉失敗", en: "Could not add holding" },
      holding_update: { "zh-Hant": "持倉更新失敗", en: "Could not update holding" },
      holding_close: { "zh-Hant": "關閉持倉失敗", en: "Could not close holding" },
      overview_load: { "zh-Hant": "帳戶總覽無法載入；持倉仍可使用", en: "Could not load account overview; holdings remain available" },
      overview_toggle_aggregate: { "zh-Hant": "帳戶總計設定更新失敗", en: "Could not update account total setting" },
      activity_load: { "zh-Hant": "活動載入失敗；請重新整理", en: "Could not load activity. Refresh to try again" },
      activity_save_annotation: { "zh-Hant": "註記未儲存；請重試", en: "Annotation was not saved. Try again" },
      activity_clear_annotation: { "zh-Hant": "註記未清除；請重試", en: "Annotation was not cleared. Try again" },
      capture_load_status: { "zh-Hant": "同步狀態載入失敗", en: "Could not load sync status" },
      capture_save_schedule: { "zh-Hant": "排程儲存失敗", en: "Could not save schedule" },
      capture_start: { "zh-Hant": "持倉同步失敗", en: "Could not start holdings sync" },
      capture_apply: { "zh-Hant": "套用同步失敗", en: "Could not apply sync" },
    };
    for (const operation of OPERATIONS) {
      const state = presentation.capturePortfolioError(operation, new Error("planted raw"));
      expect(presentation.presentPortfolioError(state, zh).title).toBe(expectedTitles[operation]["zh-Hant"]);
      expect(presentation.presentPortfolioError(state, en).title).toBe(expectedTitles[operation].en);
    }
    expect(closedLabels(presentation, zh)).toHaveLength(66);
    expect(closedLabels(presentation, en)).toHaveLength(66);
  });

  it("maps validation and empty states without parsing backend text", async () => {
    const presentation = await loadPresentation();
    const en = portfolioT("en");
    expect(presentation.portfolioValidationLabel("ticker_quantity_required", en))
      .toBe("Ticker and a non-zero quantity are required");
    expect(presentation.portfolioValidationLabel("capture_interval", en))
      .toBe("Interval must be a whole number from 5 to 1440 minutes");
    expect(presentation.portfolioEmptyStateLabel("activity", en)).toBe("No activity records");
    expect(presentation.portfolioEmptyStateLabel("account_snapshot", en)).toBe("No account snapshot yet");
    const source = readFileSync(new URL("./portfolioPresentation.ts", import.meta.url), "utf8");
    expect(source).not.toMatch(/\.message\b/u);
  });

  it("exposes only reviewed safe ApiError fields in Developer Mode", async () => {
    const presentation = await loadPresentation();
    let messageReads = 0;
    const error = {} as Record<string, unknown>;
    Object.defineProperties(error, {
      message: { get: () => { messageReads += 1; throw new Error("message read"); } },
      status: { value: 503 },
      code: { value: "activity_temporarily_unavailable" },
      path: { value: "/portfolio/activity?token=secret#private" },
      diagnostic: { value: "traceback SQL /home/user <script>\u0001" },
    });
    const state = presentation.capturePortfolioError("activity_load", error);
    expect(messageReads).toBe(0);
    expect(state).toEqual({
      operation: "activity_load",
      category: "http",
      status: 503,
      code: "activity_temporarily_unavailable",
      routeTemplate: "/portfolio/activity",
    });
    const presented = presentation.presentPortfolioError(state, portfolioT("en"), true);
    expect(presented.diagnostics).toEqual([
      { label: "Status", value: "503" },
      { label: "Code", value: "activity_temporarily_unavailable" },
      { label: "Route", value: "/portfolio/activity" },
    ]);
    const copiedState = Object.freeze({ ...state });
    expect(presentation.presentPortfolioError(copiedState, portfolioT("en"), true))
      .toEqual(presented);
    expect(presentation.capturePortfolioError("activity_save_annotation", {
      status: 409,
      path: "/portfolio/activity/annotations/order%3Aabc-123?raw=secret",
    }).routeTemplate).toBe("/portfolio/activity/annotations/{activity_id}");
    expect(presentation.capturePortfolioError("capture_load_status", {
      status: 503,
      path: "/portfolio/capture",
    }).routeTemplate).toBe("/portfolio/capture");
    expect(presentation.capturePortfolioError("capture_start", {
      status: 503,
      path: "/portfolio/capture/runs",
    }).routeTemplate).toBe("/portfolio/capture/runs");
    const apiError = Object.assign(new Error("raw backend message"), {
      path: "/portfolio/capture/runs/42/apply?token=secret",
      status: 409,
      code: "capture_review_conflict",
      diagnostic: "traceback SQL /home/user",
    });
    expect(presentation.capturePortfolioError("capture_apply", apiError)).toEqual({
      operation: "capture_apply",
      category: "http",
      status: 409,
      code: "capture_review_conflict",
      routeTemplate: "/portfolio/capture/runs/{run_id}/apply",
    });
    const forgedState = {
      operation: "capture_apply",
      category: "http",
      status: 999,
      code: "traceback SQL /home/user <script>\u0001",
      routeTemplate: "/portfolio/capture/runs/{run_id}/apply?token=secret",
    } as ReturnType<PortfolioPresentationModule["capturePortfolioError"]>;
    expect(presentation.presentPortfolioError(forgedState, portfolioT("en"), true).diagnostics)
      .toEqual([]);
  });

  it("omits arbitrary error details in normal mode", async () => {
    const presentation = await loadPresentation();
    const hostile = "traceback SQL token=/home/user?q=secret <script>\u0001";
    const state = presentation.capturePortfolioError("capture_start", {
      status: 500,
      code: "capture_failed",
      path: "/portfolio/capture?token=secret",
      diagnostic: hostile,
      message: hostile,
    });
    const output = presentation.presentPortfolioError(state, portfolioT("en"));
    expect(output.diagnostics).toEqual([]);
    expect(flattenStrings(output).join(" ")).not.toMatch(/traceback|SQL|token|secret|script|home\/user/u);
  });

  it("maps activity field IDs to local labels", async () => {
    const presentation = await loadPresentation();
    const fields = ["quantity", "avg_cost", "currency", "notes", "thesis", "tags", "market_value", "unrealized_pnl"];
    expect(fields.map((field) => presentation.portfolioActivityFieldLabel(field, portfolioT("zh-Hant"))))
      .toEqual(["數量", "均價", "幣別", "筆記", "投資論點", "標籤", "市值", "未實現損益"]);
    expect(fields.map((field) => presentation.portfolioActivityFieldLabel(field, portfolioT("en"))))
      .toEqual(["Quantity", "Average cost", "Currency", "Notes", "Thesis", "Tags", "Market value", "Unrealized P&L"]);
    expect(CLOSED_IDS.activityIntents.map((id) => presentation.portfolioActivityIntentLabel(id, portfolioT("en"))))
      .toEqual(["Take profit", "Stop loss", "Rebalance", "Thesis invalidated", "Cash need", "Other"]);
    expect(CLOSED_IDS.activitySources.map((id) => presentation.portfolioActivitySourceLabel(id, portfolioT("zh-Hant"))))
      .toEqual(["Broker", "手動紀錄", "系統覆蓋"]);
    expect(CLOSED_IDS.activityKinds.map((id) => presentation.portfolioActivityKindLabel(id, portfolioT("en"))))
      .toEqual(["Order filled", "Standalone execution", "Unmatched change", "Manual adjustment", "Coverage gap", "Activity history start"]);
    expect(CLOSED_IDS.objectiveSides.map((id) => presentation.portfolioObjectiveSideLabel(id, portfolioT("zh-Hant"))))
      .toEqual(["買進", "賣出", "混合", "方向未知"]);
    expect(CLOSED_IDS.executionCoverages.map((id) => presentation.portfolioExecutionCoverageLabel(id, portfolioT("en"))))
      .toEqual(["Complete coverage", "Coverage incomplete", "Coverage gap"]);
    expect(CLOSED_IDS.manualActions.map((id) => presentation.portfolioManualActionLabel(id, portfolioT("en"))))
      .toEqual(["Create", "Update", "Close"]);
    expect(CLOSED_IDS.positionContexts.map((id) => presentation.portfolioPositionContextLabel(id, portfolioT("zh-Hant"))))
      .toEqual(["完整", "未知"]);
    expect(CLOSED_IDS.positionContexts.map((id) => presentation.portfolioPositionContextLabel(id, portfolioT("en"))))
      .toEqual(["Complete", "Unknown"]);
    expect(CLOSED_IDS.grossNotionalKinds.map((id) => presentation.portfolioGrossNotionalKindLabel(id, portfolioT("zh-Hant"))))
      .toEqual(["確定性算術"]);
    expect(CLOSED_IDS.grossNotionalKinds.map((id) => presentation.portfolioGrossNotionalKindLabel(id, portfolioT("en"))))
      .toEqual(["Deterministic arithmetic"]);
  });

  it("keeps unknown stable IDs visible and distinguishable", async () => {
    const presentation = await loadPresentation();
    const t = portfolioT("en");
    const stableIds = ["future_state_alpha", "future_state_beta"]
      .map((id) => presentation.portfolioStableIdLabel(id, t));
    expect(stableIds).toEqual(["Unknown ID: future_state_alpha", "Unknown ID: future_state_beta"]);
    expect(new Set(stableIds).size).toBe(2);

    const schemaFields = ["raw_schema_alpha", "raw_schema_beta"]
      .map((field) => presentation.portfolioActivityFieldLabel(field, t));
    expect(schemaFields).toEqual(["Unknown field", "Unknown field"]);
    expect(schemaFields.join(" ")).not.toMatch(/raw_schema_(?:alpha|beta)/u);
  });

  it("selects reviewed one and other count copy", async () => {
    const presentation = await loadPresentation();
    const en = portfolioT("en");
    const zh = portfolioT("zh-Hant");
    expect(presentation.portfolioCountCopy("holdings", 1, en)).toBe("1 holding");
    expect(presentation.portfolioCountCopy("holdings", 2, en)).toBe("2 holdings");
    expect(presentation.portfolioCountCopy("activity", 1, en)).toBe("1 activity record loaded");
    expect(presentation.portfolioCountCopy("review_changes", 3, zh)).toBe("3 項變更");
    expect(presentation.portfolioCountCopy("recent_fields", 1, en)).toBe("1 field");
  });

  it("renders late outcomes in the active locale", async () => {
    const presentation = await loadPresentation();
    expect(presentation.portfolioOutcomeLabel("schedule_saved", portfolioT("zh-Hant"))).toBe("排程已儲存");
    expect(presentation.portfolioOutcomeLabel("schedule_saved", portfolioT("en"))).toBe("Schedule saved");
    expect(presentation.portfolioOutcomeLabel("capture_applied", portfolioT("en"))).toBe("Sync changes applied");
    expect(CLOSED_IDS.captureRunStates.map((id) => presentation.portfolioCaptureRunStateLabel(id, portfolioT("en"))))
      .toEqual(["Running", "Succeeded", "Partially completed", "Failed", "Blocked", "Aborted"]);
    expect(CLOSED_IDS.captureRunStates.map((id) => presentation.portfolioCaptureRunDetailLabel(id, portfolioT("zh-Hant"))))
      .toEqual(["同步資訊", "同步資訊", "同步資料不完整", "同步失敗", "同步已阻擋", "同步已中止"]);
    expect(CLOSED_IDS.captureTriggers.map((id) => presentation.portfolioCaptureTriggerLabel(id, portfolioT("zh-Hant"))))
      .toEqual(["啟動補抓", "排程", "手動"]);
    expect(CLOSED_IDS.captureLegStates.map((id) => presentation.portfolioCaptureLegStateLabel(id, portfolioT("en"))))
      .toEqual(["Not run", "Complete", "Partially completed", "Failed"]);
  });

  it("preserves source user and measured values", async () => {
    const presentation = await loadPresentation();
    const facts = {
      source: "IBKR / 原始來源",
      userValue: "keep <b>my note</b>\u0001 exactly",
      stableId: "DU:acct/原值",
      measuredValue: -1234.56789,
      count: 0,
    };
    expect(presentation.preservePortfolioFacts(facts)).toBe(facts);
    expect(presentation.preservePortfolioFacts(facts)).toEqual(facts);
  });

  it("uses only static Portfolio resource selectors", () => {
    const source = readFileSync(new URL("./portfolioPresentation.ts", import.meta.url), "utf8");
    expect(source).not.toMatch(/\bt\s*\(\s*["'`]/u);
    expect(source).not.toMatch(/\bt\s*\(\s*\([^)]*\)\s*=>\s*\$\s*\[/u);
    expect(source).not.toMatch(/\bt\s*\(\s*`/u);
    expect(source).not.toMatch(/\$\s*\[/u);
    const activitySemanticSelectors = ["portfolioObjectiveSideLabel", "portfolioManualActionLabel"]
      .map((name) => source.slice(source.indexOf(`export function ${name}`), source.indexOf("\n}\n", source.indexOf(`export function ${name}`)) + 3))
      .join("\n");
    expect(activitySemanticSelectors).toContain("$.activity.");
    expect(activitySemanticSelectors).not.toContain("$.recentActivity.");
    expect(source.match(/\$\.recentActivity\./gu)).toHaveLength(2);
    expect(source.match(/\$\.recentActivity\.fieldCount\.(?:one|other)/gu)).toHaveLength(2);
  });

  it("covers both locales for every closed operation branch", async () => {
    const presentation = await loadPresentation();
    expect(new Set(presentation.PORTFOLIO_OPERATIONS).size).toBe(13);
    expect(Object.fromEntries(Object.entries(presentation.PORTFOLIO_CLOSED_IDS)
      .map(([family, ids]) => [family, ids.length])))
      .toEqual(Object.fromEntries(Object.entries(CLOSED_IDS).map(([family, ids]) => [family, ids.length])));
    for (const locale of ["zh-Hant", "en"] as const) {
      const t = portfolioT(locale);
      const titles = presentation.PORTFOLIO_OPERATIONS.map((operation) =>
        presentation.presentPortfolioError(
          presentation.capturePortfolioError(operation, null),
          t,
        ).title);
      expect(titles).toHaveLength(13);
      expect(new Set(titles).size).toBe(13);
      const labels = closedLabels(presentation, t);
      expect(labels).toHaveLength(66);
      expect(labels.every((label) => label.trim().length > 0)).toBe(true);
      expect(CLOSED_IDS.captureRunStates
        .map((id) => presentation.portfolioCaptureRunDetailLabel(id, t))
        .every((label) => label.trim().length > 0)).toBe(true);
    }
  });

  it("never returns a raw resource key", async () => {
    const presentation = await loadPresentation();
    for (const locale of ["zh-Hant", "en"] as const) {
      const t = portfolioT(locale);
      const outputs: unknown[] = [
        ...OPERATIONS.map((operation) => presentation.presentPortfolioError(
          presentation.capturePortfolioError(operation, null), t, true,
        )),
        ...["quantity", "avg_cost", "currency", "notes", "thesis", "tags", "market_value", "unrealized_pnl"]
          .map((field) => presentation.portfolioActivityFieldLabel(field, t)),
        ...["holdings", "activity", "review_changes", "recent_fields"]
          .flatMap((kind) => [1, 2].map((count) => presentation.portfolioCountCopy(
            kind as "holdings" | "activity" | "review_changes" | "recent_fields", count, t,
          ))),
        ...["schedule_saved", "capture_applied", "capture_partial", "capture_blocked"]
          .map((outcome) => presentation.portfolioOutcomeLabel(
            outcome as "schedule_saved" | "capture_applied" | "capture_partial" | "capture_blocked", t,
          )),
        ...closedLabels(presentation, t),
        ...CLOSED_IDS.captureRunStates.map((id) => presentation.portfolioCaptureRunDetailLabel(id, t)),
      ];
      expect(flattenStrings(outputs).filter((value) => /^(?:portfolio:)?(?:presentation|holdings|activity|capture|accountOverview|recentActivity|tableLabels)\./u.test(value)))
        .toEqual([]);
    }
  });
});
