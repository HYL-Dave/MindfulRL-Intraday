import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Check, RefreshCw, Save } from "lucide-react";
import { useTranslation } from "react-i18next";

import {
  applyPortfolioCaptureRun,
  getPortfolioCaptureStatus,
  triggerPortfolioCapture,
  updatePortfolioCaptureSettings,
  type PortfolioCaptureReviewChange,
  type PortfolioCaptureRun,
  type PortfolioCaptureRunState,
  type PortfolioCaptureStatus,
} from "./api";
import {
  capturePortfolioError,
  portfolioCaptureLegStateLabel,
  portfolioCaptureRunDetailLabel,
  portfolioCaptureRunStateLabel,
  portfolioCaptureTriggerLabel,
  portfolioCountCopy,
  portfolioEmptyStateLabel,
  portfolioOutcomeLabel,
  presentPortfolioError,
  type PortfolioErrorState,
  type PortfolioOperation,
  type PortfolioT,
} from "./i18n/portfolioPresentation";
import {
  Button,
  DataTable,
  InlineAlert,
  StatusBadge,
  type CommonUiState,
  type DataTableColumn,
} from "./ui";

const IDLE_POLL_MS = 30_000;
const RUNNING_POLL_MS = 2_000;
const SCHEDULE_SAVED_NOTICE = "schedule_saved" as const;
const CAPTURE_APPLIED_NOTICE = "capture_applied" as const;
const CAPTURE_LOAD_OPERATION: PortfolioOperation = "capture_load_status";
const CAPTURE_SAVE_OPERATION: PortfolioOperation = "capture_save_schedule";
const CAPTURE_START_OPERATION: PortfolioOperation = "capture_start";
const CAPTURE_APPLY_OPERATION: PortfolioOperation = "capture_apply";

function runUiState(state: PortfolioCaptureRunState): CommonUiState {
  return state === "succeeded" ? "ready" : state;
}

function formatLocalTime(value?: string | null): string {
  if (!value) return "-";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function captureLegStateLabel(value: string, t: PortfolioT): string {
  switch (value) {
    case "not_attempted":
    case "complete":
    case "partial":
    case "failed":
      return portfolioCaptureLegStateLabel(value, t);
    default:
      return value;
  }
}

export function parseCaptureInterval(raw: string): number | null {
  const text = raw.trim();
  if (!text) return null;
  const value = Number(text);
  return Number.isInteger(value) && value >= 5 && value <= 1440 ? value : null;
}

function isCaptureStatus(value: unknown): value is PortfolioCaptureStatus {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<PortfolioCaptureStatus>;
  return Boolean(
    candidate.settings &&
    typeof candidate.settings.enabled === "boolean" &&
    Number.isFinite(candidate.settings.interval_minutes) &&
    Array.isArray(candidate.recent_runs),
  );
}

function formatReviewMetric(
  change: PortfolioCaptureReviewChange,
  field: string,
  fallback?: number,
): string {
  const beforeValue = change.before?.[field];
  const before = typeof beforeValue === "number" ? beforeValue : null;
  const hasAfterValue = change.after != null && Object.prototype.hasOwnProperty.call(change.after, field);
  const afterValue = hasAfterValue ? change.after?.[field] : fallback;
  const after = typeof afterValue === "number" ? afterValue : null;
  const format = (value: number) => new Intl.NumberFormat(undefined, { maximumFractionDigits: 4 }).format(value);
  const display = (value: number | null) => value == null ? "-" : format(value);
  if (change.kind === "update" && before !== after) {
    return `${display(before)} → ${display(after)}`;
  }
  const value = change.kind === "remove" ? before : (after ?? before);
  return value == null ? "-" : format(value);
}

type CaptureCopyIssue =
  | "status_shape"
  | "schedule_invalid"
  | "settings_shape";

type CaptureIssue = {
  state: CommonUiState;
  kind: "operation";
  error: PortfolioErrorState;
} | {
  state: CommonUiState;
  kind: "copy";
  copy: CaptureCopyIssue;
} | {
  state: CommonUiState;
  kind: "run";
  runState: PortfolioCaptureRunState;
};

function presentCaptureIssue(issue: CaptureIssue, t: PortfolioT) {
  if (issue.kind === "operation") {
    return { ...presentPortfolioError(issue.error, t), message: null };
  }
  if (issue.kind === "run") {
    return {
      title: portfolioCaptureRunDetailLabel(issue.runState, t),
      diagnostics: [],
      message: null,
    };
  }
  switch (issue.copy) {
    case "status_shape":
      return {
        title: presentPortfolioError(capturePortfolioError(CAPTURE_LOAD_OPERATION, null), t).title,
        diagnostics: [],
        message: t(($) => $.capture.surface.statusShapeError),
      };
    case "schedule_invalid":
      return {
        title: t(($) => $.capture.surface.scheduleInvalid),
        diagnostics: [],
        message: t(($) => $.capture.validation.interval),
      };
    case "settings_shape":
      return {
        title: presentPortfolioError(capturePortfolioError(CAPTURE_SAVE_OPERATION, null), t).title,
        diagnostics: [],
        message: t(($) => $.capture.surface.settingsShapeError),
      };
  }
}

function statusWithRun(
  current: PortfolioCaptureStatus,
  run: PortfolioCaptureRun,
): PortfolioCaptureStatus {
  return {
    ...current,
    running: run.state === "running",
    latest_run: run,
    recent_runs: [run, ...current.recent_runs.filter((item) => item.id !== run.id)].slice(0, 20),
  };
}

function shouldProjectRun(
  current: PortfolioCaptureRun | null | undefined,
  candidate: PortfolioCaptureRun,
) {
  if (!current) return true;
  if (candidate.id !== current.id) return candidate.id > current.id;
  return current.state === "running" && candidate.state !== "running";
}

export function PortfolioCapturePanel({
  onPortfolioChanged,
}: {
  onPortfolioChanged: () => void | Promise<void>;
}) {
  const { t } = useTranslation("portfolio");
  const [capture, setCapture] = useState<PortfolioCaptureStatus | null>(null);
  const [enabled, setEnabled] = useState(false);
  const [interval, setIntervalValue] = useState("15");
  const [busy, setBusy] = useState<"save" | "capture" | "apply" | null>(null);
  const [issue, setIssue] = useState<CaptureIssue | null>(null);
  const [notice, setNotice] = useState<"schedule_saved" | "capture_applied" | null>(null);
  const dirtyRef = useRef(false);
  const initializedRef = useRef(false);
  const lastTerminalRunIdRef = useRef<number | null>(null);
  const captureRef = useRef<PortfolioCaptureStatus | null>(null);
  const requestSequenceRef = useRef(0);
  const acceptedSequenceRef = useRef(0);
  const settingsRevisionRef = useRef(0);
  const issueVersionRef = useRef(0);
  const issueSequenceRef = useRef(0);
  const appliedReviewRunIdsRef = useRef(new Set<number>());
  const safeCapture = isCaptureStatus(capture) ? capture : null;
  const issueCopy = issue ? presentCaptureIssue(issue, t) : null;

  const publishIssue = useCallback((
    next: CaptureIssue,
    sequence = ++requestSequenceRef.current,
  ) => {
    if (sequence < issueSequenceRef.current) return false;
    issueSequenceRef.current = sequence;
    issueVersionRef.current += 1;
    setIssue(next);
    return true;
  }, []);

  const acceptStatus = useCallback(async (
    next: PortfolioCaptureStatus,
    sequence: number,
    settingsRevision = settingsRevisionRef.current,
  ) => {
    if (sequence < acceptedSequenceRef.current) return false;
    acceptedSequenceRef.current = sequence;
    issueSequenceRef.current = Math.max(issueSequenceRef.current, sequence);
    const current = captureRef.current;
    const projected = {
      ...next,
      settings: current && settingsRevision !== settingsRevisionRef.current
        ? {
          ...next.settings,
          enabled: current.settings.enabled,
          interval_minutes: current.settings.interval_minutes,
          source: current.settings.source,
        }
        : next.settings,
      review: next.review && appliedReviewRunIdsRef.current.has(next.review.run_id)
        ? null
        : next.review,
    };
    captureRef.current = projected;
    setCapture(projected);
    if (!dirtyRef.current) {
      setEnabled(projected.settings.enabled);
      setIntervalValue(String(projected.settings.interval_minutes));
    }

    const latest = projected.latest_run ?? null;
    const terminal = latest && latest.state !== "running" ? latest : null;
    if (!initializedRef.current) {
      initializedRef.current = true;
      lastTerminalRunIdRef.current = terminal?.id ?? null;
      return sequence === acceptedSequenceRef.current;
    }
    if (terminal && terminal.id !== lastTerminalRunIdRef.current) {
      lastTerminalRunIdRef.current = terminal.id;
      await onPortfolioChanged();
    }
    return sequence === acceptedSequenceRef.current;
  }, [onPortfolioChanged]);

  const refresh = useCallback(async () => {
    const sequence = ++requestSequenceRef.current;
    const settingsRevision = settingsRevisionRef.current;
    const issueVersion = issueVersionRef.current;
    try {
      const next: unknown = await getPortfolioCaptureStatus();
      if (!isCaptureStatus(next)) {
        publishIssue({ state: "failed", kind: "copy", copy: "status_shape" }, sequence);
        return null;
      }
      const accepted = await acceptStatus(next, sequence, settingsRevision);
      if (accepted && issueVersion === issueVersionRef.current) setIssue(null);
      return accepted ? next : null;
    } catch (reason) {
      publishIssue({
        state: "failed",
        kind: "operation",
        error: capturePortfolioError(CAPTURE_LOAD_OPERATION, reason),
      }, sequence);
      return null;
    }
  }, [acceptStatus, publishIssue]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    let cancelled = false;
    let timer = 0;
    const poll = async () => {
      await refresh();
      if (!cancelled) {
        timer = window.setTimeout(
          () => void poll(),
          safeCapture?.running ? RUNNING_POLL_MS : IDLE_POLL_MS,
        );
      }
    };
    timer = window.setTimeout(
      () => void poll(),
      safeCapture?.running ? RUNNING_POLL_MS : IDLE_POLL_MS,
    );
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [safeCapture?.running, refresh]);

  async function saveSettings() {
    const parsed = parseCaptureInterval(interval);
    if (parsed === null) {
      publishIssue({
        state: "failed",
        kind: "copy",
        copy: "schedule_invalid",
      });
      return;
    }
    setBusy("save");
    setIssue(null);
    setNotice(null);
    try {
      const next: unknown = await updatePortfolioCaptureSettings({
        enabled,
        interval_minutes: parsed,
      });
      if (!isCaptureStatus(next)) {
        publishIssue({ state: "failed", kind: "copy", copy: "settings_shape" });
        return;
      }
      settingsRevisionRef.current += 1;
      dirtyRef.current = false;
      const current = captureRef.current;
      if (current) {
        const merged = {
          ...current,
          settings: {
            ...current.settings,
            enabled: next.settings.enabled,
            interval_minutes: next.settings.interval_minutes,
            source: next.settings.source,
          },
        };
        captureRef.current = merged;
        setCapture(merged);
      }
      setEnabled(next.settings.enabled);
      setIntervalValue(String(next.settings.interval_minutes));
      setNotice(SCHEDULE_SAVED_NOTICE);
    } catch (reason) {
      publishIssue({
        state: "failed",
        kind: "operation",
        error: capturePortfolioError(CAPTURE_SAVE_OPERATION, reason),
      });
    } finally {
      setBusy(null);
    }
  }

  async function startCapture() {
    setBusy("capture");
    setIssue(null);
    setNotice(null);
    try {
      const started = await triggerPortfolioCapture();
      const current = captureRef.current;
      if (started.run && current && shouldProjectRun(current.latest_run, started.run)) {
        const sequence = ++requestSequenceRef.current;
        await acceptStatus(statusWithRun(current, started.run), sequence);
      }
      await refresh();
      if (started.error_detail && !started.run) {
        publishIssue({
          state: runUiState(started.state),
          kind: "run",
          runState: started.state,
        });
      }
    } catch (reason) {
      publishIssue({
        state: "failed",
        kind: "operation",
        error: capturePortfolioError(CAPTURE_START_OPERATION, reason),
      });
    } finally {
      setBusy(null);
    }
  }

  async function applyReview(runId: number) {
    setBusy("apply");
    setIssue(null);
    setNotice(null);
    try {
      const applied = await applyPortfolioCaptureRun(runId);
      appliedReviewRunIdsRef.current.add(applied.run_id);
      const current = captureRef.current;
      if (current?.review?.run_id === applied.run_id) {
        const next = { ...current, review: null };
        captureRef.current = next;
        setCapture(next);
      }
      await onPortfolioChanged();
      await refresh();
      setNotice(CAPTURE_APPLIED_NOTICE);
    } catch (reason) {
      publishIssue({
        state: "failed",
        kind: "operation",
        error: capturePortfolioError(CAPTURE_APPLY_OPERATION, reason),
      });
    } finally {
      setBusy(null);
    }
  }

  const runColumns = useMemo<DataTableColumn<PortfolioCaptureRun>[]>(() => [
    {
      id: "started",
      header: t(($) => $.capture.surface.runsStartedHeader),
      render: (item) => formatLocalTime(item.started_at),
    },
    {
      id: "trigger",
      header: t(($) => $.capture.surface.runsSourceHeader),
      render: (item) => portfolioCaptureTriggerLabel(item.trigger, t),
    },
    {
      id: "state",
      header: t(($) => $.capture.surface.runsStateHeader),
      render: (item) => (
        <StatusBadge
          state={runUiState(item.state)}
          label={portfolioCaptureRunStateLabel(item.state, t)}
        />
      ),
      className: "ui-data-table-status",
    },
    {
      id: "facts",
      header: t(($) => $.capture.surface.runsFactsHeader),
      render: (item) => t(($) => $.capture.surface.runsFactsSummary, {
        executionCount: item.inserted_execution_count,
        commissionCount: item.inserted_commission_count,
      }),
    },
  ], [t]);

  const reviewColumns = useMemo<DataTableColumn<PortfolioCaptureReviewChange>[]>(() => [
    {
      id: "account",
      header: t(($) => $.capture.surface.reviewAccountHeader),
      render: (item) => item.account_label
        ?? item.broker_account_id_hash?.slice(0, 8)
        ?? t(($) => $.capture.surface.reviewNewAccount),
    },
    {
      id: "kind",
      header: t(($) => $.capture.surface.reviewChangeHeader),
      render: (item) => item.kind,
    },
    {
      id: "symbol",
      header: t(($) => $.capture.surface.reviewSymbolHeader),
      render: (item) => item.symbol,
    },
    {
      id: "quantity",
      header: t(($) => $.capture.surface.reviewQuantityHeader),
      render: (item) => formatReviewMetric(item, "quantity", item.quantity),
      align: "right",
    },
    {
      id: "avg-cost",
      header: t(($) => $.tableLabels.captureAvgCost),
      render: (item) => formatReviewMetric(item, "avg_cost"),
      align: "right",
    },
    {
      id: "market-value",
      header: t(($) => $.tableLabels.captureMarketValue),
      render: (item) => formatReviewMetric(item, "market_value"),
      align: "right",
    },
    {
      id: "unrealized-pnl",
      header: t(($) => $.tableLabels.captureUnrealizedPnl),
      render: (item) => formatReviewMetric(item, "unrealized_pnl"),
      align: "right",
    },
  ], [t]);

  const latest = safeCapture?.latest_run ?? null;
  const reviewCountLabel = safeCapture?.review
    ? portfolioCountCopy("review_changes", safeCapture.review.changes.length, t)
    : "";
  const providerMissing = safeCapture?.provider_issue != null || safeCapture?.settings.provider_configured === false;

  return (
    <section className="ui-section-band portfolio-capture" data-portfolio-capture-controls>
      <div className="ui-section-head">
        <div>
          <h2>{t(($) => $.capture.surface.sectionTitle)}</h2>
          <p className="muted tiny">{t(($) => $.capture.surface.sectionNotice)}</p>
        </div>
        <Button
          icon={<RefreshCw size={15} />}
          onClick={() => void startCapture()}
          busy={busy === "capture"}
          disabled={busy != null || !safeCapture || safeCapture.running || providerMissing}
        >
          {t(($) => $.capture.surface.syncNow)}
        </Button>
      </div>

      {providerMissing ? (
        <InlineAlert
          state="blocked"
          title={t(($) => $.capture.surface.providerMissingTitle)}
        >
          {t(($) => $.capture.surface.providerMissingAction)}
        </InlineAlert>
      ) : null}
      {issue && issueCopy ? (
        <InlineAlert state={issue.state} title={issueCopy.title}>
          {issueCopy.message}
        </InlineAlert>
      ) : null}
      {notice ? (
        <InlineAlert
          state="ready"
          title={portfolioOutcomeLabel(notice, t)}
        />
      ) : null}

      <div className="portfolio-capture-settings">
        <label className="portfolio-capture-toggle">
          <input
            type="checkbox"
            aria-label={t(($) => $.capture.surface.scheduleToggleAria)}
            checked={enabled}
            disabled={!safeCapture || busy != null}
            onChange={(event) => {
              dirtyRef.current = true;
              setEnabled(event.currentTarget.checked);
            }}
          />
          {t(($) => $.capture.surface.scheduleToggleLabel)}
        </label>
        <label>
          <span>
            {t(($) => $.capture.surface.intervalLabel)}{" "}
            <span className="muted">{t(($) => $.capture.surface.intervalHint)}</span>
          </span>
          <input
            type="number"
            min={5}
            max={1440}
            step={1}
            aria-label={t(($) => $.capture.surface.intervalAria)}
            value={interval}
            disabled={!safeCapture || busy != null}
            onChange={(event) => {
              dirtyRef.current = true;
              setIntervalValue(event.currentTarget.value);
            }}
          />
        </label>
        <Button
          icon={<Save size={15} />}
          onClick={() => void saveSettings()}
          busy={busy === "save"}
          disabled={busy != null || !safeCapture}
        >
          {t(($) => $.capture.surface.scheduleSave)}
        </Button>
        <div className="portfolio-capture-next muted tiny">
          {safeCapture?.settings.enabled
            ? t(($) => $.capture.surface.nextRun, {
              timestamp: formatLocalTime(safeCapture.next_due_at),
            })
            : t(($) => $.capture.surface.scheduleDisabled)}
        </div>
      </div>

      {latest ? (
        <div className="portfolio-capture-latest">
          <div className="ui-section-head">
            <div className="ui-action-row">
              <strong>{t(($) => $.capture.surface.latestRun)}</strong>
              <StatusBadge
                state={runUiState(latest.state)}
                label={portfolioCaptureRunStateLabel(latest.state, t)}
              />
              <span className="muted tiny">{formatLocalTime(latest.finished_at ?? latest.started_at)}</span>
            </div>
          </div>
          <div className="portfolio-capture-legs">
            <span>
              {t(($) => $.capture.surface.accountLegPrefix)}{" "}
              {captureLegStateLabel(latest.account_leg_state, t)}
            </span>
            <span>
              {t(($) => $.capture.surface.executionLegPrefix)}{" "}
              {captureLegStateLabel(latest.execution_leg_state, t)}
            </span>
            <span>
              {t(($) => $.capture.surface.positionLegPrefix)}{" "}
              {captureLegStateLabel(latest.position_leg_state, t)}
            </span>
          </div>
          {latest.new_account_count > 0 ? (
            <InlineAlert
              state="partial"
              title={t(($) => $.capture.alerts.reviewTitle)}
            >
              {t(($) => $.capture.alerts.newAccounts, {
                count: latest.new_account_count,
              })}
            </InlineAlert>
          ) : null}
          {latest.archived_activity_count > 0 ? (
            <InlineAlert
              state="partial"
              title={t(($) => $.capture.alerts.archivedActivityTitle)}
            >
              {t(($) => $.capture.alerts.archivedActivityMessage, {
                count: latest.archived_activity_count,
              })}
            </InlineAlert>
          ) : null}
          {latest.error_detail ? (
            <InlineAlert
              state={runUiState(latest.state)}
              title={portfolioCaptureRunDetailLabel(latest.state, t)}
            />
          ) : null}
        </div>
      ) : <p className="muted">{portfolioEmptyStateLabel("capture_runs", t)}</p>}

      <DataTable<PortfolioCaptureRun>
        ariaLabel={t(($) => $.capture.runs.tableAria)}
        rows={safeCapture?.recent_runs ?? []}
        columns={runColumns}
        rowKey={(item) => item.id}
        rowLabel={(item) => t(($) => $.capture.runs.rowAria, { id: item.id })}
        emptyText={portfolioEmptyStateLabel("capture_runs", t)}
      />

      {safeCapture?.review && safeCapture.review.changes.length > 0 ? (
        <div className="portfolio-capture-review">
          <div className="ui-section-head">
            <div className="ui-action-row">
              <strong>{t(($) => $.capture.review.pendingTitle)}</strong>
              <StatusBadge
                state="partial"
                label={reviewCountLabel}
              />
            </div>
            {safeCapture.review.changes.length > 0 ? (
              <Button
                tone="primary"
                icon={<Check size={15} />}
                onClick={() => void applyReview(safeCapture.review!.run_id)}
                busy={busy === "apply"}
                disabled={busy != null || safeCapture.running}
              >
                {t(($) => $.capture.review.apply)}
              </Button>
            ) : null}
          </div>
          <DataTable<PortfolioCaptureReviewChange>
            ariaLabel={t(($) => $.capture.review.tableAria)}
            rows={safeCapture.review.changes}
            columns={reviewColumns}
            rowKey={(item) => `${safeCapture.review!.run_id}-${item.account_id ?? item.broker_account_id_hash}-${item.broker_con_id}-${item.kind}`}
            rowLabel={(item) => item.symbol}
            emptyText={t(($) => $.capture.review.empty)}
          />
        </div>
      ) : null}
    </section>
  );
}
