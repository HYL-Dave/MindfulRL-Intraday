import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { RefreshCw, RotateCcw } from "lucide-react";
import { useTranslation } from "react-i18next";

import {
  deletePortfolioActivityAnnotation,
  getPortfolioActivity,
  isPortfolioAnnotatableActivity,
  isPortfolioBrokerActivity,
  putPortfolioActivityAnnotation,
  type PortfolioActivityAnnotation,
  type PortfolioActivityFilters,
  type PortfolioActivityItem,
  type PortfolioActivityPage,
  type PortfolioAnnotatableActivityItem,
  type PortfolioIntentLabel,
} from "./api";
import { formatMarketTimestamp, formatSystemTimestamp } from "./timeDisplay";
import {
  PORTFOLIO_CLOSED_IDS,
  capturePortfolioError,
  portfolioActivityFieldLabel,
  portfolioActivityIntentLabel,
  portfolioActivityKindLabel,
  portfolioActivitySourceLabel,
  portfolioActivityStateLabel,
  portfolioCloseScopeLabel,
  portfolioCountCopy,
  portfolioCoverageReasonLabel,
  portfolioEmptyStateLabel,
  portfolioExecutionCoverageLabel,
  portfolioManualActionLabel,
  portfolioObjectiveOutcomeLabel,
  portfolioObjectiveSideLabel,
  portfolioPositionDirectionLabel,
  presentPortfolioError,
  type PortfolioErrorState,
  type PortfolioT,
} from "./i18n/portfolioPresentation";
import {
  Button,
  ConfirmDialog,
  DataTable,
  Drawer,
  InlineAlert,
  StatusBadge,
  type DataTableAction,
  type DataTableColumn,
} from "./ui";

const initialDraft = {
  date_from_et: "",
  date_to_et: "",
  account_id: "",
  symbol: "",
  source: "",
  state: "",
};

type FilterDraft = typeof initialDraft;
const ACTIVITY_COPY_KIND = "activity" as const;

export function PortfolioActivity({
  localTimeZone,
}: {
  localTimeZone?: string;
}) {
  const { t: portfolioT } = useTranslation("portfolio");
  useTranslation("common");
  const [page, setPage] = useState<PortfolioActivityPage | null>(null);
  const [draft, setDraft] = useState<FilterDraft>(initialDraft);
  const [activeFilters, setActiveFilters] = useState<PortfolioActivityFilters>({});
  const [loading, setLoading] = useState(true);
  const [appending, setAppending] = useState(false);
  const [readError, setReadError] = useState<PortfolioErrorState | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [editorId, setEditorId] = useState<string | null>(null);
  const [intentDraft, setIntentDraft] = useState<PortfolioIntentLabel | "">("");
  const [noteDraft, setNoteDraft] = useState("");
  const [mutationBusy, setMutationBusy] = useState(false);
  const [mutationError, setMutationError] = useState<PortfolioErrorState | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const requestGeneration = useRef(0);
  const editorReturnFocusRef = useRef<HTMLElement | null>(null);
  const deleteReturnFocusRef = useRef<HTMLButtonElement | null>(null);

  const load = useCallback(async (
    filters: PortfolioActivityFilters,
    append = false,
  ) => {
    const generation = ++requestGeneration.current;
    if (append) setAppending(true);
    else {
      setAppending(false);
      setPage(null);
      setLoading(true);
      setReadError(null);
    }
    try {
      const loaded = await getPortfolioActivity(filters);
      if (generation !== requestGeneration.current) return;
      setPage((current) => append && current
        ? appendActivityPage(current, loaded)
        : loaded);
      setReadError(null);
    } catch (error) {
      if (generation !== requestGeneration.current) return;
      setReadError(capturePortfolioError("activity_load", error));
    } finally {
      if (generation !== requestGeneration.current) return;
      if (append) setAppending(false);
      else setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load({});
  }, [load]);

  const editorItem = page?.items.find((item) => item.id === editorId);
  const annotatableEditorItem = editorItem && isPortfolioAnnotatableActivity(editorItem)
    ? editorItem
    : null;

  const updateDraft = (key: keyof FilterDraft, value: string) => {
    setDraft((current) => ({ ...current, [key]: value }));
  };

  const applyFilters = () => {
    const filters = filtersFromDraft(draft);
    setActiveFilters(filters);
    setExpandedId(null);
    void load(filters);
  };

  const resetFilters = () => {
    setDraft(initialDraft);
    setActiveFilters({});
    setExpandedId(null);
    void load({});
  };

  const openEditor = (
    item: PortfolioAnnotatableActivityItem,
    trigger: HTMLButtonElement,
  ) => {
    editorReturnFocusRef.current = trigger;
    setEditorId(item.id);
    setIntentDraft(item.annotation?.intent_label ?? "");
    setNoteDraft(item.annotation?.note ?? "");
    setMutationError(null);
  };

  const closeEditor = () => {
    if (mutationBusy) return;
    setEditorId(null);
    setConfirmDelete(false);
    setMutationError(null);
  };

  const saveAnnotation = async () => {
    if (!annotatableEditorItem) return;
    setMutationBusy(true);
    setMutationError(null);
    try {
      const annotation = await putPortfolioActivityAnnotation(
        annotatableEditorItem.id,
        { intent_label: intentDraft || null, note: noteDraft },
      );
      replaceLocalAnnotation(annotatableEditorItem.id, annotation, setPage);
      setEditorId(null);
    } catch (error) {
      setMutationError(capturePortfolioError("activity_save_annotation", error));
    } finally {
      setMutationBusy(false);
    }
  };

  const deleteAnnotation = async () => {
    if (!annotatableEditorItem) return;
    setMutationBusy(true);
    setMutationError(null);
    try {
      await deletePortfolioActivityAnnotation(annotatableEditorItem.id);
      replaceLocalAnnotation(annotatableEditorItem.id, null, setPage);
      setConfirmDelete(false);
      setEditorId(null);
    } catch (error) {
      setMutationError(capturePortfolioError("activity_clear_annotation", error));
      setConfirmDelete(false);
    } finally {
      setMutationBusy(false);
    }
  };

  const columns: DataTableColumn<PortfolioActivityItem>[] = [
    {
      id: "time",
      header: portfolioT(($) => $.activity.surface.tableTime),
      className: "portfolio-activity-time",
      render: (item) => isPortfolioBrokerActivity(item)
        ? formatMarketTimestamp(item.occurred_at_utc, { localTimeZone })
        : formatSystemTimestamp(item.occurred_at_utc, { localTimeZone }),
    },
    {
      id: "account",
      header: portfolioT(($) => $.activity.surface.tableAccount),
      render: (item) => item.account?.label ?? portfolioT(($) => $.activity.surface.tableAllAccounts),
    },
    {
      id: "event",
      header: portfolioT(($) => $.activity.surface.tableEvent),
      render: (item) => eventLabel(item, portfolioT),
    },
    {
      id: "source",
      header: portfolioT(($) => $.activity.surface.tableSource),
      render: (item) => portfolioActivitySourceLabel(item.source, portfolioT),
    },
    {
      id: "objective",
      header: portfolioT(($) => $.activity.surface.tableObjective),
      render: (item) => <ObjectiveSummary item={item} t={portfolioT} />,
    },
    {
      id: "intent",
      header: portfolioT(($) => $.activity.surface.tableIntent),
      render: (item) => <IntentSummary item={item} t={portfolioT} />,
    },
  ];

  const actions = (item: PortfolioActivityItem): DataTableAction<PortfolioActivityItem>[] => {
    const rowActions: DataTableAction<PortfolioActivityItem>[] = [{
      id: "detail",
      label: expandedId === item.id
        ? portfolioT(($) => $.activity.surface.collapseDetail)
        : portfolioT(($) => $.activity.surface.viewDetail),
      onSelect: () => setExpandedId((current) => current === item.id ? null : item.id),
    }];
    if (isPortfolioAnnotatableActivity(item)) {
      rowActions.push({
        id: "annotation",
        label: portfolioT(($) => $.activity.surface.editAnnotation),
        onSelect: (_, trigger) => openEditor(item, trigger),
      });
    }
    return rowActions;
  };

  return (
    <section className="portfolio-activity" aria-label={portfolioT(($) => $.activity.surface.pageTitle)}>
      <div className="portfolio-activity-head">
        <div>
          <h2>{portfolioT(($) => $.activity.surface.heading)}</h2>
          <p className="muted">{portfolioT(($) => $.activity.surface.timezoneNotice)}</p>
        </div>
        {loading ? (
          <StatusBadge state="loading" label={portfolioT(($) => $.activity.surface.loading)} />
        ) : page ? (
          <StatusBadge
            state={page.items.length ? "ready" : "empty"}
            label={portfolioCountCopy(ACTIVITY_COPY_KIND, page.items.length, portfolioT)}
          />
        ) : null}
      </div>

      <form
        className="portfolio-activity-filters"
        onSubmit={(event) => {
          event.preventDefault();
          applyFilters();
        }}
      >
        <label>
          {portfolioT(($) => $.activity.surface.startDateLabel)}
          <input aria-label={portfolioT(($) => $.activity.surface.startDateLabel)} type="date" value={draft.date_from_et} onChange={(event) => updateDraft("date_from_et", event.currentTarget.value)} />
        </label>
        <label>
          {portfolioT(($) => $.activity.surface.endDateLabel)}
          <input aria-label={portfolioT(($) => $.activity.surface.endDateLabel)} type="date" value={draft.date_to_et} onChange={(event) => updateDraft("date_to_et", event.currentTarget.value)} />
        </label>
        <label>
          {portfolioT(($) => $.activity.surface.filterAccountLabel)}
          <select aria-label={portfolioT(($) => $.activity.surface.filterAccountAria)} value={draft.account_id} onChange={(event) => updateDraft("account_id", event.currentTarget.value)}>
            <option value="">{portfolioT(($) => $.activity.surface.tableAllAccounts)}</option>
            {page?.accounts.map((activityAccount) => (
              <option key={activityAccount.id} value={activityAccount.id}>{activityAccount.label}</option>
            ))}
          </select>
        </label>
        <label>
          {portfolioT(($) => $.activity.surface.filterSymbolLabel)}
          <input aria-label={portfolioT(($) => $.activity.surface.filterSymbolAria)} value={draft.symbol} onChange={(event) => updateDraft("symbol", event.currentTarget.value)} />
        </label>
        <label>
          {portfolioT(($) => $.activity.surface.tableSource)}
          <select aria-label={portfolioT(($) => $.activity.surface.filterSourceAria)} value={draft.source} onChange={(event) => updateDraft("source", event.currentTarget.value)}>
            <option value="">{portfolioT(($) => $.activity.surface.filterAllSources)}</option>
            {PORTFOLIO_CLOSED_IDS.activitySources.map((source) => (
              <option key={source} value={source}>{portfolioActivitySourceLabel(source, portfolioT)}</option>
            ))}
          </select>
        </label>
        <label>
          {portfolioT(($) => $.activity.surface.filterStateLabel)}
          <select aria-label={portfolioT(($) => $.activity.surface.filterStateAria)} value={draft.state} onChange={(event) => updateDraft("state", event.currentTarget.value)}>
            <option value="">{portfolioT(($) => $.activity.surface.filterAllStates)}</option>
            {PORTFOLIO_CLOSED_IDS.activityStates.map((state) => (
              <option key={state} value={state}>{portfolioActivityStateLabel(state, portfolioT)}</option>
            ))}
          </select>
        </label>
        <div className="portfolio-activity-filter-actions">
          <Button type="submit" size="compact" icon={<RefreshCw size={15} />}>{portfolioT(($) => $.activity.surface.applyFilters)}</Button>
          <Button type="button" size="compact" tone="ghost" icon={<RotateCcw size={15} />} onClick={resetFilters}>{portfolioT(($) => $.activity.surface.resetFilters)}</Button>
        </div>
      </form>

      {readError ? (
        <InlineAlert
          state="failed"
          title={presentPortfolioError(readError, portfolioT).title}
          action={<Button size="compact" onClick={() => void load(activeFilters)}>{portfolioT(($) => $.activity.surface.refresh)}</Button>}
        />
      ) : null}

      {page ? (
        <>
          <p className="portfolio-activity-history muted">
            {page.history_started_at_utc
              ? portfolioT(($) => $.activity.surface.historyStarted, {
                timestamp: formatSystemTimestamp(page.history_started_at_utc, { localTimeZone }),
              })
              : portfolioT(($) => $.activity.surface.historyNotStarted)}
          </p>
          <DataTable<PortfolioActivityItem>
            ariaLabel={portfolioT(($) => $.activity.surface.tableAria)}
            rows={page.items}
            columns={columns}
            rowKey={(item) => item.id}
            rowLabel={(item) => eventLabel(item, portfolioT)}
            emptyText={portfolioEmptyStateLabel(ACTIVITY_COPY_KIND, portfolioT)}
            actions={actions}
            renderExpandedRow={(item) => expandedId === item.id
              ? <ActivityDetail item={item} localTimeZone={localTimeZone} t={portfolioT} />
              : null}
          />
          {page.next_cursor ? (
            <div className="portfolio-activity-more">
              <Button
                size="compact"
                busy={appending}
                onClick={() => void load({ ...activeFilters, cursor: page.next_cursor ?? undefined }, true)}
              >
                {portfolioT(($) => $.activity.surface.loadMore)}
              </Button>
            </div>
          ) : null}
        </>
      ) : null}

      <Drawer
        open={Boolean(annotatableEditorItem)}
        title={portfolioT(($) => $.activity.surface.editorTitle)}
        onClose={closeEditor}
        returnFocusRef={editorReturnFocusRef}
        footer={(
          <div className="portfolio-activity-editor-actions">
            {annotatableEditorItem?.annotation ? (
              <Button ref={deleteReturnFocusRef} tone="danger" disabled={mutationBusy} onClick={() => setConfirmDelete(true)}>{portfolioT(($) => $.activity.surface.editorClear)}</Button>
            ) : null}
            <span className="portfolio-activity-editor-spacer" />
            <Button disabled={mutationBusy} onClick={closeEditor}>{portfolioT(($) => $.activity.surface.editorCancel)}</Button>
            <Button
              tone="primary"
              busy={mutationBusy}
              disabled={!intentDraft && !noteDraft.trim()}
              onClick={() => void saveAnnotation()}
            >
              {portfolioT(($) => $.activity.surface.editorSave)}
            </Button>
          </div>
        )}
      >
        <div className="portfolio-activity-editor">
          <label>
            {portfolioT(($) => $.activity.surface.editorIntentLabel)}
            <select aria-label={portfolioT(($) => $.activity.surface.editorIntentLabel)} value={intentDraft} onChange={(event) => setIntentDraft(event.currentTarget.value as PortfolioIntentLabel | "")}>
              <option value="">{portfolioT(($) => $.activity.surface.editorUnconfirmed)}</option>
              {PORTFOLIO_CLOSED_IDS.activityIntents.map((value) => (
                <option key={value} value={value}>{portfolioActivityIntentLabel(value, portfolioT)}</option>
              ))}
            </select>
          </label>
          <label>
            {portfolioT(($) => $.activity.surface.editorNoteLabel)}
            <textarea aria-label={portfolioT(($) => $.activity.surface.editorNoteLabel)} rows={7} value={noteDraft} onChange={(event) => setNoteDraft(event.currentTarget.value)} />
          </label>
          {mutationError ? (
            <InlineAlert state="failed" title={presentPortfolioError(mutationError, portfolioT).title} />
          ) : null}
        </div>
      </Drawer>

      <ConfirmDialog
        open={confirmDelete}
        title={portfolioT(($) => $.activity.surface.clearDialogTitle)}
        consequence={portfolioT(($) => $.activity.surface.clearDialogConsequence)}
        confirmLabel={portfolioT(($) => $.activity.surface.clearDialogConfirm)}
        busy={mutationBusy}
        onConfirm={() => void deleteAnnotation()}
        onCancel={() => setConfirmDelete(false)}
        returnFocusRef={deleteReturnFocusRef}
        fallbackFocusRef={editorReturnFocusRef}
      />
    </section>
  );
}

function filtersFromDraft(draft: FilterDraft): PortfolioActivityFilters {
  return {
    ...(draft.date_from_et ? { date_from_et: draft.date_from_et } : {}),
    ...(draft.date_to_et ? { date_to_et: draft.date_to_et } : {}),
    ...(draft.account_id ? { account_id: Number(draft.account_id) } : {}),
    ...(draft.symbol.trim() ? { symbol: draft.symbol.trim() } : {}),
    ...(draft.source ? { source: draft.source as PortfolioActivityFilters["source"] } : {}),
    ...(draft.state ? { state: draft.state as PortfolioActivityFilters["state"] } : {}),
  };
}

function appendActivityPage(
  current: PortfolioActivityPage,
  loaded: PortfolioActivityPage,
): PortfolioActivityPage {
  const items = [...current.items];
  const seen = new Set(items.map((item) => item.id));
  for (const item of loaded.items) {
    if (!seen.has(item.id)) {
      seen.add(item.id);
      items.push(item);
    }
  }
  return {
    ...loaded,
    accounts: loaded.accounts.length ? loaded.accounts : current.accounts,
    history_started_at_utc: loaded.history_started_at_utc ?? current.history_started_at_utc,
    items,
    summary: { ...loaded.summary, item_count: items.length },
  };
}

function replaceLocalAnnotation(
  id: string,
  annotation: PortfolioActivityAnnotation | null,
  setPage: React.Dispatch<React.SetStateAction<PortfolioActivityPage | null>>,
) {
  setPage((current) => current ? {
    ...current,
    items: current.items.map((item) => item.id === id && isPortfolioAnnotatableActivity(item)
      ? { ...item, annotation } as PortfolioActivityItem
      : item),
  } : current);
}

function eventLabel(item: PortfolioActivityItem, t: PortfolioT): string {
  switch (item.kind) {
    case "order": return item.symbol
      ? t(($) => $.activity.surface.eventOrderWithSymbol, { symbol: item.symbol })
      : t(($) => $.activity.surface.eventOrder);
    case "execution": return item.symbol
      ? t(($) => $.activity.surface.eventExecutionWithSymbol, { symbol: item.symbol })
      : t(($) => $.activity.surface.eventExecution);
    case "unmatched": return item.symbol
      ? t(($) => $.activity.surface.eventUnmatchedWithSymbol, { symbol: item.symbol })
      : t(($) => $.activity.surface.eventUnmatched);
    case "manual_adjustment": return t(($) => $.activity.surface.eventManualWithSymbol, { symbol: item.symbol });
    case "coverage_gap": return portfolioCoverageReasonLabel(item.reason_code, t);
    case "history_start": return t(($) => $.activity.surface.eventHistoryStart);
  }
}

function ObjectiveSummary({ item, t }: { item: PortfolioActivityItem; t: PortfolioT }) {
  let content: ReactNode;
  switch (item.kind) {
    case "order":
    case "execution":
      content = (
        <>
          <strong>{portfolioObjectiveOutcomeLabel(item.objective.realized_outcome, t)}</strong>
          <span className="muted tiny">
            {portfolioObjectiveSideLabel(item.objective.side, t)} · {formatNumber(item.objective.quantity, t(($) => $.activity.surface.unknown))}
          </span>
        </>
      );
      break;
    case "unmatched":
      content = (
        <>
          <strong>{t(($) => $.activity.surface.objectiveUnmatched)}</strong>
          <span className="muted tiny">
            {t(($) => $.activity.surface.objectiveResidual)} {formatNumber(item.residual_quantity, t(($) => $.activity.surface.unknown))}
          </span>
        </>
      );
      break;
    case "manual_adjustment":
      content = (
        <>
          <strong>{portfolioActivityKindLabel(item.kind, t)}</strong>
          <span className="muted tiny">{portfolioManualActionLabel(item.action, t)}</span>
        </>
      );
      break;
    case "coverage_gap":
      content = (
        <StatusBadge
          state={item.reason_code === "broker_day_gap" ? "stale" : "partial"}
          label={t(($) => $.activity.surface.objectiveCoverageIncomplete)}
        />
      );
      break;
    case "history_start":
      content = <StatusBadge state="ready" label={t(($) => $.activity.surface.objectiveHistoryStart)} />;
      break;
  }
  return <div className="portfolio-activity-objective">{content}</div>;
}

function IntentSummary({ item, t }: { item: PortfolioActivityItem; t: PortfolioT }) {
  if (!isPortfolioAnnotatableActivity(item)) {
    return <div className="portfolio-activity-intent muted">{t(($) => $.activity.surface.intentNotApplicable)}</div>;
  }
  const label = item.annotation?.intent_label
    ? portfolioActivityIntentLabel(item.annotation.intent_label, t)
    : t(($) => $.activity.surface.intentUnconfirmed);
  return (
    <div className="portfolio-activity-intent">
      <strong>{label}</strong>
      {item.annotation?.note ? <span className="muted tiny">{item.annotation.note}</span> : null}
    </div>
  );
}

function ActivityDetail({
  item,
  localTimeZone,
  t,
}: {
  item: PortfolioActivityItem;
  localTimeZone?: string;
  t: PortfolioT;
}) {
  const unknown = t(($) => $.activity.surface.unknown);
  let content: ReactNode;
  switch (item.kind) {
    case "order":
    case "execution":
      content = (
        <div className="portfolio-activity-fill-list">
          <dl className="portfolio-activity-detail-grid">
            <Detail label={t(($) => $.activity.surface.detailAveragePrice)} value={formatNumber(item.objective.average_price, unknown)} />
            <Detail label={t(($) => $.activity.surface.detailNotional)} value={formatNumber(item.objective.gross_notional, unknown)} />
            <Detail label={t(($) => $.activity.surface.detailCommission)} value={formatAmount(item.objective.commission, item.objective.commission_currency, unknown)} />
            <Detail label={t(($) => $.activity.surface.detailRealizedPnl)} value={formatAmount(item.objective.realized_pnl, item.currency, unknown)} />
            <Detail label={t(($) => $.activity.surface.detailPositionDirection)} value={portfolioPositionDirectionLabel(item.objective.position_direction, t)} />
            <Detail label={t(($) => $.activity.surface.detailCloseScope)} value={portfolioCloseScopeLabel(item.objective.close_scope, t)} />
          </dl>
          {item.fills.map((fill) => (
            <section key={fill.family_root_id} className="portfolio-activity-fill">
              <strong>{t(($) => $.activity.surface.fillFamily)}{fill.family_root_id}</strong>
              {fill.revisions.map((execution) => (
                <div key={execution.id} className="portfolio-activity-revision">
                  <span className="portfolio-activity-revision-head">
                    {t(($) => $.activity.surface.executionPrefix)} {execution.exec_id}
                    {execution.corrects_exec_id
                      ? <>{" "}{t(($) => $.activity.surface.executionCorrection, { executionId: execution.corrects_exec_id })}</>
                      : null}
                    {execution.is_effective
                      ? <>{" "}{t(($) => $.activity.surface.executionEffective)}</>
                      : <>{" "}{t(($) => $.activity.surface.executionHistorical)}</>}
                  </span>
                  <span className="muted tiny">
                    {execution.side} {formatNumber(execution.quantity, unknown)} @ {formatNumber(execution.price, unknown)} · {formatMarketTimestamp(execution.execution_time_utc, { localTimeZone })}
                  </span>
                  <span className="muted tiny">
                    {t(($) => $.activity.surface.executionFirstObservedRun)}{execution.first_observed_run_id} · {formatSystemTimestamp(execution.first_observed_at_utc, { localTimeZone })}
                  </span>
                  {execution.commission_revisions.length ? (
                    <ul>
                      {execution.commission_revisions.map((commission) => (
                        <li key={commission.id}>
                          {t(($) => $.activity.surface.commissionPrefix)}{commission.id} · {formatAmount(commission.commission, commission.currency, unknown)} {t(($) => $.activity.surface.commissionRealizedPnl)} {formatAmount(commission.realized_pnl, commission.currency, unknown)} {t(($) => $.activity.surface.commissionFirstObservedRun)}{commission.first_observed_run_id} · {formatSystemTimestamp(commission.first_observed_at_utc, { localTimeZone })} {t(($) => $.activity.surface.commissionYield)} {formatNumber(commission.yield_value, unknown)} {t(($) => $.activity.surface.commissionRedemptionDate)} {commission.yield_redemption_date ?? unknown}{commission.is_latest ? <>{" "}{t(($) => $.activity.surface.commissionLatest)}</> : null}
                        </li>
                      ))}
                    </ul>
                  ) : <span className="muted tiny">{t(($) => $.activity.surface.commissionUnknown)}</span>}
                </div>
              ))}
            </section>
          ))}
        </div>
      );
      break;
    case "unmatched":
      content = (
        <dl className="portfolio-activity-detail-grid">
          <Detail label={t(($) => $.activity.surface.unmatchedBefore)} value={formatNumber(item.before_quantity, unknown)} />
          <Detail label={t(($) => $.activity.surface.unmatchedAfter)} value={formatNumber(item.after_quantity, unknown)} />
          <Detail label={t(($) => $.activity.surface.unmatchedExpected)} value={formatNumber(item.expected_quantity, unknown)} />
          <Detail label={t(($) => $.activity.surface.unmatchedResidual)} value={formatNumber(item.residual_quantity, unknown)} />
          <Detail label={t(($) => $.activity.surface.captureRange)} value={t(($) => $.activity.surface.runRange, { fromRun: item.from_run_id, toRun: item.to_run_id })} />
          <Detail label={t(($) => $.activity.surface.timeWindow)} value={`${formatSystemTimestamp(item.from_as_of_utc, { localTimeZone })} → ${formatSystemTimestamp(item.to_as_of_utc, { localTimeZone })}`} />
          <Detail label={t(($) => $.activity.surface.executionCoverage)} value={portfolioExecutionCoverageLabel(item.execution_coverage, t)} />
          <Detail label={t(($) => $.activity.surface.reason)} value={item.reason_code || unknown} />
        </dl>
      );
      break;
    case "manual_adjustment":
      content = (
        <div className="portfolio-activity-change-list">
          <span>{t(($) => $.activity.surface.positionPrefix)}{item.position_id} · {portfolioManualActionLabel(item.action, t)}</span>
          {item.changes.map((change, index) => (
            <div key={`${change.field}-${index}`}>
              <strong>{portfolioActivityFieldLabel(change.field, t)}</strong> {formatUnknown(change.before, unknown)} → {formatUnknown(change.after, unknown)}
            </div>
          ))}
        </div>
      );
      break;
    case "coverage_gap":
      content = (
        <dl className="portfolio-activity-detail-grid">
          <Detail label={t(($) => $.activity.surface.captureRange)} value={t(($) => $.activity.surface.runRange, { fromRun: item.from_run_id ?? unknown, toRun: item.to_run_id })} />
          <Detail label={t(($) => $.activity.surface.start)} value={item.from_as_of_utc
            ? formatSystemTimestamp(item.from_as_of_utc, { localTimeZone })
            : unknown} />
          <Detail label={t(($) => $.activity.surface.end)} value={formatSystemTimestamp(item.to_as_of_utc, { localTimeZone })} />
          <Detail label={t(($) => $.activity.surface.reason)} value={eventLabel(item, t)} />
        </dl>
      );
      break;
    case "history_start":
      content = <span>{t(($) => $.activity.surface.firstSuccessfulCaptureRun)}{item.capture_run_id} · {formatSystemTimestamp(item.occurred_at_utc, { localTimeZone })}</span>;
      break;
  }
  return <div className="portfolio-activity-detail">{content}</div>;
}

function Detail({ label, value }: { label: string; value: string }) {
  return <div><dt>{label}{" "}</dt><dd>{value}</dd></div>;
}

function formatNumber(value: number | null, unknown: string): string {
  if (value == null || !Number.isFinite(value)) return unknown;
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 4 }).format(value);
}

function formatAmount(value: number | null, currency: string | null, unknown: string): string {
  if (value == null || !Number.isFinite(value)) return unknown;
  const number = formatNumber(value, unknown);
  return currency ? `${number} ${currency}` : number;
}

function formatUnknown(value: unknown, unknownLabel: string): string {
  if (value == null) return unknownLabel;
  if (typeof value === "string") return value || unknownLabel;
  if (typeof value === "number") return Number.isFinite(value) ? formatNumber(value, unknownLabel) : unknownLabel;
  if (typeof value === "boolean") return value ? "true" : "false";
  try {
    return JSON.stringify(value) || unknownLabel;
  } catch {
    return unknownLabel;
  }
}
