import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
} from "react";
import { Plus, RefreshCw, Save, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  closePortfolioPosition,
  createManualPosition,
  getPortfolio,
  getPortfolioActivity,
  getPortfolioOverview,
  updatePortfolioAccount,
  updatePortfolioPosition,
  type PortfolioActivityPage,
  type PortfolioOverview,
  type PortfolioPosition,
  type PortfolioSnapshot,
  type PositionUpdate,
} from "./api";
import {
  PortfolioAccountDetails,
  PortfolioAccountSummary,
} from "./PortfolioAccountOverview";
import { PortfolioActivity } from "./PortfolioActivity";
import { PortfolioCapturePanel } from "./PortfolioCapturePanel";
import { PortfolioRecentActivity } from "./PortfolioRecentActivity";
import {
  capturePortfolioError,
  portfolioCountCopy,
  portfolioEmptyStateLabel,
  portfolioValidationLabel,
  portfolioViewLabel,
  presentPortfolioError,
  type PortfolioErrorState,
  type PortfolioT,
  type PortfolioView,
} from "./i18n/portfolioPresentation";
import {
  Button,
  ConfirmDialog,
  DataTable,
  IconButton,
  InlineAlert,
  PageHeader,
  StatusBadge,
  useShellOverlay,
  type DataTableColumn,
} from "./ui";

const PORTFOLIO_VIEW = {
  holdings: "holdings",
  activity: "activity",
  accountDetails: "account_details",
  syncRecords: "sync_records",
} as const satisfies Record<string, PortfolioView>;
const PORTFOLIO_VIEWS: PortfolioView[] = Object.values(PORTFOLIO_VIEW);

type HoldingsValidation = "ticker_quantity_required" | "quantity_nonzero" | "avg_cost_number";
const HOLDINGS_NOTICE_KIND = {
  validation: "validation",
  error: "error",
} as const;
const HOLDINGS_VALIDATION = {
  tickerQuantityRequired: "ticker_quantity_required",
  quantityNonzero: "quantity_nonzero",
  avgCostNumber: "avg_cost_number",
} as const satisfies Record<string, HoldingsValidation>;
type HoldingsNotice =
  | { kind: "validation"; validation: HoldingsValidation }
  | { kind: "error"; error: PortfolioErrorState };

export function HoldingsView() {
  const { t: portfolioT } = useTranslation("portfolio");
  const { t: commonT } = useTranslation("common");
  const shellOverlay = useShellOverlay();
  const [snapshot, setSnapshot] = useState<PortfolioSnapshot | null>(null);
  const [overview, setOverview] = useState<PortfolioOverview | null>(null);
  const [recentActivity, setRecentActivity] = useState<PortfolioActivityPage | null>(null);
  const [recentRevision, setRecentRevision] = useState(0);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<HoldingsNotice | null>(null);
  const [overviewErr, setOverviewErr] = useState<PortfolioErrorState | null>(null);
  const [includeClosed, setIncludeClosed] = useState(false);
  const [activeView, setActiveView] = useState<PortfolioView>(PORTFOLIO_VIEW.holdings);
  const [positionAccountId, setPositionAccountId] = useState<number | "all">("all");
  const [editing, setEditing] = useState<PortfolioPosition | null>(null);
  const [pendingClose, setPendingClose] = useState<PortfolioPosition | null>(null);
  const recentGeneration = useRef(0);
  const closeTriggerRef = useRef<HTMLElement | null>(null);
  const closedFilterRef = useRef<HTMLInputElement | null>(null);
  const tickerRef = useRef<HTMLInputElement>(null);
  const quantityRef = useRef<HTMLInputElement>(null);
  const notesRef = useRef<HTMLInputElement>(null);
  const editSymbolRef = useRef<HTMLInputElement>(null);
  const editAssetRef = useRef<HTMLInputElement>(null);
  const editQuantityRef = useRef<HTMLInputElement>(null);
  const editAvgCostRef = useRef<HTMLInputElement>(null);
  const editCurrencyRef = useRef<HTMLInputElement>(null);
  const editNotesRef = useRef<HTMLInputElement>(null);
  const editThesisRef = useRef<HTMLInputElement>(null);
  const editTagsRef = useRef<HTMLInputElement>(null);
  const tabRefs = useRef<Record<PortfolioView, HTMLButtonElement | null>>({
    holdings: null,
    activity: null,
    account_details: null,
    sync_records: null,
  });

  const manualAccount = useMemo(
    () => snapshot?.accounts.find((a) => a.broker === "manual") ?? snapshot?.accounts[0] ?? null,
    [snapshot],
  );

  const load = useCallback(async () => {
    setLoading(true);
    setErr(null);
    setOverviewErr(null);
    setOverview(null);
    try {
      setSnapshot(await getPortfolio(includeClosed));
      try {
        setOverview(await getPortfolioOverview());
      } catch (overviewError) {
        setOverviewErr(capturePortfolioError("overview_load", overviewError));
      }
    } catch (portfolioError) {
      setErr({ kind: HOLDINGS_NOTICE_KIND.error, error: capturePortfolioError("holdings_load", portfolioError) });
    } finally {
      setLoading(false);
    }
  }, [includeClosed]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    const generation = ++recentGeneration.current;
    setRecentActivity(null);
    if (activeView !== PORTFOLIO_VIEW.holdings || shellOverlay) return;

    void getPortfolioActivity({ recent: true, limit: 5 })
      .then((page) => {
        if (recentGeneration.current === generation) setRecentActivity(page);
      })
      .catch(() => {
        if (recentGeneration.current === generation) setRecentActivity(null);
      });

    return () => {
      if (recentGeneration.current === generation) recentGeneration.current += 1;
    };
  }, [activeView, recentRevision, shellOverlay]);

  const invalidateRecentActivity = useCallback(() => {
    recentGeneration.current += 1;
    setRecentActivity(null);
    setRecentRevision((current) => current + 1);
  }, []);

  async function onAddManual() {
    const symbol = tickerRef.current?.value.trim().toUpperCase() ?? "";
    const quantity = Number(quantityRef.current?.value || "0");
    if (!symbol || !Number.isFinite(quantity) || quantity === 0) {
      setErr({ kind: HOLDINGS_NOTICE_KIND.validation, validation: HOLDINGS_VALIDATION.tickerQuantityRequired });
      return;
    }
    setBusy("manual");
    setErr(null);
    try {
      await createManualPosition({
        account_id: manualAccount?.id ?? null,
        symbol,
        quantity,
        asset_class: "stock",
        currency: "USD",
        notes: notesRef.current?.value ?? "",
      });
      invalidateRecentActivity();
      if (tickerRef.current) tickerRef.current.value = "";
      if (quantityRef.current) quantityRef.current.value = "";
      if (notesRef.current) notesRef.current.value = "";
      await load();
    } catch (e) {
      setErr({ kind: HOLDINGS_NOTICE_KIND.error, error: capturePortfolioError("holding_create", e) });
    } finally {
      setBusy(null);
    }
  }

  async function onSaveEdit() {
    if (!editing) return;
    const originalPosition = editing;
    const body: PositionUpdate = {
      notes: editNotesRef.current?.value ?? "",
      thesis: editThesisRef.current?.value ?? "",
      tags: splitTags(editTagsRef.current?.value ?? ""),
    };
    if (editing.broker === "manual") {
      const quantity = Number(editQuantityRef.current?.value ?? editing.quantity);
      if (!Number.isFinite(quantity) || quantity === 0) {
        setErr({ kind: HOLDINGS_NOTICE_KIND.validation, validation: HOLDINGS_VALIDATION.quantityNonzero });
        return;
      }
      // Only a truly blank input clears avg_cost; anything non-numeric is an
      // input error, never a silent clear.
      const avgRaw = (editAvgCostRef.current?.value ?? "").trim();
      let avgCost: number | null = null;
      if (avgRaw !== "") {
        avgCost = Number(avgRaw);
        if (!Number.isFinite(avgCost)) {
          setErr({ kind: HOLDINGS_NOTICE_KIND.validation, validation: HOLDINGS_VALIDATION.avgCostNumber });
          return;
        }
      }
      body.symbol = editSymbolRef.current?.value.trim() ?? editing.symbol;
      body.asset_class = editAssetRef.current?.value.trim() ?? editing.asset_class;
      body.quantity = quantity;
      body.avg_cost = avgCost;
      body.currency = editCurrencyRef.current?.value.trim() ?? editing.currency;
    }
    setBusy(`edit-${editing.id}`);
    setErr(null);
    try {
      const persistedPosition = await updatePortfolioPosition(editing.id, body);
      if (
        originalPosition.broker === "manual"
        && hasManualFinancialChange(originalPosition, persistedPosition)
      ) {
        invalidateRecentActivity();
      }
      setEditing(null);
      await load();
    } catch (e) {
      setErr({ kind: HOLDINGS_NOTICE_KIND.error, error: capturePortfolioError("holding_update", e) });
    } finally {
      setBusy(null);
    }
  }

  async function onCloseRow(position: PortfolioPosition) {
    setBusy(`close-${position.id}`);
    setErr(null);
    try {
      await closePortfolioPosition(position.id);
      invalidateRecentActivity();
      if (editing?.id === position.id) setEditing(null);
      await load();
    } catch (e) {
      setErr({ kind: HOLDINGS_NOTICE_KIND.error, error: capturePortfolioError("holding_close", e) });
    } finally {
      setBusy(null);
      setPendingClose(null);
    }
  }

  async function onToggleAggregate(accountId: number, include: boolean) {
    setBusy(`account-${accountId}`);
    setErr(null);
    try {
      await updatePortfolioAccount(accountId, { include_in_total: include });
      await load();
    } catch (e) {
      setErr({ kind: HOLDINGS_NOTICE_KIND.error, error: capturePortfolioError("overview_toggle_aggregate", e) });
    } finally {
      setBusy(null);
    }
  }

  const positions = snapshot?.positions ?? [];
  const accounts = snapshot?.accounts ?? [];
  const accountLabels = useMemo(() => {
    const safe = new Map(
      (overview?.accounts ?? []).map((account) => [account.id, account.label]),
    );
    return new Map(
      (snapshot?.accounts ?? []).map((account) => [
        account.id,
        safe.get(account.id) ?? account.label,
      ]),
    );
  }, [overview, snapshot]);
  const filteredPositions = positionAccountId === "all"
    ? positions
    : positions.filter((position) => position.account_id === positionAccountId);
  const optionPositions = filteredPositions.filter(
    (position) => position.asset_class === "option",
  );
  const standardPositions = filteredPositions.filter(
    (position) => position.asset_class !== "option",
  );
  const showRecent = activeView === PORTFOLIO_VIEW.holdings
    && !shellOverlay
    && recentActivity != null
    && (recentActivity.items.length > 0 || recentActivity.summary.unmatched_count > 0);

  useEffect(() => {
    if (
      positionAccountId !== "all"
      && !accounts.some((account) => account.id === positionAccountId)
    ) {
      setPositionAccountId("all");
    }
  }, [accounts, positionAccountId]);

  function onTabKeyDown(
    event: ReactKeyboardEvent<HTMLButtonElement>,
    current: PortfolioView,
  ) {
    const currentIndex = PORTFOLIO_VIEWS.indexOf(current);
    let nextIndex: number | null = null;
    if (event.key === "ArrowRight") {
      nextIndex = (currentIndex + 1) % PORTFOLIO_VIEWS.length;
    }
    if (event.key === "ArrowLeft") {
      nextIndex = (
        currentIndex - 1 + PORTFOLIO_VIEWS.length
      ) % PORTFOLIO_VIEWS.length;
    }
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = PORTFOLIO_VIEWS.length - 1;
    if (nextIndex == null) return;
    event.preventDefault();
    const next = PORTFOLIO_VIEWS[nextIndex];
    setActiveView(next);
    tabRefs.current[next]?.focus();
  }
  const viewState = err
    ? { state: "failed" as const, label: portfolioT(($) => $.holdings.surface.inlineLoadFailed) }
    : snapshot == null || loading
      ? { state: "loading" as const, label: portfolioT(($) => $.holdings.surface.inlineLoading) }
      : busy
        ? { state: "running" as const, label: portfolioT(($) => $.holdings.surface.inlineUpdating) }
        : positions.length === 0
          ? { state: "empty" as const, label: portfolioEmptyStateLabel(PORTFOLIO_VIEW.holdings, portfolioT) }
          : { state: "ready" as const, label: portfolioCountCopy(PORTFOLIO_VIEW.holdings, positions.length, portfolioT) };
  const errorTitle = err?.kind === "validation"
    ? portfolioValidationLabel(err.validation, portfolioT)
    : err?.kind === "error"
      ? presentPortfolioError(err.error, portfolioT).title
      : null;
  const editorNode = editing ? (
    <div className="ui-inline-form" key={editing.id}>
      {editing.broker === "manual" && (
        <>
          <label>
            <span>{portfolioT(($) => $.holdings.surface.inlineSymbolLabel)}</span>
            <input ref={editSymbolRef} aria-label={portfolioT(($) => $.holdings.surface.inlineSymbolAria)} defaultValue={editing.symbol} />
          </label>
          <label>
            <span>{portfolioT(($) => $.holdings.surface.inlineAssetLabel)}</span>
            <input
              ref={editAssetRef}
              aria-label={portfolioT(($) => $.holdings.surface.inlineAssetAria)}
              defaultValue={editing.asset_class}
            />
          </label>
          <label>
            <span>{portfolioT(($) => $.holdings.surface.inlineQuantityLabel)}</span>
            <input
              ref={editQuantityRef}
              aria-label={portfolioT(($) => $.holdings.surface.inlineQuantityAria)}
              inputMode="decimal"
              defaultValue={String(editing.quantity)}
            />
          </label>
          <label>
            <span>{portfolioT(($) => $.holdings.surface.inlineAverageCostLabel)}</span>
            <input
              ref={editAvgCostRef}
              aria-label={portfolioT(($) => $.holdings.surface.inlineAverageCostAria)}
              inputMode="decimal"
              placeholder={portfolioT(($) => $.holdings.surface.inlineClearHint)}
              defaultValue={editing.avg_cost == null ? "" : String(editing.avg_cost)}
            />
          </label>
          <label>
            <span>{portfolioT(($) => $.holdings.surface.inlineCurrencyLabel)}</span>
            <input ref={editCurrencyRef} aria-label={portfolioT(($) => $.holdings.surface.inlineCurrencyAria)} defaultValue={editing.currency} />
          </label>
        </>
      )}
      <label>
        <span>{portfolioT(($) => $.holdings.surface.inlineNotesLabel)}</span>
        <input ref={editNotesRef} aria-label={portfolioT(($) => $.holdings.surface.inlineNotesAria)} defaultValue={editing.notes ?? ""} />
      </label>
      <label>
        <span>{portfolioT(($) => $.holdings.surface.inlineThesisLabel)}</span>
        <input ref={editThesisRef} aria-label={portfolioT(($) => $.holdings.surface.inlineThesisAria)} defaultValue={editing.thesis ?? ""} />
      </label>
      <label>
        <span>{portfolioT(($) => $.holdings.surface.inlineTagsLabel)}</span>
        <input
          ref={editTagsRef}
          aria-label={portfolioT(($) => $.holdings.surface.inlineTagsAria)}
          placeholder={portfolioT(($) => $.holdings.surface.inlineTagsHint)}
          defaultValue={(editing.tags ?? []).join(", ")}
        />
      </label>
      <Button
        tone="primary"
        icon={<Save size={15} />}
        onClick={() => void onSaveEdit()}
        busy={busy === `edit-${editing.id}`}
      >
        {portfolioT(($) => $.holdings.surface.inlineSave)}
      </Button>
      <Button icon={<X size={15} />} onClick={() => setEditing(null)}>
        {portfolioT(($) => $.holdings.surface.inlineCancel)}
      </Button>
    </div>
  ) : null;

  return (
    <main className="main">
      <PageHeader
        eyebrow={portfolioT(($) => $.holdings.surface.eyebrow)}
        title={portfolioT(($) => $.holdings.surface.pageTitle)}
        context={<StatusBadge state={viewState.state} label={viewState.label} />}
        actions={(
          <IconButton
            label={portfolioT(($) => $.holdings.surface.refresh)}
            icon={<RefreshCw size={16} />}
            onClick={() => void load()}
            disabled={loading}
          />
        )}
      />

      {err ? (
        <InlineAlert state="failed" title={errorTitle ?? portfolioT(($) => $.holdings.operations.holdingsLoad)} />
      ) : null}

      {overviewErr ? (
        <InlineAlert state="partial" title={presentPortfolioError(overviewErr, portfolioT).title}>
          {portfolioT(($) => $.holdings.surface.retryGuidance)}
        </InlineAlert>
      ) : null}

      {overview ? (
        <PortfolioAccountSummary
          overview={overview}
          busyAccountId={
            busy?.startsWith("account-") ? Number(busy.slice(8)) : null
          }
          onToggleAggregate={(accountId, include) => {
            void onToggleAggregate(accountId, include);
          }}
        />
      ) : null}

      <div className="portfolio-view-tabs" role="tablist" aria-label={portfolioT(($) => $.holdings.surface.viewAria)}>
        {PORTFOLIO_VIEWS.map((view) => (
          <button
            key={view}
            ref={(node) => { tabRefs.current[view] = node; }}
            id={`portfolio-tab-${view}`}
            className="portfolio-view-tab"
            type="button"
            role="tab"
            tabIndex={activeView === view ? 0 : -1}
            aria-selected={activeView === view}
            aria-controls={`portfolio-panel-${view}`}
            onClick={() => setActiveView(view)}
            onKeyDown={(event) => onTabKeyDown(event, view)}
          >
            {portfolioViewLabel(view, portfolioT)}
          </button>
        ))}
      </div>

      {activeView === PORTFOLIO_VIEW.holdings ? (
        <div
          id="portfolio-panel-holdings"
          role="tabpanel"
          aria-labelledby="portfolio-tab-holdings"
        >
          <div
            className="portfolio-holdings-layout"
            data-has-recent={String(showRecent)}
          >
            <div className="portfolio-holdings-primary">
          <section className="ui-section-band">
            <div className="ui-section-head">
              <h2>{portfolioT(($) => $.holdings.surface.manualFormTitle)}</h2>
            </div>
            <div className="ui-inline-form">
              <label>
                <span>{portfolioT(($) => $.holdings.surface.manualTickerLabel)}</span>
                <input
                  ref={tickerRef}
                  aria-label={portfolioT(($) => $.holdings.surface.manualTickerLabel)}
                  placeholder={commonT(($) => $.labels.ticker)}
                />
              </label>
              <label>
                <span>{portfolioT(($) => $.holdings.surface.inlineQuantityLabel)}</span>
                <input
                  ref={quantityRef}
                  aria-label={portfolioT(($) => $.holdings.surface.inlineQuantityLabel)}
                  inputMode="decimal"
                  placeholder="1"
                />
              </label>
              <label>
                <span>{portfolioT(($) => $.holdings.surface.inlineNotesLabel)}</span>
                <input
                  ref={notesRef}
                  aria-label={portfolioT(($) => $.holdings.surface.inlineNotesLabel)}
                  placeholder={portfolioT(($) => $.holdings.surface.manualOptionalHint)}
                />
              </label>
              <Button
                tone="primary"
                icon={<Plus size={15} />}
                onClick={() => void onAddManual()}
                busy={busy === "manual"}
              >
                {portfolioT(($) => $.holdings.surface.manualSubmit)}
              </Button>
            </div>
          </section>

          <section className="ui-section-band">
            <div className="ui-section-head">
              <h2>{portfolioT(($) => $.holdings.surface.positionsTitle)}</h2>
              <div className="ui-action-row">
                <label className="muted tiny">
                  <span>{portfolioT(($) => $.holdings.surface.accountFilterLabel)}</span>
                  <select
                    aria-label={portfolioT(($) => $.holdings.surface.accountFilterAria)}
                    value={positionAccountId}
                    onChange={(event) => {
                      setPositionAccountId(
                        event.currentTarget.value === "all"
                          ? "all"
                          : Number(event.currentTarget.value),
                      );
                    }}
                  >
                    <option value="all">{portfolioT(($) => $.holdings.surface.allAccounts)}</option>
                    {accounts.map((account) => (
                      <option key={account.id} value={account.id}>
                        {accountLabels.get(account.id) ?? account.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="muted tiny">
                  <input
                    ref={closedFilterRef}
                    type="checkbox"
                    aria-label={portfolioT(($) => $.holdings.surface.includeClosedAria)}
                    checked={includeClosed}
                    onChange={(event) => setIncludeClosed(event.currentTarget.checked)}
                  />
                  {portfolioT(($) => $.holdings.surface.includeClosedLabel)}
                </label>
                <span className="muted tiny">
                  {standardPositions.length} {portfolioT(($) => $.holdings.surface.rowCountSuffix)}
                </span>
              </div>
            </div>
            <PositionsTable
              positions={standardPositions}
              accountLabels={accountLabels}
              emptyText={portfolioT(($) => $.holdings.surface.positionsEmpty)}
              editingId={editing?.id ?? null}
              editor={editorNode}
              busy={busy}
              t={portfolioT}
              onEdit={(position) => setEditing(position)}
              onClose={(position, trigger) => {
                closeTriggerRef.current = trigger;
                setPendingClose(position);
              }}
            />
          </section>

          {optionPositions.length > 0 && (
            <section className="ui-section-band">
              <div className="ui-section-head">
                <h2>{portfolioT(($) => $.holdings.surface.optionsTitle)}</h2>
                <span className="muted tiny">
                  {optionPositions.length} {portfolioT(($) => $.holdings.surface.rowCountSuffix)}
                </span>
              </div>
              <p className="muted">
                {portfolioT(($) => $.holdings.surface.optionsNotice)}
              </p>
              <PositionsTable
                positions={optionPositions}
                accountLabels={accountLabels}
                emptyText={portfolioT(($) => $.holdings.surface.optionsEmpty)}
                editingId={editing?.id ?? null}
                editor={editorNode}
                busy={busy}
                t={portfolioT}
                onEdit={(position) => setEditing(position)}
                onClose={(position, trigger) => {
                  closeTriggerRef.current = trigger;
                  setPendingClose(position);
                }}
              />
            </section>
          )}

          <ConfirmDialog
            open={pendingClose != null}
            title={pendingClose
              ? portfolioT(($) => $.holdings.surface.closeAria, { symbol: pendingClose.symbol })
              : portfolioT(($) => $.holdings.surface.closeAction)}
            consequence={portfolioT(($) => $.holdings.closeDialog.consequence)}
            confirmLabel={portfolioT(($) => $.holdings.closeDialog.confirm)}
            busy={pendingClose != null && busy === `close-${pendingClose.id}`}
            returnFocusRef={closeTriggerRef}
            fallbackFocusRef={closedFilterRef}
            onCancel={() => setPendingClose(null)}
            onConfirm={() => {
              if (pendingClose) void onCloseRow(pendingClose);
            }}
          />
            </div>
            {showRecent && recentActivity ? (
              <PortfolioRecentActivity
                page={recentActivity}
                onOpenActivity={() => setActiveView(PORTFOLIO_VIEW.activity)}
              />
            ) : null}
          </div>
        </div>
      ) : null}

      {activeView === PORTFOLIO_VIEW.activity ? (
        <div
          id="portfolio-panel-activity"
          role="tabpanel"
          aria-labelledby="portfolio-tab-activity"
        >
          <PortfolioActivity />
        </div>
      ) : null}

      {activeView === PORTFOLIO_VIEW.accountDetails ? (
        <div
          id="portfolio-panel-account_details"
          role="tabpanel"
          aria-labelledby="portfolio-tab-account_details"
        >
          {overview ? (
            <PortfolioAccountDetails overview={overview} />
          ) : (
            <InlineAlert
              state={loading ? "loading" : "empty"}
              title={loading
                ? portfolioT(($) => $.holdings.surface.accountDetailsLoading)
                : portfolioT(($) => $.holdings.surface.accountDetailsUnavailable)}
            />
          )}
        </div>
      ) : null}

      {activeView === PORTFOLIO_VIEW.syncRecords ? (
        <div
          id="portfolio-panel-sync_records"
          role="tabpanel"
          aria-labelledby="portfolio-tab-sync_records"
        >
          <PortfolioCapturePanel onPortfolioChanged={load} />
        </div>
      ) : null}
    </main>
  );
}

function PositionsTable({
  positions,
  accountLabels,
  emptyText,
  editingId,
  editor,
  busy,
  t,
  onEdit,
  onClose,
}: {
  positions: PortfolioPosition[];
  accountLabels: ReadonlyMap<number, string>;
  emptyText: string;
  editingId: number | null;
  editor: ReactNode;
  busy: string | null;
  t: PortfolioT;
  onEdit: (position: PortfolioPosition) => void;
  onClose: (position: PortfolioPosition, trigger: HTMLButtonElement) => void;
}) {
  const columns: DataTableColumn<PortfolioPosition>[] = [
    {
      id: "account",
      header: t(($) => $.tableLabels.holdingsAccount),
      render: (position) => (
        accountLabels.get(position.account_id) ?? `#${position.account_id}`
      ),
    },
    { id: "symbol", header: t(($) => $.tableLabels.holdingsSymbol), render: (position) => position.symbol },
    { id: "asset", header: t(($) => $.tableLabels.holdingsAsset), render: (position) => position.asset_class },
    {
      id: "quantity",
      header: t(($) => $.tableLabels.holdingsQuantity),
      align: "right",
      render: (position) => formatNum(position.quantity),
    },
    { id: "currency", header: t(($) => $.tableLabels.holdingsCurrency), render: (position) => position.currency },
    {
      id: "avg-cost",
      header: t(($) => $.tableLabels.holdingsAvgCost),
      align: "right",
      render: (position) => formatMaybe(position.avg_cost),
    },
    {
      id: "market-value",
      header: t(($) => $.tableLabels.holdingsMarketValue),
      align: "right",
      render: (position) => formatMaybe(position.market_value),
    },
    {
      id: "unrealized-pnl",
      header: t(($) => $.holdings.surface.unrealizedPnlLabel),
      align: "right",
      render: (position) => formatMaybe(position.unrealized_pnl),
    },
    { id: "notes", header: t(($) => $.tableLabels.holdingsNotes), render: (position) => position.notes ?? "" },
    {
      id: "status",
      header: t(($) => $.tableLabels.holdingsStatus),
      className: "ui-data-table-status",
      render: (position) => position.closed_at
        ? <span className="muted tiny">{t(($) => $.holdings.surface.closedStatus)}</span>
        : position.broker === "manual"
          ? null
          : <span className="muted tiny">{t(($) => $.holdings.surface.brokerSyncStatus)}</span>,
    },
  ];

  return (
    <DataTable<PortfolioPosition>
      ariaLabel={t(($) => $.holdings.surface.pageTitle)}
      rows={positions}
      columns={columns}
      rowKey={(position) => position.id}
      rowLabel={(position) => position.symbol}
      emptyText={emptyText}
      actions={(position) => [
        {
          id: "edit",
          label: t(($) => $.holdings.surface.editAction),
          disabled: busy != null,
          onSelect: onEdit,
        },
        ...(!position.closed_at && position.broker === "manual" ? [{
          id: "close",
          label: t(($) => $.holdings.surface.closeRowAction),
          tone: "danger" as const,
          disabled: busy != null,
          onSelect: onClose,
        }] : []),
      ]}
      renderExpandedRow={(position) => editingId === position.id ? editor : null}
    />
  );
}

function hasManualFinancialChange(
  before: PortfolioPosition,
  after: PortfolioPosition,
): boolean {
  return before.symbol !== after.symbol
    || before.asset_class !== after.asset_class
    || before.quantity !== after.quantity
    || before.avg_cost !== after.avg_cost
    || before.currency !== after.currency;
}

function splitTags(raw: string): string[] {
  return raw
    .split(",")
    .map((tag) => tag.trim())
    .filter((tag) => tag.length > 0);
}

function formatMaybe(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "";
  return formatNum(value);
}

function formatNum(value: number): string {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 4 }).format(value);
}
