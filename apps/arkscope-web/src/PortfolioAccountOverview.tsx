import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";

import type {
  PortfolioAccountValueSnapshot,
  PortfolioOverview,
  PortfolioOverviewAccount,
  PortfolioTotals,
} from "./api";
import type { PortfolioT } from "./i18n/portfolioPresentation";
import { formatSystemTimestamp } from "./timeDisplay";
import { DataTable, StatusBadge, type DataTableColumn } from "./ui";


export function PortfolioAccountSummary({
  overview,
  busyAccountId,
  onToggleAggregate,
}: {
  overview: PortfolioOverview;
  busyAccountId: number | null;
  onToggleAggregate: (accountId: number, include: boolean) => void;
}) {
  const { t } = useTranslation("portfolio");
  const manualRows = currencyRows(overview.manual_subtotal.totals);
  return (
    <section
      className="ui-section-band portfolio-account-summary"
      aria-label={t(($) => $.accountOverview.surface.summaryTitle)}
    >
      <div className="ui-section-head">
        <h2>{t(($) => $.accountOverview.surface.summaryTitle)}</h2>
      </div>
      {overview.accounts.map((account) => {
        const snapshot = account.latest_snapshot;
        const currency = snapshot?.base_currency ?? account.base_currency;
        return (
          <article className="portfolio-account-row" key={account.id}>
            <div className="ui-section-head">
              <div>
                <h3>{account.label}</h3>
                <span className="muted tiny">
                  {account.broker === "manual"
                    ? t(($) => $.accountOverview.surface.manualAccount)
                    : account.sync_mode}
                </span>
              </div>
              {account.broker === "manual" ? (
                <span className="muted tiny">
                  {t(($) => $.accountOverview.surface.valueUnavailable)}
                </span>
              ) : snapshot ? (
                <StatusBadge
                  state="ready"
                  label={t(($) => $.accountOverview.surface.snapshotAvailable)}
                />
              ) : (
                <StatusBadge
                  state="empty"
                  label={t(($) => $.accountOverview.surface.snapshotMissing)}
                />
              )}
            </div>

            {account.broker !== "manual" ? (
              <div className="portfolio-account-values">
                <Metric label={t(($) => $.accountOverview.surface.metricNetLiquidation)}>
                  {formatAmount(snapshot?.net_liquidation, currency)}
                </Metric>
                <Metric label={t(($) => $.accountOverview.surface.metricTotalCash)}>
                  {formatAmount(snapshot?.total_cash_value, currency)}
                </Metric>
                <Metric label={t(($) => $.accountOverview.surface.metricBuyingPower)}>
                  {formatAmount(snapshot?.buying_power, currency)}
                </Metric>
                <Metric label={t(($) => $.accountOverview.surface.metricDailyRealized)}>
                  {formatAmount(snapshot?.daily_realized_pnl, currency)}
                </Metric>
                <Metric label={t(($) => $.accountOverview.surface.metricDailyUnrealized)}>
                  {formatAmount(snapshot?.daily_unrealized_pnl, currency)}
                </Metric>
                <Metric label={t(($) => $.accountOverview.surface.metricDailyTotal)}>
                  {formatAmount(snapshot?.daily_total_pnl, currency)}
                </Metric>
              </div>
            ) : null}

            <div className="portfolio-account-times">
              <span className="muted tiny">
                {t(($) => $.accountOverview.surface.brokerObservedLine, {
                  timestamp: formatSystemTimestamp(snapshot?.as_of_utc),
                })}
              </span>
              <span className="muted tiny">
                {t(($) => $.accountOverview.surface.canonicalSyncLine, {
                  timestamp: formatSystemTimestamp(account.canonical_last_sync_at),
                })}
              </span>
            </div>
            <label className="muted tiny">
              <input
                type="checkbox"
                aria-label={t(($) => $.accountOverview.surface.includeAccountAria, {
                  account: account.label,
                })}
                checked={account.include_in_total}
                disabled={busyAccountId === account.id}
                onChange={(event) => onToggleAggregate(
                  account.id,
                  event.currentTarget.checked,
                )}
              />
              {t(($) => $.accountOverview.surface.includeAccountLabel)}
            </label>
          </article>
        );
      })}

      <section
        className="portfolio-manual-subtotal"
        aria-label={t(($) => $.accountOverview.surface.manualSubtotalTitle)}
      >
        <div className="ui-section-head">
          <div>
            <h3>{t(($) => $.accountOverview.surface.manualSubtotalTitle)}</h3>
            <span className="muted tiny">
              {t(($) => $.accountOverview.surface.manualSubtotalNotice)}
            </span>
          </div>
        </div>
        {manualRows.length === 0 ? (
          <p className="muted">{t(($) => $.accountOverview.surface.manualSubtotalEmpty)}</p>
        ) : (
          <div className="portfolio-account-values">
            {manualRows.map(([currency, row]) => (
              <div className="ui-metric" key={currency}>
                <span className="ui-metric-label">{currency}</span>
                <strong>{formatAmount(row.market_value, currency)}</strong>
                <span className="muted tiny">
                  {t(($) => $.accountOverview.surface.manualPositionSummary, {
                    count: row.position_count,
                  })}{" "}
                  {formatAmount(row.unrealized_pnl, currency)}
                </span>
              </div>
            ))}
            {overview.manual_subtotal.totals.broker_base ? (
              <div className="ui-metric">
                <span className="ui-metric-label">
                  {t(($) => $.accountOverview.surface.brokerBaseSubtotal)}
                </span>
                <strong>
                  {formatAmount(
                    overview.manual_subtotal.totals.broker_base.market_value,
                    null,
                  )}
                </strong>
                <span className="muted tiny">
                  {t(($) => $.accountOverview.surface.unrealizedPrefix)} {formatAmount(
                    overview.manual_subtotal.totals.broker_base.unrealized_pnl,
                    null,
                  )}
                </span>
              </div>
            ) : null}
          </div>
        )}
      </section>
    </section>
  );
}


export function PortfolioAccountDetails({ overview }: { overview: PortfolioOverview }) {
  const { t } = useTranslation("portfolio");
  const moneyColumns: DataTableColumn<PortfolioOverviewAccount>[] = moneyColumnSpecs
    .map(([id, field]) => ({
      id,
      header: moneyHeader(id, t),
      align: "right" as const,
      render: (account: PortfolioOverviewAccount) => {
        const snapshot = account.latest_snapshot;
        const currency = snapshot?.base_currency ?? account.base_currency;
        return formatAmount(snapshot?.[field], currency);
      },
    }));
  const columns: DataTableColumn<PortfolioOverviewAccount>[] = [
    {
      id: "account",
      header: t(($) => $.accountOverview.surface.detailsAccountHeader),
      render: (account) => (
        <>
          <strong>{account.label}</strong>
          <br />
          <span className="muted tiny">{account.broker}</span>
        </>
      ),
    },
    {
      id: "run",
      header: t(($) => $.tableLabels.accountCaptureRun),
      align: "right",
      render: (account) => account.latest_snapshot?.capture_run_id ?? "—",
    },
    {
      id: "currency",
      header: t(($) => $.tableLabels.accountBaseCurrency),
      render: (account) => account.latest_snapshot?.base_currency
        ?? account.base_currency
        ?? "—",
    },
    ...moneyColumns,
    {
      id: "source",
      header: t(($) => $.accountOverview.surface.detailsSourceHeader),
      render: (account) => account.latest_snapshot
        ? `${account.latest_snapshot.source} · ${account.latest_snapshot.as_of_kind}`
        : account.broker === "manual"
          ? t(($) => $.accountOverview.surface.detailsManualUnavailable)
          : t(($) => $.accountOverview.empty.accountSnapshot),
    },
    {
      id: "broker-time",
      header: t(($) => $.accountOverview.surface.detailsBrokerObservedHeader),
      render: (account) => formatSystemTimestamp(account.latest_snapshot?.as_of_utc),
    },
    {
      id: "canonical-time",
      header: t(($) => $.accountOverview.surface.detailsCanonicalSyncHeader),
      render: (account) => formatSystemTimestamp(account.canonical_last_sync_at),
    },
  ];
  return (
    <section className="ui-section-band portfolio-account-details">
      <div className="ui-section-head">
        <div>
          <h2>{t(($) => $.accountOverview.surface.detailsTitle)}</h2>
          <p className="muted">
            {t(($) => $.accountOverview.surface.detailsNotice)}
          </p>
        </div>
      </div>
      <DataTable<PortfolioOverviewAccount>
        ariaLabel={t(($) => $.accountOverview.surface.detailsTableAria)}
        rows={overview.accounts}
        columns={columns}
        rowKey={(account) => account.id}
        rowLabel={(account) => account.label}
        emptyText={t(($) => $.accountOverview.empty.accounts)}
      />
    </section>
  );
}


type MoneyField = keyof Pick<
  PortfolioAccountValueSnapshot,
  | "net_liquidation"
  | "total_cash_value"
  | "settled_cash"
  | "gross_position_value"
  | "buying_power"
  | "available_funds"
  | "initial_margin_requirement"
  | "maintenance_margin_requirement"
  | "daily_realized_pnl"
  | "daily_unrealized_pnl"
  | "daily_total_pnl"
>;

type MoneyColumnId =
  | "net-liquidation"
  | "total-cash"
  | "settled-cash"
  | "gross-position"
  | "buying-power"
  | "available-funds"
  | "initial-margin"
  | "maintenance-margin"
  | "daily-realized"
  | "daily-unrealized"
  | "daily-total";

const moneyColumnSpecs: Array<[MoneyColumnId, MoneyField]> = [
  ["net-liquidation", "net_liquidation"],
  ["total-cash", "total_cash_value"],
  ["settled-cash", "settled_cash"],
  ["gross-position", "gross_position_value"],
  ["buying-power", "buying_power"],
  ["available-funds", "available_funds"],
  ["initial-margin", "initial_margin_requirement"],
  ["maintenance-margin", "maintenance_margin_requirement"],
  ["daily-realized", "daily_realized_pnl"],
  ["daily-unrealized", "daily_unrealized_pnl"],
  ["daily-total", "daily_total_pnl"],
];

function moneyHeader(id: MoneyColumnId, t: PortfolioT): string {
  switch (id) {
    case "net-liquidation": return t(($) => $.tableLabels.netLiquidation);
    case "total-cash": return t(($) => $.tableLabels.totalCash);
    case "settled-cash": return t(($) => $.tableLabels.settledCash);
    case "gross-position": return t(($) => $.tableLabels.grossPositionValue);
    case "buying-power": return t(($) => $.tableLabels.buyingPower);
    case "available-funds": return t(($) => $.tableLabels.availableFunds);
    case "initial-margin": return t(($) => $.tableLabels.initialMargin);
    case "maintenance-margin": return t(($) => $.tableLabels.maintenanceMargin);
    case "daily-realized": return t(($) => $.accountOverview.table.dailyRealized);
    case "daily-unrealized": return t(($) => $.accountOverview.table.dailyUnrealized);
    case "daily-total": return t(($) => $.accountOverview.table.dailyTotal);
  }
}


function Metric({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="ui-metric">
      <span className="ui-metric-label">{label}</span>
      <strong>{children}</strong>
    </div>
  );
}


function formatAmount(
  value: number | null | undefined,
  currency: string | null,
): string {
  if (value == null || !Number.isFinite(value)) return "—";
  if (!currency) {
    return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(value);
  }
  try {
    return new Intl.NumberFormat(undefined, {
      style: "currency",
      currency,
      maximumFractionDigits: 2,
    }).format(value);
  } catch {
    const formatted = new Intl.NumberFormat(
      undefined,
      { maximumFractionDigits: 2 },
    ).format(value);
    return `${formatted} ${currency}`;
  }
}


function currencyRows(totals: PortfolioTotals) {
  return Object.entries(totals.per_currency)
    .sort(([left], [right]) => left.localeCompare(right));
}
