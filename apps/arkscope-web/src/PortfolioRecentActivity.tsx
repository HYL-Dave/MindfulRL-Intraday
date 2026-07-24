import { ArrowRight } from "lucide-react";
import { useTranslation } from "react-i18next";

import type { PortfolioActivityItem, PortfolioActivityPage } from "./api";
import {
  portfolioCountCopy,
  type PortfolioT,
} from "./i18n/portfolioPresentation";
import { formatMarketTimestamp, formatSystemTimestamp } from "./timeDisplay";
import { Button } from "./ui";

export function PortfolioRecentActivity({
  page,
  onOpenActivity,
}: {
  page: PortfolioActivityPage;
  onOpenActivity: () => void;
}) {
  const { t } = useTranslation("portfolio");
  if (page.items.length === 0 && page.summary.unmatched_count === 0) return null;

  const days = page.summary.recent_window_days ?? 7;
  return (
    <aside
      className="portfolio-recent-activity"
      aria-labelledby="portfolio-recent-title"
    >
      <div className="portfolio-recent-head">
        <div>
          <h2 id="portfolio-recent-title">{t(($) => $.recentActivity.surface.title)}</h2>
          <p className="muted tiny">
            {t(($) => $.recentActivity.surface.windowPrefix)} {days}{" "}
            {t(($) => $.recentActivity.surface.daySuffix)}
          </p>
        </div>
        <Button
          size="compact"
          icon={<ArrowRight size={15} />}
          aria-label={t(($) => $.recentActivity.surface.openFullAria)}
          onClick={onOpenActivity}
        >
          {t(($) => $.recentActivity.surface.openFullAction)}
        </Button>
      </div>

      {page.summary.unmatched_count > 0 ? (
        <p className="portfolio-recent-unmatched">
          {t(($) => $.recentActivity.surface.unmatchedWindowPrefix)} {days}{" "}
          {t(($) => $.recentActivity.surface.unmatchedWindowMiddle)}{" "}
          {page.summary.unmatched_count}{" "}
          {t(($) => $.recentActivity.surface.unmatchedWindowSuffix)}
        </p>
      ) : null}

      {page.items.length > 0 ? (
        <ul className="portfolio-recent-list">
          {page.items.map((item) => (
            <li key={item.id}>
              <strong>{eventLabel(item, t)}</strong>
              <span>{compactFact(item, t)}</span>
              <span className="muted tiny">
                {accountLabel(item, t)} · {compactTime(item)}
              </span>
            </li>
          ))}
        </ul>
      ) : null}
    </aside>
  );
}

function eventLabel(item: PortfolioActivityItem, t: PortfolioT): string {
  switch (item.kind) {
    case "order": return item.symbol
      ? t(($) => $.recentActivity.surface.eventOrderWithSymbol, { symbol: item.symbol })
      : t(($) => $.recentActivity.surface.eventOrder);
    case "execution": return item.symbol
      ? t(($) => $.recentActivity.surface.eventExecutionWithSymbol, { symbol: item.symbol })
      : t(($) => $.recentActivity.surface.eventExecution);
    case "unmatched": return item.symbol
      ? t(($) => $.recentActivity.surface.eventUnmatchedWithSymbol, { symbol: item.symbol })
      : t(($) => $.recentActivity.surface.eventUnmatched);
    case "manual_adjustment": return t(($) => $.recentActivity.surface.eventManualWithSymbol, {
      symbol: item.symbol,
    });
    case "coverage_gap": return item.reason_code === "broker_day_gap"
      ? t(($) => $.recentActivity.surface.eventBrokerGap)
      : t(($) => $.recentActivity.surface.eventExecutionIncomplete);
    case "history_start": return t(($) => $.recentActivity.surface.eventHistoryStart);
  }
}

function compactFact(item: PortfolioActivityItem, t: PortfolioT): string {
  switch (item.kind) {
    case "order":
    case "execution":
      return [
        `${sideLabel(item.objective.side, t)} ${formatNumber(item.objective.quantity)}`,
        outcomeLabel(item.objective.realized_outcome, t),
      ].join(" · ");
    case "unmatched":
      return t(($) => $.recentActivity.surface.residualFact, {
        quantity: formatNumber(item.residual_quantity),
      });
    case "manual_adjustment":
      return `${manualActionLabel(item.action, t)} · ${portfolioCountCopy(
        "recent_fields",
        item.changes.length,
        t,
      )}`;
    case "coverage_gap":
      return item.reason_code === "broker_day_gap"
        ? t(($) => $.recentActivity.surface.brokerGapFact)
        : t(($) => $.recentActivity.surface.executionIncompleteFact);
    case "history_start":
      return t(($) => $.recentActivity.surface.historyStartFact);
  }
}

function accountLabel(item: PortfolioActivityItem, t: PortfolioT): string {
  return item.account?.label ?? t(($) => $.recentActivity.surface.allAccounts);
}

function compactTime(item: PortfolioActivityItem): string {
  const formatted = item.kind === "order" || item.kind === "execution"
    ? formatMarketTimestamp(item.occurred_at_utc)
    : formatSystemTimestamp(item.occurred_at_utc);
  return formatted.split(" · ")[0];
}

function sideLabel(value: "buy" | "sell" | "mixed" | "unknown", t: PortfolioT): string {
  switch (value) {
    case "buy": return t(($) => $.recentActivity.surface.sideBuy);
    case "sell": return t(($) => $.recentActivity.surface.sideSell);
    case "mixed": return t(($) => $.recentActivity.surface.sideMixed);
    case "unknown": return t(($) => $.recentActivity.surface.sideUnknown);
  }
}

function manualActionLabel(value: "create" | "update" | "close", t: PortfolioT): string {
  switch (value) {
    case "create": return t(($) => $.recentActivity.surface.manualCreate);
    case "update": return t(($) => $.recentActivity.surface.manualUpdate);
    case "close": return t(($) => $.recentActivity.surface.manualClose);
  }
}

function outcomeLabel(value: "gain" | "loss" | "flat" | "unknown", t: PortfolioT): string {
  switch (value) {
    case "gain": return t(($) => $.recentActivity.surface.outcomeGain);
    case "loss": return t(($) => $.recentActivity.surface.outcomeLoss);
    case "flat": return t(($) => $.recentActivity.surface.outcomeFlat);
    case "unknown": return t(($) => $.recentActivity.surface.outcomeUnknown);
  }
}

function formatNumber(value: number | null): string {
  return value == null
    ? "—"
    : new Intl.NumberFormat("en-US", { maximumFractionDigits: 6 }).format(value);
}
