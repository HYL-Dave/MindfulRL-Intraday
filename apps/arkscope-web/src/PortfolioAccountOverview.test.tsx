/** @vitest-environment jsdom */
import React, { type ReactNode } from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import i18n from "i18next";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  PortfolioAccountDetails,
  PortfolioAccountSummary,
} from "./PortfolioAccountOverview";
import { HoldingsView } from "./Holdings";
import type {
  PortfolioActivityPage,
  PortfolioOverview,
  PortfolioSnapshot,
} from "./api";
import {
  capturePortfolioError,
  presentPortfolioError,
} from "./i18n/portfolioPresentation";
import { formatSystemTimestamp } from "./timeDisplay";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
  .IS_REACT_ACT_ENVIRONMENT = true;

let root: ReturnType<typeof createRoot> | null = null;
let host: HTMLDivElement | null = null;

beforeEach(async () => {
  await i18n.changeLanguage("zh-Hant");
  document.documentElement.lang = "zh-Hant";
});

function clearRendered() {
  if (root) act(() => root!.unmount());
  root = null;
  host?.remove();
  host = null;
}

afterEach(() => {
  clearRendered();
  vi.unstubAllGlobals();
});

const overview = (over: Partial<PortfolioOverview> = {}): PortfolioOverview => ({
  accounts: [
    {
      id: 1,
      label: "Manual",
      broker: "manual",
      broker_account_id_hash: null,
      sync_mode: "manual",
      base_currency: "USD",
      include_in_total: true,
      canonical_last_sync_at: null,
      latest_snapshot: null,
    },
    {
      id: 2,
      label: "IBKR · hash-one",
      broker: "ibkr",
      broker_account_id_hash: "hash-one",
      sync_mode: "ibkr_review",
      base_currency: "USD",
      include_in_total: true,
      canonical_last_sync_at: "2026-07-14T05:01:00+00:00",
      latest_snapshot: {
        capture_run_id: 50,
        as_of_utc: "2026-07-14T05:00:00+00:00",
        as_of_kind: "capture_completed",
        source: "ibkr_gateway",
        base_currency: "USD",
        net_liquidation: 100_000,
        total_cash_value: 10_000,
        settled_cash: 9_000,
        gross_position_value: 90_000,
        buying_power: 25_000,
        available_funds: 20_000,
        initial_margin_requirement: 15_000,
        maintenance_margin_requirement: 12_000,
        daily_realized_pnl: 125,
        daily_unrealized_pnl: -25,
        daily_total_pnl: 100,
      },
    },
  ],
  manual_subtotal: {
    included_account_ids: [1],
    totals: {
      currency_basis: "per_currency",
      per_currency: {
        USD: { position_count: 1, market_value: 500, unrealized_pnl: 25 },
        TWD: { position_count: 1, market_value: 10_000, unrealized_pnl: -500 },
      },
      broker_base: null,
    },
  },
  ...over,
});

function render(node: ReactNode) {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => root!.render(node));
}

function renderSummary(
  value = overview(),
  onToggleAggregate = vi.fn(),
) {
  render(
    <PortfolioAccountSummary
      overview={value}
      busyAccountId={null}
      onToggleAggregate={onToggleAggregate}
    />,
  );
  return onToggleAggregate;
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function holdingsSnapshot(
  positions: PortfolioSnapshot["positions"] = [],
): PortfolioSnapshot {
  return {
    accounts: [{
      id: 1,
      label: "Manual",
      broker: "manual",
      sync_mode: "manual",
      base_currency: "USD",
      include_in_total: true,
    }],
    positions,
    totals: { currency_basis: "per_currency", per_currency: {}, broker_base: null },
    included_account_ids: [1],
  };
}

function emptyRecentActivity(): PortfolioActivityPage {
  return {
    accounts: [],
    history_started_at_utc: null,
    items: [],
    summary: { item_count: 0, unmatched_count: 0, recent_window_days: null },
    next_cursor: null,
  };
}

async function mountHoldingsOwner({
  portfolio,
  accountOverview = () => jsonResponse(overview()),
}: {
  portfolio: () => Response | Promise<Response>;
  accountOverview?: () => Response | Promise<Response>;
}) {
  clearRendered();
  vi.stubGlobal("fetch", vi.fn((url: unknown) => {
    const path = new URL(String(url)).pathname;
    if (path === "/portfolio/activity") return Promise.resolve(jsonResponse(emptyRecentActivity()));
    if (path === "/portfolio/overview") return Promise.resolve(accountOverview());
    if (path === "/portfolio") return Promise.resolve(portfolio());
    return Promise.resolve(jsonResponse({ detail: "unexpected test route" }, 404));
  }));
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  await act(async () => { root!.render(<HoldingsView />); });
}

async function flushOwner() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

function metricValue(label: string): string {
  const metric = Array.from(host!.querySelectorAll<HTMLElement>(".ui-metric"))
    .find((candidate) => candidate.textContent?.includes(label));
  if (!metric) throw new Error(`metric not found: ${label}`);
  return metric.querySelector("strong")?.textContent ?? "";
}

describe("Portfolio account overview", () => {
  it("renders_every_visible_account_even_without_snapshot", () => {
    const value = overview({
      accounts: overview().accounts.map((account) => ({
        ...account,
        latest_snapshot: null,
      })),
    });

    renderSummary(value);

    expect(host!.textContent).toContain("Manual");
    expect(host!.textContent).toContain("IBKR · hash-one");
    expect(host!.textContent).toContain("無帳戶價值資料");
    expect(host!.textContent).toContain("尚無帳戶快照");
  });

  it("renders_broker_values_and_both_timestamps", () => {
    renderSummary();

    expect(metricValue("Net Liquidation")).not.toBe("—");
    expect(metricValue("Total Cash")).not.toBe("—");
    expect(metricValue("Buying Power")).not.toBe("—");
    expect(host!.textContent).toContain(
      `Broker 觀察：${formatSystemTimestamp("2026-07-14T05:00:00+00:00")}`,
    );
    expect(host!.textContent).toContain(
      `本地持倉核准 / 同步：${formatSystemTimestamp("2026-07-14T05:01:00+00:00")}`,
    );
  });

  it("labels_daily_total_as_realized_plus_unrealized", () => {
    renderSummary();

    expect(host!.textContent).toContain("今日損益合計（已實現 + 未實現，ET）");
    expect(metricValue("今日損益合計（已實現 + 未實現，ET）")).not.toBe("—");
  });

  it("does_not_invent_daily_total_when_one_provider_leg_is_missing", () => {
    const value = overview();
    value.accounts[1].latest_snapshot = {
      ...value.accounts[1].latest_snapshot!,
      daily_unrealized_pnl: null,
      daily_total_pnl: null,
    };

    renderSummary(value);

    expect(metricValue("今日已實現損益（ET）")).not.toBe("—");
    expect(metricValue("今日未實現損益（ET）")).toBe("—");
    expect(metricValue("今日損益合計（已實現 + 未實現，ET）")).toBe("—");
  });

  it("renders_manual_subtotal_by_currency_without_overall_net_worth", () => {
    renderSummary();

    const subtotal = host!.querySelector<HTMLElement>(
      '[aria-label="手動帳戶持倉小計"]',
    )!;
    expect(subtotal.textContent).toContain("手動帳戶持倉小計");
    expect(subtotal.textContent).toContain("USD");
    expect(subtotal.textContent).toContain("TWD");
    expect(host!.textContent).not.toContain("整體淨值");
  });

  it("keeps_manual_subtotal_separate_from_ibkr_net_liquidation", () => {
    renderSummary();

    const subtotal = host!.querySelector<HTMLElement>(
      '[aria-label="手動帳戶持倉小計"]',
    )!;
    expect(subtotal.textContent).toContain("不與 IBKR Net Liquidation 相加");
    expect(subtotal.textContent).not.toContain("Net Liquidation$100,000");
  });

  it("renders_all_latest_snapshot_fields_in_account_details", () => {
    render(<PortfolioAccountDetails overview={overview()} />);

    for (const label of [
      "Capture Run",
      "Base Currency",
      "Net Liquidation",
      "Total Cash",
      "Settled Cash",
      "Gross Position Value",
      "Buying Power",
      "Available Funds",
      "Initial Margin",
      "Maintenance Margin",
      "今日已實現（ET）",
      "今日未實現（ET）",
      "今日合計（已實現 + 未實現，ET）",
      "Broker 觀察",
      "本地持倉核准 / 同步",
    ]) {
      expect(host!.textContent).toContain(label);
    }
  });

  it("keeps_account_details_inside_the_data_table_scroll_owner", () => {
    render(<PortfolioAccountDetails overview={overview()} />);

    const table = host!.querySelector('table[aria-label="帳戶最新快照明細"]');
    expect(table).not.toBeNull();
    expect(table!.parentElement?.classList.contains("ui-data-table-wrap")).toBe(true);
    expect(table!.closest(".portfolio-account-details")).not.toBeNull();
  });

  it("emits_include_toggle_for_each_account", () => {
    const onToggle = renderSummary(overview(), vi.fn());
    const toggles = host!.querySelectorAll<HTMLInputElement>(
      'input[type="checkbox"][aria-label$="納入總計"]',
    );

    expect(toggles).toHaveLength(2);
    act(() => toggles[1].click());
    expect(onToggle).toHaveBeenCalledWith(2, false);
  });

  it("never_renders_an_unexpected_raw_broker_account_id_property", () => {
    const value = overview();
    value.accounts[1] = {
      ...value.accounts[1],
      broker_account_id: "DU-RAW-SHOULD-NOT-RENDER",
    } as typeof value.accounts[number] & { broker_account_id: string };

    renderSummary(value);

    expect(host!.textContent).not.toContain("DU-RAW-SHOULD-NOT-RENDER");
  });

  it("renders account and value headers in both locales", async () => {
    render(
      <>
        <PortfolioAccountSummary overview={overview()} busyAccountId={null} onToggleAggregate={vi.fn()} />
        <PortfolioAccountDetails overview={overview()} />
      </>,
    );
    expect(host!.textContent).toContain("帳戶總覽");
    expect(host!.textContent).toContain("今日損益合計（已實現 + 未實現，ET）");
    expect(host!.textContent).toContain("帳戶明細");

    await act(async () => { await i18n.changeLanguage("en"); });
    expect(host!.textContent).toContain("Account overview");
    expect(host!.textContent).toContain("Today's total P&L (realized + unrealized, ET)");
    expect(host!.textContent).toContain("Account details");
  });

  it("preserves account currency and measured values", async () => {
    const value = overview();
    value.manual_subtotal.totals.per_currency = {
      USD: { position_count: 1, market_value: 500, unrealized_pnl: 25 },
      TWD: { position_count: 2, market_value: 10_000, unrealized_pnl: -500 },
    };
    renderSummary(value);
    const brokerTimestamp = formatSystemTimestamp("2026-07-14T05:00:00+00:00");
    const canonicalTimestamp = formatSystemTimestamp("2026-07-14T05:01:00+00:00");
    const sourceFacts = [
      "IBKR · hash-one",
      "USD",
      metricValue("Net Liquidation"),
      metricValue("Total Cash"),
      brokerTimestamp,
      canonicalTimestamp,
    ];
    expect(host!.textContent).toContain("1 筆 · 未實現");
    expect(host!.textContent).toContain("2 筆 · 未實現");
    expect(host!.textContent).toContain(`Broker 觀察：${brokerTimestamp}`);
    expect(host!.textContent).toContain(`本地持倉核准 / 同步：${canonicalTimestamp}`);
    await act(async () => { await i18n.changeLanguage("en"); });
    for (const fact of sourceFacts) expect(host!.textContent).toContain(fact);
    expect(host!.textContent).toContain("Positions: 1 · Unrealized");
    expect(host!.textContent).toContain("Positions: 2 · Unrealized");
    expect(host!.textContent).toContain(`Broker observed: ${brokerTimestamp}`);
    expect(host!.textContent).toContain(`Local holdings approval / sync: ${canonicalTimestamp}`);
    expect(host!.textContent).not.toContain(`Broker observed:${brokerTimestamp}`);
    expect(host!.textContent).not.toContain(`Local holdings approval / sync:${canonicalTimestamp}`);
  });

  it("localizes loading empty partial and error states", async () => {
    const value = overview({
      accounts: overview().accounts.map((account) => ({ ...account, latest_snapshot: null })),
    });
    renderSummary(value);
    expect(host!.textContent).toContain("尚無帳戶快照");
    expect(host!.textContent).toContain("無帳戶價值資料");
    expect(presentPortfolioError(
      capturePortfolioError("overview_load", { status: 503, path: "/portfolio/overview" }),
      i18n.getFixedT("zh-Hant", "portfolio"),
    ).title).toBe("帳戶總覽無法載入；持倉仍可使用");

    await act(async () => { await i18n.changeLanguage("en"); });
    expect(host!.textContent).toContain("No account snapshot yet");
    expect(host!.textContent).toContain("No account value data");
    expect(presentPortfolioError(
      capturePortfolioError("overview_load", { status: 503, path: "/portfolio/overview" }),
      i18n.getFixedT("en", "portfolio"),
    ).title).toBe("Could not load account overview; holdings remain available");

    await act(async () => { await i18n.changeLanguage("zh-Hant"); });
    const pendingPortfolio = deferred<Response>();
    await mountHoldingsOwner({ portfolio: () => pendingPortfolio.promise });
    expect(host!.querySelector('[data-state="loading"]')?.textContent).toContain("載入持倉");
    await act(async () => { await i18n.changeLanguage("en"); });
    expect(host!.querySelector('[data-state="loading"]')?.textContent).toContain("Loading holdings");
    pendingPortfolio.resolve(jsonResponse(holdingsSnapshot()));
    await flushOwner();
    expect(host!.querySelector('[data-state="empty"]')?.textContent).toContain("No holdings");
    await act(async () => { await i18n.changeLanguage("zh-Hant"); });
    expect(host!.querySelector('[data-state="empty"]')?.textContent).toContain("尚無持倉");

    await mountHoldingsOwner({
      portfolio: () => jsonResponse(holdingsSnapshot([{
        id: 10,
        account_id: 1,
        broker: "manual",
        symbol: "NVDA",
        asset_class: "stock",
        quantity: 3,
        currency: "USD",
      }])),
      accountOverview: () => jsonResponse({ detail: "partial overview planted detail" }, 503),
    });
    await flushOwner();
    expect(host!.querySelector('[data-state="partial"]')?.textContent)
      .toContain("帳戶總覽無法載入；持倉仍可使用");
    expect(host!.textContent).toContain("NVDA");
    expect(host!.textContent).not.toContain("partial overview planted detail");
    await act(async () => { await i18n.changeLanguage("en"); });
    expect(host!.querySelector('[data-state="partial"]')?.textContent)
      .toContain("Could not load account overview; holdings remain available");
    expect(host!.textContent).toContain("NVDA");

    await act(async () => { await i18n.changeLanguage("zh-Hant"); });
    await mountHoldingsOwner({
      portfolio: () => jsonResponse({ detail: "full portfolio planted detail" }, 503),
    });
    await flushOwner();
    expect(host!.querySelector('[data-state="failed"]')?.textContent).toContain("載入失敗");
    expect(host!.textContent).toContain("持倉載入失敗");
    expect(host!.textContent).not.toContain("full portfolio planted detail");
    await act(async () => { await i18n.changeLanguage("en"); });
    expect(host!.querySelector('[data-state="failed"]')?.textContent).toContain("Load failed");
    expect(host!.textContent).toContain("Could not load holdings");
    expect(host!.textContent).not.toContain("full portfolio planted detail");
  });

  it("preserves selected account across locale changes", async () => {
    renderSummary();
    const selected = host!.querySelector<HTMLInputElement>('input[aria-label="Manual 納入總計"]')!;
    selected.dataset.identity = "selected-account";
    selected.focus();
    await act(async () => { await i18n.changeLanguage("en"); });

    const after = host!.querySelector<HTMLInputElement>('input[aria-label="Include Manual in totals"]')!;
    expect(after).toBe(selected);
    expect(after.checked).toBe(true);
    expect(after.dataset.identity).toBe("selected-account");
    expect(document.activeElement).toBe(after);
  });

  it("omits raw diagnostics in normal mode", () => {
    const value = overview();
    value.accounts[1] = {
      ...value.accounts[1],
      broker_account_id: "DU-PLANTED-RAW-ID",
      error_detail: "sqlite3 traceback planted detail",
    } as typeof value.accounts[number] & { broker_account_id: string; error_detail: string };
    render(
      <>
        <PortfolioAccountSummary overview={value} busyAccountId={null} onToggleAggregate={vi.fn()} />
        <PortfolioAccountDetails overview={value} />
      </>,
    );
    expect(host!.textContent).not.toContain("DU-PLANTED-RAW-ID");
    expect(host!.textContent).not.toContain("sqlite3 traceback planted detail");
    expect(presentPortfolioError(
      capturePortfolioError("overview_load", {
        status: 503,
        code: "active_universe_unavailable",
        path: "/portfolio/overview?account=DU-PLANTED-RAW-ID",
      }),
      i18n.getFixedT("zh-Hant", "portfolio"),
      false,
    ).diagnostics).toEqual([]);
  });
});
