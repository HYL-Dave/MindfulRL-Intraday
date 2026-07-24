/** @vitest-environment jsdom */
import React, { type ComponentProps } from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import i18n from "i18next";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ResearchRunDTO } from "./api";
import { ResearchEvidenceDrawer } from "./ResearchEvidenceDrawer";
import type { Message, TraceRow } from "./researchReducer";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
  .IS_REACT_ACT_ENVIRONMENT = true;

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function run(over: Partial<ResearchRunDTO> = {}): ResearchRunDTO {
  return {
    id: "run-evidence",
    thread_id: "thread-evidence",
    status: "succeeded",
    question: "SOURCE_QUESTION",
    ticker: "MU",
    provider: "openai",
    model: "gpt-5.6-luna",
    effort: "high",
    auth_mode: "api_key",
    credential_id: "local:7",
    started_at: "2026-07-20T00:01:00Z",
    completed_at: "2026-07-20T00:02:00Z",
    error: null,
    token_usage: {
      total_input_tokens: 1234,
      total_output_tokens: 56,
    },
    created_at: "2026-07-20T00:00:00Z",
    updated_at: "2026-07-20T00:02:00Z",
    ...over,
  };
}

const PERSONALIZATION = {
  profile_active: true,
  assistant_stance: "complementary" as const,
  skill_mode: "suggest_only" as const,
  suggested_skills: ["SOURCE_SUGGESTED_SKILL"],
  applied_skills: ["SOURCE_APPLIED_SKILL"],
  context_snapshot: "SOURCE_CONTEXT_BYTES::<keep exactly>\nline-two",
};

function message(over: Partial<Message> = {}): Message {
  return {
    role: "assistant",
    content: "SOURCE_GENERATED_ANSWER",
    provider: "openai",
    model: "gpt-5.6-luna",
    effort: "high",
    tools_used: ["source_tool"],
    tool_calls: [{
      name: "source_tool",
      input: { query: "SOURCE_INPUT_BYTES", limit: 7 },
      result_preview: "SOURCE_RESULT_BYTES::<verbatim>",
    }],
    token_usage: { total_input_tokens: 1234, total_output_tokens: 56 },
    tickers: ["MU"],
    elapsed_seconds: 12.5,
    created_at: "2026-07-20T00:03:00Z",
    personalization: PERSONALIZATION,
    runId: "run-evidence",
    ...over,
  };
}

type EvidenceProps = ComponentProps<typeof ResearchEvidenceDrawer>;

let root: ReturnType<typeof createRoot> | null = null;
let host: HTMLDivElement | null = null;
let currentProps: EvidenceProps | null = null;

function stubMatchMedia(matches = false) {
  vi.stubGlobal("matchMedia", vi.fn((query: string) => ({
    matches,
    media: query,
    onchange: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(() => true),
  })));
}

function stubEvidenceFetch({
  runResponse = json({ run: run() }),
  eventsResponse = json({
    run: run(),
    events: [{
      seq: 1,
      type: "diagnostic",
      created_at: "2026-07-20T00:01:30Z",
      data: { access_token: "SECRET_DIAGNOSTIC_TOKEN", source: "SOURCE_EVENT" },
    }],
    has_more: false,
  }),
}: {
  runResponse?: Response | Promise<Response>;
  eventsResponse?: Response | Promise<Response>;
} = {}) {
  const fetchMock = vi.fn(async (input: string | URL | Request) => {
    const raw = typeof input === "string"
      ? input
      : input instanceof URL ? input.href : input.url;
    const url = new URL(raw);
    if (url.pathname === "/research/runs/run-evidence/events") {
      return await eventsResponse;
    }
    if (url.pathname === "/research/runs/run-evidence") {
      return await runResponse;
    }
    throw new Error(`unhandled test request: ${url.pathname}${url.search}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

async function flush() {
  await act(async () => {
    for (let index = 0; index < 10; index += 1) await Promise.resolve();
    await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
    for (let index = 0; index < 4; index += 1) await Promise.resolve();
  });
}

async function mountEvidence(over: Partial<EvidenceProps> = {}) {
  stubMatchMedia(false);
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  currentProps = {
    open: true,
    pinned: false,
    onClose: vi.fn(),
    onPinnedChange: vi.fn(),
    message: message(),
    activeTrace: [] as TraceRow[],
    activeRun: null,
    developerMode: false,
    ...over,
  };
  await act(async () => {
    root!.render(<ResearchEvidenceDrawer {...currentProps!} />);
  });
  await flush();
}

async function rerenderEvidence(over: Partial<EvidenceProps>) {
  currentProps = { ...currentProps!, ...over };
  await act(async () => {
    root!.render(<ResearchEvidenceDrawer {...currentProps!} />);
  });
  await flush();
}

function detailRow(label: string): HTMLDivElement | undefined {
  return Array.from(document.querySelectorAll<HTMLDivElement>(
    ".research-run-detail-list > div",
  )).find((row) => row.querySelector("dt")?.textContent === label);
}

afterEach(async () => {
  if (root) act(() => root!.unmount());
  root = null;
  host?.remove();
  host = null;
  currentProps = null;
  document.body.replaceChildren();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  await i18n.changeLanguage("zh-Hant");
});

describe("Research Evidence drawer", () => {
  it("localizes headings token statistics and timing labels in both locales", async () => {
    await i18n.changeLanguage("zh-Hant");
    stubEvidenceFetch();
    await mountEvidence();
    await vi.waitFor(() => expect(detailRow("總輸入 tokens")).toBeDefined());

    const drawer = document.querySelector("[role='dialog']")!;
    const inputTokens = detailRow("總輸入 tokens")!;
    const created = detailRow("建立")!;
    expect(drawer.textContent).toContain("證據與執行詳情");
    expect(drawer.textContent).toContain("工具證據");
    expect(inputTokens.querySelector("dd")?.textContent).toBe("1,234");

    await act(async () => { await i18n.changeLanguage("en"); });
    await flush();

    expect(document.querySelector("[role='dialog']")).toBe(drawer);
    expect(drawer.textContent).toContain("Evidence and Run Details");
    expect(drawer.textContent).toContain("Tool evidence");
    expect(detailRow("Total input tokens")).toBe(inputTokens);
    expect(inputTokens.querySelector("dd")?.textContent).toBe("1,234");
    expect(detailRow("Created")).toBe(created);
    expect(detailRow("Model elapsed time")?.querySelector("dd")?.textContent).toBe("12.5s");
  });

  it("preserves source trace evidence and context bytes", async () => {
    await i18n.changeLanguage("en");
    stubEvidenceFetch();
    const sourceSkills = ["SOURCE_SKILL_ONE", "SOURCE_SKILL_TWO"];
    const sourceTools = ["SOURCE_TOOL_ONE", "SOURCE_TOOL_TWO"];
    const sourceSkillsSnapshot = [...sourceSkills];
    const sourceToolsSnapshot = [...sourceTools];
    const sourceMessage = message({
      tools_used: sourceTools,
      personalization: { ...PERSONALIZATION, applied_skills: sourceSkills },
    });
    await mountEvidence({ message: sourceMessage });
    const input = document.querySelector(".research-evidence-input")!;
    const preview = document.querySelector(".research-evidence-preview")!;
    const context = document.querySelector(".research-personalization-context-source")!;
    expect(document.querySelector("[role='dialog']")?.textContent).toContain("Tool evidence");
    expect(input.textContent).toBe(JSON.stringify(sourceMessage.tool_calls[0].input, null, 2));
    expect(preview.textContent).toBe("SOURCE_RESULT_BYTES::<verbatim>");
    expect(context.textContent).toBe(PERSONALIZATION.context_snapshot);
    expect(detailRow("Applied skills")?.querySelector("dd")?.textContent).toBe(
      "SOURCE_SKILL_ONE, SOURCE_SKILL_TWO",
    );
    expect(detailRow("Tools")?.querySelector("dd")?.textContent).toBe(
      "SOURCE_TOOL_ONE, SOURCE_TOOL_TWO",
    );
    expect(sourceMessage.personalization?.applied_skills).toEqual(sourceSkillsSnapshot);
    expect(sourceMessage.tools_used).toEqual(sourceToolsSnapshot);

    await act(async () => { await i18n.changeLanguage("zh-Hant"); });
    await flush();
    expect(document.querySelector(".research-evidence-input")).toBe(input);
    expect(input.textContent).toBe(JSON.stringify(sourceMessage.tool_calls[0].input, null, 2));
    expect(document.querySelector(".research-evidence-preview")).toBe(preview);
    expect(preview.textContent).toBe("SOURCE_RESULT_BYTES::<verbatim>");
    expect(document.querySelector(".research-personalization-context-source")).toBe(context);
    expect(context.textContent).toBe(PERSONALIZATION.context_snapshot);
    expect(detailRow("套用技能")?.querySelector("dd")?.textContent).toBe(
      "SOURCE_SKILL_ONE、SOURCE_SKILL_TWO",
    );
    expect(detailRow("工具")?.querySelector("dd")?.textContent).toBe(
      "SOURCE_TOOL_ONE、SOURCE_TOOL_TWO",
    );
    expect(sourceMessage.personalization?.applied_skills).toEqual(sourceSkillsSnapshot);
    expect(sourceMessage.tools_used).toEqual(sourceToolsSnapshot);
  });

  it("preserves disclosure scroll and focus across locale changes", async () => {
    await i18n.changeLanguage("zh-Hant");
    const fetchMock = stubEvidenceFetch();
    await mountEvidence();
    await vi.waitFor(() => expect(detailRow("建立")).toBeDefined());
    const drawer = document.querySelector("[role='dialog']")!;
    const body = drawer.querySelector(".ui-drawer-body") as HTMLElement;
    const disclosure = drawer.querySelector(".research-personalization-context details")!;
    const summary = disclosure.querySelector("summary") as HTMLElement;
    (disclosure as HTMLDetailsElement).open = true;
    body.scrollTop = 137;
    summary.focus();
    const requestCountBeforeLocaleSwitch = fetchMock.mock.calls.length;
    expect(requestCountBeforeLocaleSwitch).toBe(1);

    await act(async () => { await i18n.changeLanguage("en"); });
    await flush();

    expect(document.querySelector("[role='dialog']")).toBe(drawer);
    expect(drawer.querySelector(".ui-drawer-body")).toBe(body);
    expect(drawer.querySelector(".research-personalization-context details")).toBe(disclosure);
    expect((disclosure as HTMLDetailsElement).open).toBe(true);
    expect(body.scrollTop).toBe(137);
    expect(disclosure.querySelector("summary")).toBe(summary);
    expect(summary.textContent).toBe("Personalization context for this run");
    expect(document.activeElement).toBe(summary);
    expect(drawer.textContent).toContain("Evidence and Run Details");
    expect(fetchMock).toHaveBeenCalledTimes(requestCountBeforeLocaleSwitch);
  });

  it("retains the existing Developer diagnostic boundary", async () => {
    await i18n.changeLanguage("en");
    const fetchMock = stubEvidenceFetch();
    await mountEvidence({ developerMode: false });
    expect(document.querySelector(".research-diagnostic")).toBeNull();
    expect(fetchMock.mock.calls.some(([input]) => (
      new URL(String(input)).pathname.endsWith("/events")
    ))).toBe(false);

    await rerenderEvidence({ developerMode: true });
    const diagnostic = document.querySelector(".research-diagnostic")!;
    expect(diagnostic.querySelector("summary")?.textContent).toBe("Diagnostic events");
    expect(document.body.textContent).not.toContain("SECRET_DIAGNOSTIC_TOKEN");
    await act(async () => {
      (diagnostic.querySelector("summary") as HTMLElement).click();
      await Promise.resolve();
    });
    await flush();
    await vi.waitFor(() => expect(diagnostic.querySelector("pre")).not.toBeNull());
    expect(diagnostic.textContent).toContain("[REDACTED]");
    expect(diagnostic.textContent).not.toContain("SECRET_DIAGNOSTIC_TOKEN");
    expect(fetchMock.mock.calls.filter(([input]) => (
      new URL(String(input)).pathname.endsWith("/events")
    ))).toHaveLength(1);
  });

  it("keeps unknown stable identifiers distinguishable", async () => {
    await i18n.changeLanguage("en");
    stubEvidenceFetch({
      runResponse: json({
        run: run({
          provider: "future-provider",
          model: "future-model/source-v9",
          effort: "future-effort",
          auth_mode: null,
        }),
      }),
    });
    await mountEvidence({
      message: message({
        provider: "future-provider",
        model: "future-model/source-v9",
        effort: "future-effort",
      }),
    });
    await vi.waitFor(() => expect(detailRow("Route")).toBeDefined());
    expect(detailRow("Route")?.querySelector("dd")?.textContent).toBe(
      "future-provider · future-model/source-v9 · future-effort",
    );
    expect(document.body.textContent).toContain("source_tool");
    expect(document.body.textContent).toContain("Evidence and Run Details");
  });

  it("renders partial Evidence without claiming completeness", async () => {
    await i18n.changeLanguage("en");
    stubEvidenceFetch({
      runResponse: json({
        detail: { code: "run_detail_unavailable", message: "RAW_PARTIAL_DIAGNOSTIC" },
      }, 503),
    });
    await mountEvidence();
    await vi.waitFor(() => expect(document.querySelector(".ui-inline-alert")?.textContent).toContain(
      "Only part of the run details loaded",
    ));
    const toolBadge = document.querySelector(".research-evidence-tool .ui-status")
      ?? document.querySelector(".research-evidence-tool [data-state]");
    expect(toolBadge?.textContent).toBe("Recorded");
    expect(document.querySelector(".ui-inline-alert")?.textContent).toContain(
      "The conversation and saved tool records remain available.",
    );
    expect(document.querySelector(".research-evidence-tool")?.textContent).not.toContain("Complete");
    expect(document.body.textContent).not.toContain("RAW_PARTIAL_DIAGNOSTIC");
  });

  it("updates shared model and personalization labels reactively", async () => {
    await i18n.changeLanguage("zh-Hant");
    stubEvidenceFetch({
      runResponse: json({
        run: run({
          auth_mode: "chatgpt_oauth",
          personalization: PERSONALIZATION,
        }),
      }),
    });
    await mountEvidence();
    await vi.waitFor(() => expect(detailRow("登入與額度")).toBeDefined());
    const auth = detailRow("登入與額度")!;
    const stance = detailRow("立場")!;
    const personalizationSummary = document.querySelector(
      ".research-personalization-context summary",
    )!;
    expect(auth.querySelector("dd")?.textContent).toContain("ChatGPT 訂閱登入");
    expect(auth.querySelector("dd")?.textContent).toContain("使用訂閱額度");
    expect(stance.querySelector("dd")?.textContent).toBe("互補投資人");

    await act(async () => { await i18n.changeLanguage("en"); });
    await flush();

    expect(detailRow("Sign-in and quota")).toBe(auth);
    expect(auth.querySelector("dd")?.textContent).toContain("ChatGPT subscription sign-in");
    expect(auth.querySelector("dd")?.textContent).toContain("Uses subscription quota, not API billing");
    expect(detailRow("Stance")).toBe(stance);
    expect(stance.querySelector("dd")?.textContent).toBe("Complementary");
    expect(document.querySelector(".research-personalization-context summary")).toBe(
      personalizationSummary,
    );
    expect(personalizationSummary.textContent).toBe("Personalization context for this run");
  });
});
