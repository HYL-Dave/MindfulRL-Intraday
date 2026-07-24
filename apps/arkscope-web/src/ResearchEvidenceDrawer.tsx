import { useEffect, useMemo, useState, type RefObject } from "react";
import { useTranslation } from "react-i18next";

import { getResearchRun, getResearchRunEvents, type ResearchRunDTO } from "./api";
import {
  presentResearchRoute,
  presentResearchSelection,
  researchEvidenceStatusLabel,
  researchEvidenceTimingLabel,
  researchEvidenceTokenRows,
} from "./i18n/researchPresentation";
import { stanceLabel } from "./personalizationDisplay";
import { ResearchPersonalizationContext } from "./ResearchPersonalizationContext";
import { sanitizeResearchDiagnostic } from "./researchErrors";
import type { Message, ToolTraceRow, TraceRow } from "./researchReducer";
import { formatSystemTimestamp } from "./timeDisplay";
import { Drawer, InlineAlert, StatusBadge } from "./ui";

interface EvidenceRow {
  name: string;
  input?: unknown;
  resultPreview?: string;
  completion: "complete" | "running" | "recorded";
}

interface FetchedRunAuthority {
  runId: string;
  run: ResearchRunDTO;
}

function freshestRunDetail(
  fetchedRun: ResearchRunDTO | null,
  activeRun: ResearchRunDTO | null,
): ResearchRunDTO | null {
  if (!fetchedRun) return activeRun;
  if (!activeRun) return fetchedRun;
  return Date.parse(activeRun.updated_at) >= Date.parse(fetchedRun.updated_at)
    ? activeRun
    : fetchedRun;
}

export function researchEvidenceRows(
  message: Message | null,
  activeTrace: readonly TraceRow[],
): EvidenceRow[] {
  if (activeTrace.length > 0) {
    return activeTrace
      .filter((row): row is ToolTraceRow => row.kind === "tool")
      .map((row) => ({
        name: row.name,
        input: row.input,
        resultPreview: row.result_preview,
        completion: row.done ? "complete" : "running",
      }));
  }
  return (message?.tool_calls ?? []).map((call) => ({
    name: call.name,
    input: call.input,
    resultPreview: call.result_preview,
    completion: "recorded",
  }));
}

function boundedPreview(value: string | undefined): string | null {
  if (!value) return null;
  return value.length > 500 ? `${value.slice(0, 500)}…` : value;
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return "[unserializable]";
  }
}

export function ResearchEvidenceDrawer({
  open,
  pinned,
  onClose,
  onPinnedChange,
  returnFocusRef,
  message,
  activeTrace,
  activeRun,
  developerMode,
}: {
  open: boolean;
  pinned: boolean;
  onClose: () => void;
  onPinnedChange: (pinned: boolean) => void;
  returnFocusRef?: RefObject<HTMLElement | null>;
  message: Message | null;
  activeTrace: readonly TraceRow[];
  activeRun: ResearchRunDTO | null;
  developerMode: boolean;
}) {
  const { t: researchT, i18n: researchI18n } = useTranslation("research");
  const { t: commonT } = useTranslation("common");
  const researchLocale = researchI18n.resolvedLanguage;
  const evidence = useMemo(
    () => researchEvidenceRows(message, activeTrace),
    [activeTrace, message],
  );
  const hasEvidence = evidence.length > 0;
  // A selected transcript turn owns its exact linkage. Never borrow a newer
  // active/latest run for a legacy message that has no persisted run_id.
  const runId = message ? (message.runId ?? null) : (activeRun?.id ?? null);
  const [fetchedAuthority, setFetchedAuthority] = useState<FetchedRunAuthority | null>(null);
  const [detailState, setDetailState] = useState<"idle" | "loading" | "ready" | "partial">("idle");
  const [diagnostic, setDiagnostic] = useState<string | null>(null);
  const [diagnosticFailed, setDiagnosticFailed] = useState(false);
  const [diagnosticLoading, setDiagnosticLoading] = useState(false);

  useEffect(() => {
    if (!open || !runId) {
      setFetchedAuthority(null);
      setDetailState("idle");
      setDiagnostic(null);
      setDiagnosticFailed(false);
      return;
    }
    let alive = true;
    setFetchedAuthority(null);
    setDetailState("loading");
    setDiagnosticFailed(false);
    void getResearchRun(runId)
      .then(({ run }) => {
        if (!alive) return;
        setFetchedAuthority({ runId, run });
        setDetailState("ready");
      })
      .catch(() => {
        if (!alive) return;
        setDetailState("partial");
      });
    return () => { alive = false; };
  }, [open, runId]);

  const loadDiagnostics = () => {
    if (!developerMode || !runId || diagnostic != null || diagnosticFailed || diagnosticLoading) return;
    setDiagnosticLoading(true);
    void getResearchRunEvents(runId, 0)
      .then((response) => {
        const safe = response.events.map((event) => ({
          seq: event.seq,
          type: event.type,
          created_at: event.created_at,
          data: event.data,
        }));
        setDiagnostic(sanitizeResearchDiagnostic(safeJson(safe), 8_000));
      })
      .catch(() => setDiagnosticFailed(true))
      .finally(() => setDiagnosticLoading(false));
  };

  const fetchedRun = fetchedAuthority?.runId === runId && fetchedAuthority.run.id === runId
    ? fetchedAuthority.run
    : null;
  const activeRunCandidate = activeRun?.id === runId ? activeRun : null;
  const details = freshestRunDetail(fetchedRun, activeRunCandidate);
  const usage = details?.token_usage ?? message?.token_usage ?? null;
  const personalization = message?.personalization ?? null;
  const messagePersonalizationKnown = message != null && message.personalization !== undefined;
  const detailPersonalizationKnown = details != null && details.personalization !== undefined;
  const personalizationContext = messagePersonalizationKnown
    ? (message.personalization ?? null)
    : (details?.personalization ?? null);
  const personalizationContextKnown = messagePersonalizationKnown || detailPersonalizationKnown;
  const hasTranscriptDetails = Boolean(details || message);
  const route = useMemo(() => {
    if (details) {
      return presentResearchRoute({
        provider: details.provider,
        model: details.model,
        effort: details.effort,
        runId: details.id,
        errorCode: details.error_code,
      }, researchT);
    }
    if (message?.provider || message?.model || message?.effort) {
      return presentResearchRoute({
        provider: message.provider,
        model: message.model,
        effort: message.effort,
        runId: message.runId,
        errorCode: message.errorCode,
      }, researchT);
    }
    return null;
  }, [details, message, researchLocale, researchT]);
  const auth = useMemo(() => {
    if (!details?.auth_mode) return null;
    const quotaKind = details.auth_mode === "chatgpt_oauth"
      || details.auth_mode === "claude_code_oauth"
      ? "subscription"
      : "api";
    return presentResearchSelection({
      provenance: null,
      authMode: details.auth_mode,
      quotaKind,
      reasonCode: null,
    }, researchT, commonT);
  }, [commonT, details?.auth_mode, researchLocale, researchT]);
  const tokenRows = useMemo(
    () => researchEvidenceTokenRows(usage, researchT),
    [researchLocale, researchT, usage],
  );

  return (
    <Drawer
      open={open}
      title={researchT(($) => $.evidence.drawerTitle)}
      onClose={onClose}
      returnFocusRef={returnFocusRef}
      pinnable={hasEvidence}
      pinned={pinned && hasEvidence}
      onPinnedChange={onPinnedChange}
    >
      <div className="research-evidence" data-has-evidence={String(hasEvidence)}>
        <section>
          <h3 className="surface-title tiny">{researchT(($) => $.evidence.toolEvidence)}</h3>
          {hasEvidence ? (
            <ul className="research-evidence-list">
              {evidence.map((row, index) => (
                <li key={`${row.name}-${index}`} className="research-evidence-tool">
                  <div className="research-evidence-tool-head">
                    <span className="mono">{row.name}</span>
                    <StatusBadge
                      state={row.completion === "running" ? "running" : "ready"}
                      label={researchEvidenceStatusLabel(row.completion, researchT)}
                    />
                  </div>
                  {row.input !== undefined ? (
                    <pre className="research-evidence-input mono tiny muted">{safeJson(row.input)}</pre>
                  ) : null}
                  {boundedPreview(row.resultPreview) ? (
                    <div className="research-evidence-preview tiny muted">{boundedPreview(row.resultPreview)}</div>
                  ) : null}
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted tiny">{researchT(($) => $.evidence.noToolEvidence)}</p>
          )}
        </section>

        <section className="research-run-details">
          <h3 className="surface-title tiny">{researchT(($) => $.evidence.runDetails)}</h3>
          {!runId ? <p className="muted tiny">{researchT(($) => $.evidence.legacyNoRunLink)}</p> : null}
          {detailState === "loading" && !details ? (
            <p className="muted tiny">{researchT(($) => $.evidence.loadingRunDetails)}</p>
          ) : null}
          {detailState === "partial" ? (
            <InlineAlert state="partial" title={researchT(($) => $.evidence.partialTitle)}>
              {researchT(($) => $.evidence.partialDetail)}
            </InlineAlert>
          ) : null}
          {hasTranscriptDetails ? (
            <dl className="research-run-detail-list">
              {route ? (
                <div>
                  <dt>{researchT(($) => $.evidence.route)}</dt>
                  <dd>{route.providerLabel} · {route.modelLabel} · {route.effortLabel}</dd>
                </div>
              ) : null}
              {auth?.authLabel && auth.billingCopy ? (
                <div>
                  <dt>{researchT(($) => $.evidence.authAndQuota)}</dt>
                  <dd>{researchT(($) => $.evidence.authQuotaSummary, {
                    label: auth.authLabel,
                    billing: auth.billingCopy,
                  })}</dd>
                </div>
              ) : null}
              {details ? (
                <div><dt>{researchEvidenceTimingLabel("created", researchT)}</dt><dd>{formatSystemTimestamp(details.created_at)}</dd></div>
              ) : null}
              {details ? (
                <div><dt>{researchEvidenceTimingLabel("started", researchT)}</dt><dd>{formatSystemTimestamp(details.started_at)}</dd></div>
              ) : null}
              {details ? (
                <div><dt>{researchEvidenceTimingLabel("completed", researchT)}</dt><dd>{formatSystemTimestamp(details.completed_at)}</dd></div>
              ) : null}
              {!details && message?.created_at ? (
                <div><dt>{researchEvidenceTimingLabel("turn_saved", researchT)}</dt><dd>{formatSystemTimestamp(message.created_at)}</dd></div>
              ) : null}
              {message?.elapsed_seconds != null ? (
                <div>
                  <dt>{researchEvidenceTimingLabel("model_elapsed", researchT)}</dt>
                  <dd>{message.elapsed_seconds.toFixed(1)}{researchT(($) => $.evidence.secondsSuffix)}</dd>
                </div>
              ) : null}
              {personalization?.profile_active && personalization.assistant_stance !== "off" ? (
                <div><dt>{researchT(($) => $.evidence.stance)}</dt><dd>{stanceLabel(personalization.assistant_stance, commonT)}</dd></div>
              ) : null}
              {personalization?.applied_skills?.length ? (
                <div><dt>{researchT(($) => $.evidence.appliedSkills)}</dt><dd>{personalization.applied_skills.join(", ")}</dd></div>
              ) : null}
              {(message?.tools_used?.length ?? 0) > 0 ? (
                <div><dt>{researchT(($) => $.evidence.tools)}</dt><dd>{message!.tools_used.join(", ")}</dd></div>
              ) : null}
              {tokenRows.map((row) => (
                <div key={row.key}><dt>{row.label}</dt><dd>{row.value.toLocaleString()}</dd></div>
              ))}
            </dl>
          ) : null}
          {personalizationContextKnown ? (
            <ResearchPersonalizationContext trace={personalizationContext} />
          ) : null}
          {developerMode && runId ? (
            <details
              className="research-diagnostic"
              onToggle={(event) => { if (event.currentTarget.open) loadDiagnostics(); }}
            >
              <summary>{researchT(($) => $.evidence.diagnosticEvents)}</summary>
              {diagnosticLoading ? <p className="muted tiny">{researchT(($) => $.evidence.loading)}</p> : null}
              {diagnostic ? <pre>{diagnostic}</pre> : null}
              {diagnosticFailed ? <p className="error-text tiny">{researchT(($) => $.evidence.diagnosticsFailed)}</p> : null}
            </details>
          ) : null}
        </section>
      </div>
    </Drawer>
  );
}
