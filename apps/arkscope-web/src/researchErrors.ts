import type { NavigationTarget } from "./shell/navigation";
import type { CommonUiState } from "./ui";
import zhHantResearch from "./i18n/resources/zh-Hant/research";
import type { ResearchT } from "./i18n/researchPresentation";

export interface ResearchErrorPresentation {
  code: string;
  state: CommonUiState;
  title: string;
  detail: string;
  actionLabel: string | null;
  target: NavigationTarget | null;
  preservePartial: boolean;
  developerDetail: string | null;
}

interface ErrorDefinition {
  state: CommonUiState;
  copy: "reauth" | "missingCredential" | "timeout" | "refusal" | "providerCallFailed"
    | "toolLimit" | "cancelled" | "interrupted";
  target?: NavigationTarget;
  preservePartial?: boolean;
}

const SETTINGS_PROVIDERS: NavigationTarget = {
  kind: "settings_section",
  section: "providers",
};
const SETTINGS_MODELS: NavigationTarget = {
  kind: "settings_section",
  section: "models",
};

const DEFINITIONS: Record<string, ErrorDefinition> = {
  reauth_required: {
    state: "blocked",
    copy: "reauth",
    target: SETTINGS_PROVIDERS,
  },
  missing_credential: {
    state: "blocked",
    copy: "missingCredential",
    target: SETTINGS_PROVIDERS,
  },
  model_timeout: {
    state: "failed",
    copy: "timeout",
    target: SETTINGS_MODELS,
  },
  model_refusal: {
    state: "failed",
    copy: "refusal",
  },
  provider_call_failed: {
    state: "failed",
    copy: "providerCallFailed",
  },
  tool_limit_reached: {
    state: "failed",
    copy: "toolLimit",
    target: SETTINGS_MODELS,
    preservePartial: true,
  },
  cancelled: {
    state: "interrupted",
    copy: "cancelled",
  },
  interrupted: {
    state: "interrupted",
    copy: "interrupted",
    preservePartial: true,
  },
  run_cancelled: {
    state: "interrupted",
    copy: "cancelled",
  },
  run_interrupted: {
    state: "interrupted",
    copy: "interrupted",
    preservePartial: true,
  },
};

function errorCopy(
  id: ErrorDefinition["copy"],
  t?: ResearchT,
): { title: string; detail: string; actionLabel: string | null } {
  switch (id) {
    case "reauth":
      return {
        title: t ? t(($) => $.errors.reauthTitle) : zhHantResearch.errors.reauthTitle,
        detail: t ? t(($) => $.errors.reauthDetail) : zhHantResearch.errors.reauthDetail,
        actionLabel: t ? t(($) => $.errors.reauthAction) : zhHantResearch.errors.reauthAction,
      };
    case "missingCredential":
      return {
        title: t ? t(($) => $.errors.missingCredentialTitle) : zhHantResearch.errors.missingCredentialTitle,
        detail: t ? t(($) => $.errors.missingCredentialDetail) : zhHantResearch.errors.missingCredentialDetail,
        actionLabel: t ? t(($) => $.errors.providerSettingsAction) : zhHantResearch.errors.providerSettingsAction,
      };
    case "timeout":
      return {
        title: t ? t(($) => $.errors.timeoutTitle) : zhHantResearch.errors.timeoutTitle,
        detail: t ? t(($) => $.errors.timeoutDetail) : zhHantResearch.errors.timeoutDetail,
        actionLabel: t ? t(($) => $.errors.runtimeSettingsAction) : zhHantResearch.errors.runtimeSettingsAction,
      };
    case "refusal":
      return {
        title: t ? t(($) => $.errors.refusalTitle) : zhHantResearch.errors.refusalTitle,
        detail: t ? t(($) => $.errors.refusalDetail) : zhHantResearch.errors.refusalDetail,
        actionLabel: null,
      };
    case "providerCallFailed":
      return {
        title: t ? t(($) => $.errors.providerCallFailedTitle) : zhHantResearch.errors.providerCallFailedTitle,
        detail: t ? t(($) => $.errors.providerCallFailedDetail) : zhHantResearch.errors.providerCallFailedDetail,
        actionLabel: null,
      };
    case "toolLimit":
      return {
        title: t ? t(($) => $.errors.maxTurnsTitle) : zhHantResearch.errors.maxTurnsTitle,
        detail: t ? t(($) => $.errors.maxTurnsDetail) : zhHantResearch.errors.maxTurnsDetail,
        actionLabel: t ? t(($) => $.errors.retryAction) : zhHantResearch.errors.retryAction,
      };
    case "cancelled":
      return {
        title: t ? t(($) => $.errors.cancelledTitle) : zhHantResearch.errors.cancelledTitle,
        detail: t ? t(($) => $.errors.cancelledDetail) : zhHantResearch.errors.cancelledDetail,
        actionLabel: null,
      };
    case "interrupted":
      return {
        title: t ? t(($) => $.errors.interruptedTitle) : zhHantResearch.errors.interruptedTitle,
        detail: t ? t(($) => $.errors.interruptedDetail) : zhHantResearch.errors.interruptedDetail,
        actionLabel: null,
      };
  }
}

export function sanitizeResearchDiagnostic(detail: string, limit = 1_500): string {
  return detail
    .slice(0, limit)
    .replace(
      /["']?(?:credential_id|access_token|refresh_token)["']?\s*[:=]\s*(?:"[^"]*"|'[^']*'|[^\s,;}]+)/gi,
      "[REDACTED]",
    )
    .replace(/\bBearer\s+[^\s,;]+/gi, "[REDACTED]")
    .replace(/\blocal:\d+\b/gi, "[REDACTED]");
}

export function presentResearchError({
  code,
  detail = null,
  developerMode = false,
}: {
  code: string | null | undefined;
  detail?: string | null;
  developerMode?: boolean;
}, t?: ResearchT): ResearchErrorPresentation {
  const normalizedCode = code && DEFINITIONS[code] ? code : "provider_call_failed";
  const definition = DEFINITIONS[normalizedCode];
  const copy = errorCopy(definition.copy, t);
  return {
    code: normalizedCode,
    state: definition.state,
    title: copy.title,
    detail: copy.detail,
    actionLabel: copy.actionLabel,
    target: definition.target ?? null,
    preservePartial: definition.preservePartial ?? false,
    developerDetail: developerMode && detail ? sanitizeResearchDiagnostic(detail) : null,
  };
}
