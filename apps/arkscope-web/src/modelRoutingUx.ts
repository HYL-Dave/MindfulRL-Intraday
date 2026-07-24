import type { TFunction } from "i18next";

import type {
  EffectiveProviderSummary,
  ModelCatalog,
  ModelProvider,
  ModelTask,
  ProviderCredential,
  TaskRoute,
} from "./api";
import zhHantCommon from "./i18n/resources/zh-Hant/common";

export type ModelCommonT = TFunction<"common">;

const DECORATED_COMPATIBILITY_ID = "decorated_suffix";

export function modelGroupLabel(id: string, t: ModelCommonT): string {
  switch (id) {
    case "available": return t(($) => $.models.groups.available);
    case "visible_disabled": return t(($) => $.models.groups.visibleDisabled);
    case "advanced": return t(($) => $.models.groups.advanced);
    case "current": return t(($) => $.models.groups.current);
    default: return id;
  }
}

export function modelReasonLabel(id: string, t: ModelCommonT): string {
  switch (id) {
    case "missing_active_credential": return t(($) => $.models.reasons.missingActiveCredential);
    case "task_auth_mode_unsupported": return t(($) => $.models.reasons.taskAuthModeUnsupported);
    case "task_test_unsupported": return t(($) => $.models.reasons.taskTestUnsupported);
    case "task_capability_missing": return t(($) => $.models.reasons.taskCapabilityMissing);
    case "model_not_visible": return t(($) => $.models.reasons.modelNotVisible);
    case "model_not_in_registry": return t(($) => $.models.reasons.modelNotInRegistry);
    case "discovery_unavailable": return t(($) => $.models.reasons.discoveryUnavailable);
    case "provider_call_failed": return t(($) => $.models.reasons.providerCallFailed);
    case "reauth_required": return t(($) => $.models.reasons.reauthRequired);
    default: return id;
  }
}

export function modelAuthModeLabel(id: string, t: ModelCommonT): string {
  switch (id) {
    case "api_key": return t(($) => $.models.authModes.apiKey);
    case "api_key_pool": return t(($) => $.models.authModes.apiKeyPool);
    case "chatgpt_oauth": return t(($) => $.models.authModes.chatgptOauth);
    case "claude_code_oauth": return t(($) => $.models.authModes.claudeCodeOauth);
    default: return id;
  }
}

export function modelThinkingModeLabel(id: string, t: ModelCommonT): string {
  switch (id) {
    case "none": return t(($) => $.models.thinkingModes.none);
    case "manual_budget": return t(($) => $.models.thinkingModes.manualBudget);
    case "adaptive_opt_in": return t(($) => $.models.thinkingModes.adaptiveOptIn);
    case "adaptive_default_on": return t(($) => $.models.thinkingModes.adaptiveDefaultOn);
    case "adaptive_always_on": return t(($) => $.models.thinkingModes.adaptiveAlwaysOn);
    default: return id;
  }
}

export function modelCompatibilityLabel(id: string, t: ModelCommonT): string {
  switch (id) {
    case "decorated_suffix": return t(($) => $.models.compatibility.decoratedSuffix);
    case "settings_notice": return t(($) => $.models.compatibility.settingsNotice);
    default: return id;
  }
}

export function modelDecoratedLabel(baseLabel: string, t: ModelCommonT): string {
  return [baseLabel, modelCompatibilityLabel(DECORATED_COMPATIBILITY_ID, t)].join(" · ");
}

export function modelEntryLabel(
  baseLabel: string,
  compatibility: "legacy_unverified" | null,
  t: ModelCommonT,
): string {
  switch (compatibility) {
    case "legacy_unverified": return modelDecoratedLabel(baseLabel, t);
    case null: return baseLabel;
  }
}

// Research presentation migrates in the next tranche steps. Until then, its
// existing zh-Hant projection references the Common resource owner directly.
export const MODEL_UX_LABELS: {
  groups: readonly string[];
  reasons: Record<string, string>;
  authModes: Record<string, string>;
  thinking: Record<string, string>;
} = {
  groups: [
    zhHantCommon.models.groups.available,
    zhHantCommon.models.groups.visibleDisabled,
    zhHantCommon.models.groups.advanced,
    zhHantCommon.models.groups.current,
  ],
  reasons: {
    missing_active_credential: zhHantCommon.models.reasons.missingActiveCredential,
    task_auth_mode_unsupported: zhHantCommon.models.reasons.taskAuthModeUnsupported,
    task_test_unsupported: zhHantCommon.models.reasons.taskTestUnsupported,
    task_capability_missing: zhHantCommon.models.reasons.taskCapabilityMissing,
    model_not_visible: zhHantCommon.models.reasons.modelNotVisible,
    model_not_in_registry: zhHantCommon.models.reasons.modelNotInRegistry,
    discovery_unavailable: zhHantCommon.models.reasons.discoveryUnavailable,
    provider_call_failed: zhHantCommon.models.reasons.providerCallFailed,
    reauth_required: zhHantCommon.models.reasons.reauthRequired,
  },
  authModes: {
    api_key: zhHantCommon.models.authModes.apiKey,
    api_key_pool: zhHantCommon.models.authModes.apiKeyPool,
    chatgpt_oauth: zhHantCommon.models.authModes.chatgptOauth,
    claude_code_oauth: zhHantCommon.models.authModes.claudeCodeOauth,
  },
  thinking: {
    none: zhHantCommon.models.thinkingModes.none,
    manual_budget: zhHantCommon.models.thinkingModes.manualBudget,
    adaptive_opt_in: zhHantCommon.models.thinkingModes.adaptiveOptIn,
    adaptive_default_on: zhHantCommon.models.thinkingModes.adaptiveDefaultOn,
    adaptive_always_on: zhHantCommon.models.thinkingModes.adaptiveAlwaysOn,
  },
};

export interface DraftRouteValue {
  provider: ModelProvider;
  model: string;
  effort: string;
  custom: boolean;
}

export interface TaskTestSnapshot {
  task: ModelTask;
  provider: ModelProvider;
  model: string;
  effort: string;
  credential_id: string;
}

export type ProviderContextMap = Record<ModelProvider, EffectiveProviderSummary | null>;

export function providerContexts(
  effective: Partial<Record<ModelProvider, EffectiveProviderSummary | null>> | undefined,
  credentials: Record<ModelProvider, ProviderCredential[]>,
): ProviderContextMap {
  if (effective) {
    return {
      openai: effective.openai ?? null,
      anthropic: effective.anthropic ?? null,
    };
  }
  const fromInventory = (provider: ModelProvider): EffectiveProviderSummary | null => {
    const active = (credentials[provider] ?? []).find((row) => row.active && row.available);
    if (!active) return null;
    return {
      credential_id: active.id,
      auth_mode: active.auth_type,
      label: active.label,
    };
  };
  return { openai: fromInventory("openai"), anthropic: fromInventory("anthropic") };
}

export function routesSemanticallyEqual(
  draft: Pick<DraftRouteValue, "provider" | "model" | "effort"> | undefined,
  baseline: Pick<TaskRoute, "provider" | "model" | "effort"> | undefined,
): boolean {
  if (!draft || !baseline) return draft === baseline;
  return (
    draft.provider === baseline.provider
    && draft.model.trim() === baseline.model.trim()
    && (draft.effort || "default") === (baseline.effort || "default")
  );
}

export function blockedRouteSaves(
  draft: Partial<Record<ModelTask, DraftRouteValue>>,
  baseline: ModelCatalog["routes"],
  contexts: ProviderContextMap,
): Array<{ task: ModelTask; reason: "missing_active_credential" }> {
  const blocked: Array<{ task: ModelTask; reason: "missing_active_credential" }> = [];
  for (const task of Object.keys(draft) as ModelTask[]) {
    const row = draft[task];
    if (!row || routesSemanticallyEqual(row, baseline[task])) continue;
    if (!contexts[row.provider]) blocked.push({ task, reason: "missing_active_credential" });
  }
  return blocked;
}

export function isTaskTestSnapshotCurrent(
  snapshot: TaskTestSnapshot,
  current: {
    task: ModelTask;
    route: DraftRouteValue;
    credentialId: string | null;
    stale: boolean;
  },
): boolean {
  return !current.stale
    && snapshot.task === current.task
    && snapshot.provider === current.route.provider
    && snapshot.model === current.route.model
    && snapshot.effort === (current.route.effort || "default")
    && snapshot.credential_id === current.credentialId;
}
