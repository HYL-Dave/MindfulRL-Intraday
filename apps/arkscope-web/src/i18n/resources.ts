import type { i18n } from "i18next";
import { initReactI18next } from "react-i18next";

import enCommon from "./resources/en/common";
import enExplore from "./resources/en/explore";
import enPortfolio from "./resources/en/portfolio";
import enResearch from "./resources/en/research";
import enSettings from "./resources/en/settings";
import enShell from "./resources/en/shell";
import enSystem from "./resources/en/system";
import zhHantCommon from "./resources/zh-Hant/common";
import zhHantExplore from "./resources/zh-Hant/explore";
import zhHantPortfolio from "./resources/zh-Hant/portfolio";
import zhHantResearch from "./resources/zh-Hant/research";
import zhHantSettings from "./resources/zh-Hant/settings";
import zhHantShell from "./resources/zh-Hant/shell";
import zhHantSystem from "./resources/zh-Hant/system";

export const defaultNamespace = "common" as const;
export const resourceNamespaces = [
  "common",
  "shell",
  "settings",
  "research",
  "explore",
  "portfolio",
  "system",
] as const;

export const resources = {
  "zh-Hant": {
    common: zhHantCommon,
    shell: zhHantShell,
    settings: zhHantSettings,
    research: zhHantResearch,
    explore: zhHantExplore,
    portfolio: zhHantPortfolio,
    system: zhHantSystem,
  },
  en: {
    common: enCommon,
    shell: enShell,
    settings: enSettings,
    research: enResearch,
    explore: enExplore,
    portfolio: enPortfolio,
    system: enSystem,
  },
} as const;

type ResourceLocale = keyof typeof resources;

function safeMissingCopy(instance: i18n): string {
  const locale: ResourceLocale = instance.language === "en" ? "en" : "zh-Hant";
  return resources[locale].common.i18n.missingTranslation;
}

export function initializeI18n(
  instance: i18n,
  initialLocale: ResourceLocale,
): i18n {
  instance.use(initReactI18next);
  void instance.init({
    resources,
    lng: initialLocale,
    fallbackLng: "zh-Hant",
    supportedLngs: ["zh-Hant", "en"],
    load: "currentOnly",
    ns: resourceNamespaces,
    defaultNS: defaultNamespace,
    initAsync: false,
    debug: import.meta.env.DEV,
    returnEmptyString: false,
    parseMissingKeyHandler: () => safeMissingCopy(instance),
    interpolation: {
      escapeValue: false,
    },
    react: {
      useSuspense: false,
    },
  });
  return instance;
}
