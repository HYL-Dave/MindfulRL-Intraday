import i18n from "i18next";
import { useState, type ReactElement, type ReactNode } from "react";

import {
  DEFAULT_UI_LOCALE,
  LocaleProvider,
  createUiLocaleController,
  isUiLocale,
  type UiLocale,
} from "../i18n";

function currentLocale(): UiLocale {
  return isUiLocale(i18n.resolvedLanguage)
    ? i18n.resolvedLanguage
    : DEFAULT_UI_LOCALE;
}

export function TestUiLocaleProvider({ children }: { children: ReactNode }) {
  const [controller] = useState(() => {
    const initialLocale = currentLocale();
    return createUiLocaleController({
      initialLocale,
      authority: {
        get: async () => ({ locale: initialLocale, source: "stored" }),
        put: async (locale) => ({ locale, source: "stored" }),
      },
      applyLocale: (locale) => {
        void i18n.changeLanguage(locale);
        if (typeof document !== "undefined") {
          document.documentElement.lang = locale;
        }
      },
      writeCache: () => true,
    });
  });

  return <LocaleProvider controller={controller}>{children}</LocaleProvider>;
}

export function withTestUiLocale(children: ReactNode): ReactElement {
  return <TestUiLocaleProvider>{children}</TestUiLocaleProvider>;
}
