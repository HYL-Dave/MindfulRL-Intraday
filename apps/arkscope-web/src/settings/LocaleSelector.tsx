import { type ChangeEvent } from "react";
import { useTranslation } from "react-i18next";

import {
  SUPPORTED_UI_LOCALES,
  isUiLocale,
  useUiLocale,
  type UiLocale,
} from "../i18n";

export function LocaleSelector() {
  const { t, i18n } = useTranslation("settings");
  const { locale, busy, errorCode, setLocale } = useUiLocale();
  const label = t(($) => $.locale.label);

  const localeSelfName = (value: UiLocale): string => {
    switch (value) {
      case "zh-Hant":
        return i18n.getFixedT("zh-Hant", "settings")(
          ($) => $.locale.selfName,
        );
      case "en":
        return i18n.getFixedT("en", "settings")(
          ($) => $.locale.selfName,
        );
    }
  };

  const onChange = (event: ChangeEvent<HTMLSelectElement>) => {
    const next = event.currentTarget.value;
    if (busy || !isUiLocale(next) || next === locale) return;
    void setLocale(next);
  };

  return (
    <div className="ui-inline-form" data-testid="locale-selector">
      <label>
        <span>{label}</span>
        <select
          aria-label={label}
          value={locale}
          disabled={busy}
          onChange={onChange}
        >
          {SUPPORTED_UI_LOCALES.map((value) => (
            <option key={value} value={value}>
              {localeSelfName(value)}
            </option>
          ))}
        </select>
      </label>
      {errorCode === "write_failed" ? (
        <span className="error-text" role="alert">
          {t(($) => $.locale.writeFailed)}
        </span>
      ) : null}
    </div>
  );
}
