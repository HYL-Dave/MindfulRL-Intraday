/** @vitest-environment jsdom */
import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import i18n from "i18next";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  LocaleProvider,
  createUiLocaleController,
  type UiLocale,
} from "../i18n";
import type { UiLocaleResponse } from "../i18n/localeController";
import { LocaleSelector } from "./LocaleSelector";

type PutLocale = (locale: UiLocale) => Promise<UiLocaleResponse>;

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

let root: ReturnType<typeof createRoot> | null = null;
let host: HTMLDivElement | null = null;

async function renderSelector({
  initialLocale = "zh-Hant",
  put = vi.fn<PutLocale>(async (locale) => ({ locale, source: "stored" })),
}: {
  initialLocale?: UiLocale;
  put?: ReturnType<typeof vi.fn<PutLocale>>;
} = {}) {
  await i18n.changeLanguage(initialLocale);
  document.documentElement.lang = initialLocale;

  const get = vi.fn(async () => ({
    locale: initialLocale,
    source: "stored" as const,
  }));
  const writeCache = vi.fn(() => true);
  const applyLocale = vi.fn((locale: UiLocale) => {
    void i18n.changeLanguage(locale);
    document.documentElement.lang = locale;
  });
  const controller = createUiLocaleController({
    initialLocale,
    authority: { get, put },
    applyLocale,
    writeCache,
  });

  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  await act(async () => {
    root!.render(
      <LocaleProvider controller={controller}>
        <LocaleSelector />
      </LocaleProvider>,
    );
    await controller.reconcile();
  });
  get.mockClear();
  writeCache.mockClear();
  applyLocale.mockClear();

  return {
    controller,
    put,
    writeCache,
    applyLocale,
    select: host.querySelector<HTMLSelectElement>("select")!,
  };
}

async function changeSelect(select: HTMLSelectElement, value: string) {
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(
      HTMLSelectElement.prototype,
      "value",
    )!.set!;
    setter.call(select, value);
    select.dispatchEvent(new Event("change", { bubbles: true }));
    await Promise.resolve();
  });
}

afterEach(() => {
  if (root) act(() => root!.unmount());
  host?.remove();
  root = null;
  host = null;
});

describe("LocaleSelector", () => {
  it("renders fixed-locale autonyms and current-locale labels in both locales", async () => {
    const { select } = await renderSelector();
    expect(select.getAttribute("aria-label")).toBe("介面語言");
    expect(Array.from(select.options).map(({ value, text }) => [value, text])).toEqual([
      ["zh-Hant", "繁體中文"],
      ["en", "English"],
    ]);
    expect(host!.querySelector('[role="alert"]')).toBeNull();

    await changeSelect(select, "en");

    expect(select.getAttribute("aria-label")).toBe("Interface language");
    expect(Array.from(select.options).map(({ value, text }) => [value, text])).toEqual([
      ["zh-Hant", "繁體中文"],
      ["en", "English"],
    ]);
  });

  it("offers exactly the two supported locale values in reviewed order", async () => {
    const { select } = await renderSelector();

    expect(Array.from(select.options).map((option) => option.value)).toEqual([
      "zh-Hant",
      "en",
    ]);
  });

  it("delegates a supported change only through setLocale", async () => {
    const put = vi.fn<PutLocale>(async (locale) => ({ locale, source: "stored" }));
    const { select, writeCache } = await renderSelector({ put });

    await changeSelect(select, "en");

    expect(put).toHaveBeenCalledOnce();
    expect(put).toHaveBeenCalledWith("en");
    expect(writeCache).toHaveBeenCalledOnce();
    expect(writeCache).toHaveBeenCalledWith("en");
    expect(document.documentElement.lang).toBe("en");
  });

  it("ignores invalid and same-locale DOM values", async () => {
    const put = vi.fn<PutLocale>();
    const { select } = await renderSelector({ put });
    const unsupported = document.createElement("option");
    unsupported.value = "fr";
    unsupported.textContent = "fr";
    select.appendChild(unsupported);

    await changeSelect(select, "fr");
    await changeSelect(select, "zh-Hant");

    expect(put).not.toHaveBeenCalled();
  });

  it("disables while a write is pending and prevents duplicate writes", async () => {
    const pending = deferred<UiLocaleResponse>();
    const put = vi.fn<PutLocale>(() => pending.promise);
    const { select } = await renderSelector({ put });

    await changeSelect(select, "en");
    expect(select.disabled).toBe(true);
    await changeSelect(select, "zh-Hant");
    expect(put).toHaveBeenCalledOnce();

    await act(async () => {
      pending.resolve({ locale: "en", source: "stored" });
      await pending.promise;
    });
    expect(select.disabled).toBe(false);
  });

  it("rolls back to zh-Hant and renders failure copy in the restored locale", async () => {
    const put = vi.fn<PutLocale>(async () => {
      throw new Error("PLANTED_RAW_LOCALE_FAILURE");
    });
    const { controller, select } = await renderSelector({ put });

    await changeSelect(select, "en");

    expect(controller.getSnapshot().errorCode).toBe("write_failed");
    expect(document.documentElement.lang).toBe("zh-Hant");
    expect(select.value).toBe("zh-Hant");
    expect(host!.textContent).toContain("無法儲存介面語言，已還原先前設定。");
    expect(host!.textContent).not.toContain("PLANTED_RAW_LOCALE_FAILURE");
  });

  it("rolls back to English and clears the error after a successful retry", async () => {
    const put = vi
      .fn<PutLocale>()
      .mockRejectedValueOnce(new Error("PLANTED_RAW_LOCALE_FAILURE"))
      .mockResolvedValueOnce({ locale: "zh-Hant", source: "stored" });
    const { controller, select } = await renderSelector({
      initialLocale: "en",
      put,
    });

    await changeSelect(select, "zh-Hant");
    expect(document.documentElement.lang).toBe("en");
    expect(host!.textContent).toContain(
      "Could not save the interface language. The previous setting was restored.",
    );

    await changeSelect(select, "zh-Hant");

    expect(put).toHaveBeenCalledTimes(2);
    expect(controller.getSnapshot().errorCode).toBeNull();
    expect(document.documentElement.lang).toBe("zh-Hant");
    expect(host!.querySelector('[role="alert"]')).toBeNull();
  });
});
