import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type RefObject,
} from "react";
import {
  Archive,
  ArchiveRestore,
  Check,
  Pencil,
  RefreshCw,
  Trash2,
  X,
} from "lucide-react";
import { useTranslation } from "react-i18next";

import {
  ApiError,
  deleteResearchThread,
  queryResearchThreads,
  updateResearchThread,
  type ResearchHistoryArchiveMode,
  type ResearchHistoryRunState,
  type ResearchThreadDTO,
  type ResearchThreadQueryParams,
} from "./api";
import { researchHistoryStatus } from "./i18n/researchPresentation";
import { Button, IconButton } from "./ui/Button";
import { ConfirmDialog } from "./ui/ConfirmDialog";
import { Drawer } from "./ui/Drawer";
import { StatusBadge } from "./ui/Status";

const PAGE_LIMIT = 50;
const THREAD_MUTATION_PATH = /^\/research\/threads\/[^/?#]+$/;

interface HistoryFilters {
  q: string;
  ticker: string;
  updatedFrom: string;
  updatedThrough: string;
  runState: ResearchHistoryRunState;
  archived: ResearchHistoryArchiveMode;
}

type MutationFailure = "active_run_conflict" | "update_failed";
type RenameFailure = MutationFailure | "empty_name";

const INITIAL_FILTERS: HistoryFilters = {
  q: "",
  ticker: "",
  updatedFrom: "",
  updatedThrough: "",
  runState: "all",
  archived: "current",
};

function isInitialHistoryQuery(filters: HistoryFilters): boolean {
  return filters.q === ""
    && filters.ticker === ""
    && filters.updatedFrom === ""
    && filters.updatedThrough === ""
    && filters.runState === "all"
    && filters.archived === "current";
}

export interface ResearchHistoryDrawerProps {
  open: boolean;
  onClose: () => void;
  activeThreadId: string | null;
  activeRunIds: ReadonlySet<string>;
  onInitialRowsReady: (rows: readonly ResearchThreadDTO[]) => void;
  onSelect: (thread: ResearchThreadDTO) => void;
  onThreadUpdated: (thread: ResearchThreadDTO) => void;
  onThreadDeleted: (id: string) => void;
  returnFocusRef?: RefObject<HTMLElement | null>;
}

function localDateIso(value: string, nextDay: boolean): string | undefined {
  const parts = value.split("-").map(Number);
  if (parts.length !== 3 || parts.some((part) => !Number.isInteger(part))) return undefined;
  const [year, month, day] = parts;
  const date = new Date(year, month - 1, day + (nextDay ? 1 : 0));
  if (!nextDay && (
    date.getFullYear() !== year
    || date.getMonth() !== month - 1
    || date.getDate() !== day
  )) return undefined;
  return date.toISOString();
}

function queryFor(filters: HistoryFilters, offset: number): ResearchThreadQueryParams {
  return {
    ...(filters.q.trim() ? { q: filters.q.trim() } : {}),
    ...(filters.ticker.trim() ? { ticker: filters.ticker.trim().toUpperCase() } : {}),
    ...(filters.updatedFrom
      ? { updated_from: localDateIso(filters.updatedFrom, false) }
      : {}),
    ...(filters.updatedThrough
      ? { updated_before: localDateIso(filters.updatedThrough, true) }
      : {}),
    run_state: filters.runState,
    archived: filters.archived,
    limit: PAGE_LIMIT,
    offset,
  };
}

function appendUnique(
  current: readonly ResearchThreadDTO[],
  incoming: readonly ResearchThreadDTO[],
): ResearchThreadDTO[] {
  const next = [...current];
  const seen = new Set(current.map((thread) => thread.id));
  for (const thread of incoming) {
    if (seen.has(thread.id)) continue;
    seen.add(thread.id);
    next.push(thread);
  }
  return next;
}

function formatLocalTime(value: string): string {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function mutationFailure(error: unknown): MutationFailure {
  if (
    error instanceof ApiError
    && error.status === 409
    && THREAD_MUTATION_PATH.test(error.path)
    && (error.code === null || error.code === "active_run_conflict")
  ) {
    return "active_run_conflict";
  }
  return "update_failed";
}

export function ResearchHistoryDrawer({
  open,
  onClose,
  activeThreadId,
  activeRunIds,
  onInitialRowsReady,
  onSelect,
  onThreadUpdated,
  onThreadDeleted,
  returnFocusRef,
}: ResearchHistoryDrawerProps) {
  const { t: researchT } = useTranslation("research");
  const [filters, setFilters] = useState<HistoryFilters>(INITIAL_FILTERS);
  const [rows, setRows] = useState<ResearchThreadDTO[]>([]);
  const [total, setTotal] = useState(0);
  const [nextOffset, setNextOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [appending, setAppending] = useState(false);
  const [stale, setStale] = useState(false);
  const [loadError, setLoadError] = useState(false);
  const [mutationError, setMutationError] = useState<MutationFailure | null>(null);
  const [mutationId, setMutationId] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [renameError, setRenameError] = useState<RenameFailure | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ResearchThreadDTO | null>(null);
  const [deleteReturnFocus, setDeleteReturnFocus] = useState<HTMLButtonElement | null>(null);
  const [pendingFocusRequest, setPendingFocusRequest] = useState<{
    target: "refresh" | string;
    sequence: number;
  } | null>(null);

  const rowsRef = useRef<ResearchThreadDTO[]>([]);
  const filtersRef = useRef(filters);
  filtersRef.current = filters;
  const requestSequenceRef = useRef(0);
  const initialRowsNotifiedRef = useRef(false);
  const initialRowsCallbackRef = useRef(onInitialRowsReady);
  initialRowsCallbackRef.current = onInitialRowsReady;
  const deleteReturnFocusRef = useMemo(
    () => ({ current: deleteReturnFocus }),
    [deleteReturnFocus],
  );
  const refreshButtonRef = useRef<HTMLButtonElement>(null);
  const rowSelectRefs = useRef(new Map<string, HTMLButtonElement>());
  const focusSequenceRef = useRef(0);
  const renameMutationRef = useRef<string | null>(null);

  const focusAfterRender = useCallback((target: "refresh" | string) => {
    setPendingFocusRequest({ target, sequence: ++focusSequenceRef.current });
  }, []);

  useEffect(() => {
    if (pendingFocusRequest === null) return undefined;
    const timer = window.setTimeout(() => {
      const target = pendingFocusRequest.target === "refresh"
        ? refreshButtonRef.current
        : rowSelectRefs.current.get(pendingFocusRequest.target) ?? refreshButtonRef.current;
      target?.focus();
      setPendingFocusRequest((current) => (
        current?.sequence === pendingFocusRequest.sequence ? null : current
      ));
    }, 0);
    return () => window.clearTimeout(timer);
  }, [pendingFocusRequest, rows]);

  const commitRows = useCallback((next: ResearchThreadDTO[]) => {
    rowsRef.current = next;
    setRows(next);
  }, []);

  const invalidateLoads = useCallback(() => {
    requestSequenceRef.current += 1;
    setLoading(false);
    setAppending(false);
  }, []);

  const loadPage = useCallback(async (
    requestedFilters: HistoryFilters,
    offset: number,
    append: boolean,
  ) => {
    const sequence = ++requestSequenceRef.current;
    if (append) setAppending(true);
    else {
      setAppending(false);
      setLoading(true);
    }
    setLoadError(false);
    try {
      const page = await queryResearchThreads(queryFor(requestedFilters, offset));
      if (sequence !== requestSequenceRef.current) return;
      const nextRows = append
        ? appendUnique(rowsRef.current, page.threads)
        : [...page.threads];
      commitRows(nextRows);
      setTotal(page.total);
      setNextOffset(page.offset + page.limit);
      setStale(false);
      if (
        !initialRowsNotifiedRef.current
        && page.offset === 0
        && isInitialHistoryQuery(requestedFilters)
      ) {
        initialRowsNotifiedRef.current = true;
        initialRowsCallbackRef.current(page.threads);
      }
    } catch {
      if (sequence !== requestSequenceRef.current) return;
      setLoadError(true);
      setStale(rowsRef.current.length > 0);
    } finally {
      if (sequence === requestSequenceRef.current) {
        if (append) setAppending(false);
        else setLoading(false);
      }
    }
  }, [commitRows]);

  useEffect(() => {
    setNextOffset(0);
    void loadPage(filters, 0, false);
  }, [filters, loadPage]);

  useEffect(() => () => {
    requestSequenceRef.current += 1;
  }, []);

  const updateFilter = useCallback(<K extends keyof HistoryFilters>(
    key: K,
    value: HistoryFilters[K],
  ) => {
    setNextOffset(0);
    setFilters((current) => ({ ...current, [key]: value }));
  }, []);

  const replaceRow = useCallback((updated: ResearchThreadDTO) => {
    commitRows(rowsRef.current.map((thread) => (
      thread.id === updated.id ? updated : thread
    )));
  }, [commitRows]);

  const removeRow = useCallback((id: string) => {
    const next = rowsRef.current.filter((thread) => thread.id !== id);
    if (next.length === rowsRef.current.length) return;
    commitRows(next);
    setTotal((current) => Math.max(0, current - 1));
    setNextOffset((current) => Math.max(0, current - 1));
  }, [commitRows]);

  const beginRename = useCallback((thread: ResearchThreadDTO) => {
    setRenamingId(thread.id);
    setRenameDraft(thread.title);
    setRenameError(null);
    setMutationError(null);
  }, []);

  const cancelRename = useCallback((focusThreadId?: string) => {
    setRenamingId(null);
    setRenameDraft("");
    setRenameError(null);
    if (focusThreadId) focusAfterRender(focusThreadId);
  }, [focusAfterRender]);

  const saveRename = useCallback(async (thread: ResearchThreadDTO) => {
    if (renameMutationRef.current !== null) return;
    const title = renameDraft.trim();
    if (!title) {
      setRenameError("empty_name");
      return;
    }
    renameMutationRef.current = thread.id;
    setMutationId(thread.id);
    setMutationError(null);
    setRenameError(null);
    try {
      const { thread: updated } = await updateResearchThread(thread.id, { title });
      invalidateLoads();
      replaceRow(updated);
      cancelRename();
      onThreadUpdated(updated);
      await loadPage(filtersRef.current, 0, false);
      focusAfterRender(updated.id);
    } catch (error) {
      setRenameError(mutationFailure(error));
    } finally {
      if (renameMutationRef.current === thread.id) renameMutationRef.current = null;
      setMutationId(null);
    }
  }, [cancelRename, focusAfterRender, invalidateLoads, loadPage, onThreadUpdated, renameDraft, replaceRow]);

  const changeArchive = useCallback(async (thread: ResearchThreadDTO) => {
    const archived = !thread.archived_at;
    setMutationId(thread.id);
    setMutationError(null);
    try {
      const { thread: updated } = await updateResearchThread(thread.id, { archived });
      invalidateLoads();
      removeRow(thread.id);
      if (deleteTarget?.id === thread.id) setDeleteTarget(null);
      onThreadUpdated(updated);
      await loadPage(filtersRef.current, 0, false);
      focusAfterRender("refresh");
    } catch (error) {
      setMutationError(mutationFailure(error));
    } finally {
      setMutationId(null);
    }
  }, [deleteTarget?.id, focusAfterRender, invalidateLoads, loadPage, onThreadUpdated, removeRow]);

  const confirmDelete = useCallback(async () => {
    if (!deleteTarget) return;
    setMutationId(deleteTarget.id);
    setMutationError(null);
    try {
      await deleteResearchThread(deleteTarget.id);
      const deletedId = deleteTarget.id;
      invalidateLoads();
      removeRow(deletedId);
      setDeleteTarget(null);
      onThreadDeleted(deletedId);
      await loadPage(filtersRef.current, 0, false);
      focusAfterRender("refresh");
    } catch (error) {
      setMutationError(mutationFailure(error));
    } finally {
      setMutationId(null);
    }
  }, [deleteTarget, focusAfterRender, invalidateLoads, loadPage, onThreadDeleted, removeRow]);

  const activeIds = useMemo(() => activeRunIds, [activeRunIds]);
  const isActive = useCallback((thread: ResearchThreadDTO) => {
    const activeRun = thread.active_run;
    if (!activeRun) return false;
    return activeIds.has(activeRun.id)
      || activeRun.status === "queued"
      || activeRun.status === "running";
  }, [activeIds]);

  const hasMore = nextOffset < total;
  const statusLabel = `${rows.length} / ${total}`;
  const mutationErrorLabel = mutationError === null
    ? null
    : mutationError === "active_run_conflict"
      ? researchT(($) => $.history.activeMutationBlocked)
      : researchT(($) => $.history.updateFailed);
  const renameErrorLabel = renameError === null
    ? null
    : renameError === "empty_name"
      ? researchT(($) => $.history.emptyName)
      : renameError === "active_run_conflict"
        ? researchT(($) => $.history.activeMutationBlocked)
        : researchT(($) => $.history.updateFailed);
  const deleteTargetTitle = deleteTarget?.title.trim()
    ? deleteTarget.title
    : researchT(($) => $.history.unnamedFallback);

  return (
    <>
      <Drawer
        open={open}
        title={researchT(($) => $.history.drawerTitle)}
        onClose={onClose}
        returnFocusRef={returnFocusRef}
        footer={(
          <div className="research-history-footer">
            <StatusBadge
              state={stale ? "stale" : rows.length ? "ready" : "empty"}
              label={statusLabel}
            />
            {hasMore ? (
              <Button
                size="compact"
                busy={appending}
                disabled={loading}
                onClick={() => void loadPage(filters, nextOffset, true)}
              >
                {researchT(($) => $.history.loadMore)}
              </Button>
            ) : null}
          </div>
        )}
      >
        <section
          className="research-history"
          aria-label={researchT(($) => $.history.listAria)}
        >
          <div className="research-history-toolbar">
            <div className="research-history-filters">
              <label className="research-history-search">
                <span>{researchT(($) => $.history.searchLabel)}</span>
                <input
                  type="search"
                  aria-label={researchT(($) => $.history.searchAria)}
                  value={filters.q}
                  onChange={(event) => updateFilter("q", event.currentTarget.value)}
                />
              </label>
              <label>
                <span>{researchT(($) => $.history.tickerLabel)}</span>
                <input
                  aria-label={researchT(($) => $.history.tickerFilterAria)}
                  value={filters.ticker}
                  onChange={(event) => updateFilter("ticker", event.currentTarget.value)}
                />
              </label>
              <label>
                <span>{researchT(($) => $.history.updatedFromLabel)}</span>
                <input
                  type="date"
                  aria-label={researchT(($) => $.history.updatedFromAria)}
                  value={filters.updatedFrom}
                  onChange={(event) => updateFilter("updatedFrom", event.currentTarget.value)}
                />
              </label>
              <label>
                <span>{researchT(($) => $.history.updatedToLabel)}</span>
                <input
                  type="date"
                  aria-label={researchT(($) => $.history.updatedToAria)}
                  value={filters.updatedThrough}
                  onChange={(event) => updateFilter("updatedThrough", event.currentTarget.value)}
                />
              </label>
              <label>
                <span>{researchT(($) => $.history.runStatusLabel)}</span>
                <select
                  aria-label={researchT(($) => $.history.runStatusFilterAria)}
                  value={filters.runState}
                  onChange={(event) => updateFilter(
                    "runState",
                    event.currentTarget.value as ResearchHistoryRunState,
                  )}
                >
                  <option value="all">{researchT(($) => $.history.allStatuses)}</option>
                  <option value="active">{researchT(($) => $.history.runningFilter)}</option>
                  <option value="succeeded">{researchT(($) => $.history.completedFilter)}</option>
                  <option value="failed">{researchT(($) => $.history.failedFilter)}</option>
                  <option value="interrupted">{researchT(($) => $.history.interruptedFilter)}</option>
                  <option value="no_run">{researchT(($) => $.history.noRunFilter)}</option>
                </select>
              </label>
              <label>
                <span>{researchT(($) => $.history.archiveStatusLabel)}</span>
                <select
                  aria-label={researchT(($) => $.history.archiveFilterAria)}
                  value={filters.archived}
                  onChange={(event) => updateFilter(
                    "archived",
                    event.currentTarget.value as ResearchHistoryArchiveMode,
                  )}
                >
                  <option value="current">{researchT(($) => $.history.currentThread)}</option>
                  <option value="archived">{researchT(($) => $.history.archivedFilter)}</option>
                </select>
              </label>
            </div>
            <IconButton
              ref={refreshButtonRef}
              label={researchT(($) => $.history.refreshAria)}
              tone="ghost"
              busy={loading}
              icon={<RefreshCw size={17} />}
              onClick={() => void loadPage(filters, 0, false)}
            />
          </div>

          {stale ? (
            <div className="research-history-notice" role="status">
              <StatusBadge
                state="stale"
                label={researchT(($) => $.history.staleDataTitle)}
              />
              <Button size="compact" onClick={() => void loadPage(filters, 0, false)}>
                {researchT(($) => $.history.retry)}
              </Button>
            </div>
          ) : null}
          {loadError && !stale ? (
            <div className="research-history-notice" role="alert">
              <StatusBadge
                state="failed"
                label={researchT(($) => $.history.loadFailedTitle)}
              />
              <Button size="compact" onClick={() => void loadPage(filters, 0, false)}>
                {researchT(($) => $.history.retry)}
              </Button>
            </div>
          ) : null}
          {mutationErrorLabel && !deleteTarget ? (
            <div className="research-history-notice" role="alert">
              <StatusBadge state="blocked" label={mutationErrorLabel} />
            </div>
          ) : null}

          {loading && rows.length === 0 ? (
            <div className="research-history-state">
              <StatusBadge
                state="loading"
                label={researchT(($) => $.history.loadingAria)}
              />
            </div>
          ) : rows.length === 0 && !loadError ? (
            <div className="research-history-state muted">
              {researchT(($) => $.history.emptyFiltered)}
            </div>
          ) : (
            <ul className="research-history-list">
              {rows.map((thread) => {
                const title = thread.title.trim()
                  ? thread.title
                  : researchT(($) => $.history.unnamedFallback);
                const active = isActive(thread);
                const busy = mutationId === thread.id;
                const status = researchHistoryStatus(
                  thread.latest_run_status ?? null,
                  researchT,
                );
                return (
                  <li
                    key={thread.id}
                    className={`research-history-row${thread.id === activeThreadId ? " active" : ""}`}
                    data-research-history-row={thread.id}
                  >
                    {renamingId === thread.id ? (
                      <div className="research-history-rename">
                        <label>
                          <span>{researchT(($) => $.history.conversationNameLabel)}</span>
                          <input
                            autoFocus
                            aria-label={researchT(($) => $.history.titleFilterAria)}
                            value={renameDraft}
                            disabled={busy}
                            onChange={(event) => {
                              setRenameDraft(event.currentTarget.value);
                              setRenameError(null);
                            }}
                            onKeyDown={(event) => {
                              if (renameMutationRef.current === thread.id) {
                                if (event.key === "Enter" || event.key === "Escape") {
                                  event.preventDefault();
                                  event.stopPropagation();
                                }
                                return;
                              }
                              if (event.key === "Enter") void saveRename(thread);
                              if (event.key === "Escape") {
                                event.preventDefault();
                                event.stopPropagation();
                                cancelRename(thread.id);
                              }
                            }}
                          />
                        </label>
                        <div className="research-history-rename-actions">
                          <Button
                            size="compact"
                            tone="primary"
                            busy={busy}
                            icon={<Check size={15} />}
                            onClick={() => void saveRename(thread)}
                          >
                            {researchT(($) => $.history.saveName)}
                          </Button>
                          <IconButton
                            label={researchT(($) => $.history.cancelRenameAria)}
                            size="compact"
                            tone="ghost"
                            icon={<X size={15} />}
                            disabled={busy}
                            onClick={() => cancelRename(thread.id)}
                          />
                        </div>
                        {renameErrorLabel
                          ? <p className="error-text tiny">{renameErrorLabel}</p>
                          : null}
                      </div>
                    ) : (
                      <>
                        <Button
                          ref={(node) => {
                            if (node) rowSelectRefs.current.set(thread.id, node);
                            else rowSelectRefs.current.delete(thread.id);
                          }}
                          tone="ghost"
                          className="research-history-select"
                          aria-label={researchT(($) => $.history.openThreadAria, { title })}
                          onClick={() => onSelect(thread)}
                        >
                          <span className="research-history-title">{title}</span>
                          <span className="research-history-summary">
                            {thread.ticker ? <span className="list-chip">{thread.ticker}</span> : null}
                            <StatusBadge state={status.state} label={status.label} />
                          </span>
                          <span className="research-history-times">
                            <time dateTime={thread.created_at}>
                              {researchT(($) => $.history.createdLabel)}{" "}
                              {formatLocalTime(thread.created_at)}
                            </time>
                            <time dateTime={thread.updated_at}>
                              {researchT(($) => $.history.updatedLabel)}{" "}
                              {formatLocalTime(thread.updated_at)}
                            </time>
                          </span>
                        </Button>
                        <div className="research-history-actions">
                          <IconButton
                            label={researchT(($) => $.history.renameThreadAria, { title })}
                            size="compact"
                            tone="ghost"
                            icon={<Pencil size={15} />}
                            disabled={busy}
                            onClick={() => beginRename(thread)}
                          />
                          <IconButton
                            label={researchT(($) => $.history.archiveToggleAria, {
                              action: thread.archived_at
                                ? researchT(($) => $.history.unarchiveAria)
                                : researchT(($) => $.history.archiveAria),
                              title,
                            })}
                            size="compact"
                            tone="ghost"
                            icon={thread.archived_at
                              ? <ArchiveRestore size={15} />
                              : <Archive size={15} />}
                            disabled={busy || active}
                            title={active
                              ? researchT(($) => $.history.activeThreadWarning)
                              : undefined}
                            onClick={() => void changeArchive(thread)}
                          />
                          <IconButton
                            label={researchT(($) => $.history.deleteThreadAria, { title })}
                            size="compact"
                            tone="danger"
                            icon={<Trash2 size={15} />}
                            disabled={busy || active}
                            title={active
                              ? researchT(($) => $.history.activeThreadWarning)
                              : undefined}
                            onClick={(event) => {
                              setDeleteReturnFocus(event.currentTarget);
                              setMutationError(null);
                              setDeleteTarget(thread);
                            }}
                          />
                        </div>
                      </>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </section>
      </Drawer>

      <ConfirmDialog
        open={Boolean(deleteTarget)}
        title={researchT(($) => $.history.deleteDialogAria)}
        consequence={(
          <div className="research-history-delete-consequence">
            <span>{researchT(($) => $.history.deleteConsequence, {
              title: deleteTargetTitle,
            })}</span>
            {mutationErrorLabel
              ? <StatusBadge state="blocked" label={mutationErrorLabel} />
              : null}
          </div>
        )}
        confirmLabel={researchT(($) => $.history.deleteAction)}
        busy={Boolean(deleteTarget && mutationId === deleteTarget.id)}
        onConfirm={() => void confirmDelete()}
        onCancel={() => {
          if (mutationId) return;
          setDeleteTarget(null);
          setMutationError(null);
        }}
        returnFocusRef={deleteReturnFocusRef}
      />
    </>
  );
}
