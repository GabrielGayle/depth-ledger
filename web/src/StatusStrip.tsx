import { useEffect, useMemo, useRef, useState } from "react";
import {
  fetchStatus,
  postUiHeartbeat,
  type SystemStatus,
} from "./api";

type Tone = "ok" | "warn" | "err" | "unknown";

function toneFor(
  key: "modal" | "local" | "ui" | "api" | "blender",
  s: SystemStatus | null
): Tone {
  if (!s) return "unknown";
  if (key === "api") return s.api.ok ? "ok" : "err";
  if (key === "ui") {
    if (s.ui.vite_up === "up") return "ok";
    if (s.ui.vite_up === "down") return "err";
    return "unknown";
  }
  if (key === "local") {
    const st = s.local_infer.state;
    if (st === "running" || st === "loading") return "warn";
    if (st === "failed") return "err";
    if (st === "idle") return "ok";
    return "unknown";
  }
  if (key === "modal") {
    if (!s.modal.available || s.modal.state === "undeployed") return "unknown";
    if (s.modal.state === "error") return "err";
    if (s.modal.state === "ready") return "ok";
    return "unknown";
  }
  // blender
  if (s.blender_mcp.addon === "listening") {
    if (s.blender_mcp.bpy === "error") return "warn";
    return "ok";
  }
  if (s.blender_mcp.addon === "error") return "err";
  return "unknown";
}

function formatUsd(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "$0";
  if (n < 0.01) return `$${n.toFixed(4)}`;
  return `$${n.toFixed(3)}`;
}

type Props = {
  mode: "kitchen" | "operator" | "video" | "batch";
  runId: string | null;
  lastEvaluateAt: string | null;
};

export function StatusStrip({ mode, runId, lastEvaluateAt }: Props) {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [open, setOpen] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const deepOnce = useRef(false);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const tick = async (deep = false) => {
      try {
        const [s] = await Promise.all([
          fetchStatus(deep),
          postUiHeartbeat({
            mode,
            last_evaluate_at: lastEvaluateAt,
            last_run_id: runId,
          }).catch(() => null),
        ]);
        if (cancelled) return;
        setStatus(s);
        setErr(null);
        const hint = Math.max(1500, Number(s.poll_hint_ms) || 5000);
        timer = window.setTimeout(() => void tick(false), hint);
      } catch (e) {
        if (cancelled) return;
        setErr(e instanceof Error ? e.message : String(e));
        timer = window.setTimeout(() => void tick(false), 5000);
      }
    };

    void tick(!deepOnce.current);
    deepOnce.current = true;

    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [mode, runId, lastEvaluateAt]);

  // Refresh deep blender probe when drawer opens
  useEffect(() => {
    if (!open) return;
    void fetchStatus(true)
      .then(setStatus)
      .catch(() => undefined);
  }, [open]);

  const items = useMemo(
    () =>
      (
        [
          ["Modal", "modal"],
          ["Local", "local"],
          ["UI", "ui"],
          ["API", "api"],
          ["Blender", "blender"],
        ] as const
      ).map(([label, key]) => ({
        label,
        key,
        tone: toneFor(key, status),
      })),
    [status]
  );

  const title = err
    ? `Status error: ${err}`
    : "Live Modal · Local · UI · API · Blender MCP";

  return (
    <div className={`status-strip${open ? " open" : ""}`} title={title}>
      <button
        type="button"
        className="status-strip-btn"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        {items.map((it) => (
          <span key={it.key} className="status-pill">
            <span className={`status-dot ${it.tone}`} aria-hidden />
            {it.label}
          </span>
        ))}
      </button>
      {open && status ? (
        <div className="status-drawer" role="dialog">
          <div>
            <strong>API</strong> ok={String(status.api.ok)} · warmup=
            {status.api.warmup} · jobs={status.api.active_jobs}
            {status.api.active_runs.length
              ? ` · runs ${status.api.active_runs
                  .map((r) => `${r.run_id}:${r.infer_status}`)
                  .join(", ")}`
              : ""}
          </div>
          <div>
            <strong>Local</strong> {status.local_infer.state}
            {status.local_infer.device ? ` · ${status.local_infer.device}` : ""}
            {status.local_infer.last_run_id
              ? ` · ${status.local_infer.last_run_id}`
              : ""}
            {status.local_infer.last_error
              ? ` · ${status.local_infer.last_error}`
              : ""}
          </div>
          <div>
            <strong>Modal</strong>{" "}
            {status.modal.available ? status.modal.state : "undeployed"}
            {status.modal.last_job ? ` · last ${status.modal.last_job}` : ""}
            {" · "}
            {formatUsd(Number(status.modal.usage?.cost_usd || 0))} session
          </div>
          <div>
            <strong>UI</strong> {status.ui.vite_up} · mode={status.ui.mode}
            {status.ui.last_run_id ? ` · ${status.ui.last_run_id}` : ""}
            {status.ui.last_evaluate_at
              ? ` · eval ${status.ui.last_evaluate_at}`
              : ""}
          </div>
          <div>
            <strong>Blender MCP</strong> addon={status.blender_mcp.addon}
            {" · "}bpy={status.blender_mcp.bpy}
            {status.blender_mcp.version
              ? ` · ${status.blender_mcp.version}`
              : ""}
            {status.blender_mcp.blender_path_ok ? "" : " · binary missing"}
            {status.blender_mcp.last_error
              ? ` · ${status.blender_mcp.last_error}`
              : ""}
          </div>
          <div>
            <strong>Cache</strong>{" "}
            {status.cache
              ? `${status.cache.outputs_mb ?? "?"} / ${status.cache.disk_budget_mb ?? "?"} MB disk · ${status.cache.infer_entries ?? 0} infer · ram ${status.cache.hot_arrays_mb ?? 0} MB · rss ${status.cache.rss_mb ?? "?"} MB · ${status.cache.hits ?? 0} hits`
              : "—"}
          </div>
        </div>
      ) : null}
    </div>
  );
}
