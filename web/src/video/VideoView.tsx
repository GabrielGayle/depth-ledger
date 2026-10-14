import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import {
  bust,
  cancelJob,
  exportFilm,
  fetchFilm,
  pollJob,
  prepareFilm,
  renderFilm,
  sampleFilm,
  startFilmInfer,
  startFilmPreview,
  type FilmParams,
  type FilmParamsPatch,
  type FilmResponse,
  type Meta,
} from "../api";

function pad(i: number) {
  return String(i).padStart(6, "0");
}

function defaultParams(meta: Meta | null): FilmParams {
  const f = meta?.film;
  return {
    pipeline: "film",
    anchor_focal: f?.anchor_focal ?? true,
    focallength_px: null,
    focallength_override: null,
    vis_min_m: f?.vis_min_m ?? 0.5,
    vis_max_m: f?.vis_max_m ?? 10,
    suggested_vis_min_m: null,
    suggested_vis_max_m: null,
    bounds_locked: false,
    ema_enabled: f?.ema_enabled ?? true,
    ema_alpha: f?.ema_alpha ?? 0.45,
    gamma: f?.gamma ?? 1,
    cmap: f?.cmap ?? "gray",
    preview_frames: f?.preview_frames ?? 8,
    phase: "idle",
    preview_ready: 0,
  };
}

function frameUrl(runId: string, index: number, kind: "rgb" | "vis") {
  const name = kind === "rgb" ? "rgb.jpg" : "vis.png";
  return `/api/runs/${runId}/files/frames/${pad(index)}/${name}`;
}

export type VideoViewProps = {
  meta: Meta | null;
  modelKey: string;
  maxSize: number;
  setMaxSize: (n: number) => void;
  inferBackend: "local" | "modal";
  historyRunId?: string | null;
  onStatus: (msg: string) => void;
  onProgress?: (p: number | null) => void;
  onBusy?: (busy: boolean) => void;
};

export function VideoView({
  meta,
  modelKey,
  maxSize,
  setMaxSize,
  inferBackend,
  historyRunId,
  onStatus,
  onProgress,
  onBusy,
}: VideoViewProps) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [run, setRun] = useState<FilmResponse | null>(null);
  const [params, setParams] = useState<FilmParams>(() => defaultParams(meta));
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [busy, setBusy] = useState(false);
  const [renderToken, setRenderToken] = useState(0);
  const [sample, setSample] = useState<{ depth_m: number; pct: number } | null>(
    null
  );
  const [exportUrl, setExportUrl] = useState<string | null>(null);
  const [targetFps, setTargetFps] = useState(meta?.film?.video_target_fps ?? 10);
  const [maxFrames, setMaxFrames] = useState(meta?.film?.video_max_frames ?? 60);
  const [fileLabel, setFileLabel] = useState<string | null>(null);
  const [overrideText, setOverrideText] = useState("");
  const renderTimer = useRef<number | null>(null);
  const jobIdRef = useRef<string | null>(null);
  const paramsRef = useRef(params);
  paramsRef.current = params;

  const setBusyAll = useCallback(
    (v: boolean) => {
      setBusy(v);
      onBusy?.(v);
    },
    [onBusy]
  );

  const applyRun = useCallback((res: FilmResponse) => {
    setRun(res);
    if (res.params) {
      setParams(res.params);
      setOverrideText(
        res.params.focallength_override != null
          ? String(Math.round(res.params.focallength_override * 10) / 10)
          : ""
      );
    }
    setRenderToken((t) => t + 1);
    const vis = res.vis_ready || [];
    setFrameIndex((i) => {
      if (vis.includes(i)) return i;
      return vis[0] ?? 0;
    });
  }, []);

  useEffect(() => {
    if (!historyRunId) return;
    let cancelled = false;
    setBusyAll(true);
    onStatus(`Loading ${historyRunId}…`);
    fetchFilm(historyRunId)
      .then((res) => {
        if (cancelled) return;
        applyRun(res);
        setFileLabel(historyRunId);
        onStatus(`Loaded ${historyRunId}`);
      })
      .catch((e) => {
        if (!cancelled) onStatus(e?.message || "Load failed");
      })
      .finally(() => {
        if (!cancelled) setBusyAll(false);
      });
    return () => {
      cancelled = true;
    };
  }, [historyRunId, applyRun, onStatus, setBusyAll]);

  const patch: FilmParamsPatch = useMemo(
    () => ({
      model_key: modelKey,
      max_size: maxSize,
      infer_backend: inferBackend,
      anchor_focal: params.anchor_focal,
      focallength_override: overrideText
        ? Number(overrideText) || null
        : null,
      vis_min_m: params.vis_min_m,
      vis_max_m: params.vis_max_m,
      bounds_locked: params.bounds_locked,
      ema_enabled: params.ema_enabled,
      ema_alpha: params.ema_alpha,
      gamma: params.gamma,
      cmap: params.cmap,
      preview_frames: params.preview_frames,
    }),
    [
      modelKey,
      maxSize,
      inferBackend,
      params.anchor_focal,
      params.vis_min_m,
      params.vis_max_m,
      params.bounds_locked,
      params.ema_enabled,
      params.ema_alpha,
      params.gamma,
      params.cmap,
      params.preview_frames,
      overrideText,
    ]
  );

  const visReady = run?.vis_ready || [];
  const frameCount = run?.frame_count || 0;
  const readyCount = run?.ready_count || 0;
  const phase = params.phase || run?.infer_status || "idle";
  const hasDepth = readyCount > 0;
  const allInferred = frameCount > 0 && readyCount >= frameCount;
  const allVis = frameCount > 0 && visReady.length >= frameCount;
  const maxSizes = meta?.max_sizes || [518, 756, 1008, 1536];
  const maxRes = Math.max(...maxSizes);
  const modeLabel =
    inferBackend === "modal" || run?.infer_mode === "modal_parallel"
      ? "Modal · parallel"
      : "Local · sequential";

  const scheduleRender = useCallback(
    (next?: Partial<FilmParams>) => {
      if (!run?.run_id || !hasDepth) return;
      if (next) paramsRef.current = { ...paramsRef.current, ...next };
      if (renderTimer.current) window.clearTimeout(renderTimer.current);
      renderTimer.current = window.setTimeout(() => {
        const id = run.run_id;
        renderFilm(id, paramsRef.current)
          .then((res) => {
            applyRun(res);
            onStatus("Look updated");
          })
          .catch((e) => onStatus(e?.message || "Render failed"));
      }, 220);
    },
    [run?.run_id, hasDepth, applyRun, onStatus]
  );

  const onLoadFile = async (file: File) => {
    setBusyAll(true);
    onProgress?.(0);
    onStatus(`Loading ${file.name}…`);
    setExportUrl(null);
    setSample(null);
    setPlaying(false);
    try {
      const res = await prepareFilm(file, {
        modelKey,
        maxSize,
        inferBackend,
        targetFps,
        maxFrames,
      });
      applyRun(res);
      setFileLabel(file.name);
      setFrameIndex(0);
      onStatus(
        `Loaded ${res.frame_count} frames · ${res.sample_fps.toFixed(1)} fps — Preview look next`
      );
    } catch (e: any) {
      onStatus(e?.message || "Load failed");
    } finally {
      setBusyAll(false);
      onProgress?.(null);
    }
  };

  const runJob = async (kind: "preview" | "infer") => {
    if (!run?.run_id) return;
    setBusyAll(true);
    setPlaying(false);
    setExportUrl(null);
    onProgress?.(0);
    onStatus(
      kind === "preview"
        ? "Previewing look…"
        : inferBackend === "modal"
          ? "Modal · parallel Depth Pro · locked f_px…"
          : "Inferring full clip…"
    );
    try {
      const start = kind === "preview" ? startFilmPreview : startFilmInfer;
      const { job_id } = await start(run.run_id, patch);
      jobIdRef.current = job_id;
      const job = await pollJob(
        job_id,
        (j) => {
          onProgress?.(typeof j.progress === "number" ? j.progress : null);
          if (j.message) onStatus(j.message);
          if (j.result && (j.result as FilmResponse).run_id) {
            applyRun(j.result as FilmResponse);
          }
        },
        400
      );
      if (job.result) applyRun(job.result as FilmResponse);
      onStatus(kind === "preview" ? "Preview ready — tweak the stack" : "Infer done");
    } catch (e: any) {
      onStatus(e?.message || "Job failed");
    } finally {
      jobIdRef.current = null;
      setBusyAll(false);
      onProgress?.(null);
    }
  };

  const onExport = async () => {
    if (!run?.run_id) return;
    setBusyAll(true);
    onStatus("Exporting MP4…");
    try {
      const res = await exportFilm(run.run_id);
      setExportUrl(bust(res.export_url, Date.now()));
      onStatus(`Exported ${res.frame_count} frames @ ${res.fps.toFixed(1)} fps`);
    } catch (e: any) {
      onStatus(e?.message || "Export failed");
    } finally {
      setBusyAll(false);
    }
  };

  const onCancel = () => {
    const id = jobIdRef.current;
    if (id) void cancelJob(id);
  };

  useEffect(() => {
    if (!playing || visReady.length < 2) return;
    const fps = Math.max(1, run?.sample_fps || 10);
    const id = window.setInterval(() => {
      setFrameIndex((i) => {
        const pos = visReady.indexOf(i);
        const next = visReady[(pos + 1) % visReady.length];
        return next;
      });
    }, 1000 / fps);
    return () => window.clearInterval(id);
  }, [playing, visReady, run?.sample_fps]);

  const onVisClick = async (e: MouseEvent<HTMLButtonElement>) => {
    if (!run?.run_id || !hasDepth) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const u = (e.clientX - rect.left) / Math.max(rect.width, 1);
    const v = (e.clientY - rect.top) / Math.max(rect.height, 1);
    try {
      const s = await sampleFilm(run.run_id, u, v, frameIndex);
      setSample({ depth_m: s.depth_m, pct: s.pct });
    } catch {
      setSample(null);
    }
  };

  const step =
    !run ? 0 : !hasDepth ? 1 : !allInferred ? 2 : allVis ? 3 : 2;

  const rgbSrc = run
    ? bust(frameUrl(run.run_id, frameIndex, "rgb"), renderToken)
    : null;
  const visSrc =
    run && visReady.includes(frameIndex)
      ? bust(frameUrl(run.run_id, frameIndex, "vis"), renderToken)
      : null;

  const fpx =
    params.focallength_override ?? params.focallength_px ?? null;

  return (
    <div className="workspace film-workspace">
      <aside className="rail rail-left film-rail">
        <h2>Clip</h2>
        <div
          className="film-drop"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            const f = e.dataTransfer.files?.[0];
            if (f) void onLoadFile(f);
          }}
        >
          <input
            ref={fileRef}
            type="file"
            accept="video/mp4,video/quicktime,video/webm,video/*"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void onLoadFile(f);
              e.target.value = "";
            }}
          />
          <button
            type="button"
            className="primary"
            disabled={busy}
            onClick={() => fileRef.current?.click()}
          >
            Load video
          </button>
          <p className="muted">
            {fileLabel || "Drop an mp4 / mov / webm"}
          </p>
        </div>

        <div className="film-field">
          <label>
            Sample fps
            <input
              type="number"
              min={4}
              max={24}
              step={1}
              value={targetFps}
              disabled={busy || !!run}
              onChange={(e) => setTargetFps(Number(e.target.value))}
            />
          </label>
          <label>
            Max frames
            <input
              type="number"
              min={4}
              max={meta?.film?.video_max_frames ?? 60}
              step={1}
              value={maxFrames}
              disabled={busy || !!run}
              onChange={(e) => setMaxFrames(Number(e.target.value))}
            />
          </label>
          <label>
            Preview frames
            <input
              type="number"
              min={2}
              max={24}
              step={1}
              value={params.preview_frames}
              disabled={busy}
              onChange={(e) =>
                setParams((p) => ({
                  ...p,
                  preview_frames: Number(e.target.value),
                }))
              }
            />
          </label>
        </div>

        <div className="film-res">
          <label title="Longest edge cap for Depth Pro (re-extracts on Preview / Infer all if changed)">
            Max
            <select
              value={maxSize}
              disabled={busy}
              onChange={(e) => setMaxSize(parseInt(e.target.value, 10))}
            >
              {maxSizes.map((n) => (
                <option key={n} value={n}>
                  {n}
                  {n === maxRes ? " (max)" : ""}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            disabled={busy || maxSize === maxRes}
            title={`Use Depth Pro native max (${maxRes})`}
            onClick={() => setMaxSize(maxRes)}
          >
            Max res
          </button>
        </div>

        {run ? (
          <p className="muted film-meta-line">
            {frameCount} frames · {run.sample_fps.toFixed(1)} fps
            {run.frame_width
              ? ` · ${run.frame_width}×${run.frame_height}`
              : ""}
            <br />
            Depth {readyCount}/{frameCount}
            {phase !== "idle" ? ` · ${phase}` : ""}
            {` · ${modeLabel}`}
          </p>
        ) : null}

        {run?.preview_gif_url ? (
          <img
            className="film-gif"
            src={bust(run.preview_gif_url, renderToken)}
            alt="RGB preview"
          />
        ) : null}

        <div className="film-actions">
          <button
            type="button"
            className="primary"
            disabled={busy || !run}
            onClick={() => void runJob("preview")}
          >
            Preview look
          </button>
          <button
            type="button"
            disabled={busy || !run || !hasDepth}
            title={
              inferBackend === "modal"
                ? "Depth Pro on remaining frames in parallel on Modal, locked to frame-0 focal"
                : "Depth Pro on remaining frames sequentially, locked to frame-0 focal"
            }
            onClick={() => void runJob("infer")}
          >
            Infer all
          </button>
          <button
            type="button"
            disabled={busy || !allVis}
            onClick={() => void onExport()}
          >
            Export MP4
          </button>
          {busy ? (
            <button type="button" onClick={onCancel}>
              Cancel
            </button>
          ) : null}
        </div>

        {exportUrl ? (
          <a className="film-export-link" href={exportUrl} download>
            Download MP4
          </a>
        ) : null}

        <ol className="film-steps">
          <li className={step === 0 ? "on" : step > 0 ? "done" : ""}>Load</li>
          <li className={step === 1 ? "on" : step > 1 ? "done" : ""}>Preview</li>
          <li className={step === 2 ? "on" : step > 2 ? "done" : ""}>Tweak</li>
          <li className={step === 3 ? "on" : ""}>Export</li>
        </ol>
      </aside>

      <section className="film-stage">
        <div className="film-plates">
          <div className="film-plate">
            <div className="film-plate-label">Source</div>
            {rgbSrc ? (
              <img src={rgbSrc} alt="Source frame" />
            ) : (
              <div className="film-empty">Load a clip to begin</div>
            )}
          </div>
          <div className="film-plate">
            <div className="film-plate-label">
              Depth
              {sample
                ? ` · ${sample.depth_m.toFixed(2)} m (${sample.pct.toFixed(0)}th)`
                : " · click to sample meters"}
            </div>
            {visSrc ? (
              <button
                type="button"
                className="film-vis-hit"
                onClick={(e) => void onVisClick(e)}
              >
                <img src={visSrc} alt="Depth vis" />
              </button>
            ) : (
              <div className="film-empty">
                {run
                  ? "Preview look to see locked depth"
                  : "No depth yet"}
              </div>
            )}
          </div>
        </div>

        {frameCount > 0 ? (
          <div className="film-transport">
            <button
              type="button"
              disabled={visReady.length < 2}
              onClick={() => setPlaying((p) => !p)}
            >
              {playing ? "Pause" : "Play"}
            </button>
            <input
              type="range"
              min={0}
              max={Math.max(frameCount - 1, 0)}
              value={frameIndex}
              onChange={(e) => {
                setPlaying(false);
                setFrameIndex(Number(e.target.value));
              }}
            />
            <span className="muted">
              {frameIndex + 1}/{frameCount}
            </span>
          </div>
        ) : null}

        {frameCount > 0 ? (
          <div className="film-strip">
            {Array.from({ length: frameCount }, (_, i) => {
              const ready = visReady.includes(i);
              return (
                <button
                  key={i}
                  type="button"
                  className={`film-thumb${i === frameIndex ? " on" : ""}${
                    ready ? "" : " wait"
                  }`}
                  onClick={() => {
                    setPlaying(false);
                    setFrameIndex(i);
                  }}
                >
                  {ready ? (
                    <img
                      src={bust(frameUrl(run!.run_id, i, "vis"), renderToken)}
                      alt={`f${i + 1}`}
                    />
                  ) : (
                    <span>{i + 1}</span>
                  )}
                </button>
              );
            })}
          </div>
        ) : null}

        {sample && hasDepth ? (
          <div className="film-sample-actions">
            Sample {sample.depth_m.toFixed(2)} m
            <button
              type="button"
              disabled={busy}
              onClick={() => {
                const vis_min_m = sample.depth_m;
                setParams((p) => ({
                  ...p,
                  vis_min_m,
                  bounds_locked: true,
                }));
                scheduleRender({ vis_min_m, bounds_locked: true });
              }}
            >
              Set near
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() => {
                const vis_max_m = sample.depth_m;
                setParams((p) => ({
                  ...p,
                  vis_max_m,
                  bounds_locked: true,
                }));
                scheduleRender({ vis_max_m, bounds_locked: true });
              }}
            >
              Set far
            </button>
          </div>
        ) : null}
      </section>

      <aside className="rail film-stack">
        <h2>Flicker-free stack</h2>
        <p className="muted film-stack-lead">
          Independent of Kitchen looks and Operator oscillators. Lock scale,
          lock the window, then smooth.
        </p>

        <article className="film-card">
          <header>
            <span className="film-rule">1</span>
            <div>
              <strong>Anchor focal length</strong>
              <p>Estimate f_px on frame 0, reuse that exact value on every later frame.</p>
            </div>
          </header>
          <label className="film-check">
            <input
              type="checkbox"
              checked={params.anchor_focal}
              disabled={busy}
              onChange={(e) =>
                setParams((p) => ({ ...p, anchor_focal: e.target.checked }))
              }
            />
            Lock frame-0 focal
          </label>
          <div className="film-readout">
            {fpx != null ? `${fpx.toFixed(1)} px` : "not estimated yet"}
          </div>
          <label>
            Override (px)
            <input
              type="number"
              min={20}
              step={1}
              placeholder="auto from frame 0"
              value={overrideText}
              disabled={busy}
              onChange={(e) => setOverrideText(e.target.value)}
            />
          </label>
        </article>

        <article className="film-card">
          <header>
            <span className="film-rule">2</span>
            <div>
              <strong>Fixed vis bounds</strong>
              <p>Global min/max in meters for the whole clip — never per-frame 0–255.</p>
            </div>
          </header>
          <label>
            Near (m)
            <input
              type="number"
              min={0.05}
              max={80}
              step={0.05}
              value={Number(params.vis_min_m.toFixed(3))}
              disabled={busy}
              onChange={(e) => {
                const vis_min_m = Number(e.target.value);
                setParams((p) => ({ ...p, vis_min_m, bounds_locked: true }));
                scheduleRender({ vis_min_m, bounds_locked: true });
              }}
            />
          </label>
          <input
            type="range"
            min={0.05}
            max={20}
            step={0.05}
            value={Math.min(params.vis_min_m, 20)}
            disabled={busy}
            onChange={(e) => {
              const vis_min_m = Number(e.target.value);
              setParams((p) => ({ ...p, vis_min_m, bounds_locked: true }));
              scheduleRender({ vis_min_m, bounds_locked: true });
            }}
          />
          <label>
            Far (m)
            <input
              type="number"
              min={0.2}
              max={200}
              step={0.1}
              value={Number(params.vis_max_m.toFixed(3))}
              disabled={busy}
              onChange={(e) => {
                const vis_max_m = Number(e.target.value);
                setParams((p) => ({ ...p, vis_max_m, bounds_locked: true }));
                scheduleRender({ vis_max_m, bounds_locked: true });
              }}
            />
          </label>
          <input
            type="range"
            min={0.5}
            max={40}
            step={0.1}
            value={Math.min(params.vis_max_m, 40)}
            disabled={busy}
            onChange={(e) => {
              const vis_max_m = Number(e.target.value);
              setParams((p) => ({ ...p, vis_max_m, bounds_locked: true }));
              scheduleRender({ vis_max_m, bounds_locked: true });
            }}
          />
          {params.suggested_vis_min_m != null ? (
            <button
              type="button"
              disabled={busy || !hasDepth}
              onClick={() => {
                const vis_min_m = params.suggested_vis_min_m ?? params.vis_min_m;
                const vis_max_m = params.suggested_vis_max_m ?? params.vis_max_m;
                setParams((p) => ({
                  ...p,
                  vis_min_m,
                  vis_max_m,
                  bounds_locked: false,
                }));
                scheduleRender({ vis_min_m, vis_max_m, bounds_locked: false });
              }}
            >
              Use clip suggestion
              {` ${params.suggested_vis_min_m.toFixed(2)}–${params.suggested_vis_max_m?.toFixed(2)} m`}
            </button>
          ) : null}
          <label>
            Gamma
            <input
              type="range"
              min={0.4}
              max={2.2}
              step={0.05}
              value={params.gamma}
              disabled={busy}
              onChange={(e) => {
                const gamma = Number(e.target.value);
                setParams((p) => ({ ...p, gamma }));
                scheduleRender({ gamma });
              }}
            />
            <span>{params.gamma.toFixed(2)}</span>
          </label>
          <label>
            Cmap
            <select
              value={params.cmap}
              disabled={busy}
              onChange={(e) => {
                const cmap = e.target.value;
                setParams((p) => ({ ...p, cmap }));
                scheduleRender({ cmap });
              }}
            >
              {(meta?.film?.cmaps || ["gray", "magma", "turbo"]).map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
        </article>

        <article className="film-card">
          <header>
            <span className="film-rule">3</span>
            <div>
              <strong>Temporal EMA</strong>
              <p>After scale is locked, average pixels over time to kill micro-jitter.</p>
            </div>
          </header>
          <label className="film-check">
            <input
              type="checkbox"
              checked={params.ema_enabled}
              disabled={busy}
              onChange={(e) => {
                const ema_enabled = e.target.checked;
                setParams((p) => ({ ...p, ema_enabled }));
                scheduleRender({ ema_enabled });
              }}
            />
            Enable EMA
          </label>
          <label>
            Alpha (current frame)
            <input
              type="range"
              min={0.05}
              max={1}
              step={0.05}
              value={params.ema_alpha}
              disabled={busy || !params.ema_enabled}
              onChange={(e) => {
                const ema_alpha = Number(e.target.value);
                setParams((p) => ({ ...p, ema_alpha }));
                scheduleRender({ ema_alpha });
              }}
            />
            <span>{params.ema_alpha.toFixed(2)}</span>
          </label>
          <p className="muted">
            Lower alpha sticks to the previous frame (smoother). Higher follows
            motion (snappier).
          </p>
        </article>
      </aside>
    </div>
  );
}
