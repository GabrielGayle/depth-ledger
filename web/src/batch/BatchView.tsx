import { useCallback, useMemo, useRef, useState, type MouseEvent } from "react";
import {
  bust,
  cancelJob,
  expandBatchFolders,
  openBatchFolder,
  pickBatchFolder,
  pollJob,
  renderBatch,
  sampleBatch,
  startBatchFolders,
  startBatchPreview,
  startBatchRun,
  type BatchFolderInfo,
  type BatchParamsPatch,
  type BatchResponse,
  type FilmParams,
  type Meta,
} from "../api";

function pad(i: number) {
  return String(i).padStart(6, "0");
}

function frameUrl(runId: string, index: number, kind: "rgb" | "vis") {
  const name = kind === "rgb" ? "rgb.jpg" : "vis.png";
  return `/api/runs/${runId}/files/frames/${pad(index)}/${name}`;
}

function defaultParams(meta: Meta | null): FilmParams {
  const f = meta?.film;
  return {
    pipeline: "batch",
    anchor_focal: true,
    focallength_px: null,
    focallength_override: null,
    vis_min_m: f?.vis_min_m ?? 0.5,
    vis_max_m: f?.vis_max_m ?? 10,
    suggested_vis_min_m: null,
    suggested_vis_max_m: null,
    bounds_locked: false,
    ema_enabled: false,
    ema_alpha: f?.ema_alpha ?? 0.45,
    gamma: f?.gamma ?? 1,
    cmap: f?.cmap ?? "gray",
    preview_frames: 1,
    phase: "idle",
    preview_ready: 0,
  };
}

export type BatchViewProps = {
  meta: Meta | null;
  modelKey: string;
  maxSize: number;
  setMaxSize: (n: number) => void;
  inferBackend: "local" | "modal";
  onStatus: (msg: string) => void;
  onProgress?: (p: number | null) => void;
  onBusy?: (busy: boolean) => void;
};

export function BatchView({
  meta,
  modelKey,
  maxSize,
  setMaxSize,
  inferBackend,
  onStatus,
  onProgress,
  onBusy,
}: BatchViewProps) {
  const [run, setRun] = useState<BatchResponse | null>(null);
  const [params, setParams] = useState<FilmParams>(() => defaultParams(meta));
  const [frameIndex, setFrameIndex] = useState(0);
  const [busy, setBusy] = useState(false);
  const [renderToken, setRenderToken] = useState(0);
  const [sample, setSample] = useState<{ depth_m: number; pct: number } | null>(null);
  const [folderName, setFolderName] = useState("depth");
  const [pathText, setPathText] = useState("");
  const [queue, setQueue] = useState<BatchFolderInfo[]>([]);
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

  const applyRun = useCallback((res: BatchResponse) => {
    setRun(res);
    if (res.params) {
      setParams({ ...res.params, ema_enabled: false, pipeline: "batch" });
      setOverrideText(
        res.params.focallength_override != null
          ? String(Math.round(res.params.focallength_override * 10) / 10)
          : ""
      );
    }
    setRenderToken((t) => t + 1);
    const vis = res.vis_ready || [];
    setFrameIndex((i) => (vis.includes(i) || i === 0 ? i : vis[0] ?? 0));
  }, []);

  const patch: BatchParamsPatch = useMemo(
    () => ({
      model_key: modelKey.startsWith("promptda") ? "depth-pro" : modelKey,
      max_size: maxSize,
      infer_backend: inferBackend,
      folder_name: folderName.trim() || "depth",
      anchor_focal: params.anchor_focal,
      focallength_override: overrideText ? Number(overrideText) || null : null,
      vis_min_m: params.vis_min_m,
      vis_max_m: params.vis_max_m,
      bounds_locked: params.bounds_locked,
      ema_enabled: false,
      gamma: params.gamma,
      cmap: params.cmap,
    }),
    [
      modelKey,
      maxSize,
      inferBackend,
      folderName,
      params.anchor_focal,
      params.vis_min_m,
      params.vis_max_m,
      params.bounds_locked,
      params.gamma,
      params.cmap,
      overrideText,
    ]
  );

  const visReady = run?.vis_ready || [];
  const readyIndices = run?.ready_indices || [];
  const frameCount = run?.frame_count || 0;
  const readyCount = run?.ready_count || 0;
  const names = run?.names || [];
  const hasDepth = readyCount > 0;
  const allInferred = frameCount > 0 && readyCount >= frameCount;
  const saved = (run?.saved_count || 0) > 0 && Boolean(run?.export_dir);
  const maxSizes = meta?.max_sizes || [518, 756, 1008, 1536];
  const maxRes = Math.max(...maxSizes);
  const fpx = params.focallength_override ?? params.focallength_px;
  const sizeStale = Boolean(run && run.max_size && run.max_size !== maxSize);

  const scheduleRender = useCallback(
    (next?: Partial<FilmParams>) => {
      if (!run?.run_id || readyCount < 1) return;
      if (next) paramsRef.current = { ...paramsRef.current, ...next };
      if (renderTimer.current) window.clearTimeout(renderTimer.current);
      renderTimer.current = window.setTimeout(() => {
        const id = run.run_id;
        const body: BatchParamsPatch = {
          ...patch,
          vis_min_m: paramsRef.current.vis_min_m,
          vis_max_m: paramsRef.current.vis_max_m,
          gamma: paramsRef.current.gamma,
          cmap: paramsRef.current.cmap,
          bounds_locked: true,
          ema_enabled: false,
        };
        renderBatch(id, body)
          .then((res) => {
            applyRun(res);
            onStatus("Look updated on the preview");
          })
          .catch((e) => onStatus(e?.message || "Render failed"));
      }, 220);
    },
    [run?.run_id, readyCount, patch, applyRun, onStatus]
  );

  const finishJob = async (jobId: string) => {
    jobIdRef.current = jobId;
    const job = await pollJob(
      jobId,
      (j) => {
        onProgress?.(typeof j.progress === "number" ? j.progress : null);
        if (j.message) onStatus(j.message);
        const result = j.result as BatchResponse | null | undefined;
        if (result?.run_id) applyRun(result);
      },
      400
    );
    if (job.status === "error") {
      throw new Error(job.error || job.message || "Job failed");
    }
    if (job.status === "cancelled") {
      onStatus("Cancelled");
      return null;
    }
    if (job.result) applyRun(job.result as BatchResponse);
    return job;
  };

  const depthModel = modelKey.startsWith("promptda") ? "depth-pro" : modelKey;

  const previewPath = async (info: BatchFolderInfo) => {
    setBusyAll(true);
    onProgress?.(0);
    onStatus(`Loading ${info.name}…`);
    setSample(null);
    try {
      const res = await openBatchFolder(info.path, {
        modelKey: depthModel,
        maxSize,
        inferBackend,
        folderName: folderName.trim() || "depth",
      });
      applyRun(res);
      setFrameIndex(0);
      setFileLabel(info.name);
      const dropped = res.omitted || 0;
      onStatus(
        dropped
          ? `${info.name}: ${res.frame_count} images, ${dropped} left out. Previewing the first…`
          : `${info.name}: ${res.frame_count} images. Previewing the first…`
      );
      const { job_id } = await startBatchPreview(res.run_id, {
        model_key: depthModel,
        max_size: maxSize,
        infer_backend: inferBackend,
        folder_name: folderName.trim() || "depth",
        anchor_focal: true,
        ema_enabled: false,
        bounds_locked: false,
      });
      const job = await finishJob(job_id);
      if (job) onStatus("First image ready — tweak the look, then run the folders");
    } catch (e: any) {
      onStatus(e?.message || "Import failed");
    } finally {
      jobIdRef.current = null;
      setBusyAll(false);
      onProgress?.(null);
    }
  };

  const addPaths = async (raws: string[]) => {
    const paths = raws.map((p) => p.trim()).filter(Boolean);
    if (!paths.length) return;
    const shouldPreview = queue.length === 0;
    setBusyAll(true);
    onStatus("Reading folders…");
    try {
      const found = await expandBatchFolders(paths);
      const folders = found.folders || [];
      setQueue((q) => {
        const seen = new Set(q.map((f) => f.path));
        const next = [...q];
        for (const info of folders) {
          if (!seen.has(info.path)) next.push({ ...info, status: "ready" });
        }
        return next;
      });
      setPathText("");
      const extra = found.omitted_folders
        ? ` ${found.omitted_folders} more folders left out.`
        : "";
      if (shouldPreview && folders[0]) {
        setBusyAll(false);
        onStatus(
          folders.length > 1
            ? `Added ${folders.length} folders.${extra} Previewing ${folders[0].name}…`
            : `Added ${folders[0].name} · ${folders[0].count} images`
        );
        await previewPath(folders[0]);
        return;
      }
      onStatus(`Added ${folders.length} folders.${extra}`);
    } catch (e: any) {
      onStatus(e?.message || "Could not read that folder");
    } finally {
      setBusyAll(false);
      onProgress?.(null);
    }
  };

  const onBrowse = async () => {
    setBusyAll(true);
    onStatus("Choose folders… Command-click to select more than one.");
    try {
      const picked = await pickBatchFolder();
      if (!picked.paths?.length) {
        onStatus(queue.length ? "Folder picker cancelled" : "Choose a folder to begin");
        return;
      }
      setBusyAll(false);
      await addPaths(picked.paths);
    } catch (e: any) {
      onStatus(e?.message || "Folder picker failed");
    } finally {
      setBusyAll(false);
    }
  };

  const onPreview = async () => {
    if (!run?.run_id) return;
    setBusyAll(true);
    setSample(null);
    onProgress?.(0);
    onStatus(sizeStale ? "Re-previewing the first image at the new max…" : "Previewing the first image…");
    try {
      const { job_id } = await startBatchPreview(run.run_id, patch);
      const job = await finishJob(job_id);
      if (job) onStatus("First image ready — tweak the look, then run the batch");
    } catch (e: any) {
      onStatus(e?.message || "Preview failed");
    } finally {
      jobIdRef.current = null;
      setBusyAll(false);
      onProgress?.(null);
    }
  };

  const onRun = async () => {
    if (!run?.run_id) return;
    const name = folderName.trim();
    if (!name) {
      onStatus("Name the save folder first");
      return;
    }
    setBusyAll(true);
    setSample(null);
    onProgress?.(0);
    onStatus(
      inferBackend === "modal"
        ? `Running the look across ${frameCount} images on Modal…`
        : `Running the look across ${frameCount} images…`
    );
    try {
      const { job_id } = await startBatchRun(run.run_id, { ...patch, folder_name: name });
      const job = await finishJob(job_id);
      const savedDir = (job?.result as BatchResponse | undefined)?.export_dir;
      const count = (job?.result as BatchResponse | undefined)?.saved_count;
      if (job && savedDir) {
        onStatus(`Saved ${count ?? frameCount} images to ${savedDir}`);
      }
    } catch (e: any) {
      onStatus(e?.message || "Batch failed");
    } finally {
      jobIdRef.current = null;
      setBusyAll(false);
      onProgress?.(null);
    }
  };

  const onRunAll = async () => {
    const name = folderName.trim();
    if (!queue.length || !name) {
      onStatus(name ? "Add a folder first" : "Name the save folder first");
      return;
    }
    setBusyAll(true);
    setSample(null);
    onProgress?.(0);
    onStatus(
      inferBackend === "modal"
        ? `Running ${queue.length} folders together on Modal…`
        : `Running ${queue.length} folders…`
    );
    try {
      const { job_id } = await startBatchFolders({
        ...patch,
        folder_name: name,
        paths: queue.map((f) => f.path),
      });
      jobIdRef.current = job_id;
      const job = await pollJob(
        job_id,
        (j) => {
          onProgress?.(typeof j.progress === "number" ? j.progress : null);
          if (j.message) onStatus(j.message);
          const folders = (j.result as { folders?: BatchFolderInfo[] } | null)?.folders;
          if (folders?.length) setQueue(folders);
        },
        400
      );
      const folders = (job.result as { folders?: BatchFolderInfo[] } | null)?.folders;
      if (folders?.length) setQueue(folders);
      if (job.status === "error") {
        throw new Error(job.error || job.message || "Batch failed");
      }
      if (job.status === "cancelled") {
        onStatus("Cancelled");
        return;
      }
      const saved = (job.result as { saved_count?: number } | null)?.saved_count;
      onStatus(
        typeof saved === "number"
          ? `Saved ${saved} images into each source folder / ${name}`
          : "Folders finished"
      );
    } catch (e: any) {
      onStatus(e?.message || "Batch failed");
    } finally {
      jobIdRef.current = null;
      setBusyAll(false);
      onProgress?.(null);
    }
  };

  const onCancel = () => {
    const id = jobIdRef.current;
    if (id) void cancelJob(id);
  };

  const onVisClick = async (e: MouseEvent<HTMLButtonElement>) => {
    if (!run?.run_id || !readyIndices.includes(frameIndex)) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const u = (e.clientX - rect.left) / Math.max(rect.width, 1);
    const v = (e.clientY - rect.top) / Math.max(rect.height, 1);
    try {
      const s = await sampleBatch(run.run_id, u, v, frameIndex);
      setSample({ depth_m: s.depth_m, pct: s.pct });
    } catch {
      setSample(null);
    }
  };

  const step = !run ? 0 : !hasDepth ? 1 : !allInferred || !saved ? 2 : 3;
  const currentName = names[frameIndex] || (frameCount ? `Image ${frameIndex + 1}` : "");
  const rgbSrc = run ? bust(frameUrl(run.run_id, frameIndex, "rgb"), renderToken) : null;
  const visSrc =
    run && visReady.includes(frameIndex)
      ? bust(frameUrl(run.run_id, frameIndex, "vis"), renderToken)
      : null;

  return (
    <div className="workspace film-workspace">
      <aside className="rail film-rail">
        <h2>Batch</h2>
        <p className="muted film-stack-lead">
          Add clip folders, or one parent folder that contains them. The first
          image is the preview. Run all copies that look into every clip at once.
        </p>
        <div className="film-drop">
          <div className="batch-add">
            <input
              type="text"
              value={pathText}
              disabled={busy}
              placeholder="/path/to/folder"
              onChange={(e) => setPathText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void addPaths([pathText]);
              }}
            />
            <button type="button" disabled={busy || !pathText.trim()} onClick={() => void addPaths([pathText])}>
              Add
            </button>
          </div>
          <button type="button" className="primary" disabled={busy} onClick={() => void onBrowse()}>
            Add folder
          </button>
          <p className="muted">
            {fileLabel ||
              "Command-click several folders, or pick one folder that contains the clip folders."}
          </p>
        </div>
        {queue.length ? (
          <ul className="batch-queue">
            {queue.map((folder) => (
              <li key={folder.path}>
                <div>
                  <strong>{folder.name}</strong>
                  <span className="muted">
                    {" "}
                    · {folder.count} images
                    {folder.status && folder.status !== "ready" ? ` · ${folder.status}` : ""}
                    {folder.saved_count ? ` · ${folder.saved_count} saved` : ""}
                  </span>
                  {folder.error ? <div className="muted">{folder.error}</div> : null}
                </div>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => setQueue((q) => q.filter((f) => f.path !== folder.path))}
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        ) : null}

        <div className="film-field">
          <label>
            Save into
            <span className="muted"> new folder inside each source folder</span>
            <input
              type="text"
              value={folderName}
              disabled={busy}
              placeholder="depth"
              onChange={(e) => setFolderName(e.target.value)}
            />
          </label>
        </div>

        <div className="film-res">
          <label title="Longest edge for Depth Pro. Max res uses the native cap.">
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
            {frameCount} images
            {run.frame_width ? ` · ${run.frame_width}×${run.frame_height}` : ""}
            <br />
            Depth {readyCount}/{frameCount}
            {sizeStale ? " · max changed — preview or run to apply" : ""}
            {run.omitted ? ` · ${run.omitted} left out` : ""}
          </p>
        ) : null}

        <div className="film-actions">
          <button
            type="button"
            disabled={busy || !run}
            title="Depth-map the first image so you can set the look"
            onClick={() => void onPreview()}
          >
            Preview first
          </button>
          <button
            type="button"
            className="primary"
            disabled={busy || !run || !hasDepth || !folderName.trim()}
            title="Copy this look onto the previewed folder and save inside it"
            onClick={() => void onRun()}
          >
            Run batch
          </button>
          <button
            type="button"
            disabled={busy || queue.length < 1 || !folderName.trim()}
            title={
              inferBackend === "modal"
                ? "Depth-map every folder at the same time on Modal, using this look"
                : "Depth-map every folder with this look. Local uses one GPU pass at a time."
            }
            onClick={() => void onRunAll()}
          >
            Run all folders
          </button>
          {busy ? (
            <button type="button" onClick={onCancel}>
              Cancel
            </button>
          ) : null}
        </div>

        {saved && run?.export_dir ? (
          <label className="film-field batch-path">
            Saved {run.saved_count} images
            <input readOnly value={run.export_dir} onFocus={(e) => e.currentTarget.select()} />
          </label>
        ) : (
          <p className="muted">
            Plates save as png, named after each source file, in a new folder inside the
            folder those images came from.
          </p>
        )}

        <ol className="film-steps">
          <li className={step === 0 ? "on" : step > 0 ? "done" : ""}>Import</li>
          <li className={step === 1 ? "on" : step > 1 ? "done" : ""}>Preview</li>
          <li className={step === 2 ? "on" : step > 2 ? "done" : ""}>Tweak</li>
          <li className={step === 3 ? "on" : ""}>Save</li>
        </ol>
      </aside>

      <section className="film-stage">
        <div className="film-plates">
          <div className="film-plate">
            <div className="film-plate-label">Source{currentName ? ` · ${currentName}` : ""}</div>
            {rgbSrc ? (
              <img src={rgbSrc} alt="Source" />
            ) : (
              <div className="film-empty">Import a folder to begin</div>
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
              <button type="button" className="film-vis-hit" onClick={(e) => void onVisClick(e)}>
                <img src={visSrc} alt="Depth" />
              </button>
            ) : (
              <div className="film-empty">
                {run ? "The first image’s depth shows up here" : "No depth yet"}
              </div>
            )}
          </div>
        </div>

        {frameCount > 0 ? (
          <div className="film-strip">
            {Array.from({ length: frameCount }, (_, i) => {
              const ready = visReady.includes(i);
              return (
                <button
                  key={i}
                  type="button"
                  className={`film-thumb${i === frameIndex ? " on" : ""}${ready ? "" : " wait"}`}
                  title={names[i] || `Image ${i + 1}`}
                  onClick={() => setFrameIndex(i)}
                >
                  <img
                    src={bust(frameUrl(run!.run_id, i, ready ? "vis" : "rgb"), renderToken)}
                    alt={names[i] || `image ${i + 1}`}
                  />
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
                setParams((p) => ({ ...p, vis_min_m, bounds_locked: true }));
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
                setParams((p) => ({ ...p, vis_max_m, bounds_locked: true }));
                scheduleRender({ vis_max_m, bounds_locked: true });
              }}
            >
              Set far
            </button>
          </div>
        ) : null}
      </section>

      <aside className="rail film-stack">
        <h2>Locked look</h2>
        <p className="muted film-stack-lead">
          Set this on the first image. Run batch copies the same focal length,
          meter window, gamma, and cmap onto every later file.
        </p>

        <article className="film-card">
          <header>
            <span className="film-rule">1</span>
            <div>
              <strong>Anchor focal length</strong>
              <p>Estimate f_px on the first image and reuse it for the series.</p>
            </div>
          </header>
          <label className="film-check">
            <input
              type="checkbox"
              checked={params.anchor_focal}
              disabled={busy}
              onChange={(e) => setParams((p) => ({ ...p, anchor_focal: e.target.checked }))}
            />
            Lock first-image focal
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
              placeholder="auto from first image"
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
              <p>One near/far in meters for the whole folder.</p>
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
              Use suggestion
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
                setParams((p) => ({ ...p, gamma, bounds_locked: true }));
                scheduleRender({ gamma, bounds_locked: true });
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
                setParams((p) => ({ ...p, cmap, bounds_locked: true }));
                scheduleRender({ cmap, bounds_locked: true });
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
      </aside>
    </div>
  );
}
