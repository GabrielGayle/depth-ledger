export type LookMeta = {
  key: string;
  label: string;
  cmap: string;
  lo_pct: number;
  hi_pct: number;
  gamma: number;
  grade?: string | null;
  blur_amount?: number;
};

export type ModalBilling = {
  gpu_type?: string;
  gpu_seconds?: number;
  cost_usd?: number;
  cold_start?: boolean;
  cost_usd_so_far?: number;
  gpu_seconds_so_far?: number;
  rolls?: number;
};

export type ModalUsage = {
  calls: number;
  gpu_seconds: number;
  cost_usd: number;
};

export type SystemStatus = {
  ts: string;
  poll_hint_ms: number;
  api: {
    ok: boolean;
    warmup: string;
    infer_backend_default: string;
    active_jobs: number;
    active_runs: { run_id: string; infer_status: string }[];
    warmup_error?: string | null;
  };
  local_infer: {
    state: string;
    device: string | null;
    last_run_id: string | null;
    last_error: string | null;
  };
  modal: {
    available: boolean;
    state: string;
    last_job: string | null;
    usage: ModalUsage;
    pricing?: { gpu_rates: Record<string, number>; disclaimer: string };
    last_error?: string | null;
  };
  ui: {
    mode: string;
    vite_up: string;
    last_evaluate_at: string | null;
    last_run_id: string | null;
  };
  blender_mcp: {
    blender_path_ok: boolean;
    addon: string;
    bpy: string;
    version: string | null;
    last_probe_ms: number;
    last_error: string | null;
    host?: string;
    port?: number;
  };
  cache?: {
    hits?: number;
    misses?: number;
    infer_entries?: number;
    infer_cache_mb?: number;
    outputs_mb?: number;
    disk_budget_mb?: number;
    ram_run_limit?: number;
    rss_mb?: number | null;
    run_dirs?: number;
    runs_cached?: number;
    hot_runs?: string[];
    hot_arrays_mb?: number;
  };
};

export type FilmMeta = {
  preview_frames: number;
  ema_alpha: number;
  vis_min_m: number;
  vis_max_m: number;
  gamma: number;
  cmap: string;
  cmaps: string[];
  anchor_focal: boolean;
  ema_enabled: boolean;
  video_max_frames: number;
  video_target_fps: number;
};

export type FilmParams = {
  pipeline?: string;
  anchor_focal: boolean;
  focallength_px: number | null;
  focallength_override: number | null;
  inferred_with_f_px?: number | null;
  vis_min_m: number;
  vis_max_m: number;
  suggested_vis_min_m: number | null;
  suggested_vis_max_m: number | null;
  bounds_locked: boolean;
  ema_enabled: boolean;
  ema_alpha: number;
  gamma: number;
  cmap: string;
  preview_frames: number;
  phase: string;
  preview_ready: number;
};

export type FilmResponse = {
  run_id: string;
  pipeline: "film" | string;
  kind: "video" | string;
  frame_count: number;
  ready_count: number;
  ready_indices: number[];
  vis_ready: number[];
  infer_status: string;
  sample_fps: number;
  source_fps?: number;
  duration_s?: number;
  max_size?: number;
  frame_width?: number;
  frame_height?: number;
  model_key?: string;
  infer_mode?: string | null;
  infer_backend?: string | null;
  params: FilmParams;
  rgb_url?: string | null;
  output_url?: string | null;
  preview_gif_url?: string | null;
  source_url?: string | null;
  vis_urls?: string[];
  rgb_urls?: string[];
  modal_usage?: ModalUsage;
  modal_billing?: ModalBilling;
};

export type FilmParamsPatch = Partial<
  Omit<FilmParams, "pipeline" | "phase" | "preview_ready">
> & {
  model_key?: string;
  max_size?: number;
  infer_backend?: "local" | "modal";
};

export type OperatorMeta = {
  algorithms: { id: number; label: string; desc: string }[];
  filter_types: string[];
  shaper_curves: string[];
  lfo_shapes: string[];
  mod_dests: string[];
  voices: string[];
  ndi_available?: boolean;
};

export type OperatorOscillator = {
  on: boolean;
  level: number;
  preset: string;
  lo_pct: number;
  hi_pct: number;
  gamma: number;
  cmap: string;
  blur_amount?: number;
};

export type OperatorPatch = {
  version: number;
  algorithm: number;
  oscillators: Record<"A" | "B" | "C" | "D", OperatorOscillator>;
  filter: { type: string; cutoff: number; drive: number };
  shaper: { curve: string; amount: number };
  envelope: {
    attack: number;
    decay: number;
    sustain: number;
    release: number;
    amount: number;
    dest: string[];
  };
  lfo: {
    rate: number;
    shape: string;
    amount: number;
    sync: boolean;
    dest: string[];
  };
  global: {
    time: number;
    spread: number;
    voices: string[];
  };
  t?: number | null;
};

export type OperatorEvaluateResponse = {
  run_id: string;
  output_url: string;
  elapsed_ms: number;
  osc_previews: Record<string, string>;
  voices: Record<string, string | null | undefined>;
  patch: OperatorPatch;
  spread?: number;
  ndi_sent?: boolean;
};

export type Meta = {
  models: { label: string; key: string }[];
  default_model: string;
  max_sizes: number[];
  default_max_size: number;
  infer_backends?: ("local" | "modal")[];
  default_infer_backend?: "local" | "modal";
  modal_available?: boolean;
  modal_pricing?: {
    gpu_rates: Record<string, number>;
    disclaimer: string;
  };
  looks: LookMeta[];
  blend_modes: { key: string; label: string }[];
  scenery_modes: { key: string; label: string }[];
  cmaps: string[];
  node_types: string[];
  relief?: {
    default_gc_alpha: number;
    default_invert: boolean;
    default_max_side: number;
    max_sides: number[];
  };
  smooth?: {
    modes?: { key: string; label: string }[];
    window_modes: { key: string; label: string }[];
    temporal_modes: { key: string; label: string }[];
    window_locks: { key: string; label: string }[];
    defaults: {
      mode?: string;
      window_mode: string;
      window_lock: string;
      spline_interval: number;
      lo_pct: number;
      hi_pct: number;
      temporal: string;
      ema_alpha: number;
      flow_prior?: number;
      occ_thresh?: number;
      flow_weight?: number;
      motion_gate?: number;
    };
  };
  freq_fuse?: {
    defaults: { beta: number; blur_sigma: number };
    backend_label?: string;
  };
  video_max_frames?: number;
  video_target_fps?: number;
  has_custom_default?: boolean;
  has_custom_video_default?: boolean;
  operator?: OperatorMeta;
  has_custom_operator_default?: boolean;
  film?: FilmMeta;
  arena?: {
    authenticated: boolean;
    user?: {
      id?: number;
      slug?: string;
      name?: string;
      email?: string;
    } | null;
    error?: string;
  };
};

export type ArenaChannelOption = {
  id: number;
  slug: string;
  title: string;
  visibility?: string;
  updated_at?: string;
  arena_url?: string;
  can_add?: boolean;
};

export type Graph = {
  version: number;
  nodes: any[];
  edges: any[];
};

export type InferResponse = {
  run_id: string;
  model_key: string;
  graph: Graph;
  looks: Record<string, string>;
  kind?: "still" | "video";
  frame_count?: number;
  frame_index?: number;
  ready_count?: number;
  ready_indices?: number[];
  infer_status?: string;
  infer_mode?: string;
  sample_fps?: number;
  max_size?: number;
  frame_width?: number;
  frame_height?: number;
  rgb_url?: string;
  blend_url?: string;
  output_url?: string;
  metric_url?: string | null;
  ply_url?: string | null;
  preview_gif_url?: string | null;
  source_url?: string | null;
  frame_urls?: string[];
  arena?: ArenaResolveResponse;
  infer_backend?: "local" | "modal" | string;
  infer_device?: string;
  modal_billing?: ModalBilling;
  modal_usage?: ModalUsage;
};

export type BakeResponse = {
  run_id: string;
  frame_count: number;
  ready_count: number;
  baked_count: number;
  frame_urls: string[];
  output_url: string;
};

export type ArenaBlockSummary = {
  id: number;
  class: string;
  title: string;
  has_image: boolean;
  thumb_url?: string | null;
  image_url?: string | null;
  source_url?: string | null;
  arena_url?: string;
};

export type ArenaChannelSummary = {
  id?: number;
  slug: string;
  title: string;
  length?: number;
  user?: string;
  arena_url?: string;
  imageable_count?: number;
};

export type ArenaResolveResponse = {
  kind: "block" | "channel";
  input: string;
  mode: string;
  block: ArenaBlockSummary;
  channel: ArenaChannelSummary | null;
  imageable_count: number;
  /** True when shuffle used the in-memory channel block cache. */
  cached?: boolean;
};

export type EvaluateResponse = {
  run_id: string;
  output_url: string;
  elapsed_ms: number;
  node_previews: Record<string, string>;
};

export type BatchRollItem = {
  index: number;
  status: string;
  title?: string | null;
  run_id?: string | null;
  arena_url?: string | null;
  error?: string | null;
};

export type JobStatus = {
  job_id: string;
  status: "running" | "done" | "error" | "cancelled" | string;
  progress: number;
  message?: string;
  run_id?: string | null;
  error?: string | null;
  result?: InferResponse | (Record<string, unknown> & { run_id?: string }) | null;
  kind?: string;
  phase?: string;
  i?: number;
  n?: number;
  saved?: number;
  downloaded?: number;
  inferred?: number;
  cached?: number;
  cooked?: number;
  published?: number;
  ready_frames?: number;
  total_frames?: number;
  ready_indices?: number[];
  infer_mode?: string;
  items?: BatchRollItem[];
  modal_billing?: ModalBilling;
  modal_usage?: ModalUsage;
};

export type FrameResponse = {
  run_id: string;
  frame_index: number;
  frame_count: number;
  rgb_url: string;
  output_url: string;
  node_previews: Record<string, string>;
  elapsed_ms?: number;
};

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function fetchMeta(): Promise<Meta> {
  return json(await fetch("/api/meta"));
}

export async function fetchStatus(deep = false): Promise<SystemStatus> {
  const q = deep ? "?deep=1" : "";
  return json(await fetch(`/api/status${q}`));
}

export async function postUiHeartbeat(body: {
  mode?: string;
  last_evaluate_at?: string | null;
  last_run_id?: string | null;
}): Promise<{ ok: boolean }> {
  return json(
    await fetch("/api/status/ui", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
  );
}

export async function exportBlender(runId: string): Promise<{
  ok: boolean;
  run_id: string;
  depth_exr_url?: string | null;
  inv_smooth_exr_url?: string | null;
  near_far_url?: string | null;
  rgb_url?: string | null;
  near_far?: Record<string, unknown>;
}> {
  return json(
    await fetch(`/api/runs/${runId}/export/blender`, { method: "POST" })
  );
}

export async function openInBlender(
  runId: string,
  opts?: { preset?: string }
): Promise<{
  ok: boolean;
  run_id: string;
  opened?: boolean;
  launch?: {
    pid: number;
    blend: string;
    run_dir: string;
    preset: string;
    log?: string;
  };
  near_far?: Record<string, unknown>;
}> {
  return json(
    await fetch(`/api/runs/${runId}/open/blender`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        preset: opts?.preset || "ledger",
        write_inv: true,
        write_matte: true,
      }),
    })
  );
}

export async function inferRgb(
  file: File,
  modelKey: string,
  maxSize: number,
  inferBackend: "local" | "modal" = "local"
): Promise<InferResponse> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("model_key", modelKey);
  fd.append("max_size", String(maxSize));
  fd.append("infer_backend", inferBackend);
  return json(
    await fetch("/api/infer/rgb", {
      method: "POST",
      body: fd,
    })
  );
}

export async function resolveArena(
  url: string,
  mode: "auto" | "shuffle" = "auto"
): Promise<ArenaResolveResponse> {
  return json(
    await fetch("/api/arena/resolve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, mode }),
    })
  );
}

export async function inferArena(
  url: string,
  modelKey: string,
  maxSize: number,
  mode: "auto" | "shuffle" = "auto",
  inferBackend: "local" | "modal" = "local"
): Promise<InferResponse> {
  return json(
    await fetch("/api/infer/arena", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url,
        mode,
        model_key: modelKey,
        max_size: maxSize,
        infer_backend: inferBackend,
      }),
    })
  );
}

export async function fetchArenaChannels(per = 24): Promise<{
  channels: ArenaChannelOption[];
  authenticated: boolean;
  user?: { id?: number; slug?: string; name?: string } | null;
}> {
  return json(await fetch(`/api/arena/channels?per=${per}`));
}

export type ArenaKitchenEntry = {
  block_id?: number;
  title?: string;
  arena_url?: string | null;
  run_name?: string | null;
  summary?: string;
  model_key?: string | null;
  max_size?: number | null;
  recipe: {
    app?: string;
    version?: number;
    run_name?: string;
    model_key?: string;
    max_size?: number;
    infer_backend?: string;
    graph?: Graph;
    summary?: string;
    arena?: Record<string, unknown>;
  };
};

export async function fetchArenaKitchens(
  url: string,
  limit = 40
): Promise<{
  kind: string;
  input: string;
  channel?: { id?: number; slug?: string; title?: string } | null;
  count: number;
  kitchens: ArenaKitchenEntry[];
}> {
  return json(
    await fetch("/api/arena/kitchens", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, limit }),
    })
  );
}

export async function publishToArena(
  runId: string,
  channelId: string,
  title?: string
): Promise<{
  ok: boolean;
  block: {
    id: number;
    title?: string;
    arena_url?: string;
    state?: string;
    channel_id?: string;
  };
  file?: string;
  title?: string;
  run_label?: string;
}> {
  return json(
    await fetch("/api/arena/publish", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        run_id: runId,
        channel_id: channelId,
        title: title || undefined,
      }),
    })
  );
}

export async function inferZip(
  file: File,
  modelKey: string,
  maxSize: number,
  inferBackend: "local" | "modal" = "local"
): Promise<InferResponse> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("model_key", modelKey);
  fd.append("max_size", String(maxSize));
  fd.append("infer_backend", inferBackend);
  return json(
    await fetch("/api/infer/zip", {
      method: "POST",
      body: fd,
    })
  );
}

export async function startVideoInfer(
  file: File,
  modelKey: string,
  maxSize: number,
  inferBackend: "local" | "modal" = "local"
): Promise<{ job_id: string }> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("model_key", modelKey);
  fd.append("max_size", String(maxSize));
  fd.append("infer_backend", inferBackend);
  return json(
    await fetch("/api/infer/video", {
      method: "POST",
      body: fd,
    })
  );
}

export async function prepareVideo(
  file: File,
  modelKey: string,
  maxSize: number,
  inferBackend: "local" | "modal" = "local"
): Promise<InferResponse> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("model_key", modelKey);
  fd.append("max_size", String(maxSize));
  fd.append("infer_backend", inferBackend);
  return json(
    await fetch("/api/video/prepare", {
      method: "POST",
      body: fd,
    })
  );
}

export async function startPreparedInfer(
  runId: string,
  opts?: {
    model_key?: string;
    max_size?: number;
    infer_backend?: "local" | "modal";
  }
): Promise<{ job_id: string; run_id: string }> {
  return json(
    await fetch(`/api/runs/${runId}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(opts || {}),
    })
  );
}

export async function bakeRun(
  runId: string,
  graph: Graph
): Promise<BakeResponse> {
  return json(
    await fetch(`/api/runs/${runId}/bake`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ graph }),
    })
  );
}

export async function startArenaRoll(body: {
  arena_url: string;
  count: number;
  model_key: string;
  max_size: number;
  infer_backend: "local" | "modal";
  wackiness: number;
  max_fidelity?: boolean;
  channel_id: string;
  title?: string;
  graph: Graph;
}): Promise<{ job_id: string }> {
  return json(
    await fetch("/api/batch/arena-roll", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
  );
}

export async function cancelJob(jobId: string): Promise<{ ok: boolean }> {
  return json(
    await fetch(`/api/jobs/${jobId}/cancel`, {
      method: "POST",
    })
  );
}

export async function fetchModalUsage(): Promise<ModalUsage> {
  return json(await fetch("/api/modal/usage"));
}

export async function pollJob(
  jobId: string,
  onProgress?: (job: JobStatus) => void,
  intervalMs = 500,
  shouldStop?: () => boolean
): Promise<JobStatus> {
  for (;;) {
    if (shouldStop?.()) {
      await cancelJob(jobId).catch(() => undefined);
    }
    const job = await fetchJob(jobId);
    onProgress?.(job);
    if (job.status === "done" || job.status === "error" || job.status === "cancelled") {
      return job;
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

export async function fetchJob(jobId: string): Promise<JobStatus> {
  return json(await fetch(`/api/jobs/${jobId}`));
}

export async function pollVideoJob(
  jobId: string,
  onProgress?: (job: JobStatus) => void,
  onPreview?: (result: InferResponse, job: JobStatus) => void,
  intervalMs = 400
): Promise<InferResponse> {
  let previewed = false;
  for (;;) {
    const job = await fetchJob(jobId);
    onProgress?.(job);
    if (job.result && !previewed && (job.phase === "preview" || job.status === "done")) {
      previewed = true;
      onPreview?.(job.result as InferResponse, job);
    } else if (job.result && previewed && job.status === "running") {
      // Grow scrubbable range as more frames land
      onProgress?.(job);
    }
    if (job.status === "done" && job.result) return job.result as InferResponse;
    if (job.status === "error") {
      throw new Error(job.error || job.message || "Video infer failed");
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

export async function setRunFrame(
  runId: string,
  index: number,
  graph?: Graph
): Promise<FrameResponse> {
  return json(
    await fetch(`/api/runs/${runId}/frame`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ index, graph }),
    })
  );
}

export async function exportRunVideo(
  runId: string,
  graph: Graph
): Promise<{ export_url: string; frame_count: number; fps: number }> {
  return json(
    await fetch(`/api/runs/${runId}/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ graph }),
    })
  );
}

export async function evaluateGraph(
  runId: string,
  graph: Graph
): Promise<EvaluateResponse> {
  return json(
    await fetch(`/api/runs/${runId}/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ graph }),
    })
  );
}

export async function evaluateOperator(
  runId: string,
  patch: OperatorPatch
): Promise<OperatorEvaluateResponse> {
  return json(
    await fetch(`/api/runs/${runId}/operator/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    })
  );
}

export async function exportOperatorVideo(
  runId: string,
  patch: OperatorPatch
): Promise<{ export_url: string; frame_count: number }> {
  return json(
    await fetch(`/api/runs/${runId}/operator/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    })
  );
}

export async function fetchDefaultOperator(): Promise<{
  has_custom: boolean;
  patch: OperatorPatch;
}> {
  return json(await fetch("/api/operator/default"));
}

export async function saveDefaultOperator(
  patch: OperatorPatch
): Promise<{ ok: boolean; has_custom: boolean }> {
  return json(
    await fetch("/api/operator/default", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    })
  );
}

export async function clearDefaultOperator(): Promise<{
  ok: boolean;
  cleared: boolean;
}> {
  return json(await fetch("/api/operator/default", { method: "DELETE" }));
}

export async function saveDefaultGraph(
  graph: Graph,
  kind: "still" | "video" = "still"
): Promise<{ ok: boolean; has_custom: boolean; kind?: string }> {
  return json(
    await fetch(`/api/graph/default?kind=${kind}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(graph),
    })
  );
}

export async function fetchDefaultGraph(
  kind: "still" | "video" = "still"
): Promise<{
  has_custom: boolean;
  kind?: string;
  graph: Graph | null;
}> {
  return json(await fetch(`/api/graph/default?kind=${kind}`));
}

export async function clearDefaultGraph(
  kind: "still" | "video" = "still"
): Promise<{
  ok: boolean;
  cleared: boolean;
  kind?: string;
}> {
  return json(
    await fetch(`/api/graph/default?kind=${kind}`, {
      method: "DELETE",
    })
  );
}

export type VideoRecipe = {
  id: string;
  label: string;
  summary: string;
  graph: Graph;
};

export async function fetchVideoRecipes(): Promise<VideoRecipe[]> {
  const data = await json<{ recipes: VideoRecipe[] }>(
    await fetch("/api/video/recipes")
  );
  return data.recipes || [];
}

export async function listRuns(): Promise<
  { run_id: string; label: string; has_recipe: boolean }[]
> {
  const data = await json<{ runs: any[] }>(await fetch("/api/runs"));
  return data.runs;
}

export type DepthSample = {
  depth_m: number;
  pct: number;
  u: number;
  v: number;
  px: number;
  py: number;
  w: number;
  h: number;
  frame_index: number;
};

export async function sampleDepthAt(
  runId: string,
  u: number,
  v: number,
  frame?: number
): Promise<DepthSample> {
  const params = new URLSearchParams({
    u: String(u),
    v: String(v),
  });
  if (frame != null) params.set("frame", String(frame));
  return json(await fetch(`/api/runs/${runId}/depth-sample?${params}`));
}

export async function fetchTunePreview(
  runId: string,
  opts?: { frame?: number; nodeId?: string }
): Promise<{ url: string; frame_index: number }> {
  const params = new URLSearchParams();
  if (opts?.frame != null) params.set("frame", String(opts.frame));
  if (opts?.nodeId) params.set("node_id", opts.nodeId);
  const q = params.toString();
  return json(
    await fetch(`/api/runs/${runId}/tune-preview${q ? `?${q}` : ""}`)
  );
}

export async function loadRun(runId: string): Promise<InferResponse> {
  return json(await fetch(`/api/runs/${runId}`));
}

export async function prepareFilm(
  file: File,
  opts: {
    modelKey: string;
    maxSize: number;
    inferBackend: "local" | "modal";
    targetFps?: number;
    maxFrames?: number;
  }
): Promise<FilmResponse> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("model_key", opts.modelKey);
  fd.append("max_size", String(opts.maxSize));
  fd.append("infer_backend", opts.inferBackend);
  if (opts.targetFps != null) fd.append("target_fps", String(opts.targetFps));
  if (opts.maxFrames != null) fd.append("max_frames", String(opts.maxFrames));
  return json(await fetch("/api/film/prepare", { method: "POST", body: fd }));
}

export async function fetchFilm(runId: string): Promise<FilmResponse> {
  return json(await fetch(`/api/film/${runId}`));
}

export async function startFilmPreview(
  runId: string,
  params: FilmParamsPatch
): Promise<{ job_id: string; run_id: string }> {
  return json(
    await fetch(`/api/film/${runId}/preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    })
  );
}

export async function startFilmInfer(
  runId: string,
  params: FilmParamsPatch
): Promise<{ job_id: string; run_id: string }> {
  return json(
    await fetch(`/api/film/${runId}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    })
  );
}

export async function renderFilm(
  runId: string,
  params: FilmParamsPatch
): Promise<FilmResponse> {
  return json(
    await fetch(`/api/film/${runId}/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    })
  );
}

export type BatchResponse = {
  run_id: string;
  pipeline: "batch" | string;
  kind: "batch" | string;
  frame_count: number;
  ready_count: number;
  ready_indices: number[];
  vis_ready: number[];
  infer_status: string;
  names: string[];
  max_size?: number;
  frame_width?: number;
  frame_height?: number;
  model_key?: string;
  infer_mode?: string | null;
  infer_backend?: string | null;
  export_folder?: string;
  export_dir?: string | null;
  source_dir?: string | null;
  saved_count?: number;
  saved_names?: string[];
  omitted?: number;
  params: FilmParams;
  rgb_urls?: string[];
  vis_urls?: string[];
  modal_billing?: ModalBilling;
};

export type BatchParamsPatch = FilmParamsPatch & {
  folder_name?: string;
};

export async function prepareBatch(
  files: File[],
  names: string[],
  opts: {
    modelKey: string;
    maxSize: number;
    inferBackend: "local" | "modal";
    folderName: string;
  }
): Promise<BatchResponse> {
  const fd = new FormData();
  files.forEach((file) => fd.append("files", file, file.name));
  fd.append("names", JSON.stringify(names));
  fd.append("model_key", opts.modelKey);
  fd.append("max_size", String(opts.maxSize));
  fd.append("infer_backend", opts.inferBackend);
  fd.append("folder_name", opts.folderName);
  return json(await fetch("/api/batch/stills/prepare", { method: "POST", body: fd }));
}

export async function startBatchPreview(
  runId: string,
  params: BatchParamsPatch
): Promise<{ job_id: string; run_id: string }> {
  return json(
    await fetch(`/api/batch/stills/${runId}/preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    })
  );
}

export async function startBatchRun(
  runId: string,
  params: BatchParamsPatch
): Promise<{ job_id: string; run_id: string }> {
  return json(
    await fetch(`/api/batch/stills/${runId}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    })
  );
}

export async function renderBatch(
  runId: string,
  params: BatchParamsPatch
): Promise<BatchResponse> {
  return json(
    await fetch(`/api/batch/stills/${runId}/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    })
  );
}

export type BatchFolderInfo = {
  path: string;
  name: string;
  count: number;
  omitted: number;
  names?: string[];
  status?: "queued" | "running" | "done" | "error" | "cancelled" | "ready" | string;
  progress?: number;
  saved_count?: number;
  export_dir?: string | null;
  error?: string | null;
};

export async function pickBatchFolder(): Promise<{ paths: string[]; cancelled?: boolean }> {
  return json(await fetch("/api/batch/stills/pick", { method: "POST" }));
}

export async function expandBatchFolders(
  paths: string[]
): Promise<{ folders: BatchFolderInfo[]; omitted_folders?: number }> {
  return json(
    await fetch("/api/batch/stills/inspect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paths }),
    })
  );
}

export async function openBatchFolder(
  path: string,
  opts: {
    modelKey: string;
    maxSize: number;
    inferBackend: "local" | "modal";
    folderName: string;
  }
): Promise<BatchResponse> {
  return json(
    await fetch("/api/batch/stills/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        path,
        model_key: opts.modelKey,
        max_size: opts.maxSize,
        infer_backend: opts.inferBackend,
        folder_name: opts.folderName,
      }),
    })
  );
}

export async function startBatchFolders(
  params: BatchParamsPatch & { paths: string[] }
): Promise<{ job_id: string }> {
  return json(
    await fetch("/api/batch/stills/run-folders", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    })
  );
}

export async function sampleBatch(
  runId: string,
  u: number,
  v: number,
  frame: number
): Promise<{ depth_m: number; pct: number; frame_index: number }> {
  const q = new URLSearchParams({
    u: String(u),
    v: String(v),
    frame: String(frame),
  });
  return json(await fetch(`/api/batch/stills/${runId}/sample?${q}`));
}

export async function exportFilm(
  runId: string
): Promise<{ export_url: string; frame_count: number; fps: number }> {
  return json(await fetch(`/api/film/${runId}/export`, { method: "POST" }));
}

export async function sampleFilm(
  runId: string,
  u: number,
  v: number,
  frame: number
): Promise<{ depth_m: number; pct: number; frame_index: number }> {
  const q = new URLSearchParams({
    u: String(u),
    v: String(v),
    frame: String(frame),
  });
  return json(await fetch(`/api/film/${runId}/sample?${q}`));
}

export function bust(url: string, token: number | string): string {
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}t=${token}`;
}
