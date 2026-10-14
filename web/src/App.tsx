import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  addEdge,
  useEdgesState,
  useNodesState,
  type Connection,
  type Edge,
  type Node,
  type OnSelectionChangeParams,
  type NodeChange,
  type EdgeChange,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import {
  bakeRun,
  bust,
  cancelJob,
  clearDefaultGraph,
  evaluateGraph,
  exportRunVideo,
  fetchArenaChannels,
  fetchArenaKitchens,
  fetchDefaultGraph,
  fetchMeta,
  fetchModalUsage,
  fetchVideoRecipes,
  inferArena,
  inferRgb,
  inferZip,
  listRuns,
  loadRun,
  pollJob,
  pollVideoJob,
  prepareVideo,
  publishToArena,
  resolveArena,
  saveDefaultGraph,
  setRunFrame,
  startArenaRoll,
  startPreparedInfer,
  openInBlender,
  type ArenaChannelOption,
  type ArenaKitchenEntry,
  type Graph,
  type InferResponse,
  type Meta,
  type ModalBilling,
  type VideoRecipe,
} from "./api";
import { type BlendEdgeData } from "./edges";
import { KitchenContext, clampWindowPct, lookDefaults } from "./kitchenContext";
import type { WindowTuneMode, WindowTuneState } from "./kitchenContext";
import { KitchenView } from "./KitchenView";
import { OperatorView } from "./operator/OperatorView";
import { VideoView } from "./video/VideoView";
import { BatchView } from "./batch/BatchView";
import { arenaDataFromResolve, mergeArenaNodeData } from "./arenaUtils";
import { StatusStrip } from "./StatusStrip";

const PROTECTED_TYPES = new Set(["depth", "output", "video_out"]);
const IMAGE_TYPES = new Set(["look", "scene", "blend", "relief"]);

function uid(prefix: string) {
  return `${prefix}_${Math.random().toString(36).slice(2, 8)}`;
}

function formatElapsed(ms: number): string {
  const s = ms / 1000;
  return s < 10 ? `${s.toFixed(1)}s` : `${Math.round(s)}s`;
}

/** Live status clock while a long infer waits (Modal cold start, download, etc.). */
function startStatusTimer(
  setStatus: (s: string) => void,
  label: string
): { stop: () => number } {
  const t0 = performance.now();
  let stopped = false;
  const tick = () => {
    if (stopped) return;
    setStatus(`${label} · ${formatElapsed(performance.now() - t0)}`);
  };
  tick();
  const id = window.setInterval(tick, 250);
  return {
    stop: () => {
      if (stopped) return performance.now() - t0;
      stopped = true;
      window.clearInterval(id);
      return performance.now() - t0;
    },
  };
}

function isBlendableEdge(
  sourceType: string | undefined,
  targetType: string | undefined
) {
  return (
    Boolean(sourceType && IMAGE_TYPES.has(sourceType)) &&
    Boolean(targetType && (targetType === "look" || targetType === "scene"))
  );
}

function annotateEdge(
  e: Edge,
  nodesById: Map<string, Node>
): Edge {
  const src = nodesById.get(e.source);
  const tgt = nodesById.get(e.target);
  const blendable = isBlendableEdge(src?.type, tgt?.type);
  const prev = (e.data || {}) as BlendEdgeData;
  const data: BlendEdgeData = blendable
    ? {
        blend_mode: prev.blend_mode || "normal",
        opacity: prev.opacity ?? 1,
        blendable: true,
      }
    : { blendable: false };
  return {
    ...e,
    type: "blend",
    data,
    deletable: true,
    selectable: true,
  };
}

// Saved graphs use the compact coordinate space; the canvas spreads them out
// so the larger node panels don't overlap.
const LAYOUT_SCALE = 1.5;

function toCanvasPos(p: { x: number; y: number }) {
  return { x: p.x * LAYOUT_SCALE, y: p.y * LAYOUT_SCALE };
}

function toGraphPos(p: { x: number; y: number }) {
  return { x: p.x / LAYOUT_SCALE, y: p.y / LAYOUT_SCALE };
}

function toFlow(graph: Graph, meta: Meta | null): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = (graph.nodes || []).map((n) => {
    const data = { ...(n.data || {}) };
    if (n.type === "look") {
      const preset = data.preset || "ledger";
      const defaults = lookDefaults(meta, preset);
      if (data.collapsed === undefined) data.collapsed = true;
      if (data.lo_pct == null && defaults) data.lo_pct = defaults.lo_pct;
      if (data.hi_pct == null && defaults) data.hi_pct = defaults.hi_pct;
      if (data.gamma == null && defaults) data.gamma = defaults.gamma;
      if (!data.cmap) data.cmap = "inherit";
      delete data.blend_mode;
      delete data.opacity;
    }
    if (n.type === "scene") {
      if (data.collapsed === undefined) data.collapsed = true;
      delete data.blend_mode;
      delete data.opacity;
    }
    if (n.type === "relief") {
      if (data.collapsed === undefined) data.collapsed = true;
      if (data.gc_alpha == null) {
        data.gc_alpha = meta?.relief?.default_gc_alpha ?? 10;
      }
      if (data.invert == null) {
        data.invert = meta?.relief?.default_invert ?? false;
      }
      if (data.max_side == null) {
        data.max_side = meta?.relief?.default_max_side ?? 512;
      }
    }
    if (n.type === "smooth") {
      const d = meta?.smooth?.defaults;
      if (data.collapsed === undefined) data.collapsed = true;
      if (data.mode == null) data.mode = d?.mode ?? "ema";
      if (data.window_mode == null) data.window_mode = d?.window_mode ?? "locked";
      if (data.window_lock == null) data.window_lock = d?.window_lock ?? "first";
      if (data.spline_interval == null) {
        data.spline_interval = d?.spline_interval ?? 30;
      }
      if (data.lo_pct == null) data.lo_pct = d?.lo_pct ?? 2;
      if (data.hi_pct == null) data.hi_pct = d?.hi_pct ?? 98;
      if (data.temporal == null) data.temporal = d?.temporal ?? "ema";
      if (data.ema_alpha == null) data.ema_alpha = d?.ema_alpha ?? 0.45;
      if (data.flow_prior == null) {
        data.flow_prior =
          d?.flow_prior ??
          (data.flow_weight != null ? 1 - Number(data.flow_weight) : 0.75);
      }
      if (data.occ_thresh == null) data.occ_thresh = d?.occ_thresh ?? 1.5;
    }
    if (n.type === "freq_fuse") {
      const d = meta?.freq_fuse?.defaults;
      if (data.collapsed === undefined) data.collapsed = true;
      if (data.beta == null) data.beta = d?.beta ?? 1;
      if (data.blur_sigma == null) data.blur_sigma = d?.blur_sigma ?? 4;
    }
    if (
      (n.type === "output" || n.type === "video_out") &&
      data.preview_width == null
    ) {
      data.preview_width = 420;
    }
    return {
      id: n.id,
      type: n.type,
      position: toCanvasPos(n.position || { x: 0, y: 0 }),
      data,
      selected: false,
      deletable: !PROTECTED_TYPES.has(n.type || ""),
    };
  });
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const edges = (graph.edges || []).map((e) =>
    annotateEdge(
      {
        id: e.id,
        source: e.source,
        target: e.target,
        sourceHandle: e.sourceHandle ?? undefined,
        targetHandle: e.targetHandle ?? undefined,
        data: e.data || {},
      },
      byId
    )
  );
  return { nodes, edges };
}

function fromFlow(nodes: Node[], edges: Edge[]): Graph {
  return {
    version: 2,
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.type,
      position: toGraphPos(n.position),
      data: n.data,
    })),
    edges: edges.map((e) => {
      const d = (e.data || {}) as BlendEdgeData;
      const data =
        d.blendable === false
          ? undefined
          : {
              blend_mode: d.blend_mode || "normal",
              opacity: d.opacity ?? 1,
            };
      return {
        id: e.id,
        source: e.source,
        target: e.target,
        sourceHandle: e.sourceHandle,
        targetHandle: e.targetHandle,
        ...(data ? { data } : {}),
      };
    }),
  };
}

function replaceTargetEdge(
  edges: Edge[],
  connection: Connection,
  nodes: Node[]
): Edge[] {
  const handle = connection.targetHandle ?? null;
  const filtered = edges.filter(
    (e) =>
      !(
        e.target === connection.target &&
        (e.targetHandle ?? null) === handle
      )
  );
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const draft: Edge = {
    id: uid("e"),
    source: connection.source!,
    target: connection.target!,
    sourceHandle: connection.sourceHandle,
    targetHandle: connection.targetHandle,
    data: {},
  };
  return addEdge(annotateEdge(draft, byId), filtered);
}

type ProgressPhase = null | "upload" | "infer" | "evaluate";

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [modelKey, setModelKey] = useState("depth-pro");
  const [maxSize, setMaxSize] = useState(756);
  const [inferBackend, setInferBackend] = useState<"local" | "modal">(() => {
    try {
      const stored = localStorage.getItem("depth_ledger_infer_backend");
      if (stored === "modal" || stored === "local") return stored;
    } catch {
      /* ignore */
    }
    return "local";
  });
  const [modalLastCost, setModalLastCost] = useState<number | null>(null);
  const [modalSessionCost, setModalSessionCost] = useState(0);
  const [modalBatchBilling, setModalBatchBilling] =
    useState<ModalBilling | null>(null);
  const [inferDevice, setInferDevice] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [nodePreviews, setNodePreviews] = useState<Record<string, string>>({});
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const [rgbUrl, setRgbUrl] = useState<string | null>(null);
  const [lightbox, setLightbox] = useState<{
    url: string;
    title: string;
  } | null>(null);
  const [hasCustomDefault, setHasCustomDefault] = useState(false);
  const [hasCustomVideoDefault, setHasCustomVideoDefault] = useState(false);
  const [status, setStatus] = useState("Booting…");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<ProgressPhase>(null);
  const [evalToken, setEvalToken] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [edgeModalId, setEdgeModalId] = useState<string | null>(null);
  const [runs, setRuns] = useState<{ run_id: string; label: string }[]>([]);
  const [frameCount, setFrameCount] = useState(1);
  const [readyFrames, setReadyFrames] = useState(1);
  const [readyIndices, setReadyIndices] = useState<number[]>([]);
  const [inferMode, setInferMode] = useState<string | null>(null);
  const [frameWidth, setFrameWidth] = useState<number | null>(null);
  const [frameHeight, setFrameHeight] = useState<number | null>(null);
  const [frameStatusExpanded, setFrameStatusExpanded] = useState(false);
  const [inferStatus, setInferStatus] = useState<string>("done");
  const [frameIndex, setFrameIndex] = useState(0);
  const [sampleFps, setSampleFps] = useState(10);
  const [playing, setPlaying] = useState(false);
  const [exportUrl, setExportUrl] = useState<string | null>(null);
  const [videoProgress, setVideoProgress] = useState<number | null>(null);
  const [previewGifUrl, setPreviewGifUrl] = useState<string | null>(null);
  const [frameUrls, setFrameUrls] = useState<string[]>([]);
  const [arenaChannels, setArenaChannels] = useState<ArenaChannelOption[]>([]);
  const [arenaAuth, setArenaAuth] = useState(false);
  const [kitchenUrl, setKitchenUrl] = useState("");
  const [arenaKitchens, setArenaKitchens] = useState<ArenaKitchenEntry[]>([]);
  const [videoRecipes, setVideoRecipes] = useState<VideoRecipe[]>([]);
  const [kitchenPick, setKitchenPick] = useState("");
  const [windowTune, setWindowTune] = useState<WindowTuneState>(null);
  const [appMode, setAppMode] = useState<
    "kitchen" | "operator" | "video" | "batch"
  >("kitchen");
  const [filmHistoryId, setFilmHistoryId] = useState<string | null>(null);
  const [lastEvaluateAt, setLastEvaluateAt] = useState<string | null>(null);

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const evalTimer = useRef<number | null>(null);
  const evalOverlayTimer = useRef<number | null>(null);
  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  const fileRgb = useRef<HTMLInputElement>(null);
  const fileZip = useRef<HTMLInputElement>(null);
  const playTimer = useRef<number | null>(null);
  const frameIndexRef = useRef(0);
  const seekingRef = useRef(false);
  const batchAbortRef = useRef(false);
  const batchJobIdRef = useRef<string | null>(null);
  const frameCountRef = useRef(1);
  const readyFramesRef = useRef(1);
  const inferStatusRef = useRef("done");

  const formatUsd = (n: number) => {
    if (n < 0.01) return `~$${n.toFixed(4)}`;
    return `~$${n.toFixed(3)}`;
  };

  const applyModalBilling = useCallback((res: {
    modal_billing?: ModalBilling | null;
    modal_usage?: { cost_usd?: number } | null;
    infer_device?: string | null;
    infer_backend?: string | null;
  }) => {
    if (res.modal_billing?.cost_usd != null) {
      setModalLastCost(Number(res.modal_billing.cost_usd));
    }
    if (res.modal_usage?.cost_usd != null) {
      setModalSessionCost(Number(res.modal_usage.cost_usd));
      try {
        sessionStorage.setItem(
          "depth_ledger_modal_session",
          String(res.modal_usage.cost_usd)
        );
      } catch {
        /* ignore */
      }
    }
    if (res.infer_device) setInferDevice(res.infer_device);
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("depth_ledger_infer_backend", inferBackend);
    } catch {
      /* ignore */
    }
  }, [inferBackend]);

  useEffect(() => {
    nodesRef.current = nodes;
  }, [nodes]);
  useEffect(() => {
    edgesRef.current = edges;
  }, [edges]);
  useEffect(() => {
    frameIndexRef.current = frameIndex;
  }, [frameIndex]);
  useEffect(() => {
    frameCountRef.current = frameCount;
  }, [frameCount]);
  useEffect(() => {
    readyFramesRef.current = readyFrames;
  }, [readyFrames]);
  useEffect(() => {
    inferStatusRef.current = inferStatus;
  }, [inferStatus]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [m, d, recipes] = await Promise.all([
          fetchMeta(),
          fetchDefaultGraph(),
          fetchVideoRecipes().catch(() => [] as VideoRecipe[]),
        ]);
        if (cancelled) return;
        setMeta(m);
        setVideoRecipes(recipes);
        setModelKey(m.default_model);
        setMaxSize(m.default_max_size);
        if (m.default_infer_backend === "modal" || m.default_infer_backend === "local") {
          try {
            const stored = localStorage.getItem("depth_ledger_infer_backend");
            if (!stored) setInferBackend(m.default_infer_backend);
          } catch {
            setInferBackend(m.default_infer_backend);
          }
        }
        setHasCustomDefault(d.has_custom);
        setHasCustomVideoDefault(Boolean(m.has_custom_video_default));
        setArenaAuth(Boolean(m.arena?.authenticated));
        if (m.modal_available) {
          fetchModalUsage()
            .then((u) => {
              if (!cancelled) setModalSessionCost(u.cost_usd || 0);
            })
            .catch(() => undefined);
        } else {
          try {
            const s = sessionStorage.getItem("depth_ledger_modal_session");
            if (s) setModalSessionCost(parseFloat(s) || 0);
          } catch {
            /* ignore */
          }
        }
        if (d.graph) {
          const flow = toFlow(d.graph, m);
          setNodes(flow.nodes);
          setEdges(flow.edges);
          setStatus(
            m.arena?.authenticated
              ? `Ready · Are.na @${m.arena.user?.slug || "ok"}`
              : "Ready — Are.na or Generate to load depth"
          );
        } else {
          setStatus("Ready");
        }
        if (m.arena?.authenticated) {
          fetchArenaChannels(24)
            .then((res) => {
              if (!cancelled) {
                setArenaChannels(res.channels || []);
                setArenaAuth(Boolean(res.authenticated));
              }
            })
            .catch(() => undefined);
        }
      } catch (e: any) {
        if (!cancelled) setStatus(`Boot error: ${e.message}`);
      }
      listRuns()
        .then((r) => {
          if (!cancelled) setRuns(r);
        })
        .catch(() => undefined);
    })();
    return () => {
      cancelled = true;
    };
  }, [setEdges, setNodes]);

  const selectedNode = useMemo(
    () => nodes.find((n) => n.id === selectedId) || null,
    [nodes, selectedId]
  );

  const modalEdge = useMemo(
    () => edges.find((e) => e.id === edgeModalId) || null,
    [edges, edgeModalId]
  );

  const ingredientNodes = useMemo(
    () =>
      nodes.filter(
        (n) =>
          n.type === "look" ||
          n.type === "scene" ||
          n.type === "blend" ||
          n.type === "relief" ||
          n.type === "smooth" ||
          n.type === "freq_fuse"
      ),
    [nodes]
  );

  const scheduleEvaluate = useCallback(
    (nextNodes: Node[], nextEdges: Edge[], rid: string | null, opts?: { bake?: boolean }) => {
      if (!rid) return;
      if (evalTimer.current) window.clearTimeout(evalTimer.current);
      if (evalOverlayTimer.current) window.clearTimeout(evalOverlayTimer.current);

      evalTimer.current = window.setTimeout(async () => {
        setStatus("Cooking graph…");
        evalOverlayTimer.current = window.setTimeout(() => {
          setProgress("evaluate");
        }, 300);
        try {
          const graph = fromFlow(nextNodes, nextEdges);
          const res = await evaluateGraph(rid, graph);
          setOutputUrl(res.output_url);
          setNodePreviews(res.node_previews || {});
          setEvalToken(Date.now());
          setLastEvaluateAt(new Date().toISOString());

          const fc = frameCountRef.current;
          const ready = readyFramesRef.current;
          const status = inferStatusRef.current;
          const shouldBake =
            opts?.bake === true ||
            (fc > 1 && ready >= 1 && status === "done");
          if (shouldBake && fc > 1 && ready >= 1) {
            setStatus("Baking video frames…");
            const baked = await bakeRun(rid, graph);
            setFrameUrls(baked.frame_urls || []);
            if (baked.frame_urls?.[0]) {
              setOutputUrl(baked.frame_urls[0]);
            }
            setStatus(
              `Baked ${baked.baked_count} frames · eval ${res.elapsed_ms} ms`
            );
          } else {
            setStatus(`Evaluated in ${res.elapsed_ms} ms`);
          }
        } catch (e: any) {
          setStatus(`Eval error: ${e.message}`);
        } finally {
          if (evalOverlayTimer.current) {
            window.clearTimeout(evalOverlayTimer.current);
            evalOverlayTimer.current = null;
          }
          setProgress((p) => (p === "evaluate" ? null : p));
        }
      }, 150);
    },
    []
  );

  const evaluateNow = useCallback(
    async (nextNodes: Node[], nextEdges: Edge[], rid: string) => {
      if (evalTimer.current) {
        window.clearTimeout(evalTimer.current);
        evalTimer.current = null;
      }
      if (evalOverlayTimer.current) {
        window.clearTimeout(evalOverlayTimer.current);
        evalOverlayTimer.current = null;
      }
      setProgress("evaluate");
      const res = await evaluateGraph(rid, fromFlow(nextNodes, nextEdges));
      setOutputUrl(res.output_url);
      setNodePreviews(res.node_previews || {});
      setEvalToken(Date.now());
      setProgress(null);
      return res;
    },
    []
  );

  const patchNodeData = useCallback(
    (nodeId: string, patch: Record<string, unknown>) => {
      setNodes((nds) => {
        const next = nds.map((n) =>
          n.id === nodeId ? { ...n, data: { ...n.data, ...patch } } : n
        );
        scheduleEvaluate(next, edgesRef.current, runId);
        return next;
      });
    },
    [runId, scheduleEvaluate, setNodes]
  );

  const patchNodeUi = useCallback(
    (nodeId: string, patch: Record<string, unknown>) => {
      setNodes((nds) =>
        nds.map((n) =>
          n.id === nodeId ? { ...n, data: { ...n.data, ...patch } } : n
        )
      );
    },
    [setNodes]
  );

  const openLightbox = useCallback((url: string, title?: string) => {
    setLightbox({ url, title: title || "Preview" });
  }, []);

  const startWindowTune = useCallback((nodeId: string) => {
    setWindowTune({ nodeId, mode: "near" });
  }, []);

  const stopWindowTune = useCallback(() => {
    setWindowTune(null);
  }, []);

  const setWindowTuneMode = useCallback((mode: WindowTuneMode) => {
    setWindowTune((t) => (t ? { ...t, mode } : t));
  }, []);

  const applyWindowPct = useCallback(
    (nodeId: string, field: "lo_pct" | "hi_pct", pct: number) => {
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id !== nodeId) return n;
          const d = n.data as Record<string, unknown>;
          const lo = Number(d.lo_pct ?? 2);
          const hi = Number(d.hi_pct ?? 98);
          const value = clampWindowPct(field, pct, lo, hi);
          return { ...n, data: { ...d, [field]: value } };
        })
      );
    },
    [setNodes]
  );

  const commitWindowTune = useCallback(async () => {
    if (!runId) return;
    try {
      await evaluateNow(nodesRef.current, edgesRef.current, runId);
    } catch (e: any) {
      setStatus(`Eval error: ${e.message}`);
    }
  }, [evaluateNow, runId]);

  const openBlender = useCallback(async () => {
    if (!runId) {
      setStatus("Generate or load a run first");
      return;
    }
    setBusy(true);
    setStatus("Exporting depth for Blender…");
    try {
      const res = await openInBlender(runId, { preset: "ledger" });
      const pid = res.launch?.pid;
      setStatus(
        pid
          ? `Opened Blender (pid ${pid}) · ${runId}`
          : `Opened Blender · ${runId}`
      );
    } catch (e: any) {
      setStatus(`Blender open failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }, [runId]);

  const graphKind = frameCount > 1 ? "video" : "still";
  const activeHasCustomDefault =
    graphKind === "video" ? hasCustomVideoDefault : hasCustomDefault;

  const saveAsDefault = useCallback(async () => {
    if (!nodesRef.current.length) {
      setStatus("Nothing to save");
      return;
    }
    const kind = frameCount > 1 ? "video" : "still";
    try {
      await saveDefaultGraph(fromFlow(nodesRef.current, edgesRef.current), kind);
      if (kind === "video") setHasCustomVideoDefault(true);
      else setHasCustomDefault(true);
      setStatus(
        kind === "video"
          ? "Saved current graph as default video kitchen"
          : "Saved current graph as default kitchen"
      );
    } catch (e: any) {
      setStatus(`Save default failed: ${e.message}`);
    }
  }, [frameCount]);

  const resetDefault = useCallback(async () => {
    const kind = frameCount > 1 ? "video" : "still";
    try {
      await clearDefaultGraph(kind);
      if (kind === "video") setHasCustomVideoDefault(false);
      else setHasCustomDefault(false);
      const d = await fetchDefaultGraph(kind);
      if (d.graph) {
        const flow = toFlow(d.graph, meta);
        setNodes(flow.nodes);
        setEdges(flow.edges);
      }
      setStatus(
        kind === "video"
          ? "Restored built-in video kitchen"
          : "Restored built-in default kitchen"
      );
    } catch (e: any) {
      setStatus(`Reset default failed: ${e.message}`);
    }
  }, [frameCount, meta, setEdges, setNodes]);

  const applyArenaKitchen = useCallback(
    (entry: ArenaKitchenEntry) => {
      const recipe = entry.recipe || {};
      const g = recipe.graph;
      if (!g || !Array.isArray(g.nodes) || g.nodes.length === 0) {
        setStatus("Kitchen recipe has no graph");
        return;
      }

      const rid = runId || "";
      const current = fromFlow(nodesRef.current, edgesRef.current);
      const preserveTypes = new Set(["output", "arena_out"]);
      const preserved = (current.nodes || []).filter((n) =>
        preserveTypes.has(String(n.type || ""))
      );

      const recipeNodes = (g.nodes || []).map((n) => {
        const data = { ...(n.data || {}) };
        if (n.type === "depth") data.run_id = rid;
        return { ...n, data };
      });
      const byId = new Map(recipeNodes.map((n) => [n.id, n]));
      for (const p of preserved) {
        if (!byId.has(p.id)) {
          recipeNodes.push(p);
          byId.set(p.id, p);
        } else if (p.type === "arena_out") {
          // Keep channel / title from the live canvas.
          const existing = byId.get(p.id)!;
          existing.data = {
            ...(existing.data || {}),
            channel_id: (p.data as any)?.channel_id || "",
            title: (p.data as any)?.title || "",
            batch_count: (p.data as any)?.batch_count,
            wackiness: (p.data as any)?.wackiness,
            max_fidelity: (p.data as any)?.max_fidelity,
          };
        }
      }

      // Ensure output exists.
      if (![...byId.values()].some((n) => n.type === "output")) {
        const maxX = Math.max(
          0,
          ...recipeNodes.map((n) => Number(n.position?.x || 0))
        );
        recipeNodes.push({
          id: "output",
          type: "output",
          position: { x: maxX + 280, y: 40 },
          data: { preview_width: 420 },
        });
        byId.set("output", recipeNodes[recipeNodes.length - 1]);
      }

      let nextEdges = [...(g.edges || [])];
      const edgeIds = new Set(nextEdges.map((e) => e.id));
      const imageTypes = new Set(["look", "scene", "blend", "relief"]);
      const terminals = recipeNodes.filter((n) => imageTypes.has(String(n.type)));
      const terminal =
        terminals.find((n) => n.type === "relief") ||
        terminals[terminals.length - 1];
      const outNode = recipeNodes.find((n) => n.type === "output");
      const arenaOut = recipeNodes.find((n) => n.type === "arena_out");

      const ensureEdge = (
        id: string,
        source: string,
        target: string,
        data?: Record<string, unknown>
      ) => {
        if (nextEdges.some((e) => e.source === source && e.target === target)) {
          return;
        }
        let eid = id;
        while (edgeIds.has(eid)) eid = `${id}_${Math.random().toString(36).slice(2, 5)}`;
        edgeIds.add(eid);
        nextEdges.push({ id: eid, source, target, data });
      };

      if (terminal && outNode) {
        ensureEdge(`e_${terminal.id}_o`, terminal.id, outNode.id);
      }
      if (terminal && arenaOut) {
        ensureEdge(`e_${terminal.id}_ao`, terminal.id, arenaOut.id);
      }

      const graph: Graph = {
        version: 2,
        nodes: recipeNodes,
        edges: nextEdges,
      };
      const flow = toFlow(graph, meta);
      setNodes(flow.nodes);
      setEdges(flow.edges);
      setAppMode("kitchen");

      if (recipe.model_key) setModelKey(String(recipe.model_key));
      if (recipe.max_size) setMaxSize(Number(recipe.max_size));

      const label = entry.run_name || entry.title || entry.summary || "kitchen";
      setStatus(`Applied kitchen · ${label}`);
      if (rid) scheduleEvaluate(flow.nodes, flow.edges, rid);
    },
    [meta, runId, scheduleEvaluate, setEdges, setNodes]
  );

  const applyVideoRecipe = useCallback(
    (recipe: VideoRecipe) => {
      const g = recipe.graph;
      if (!g || !Array.isArray(g.nodes) || g.nodes.length === 0) {
        setStatus("Video recipe has no graph");
        return;
      }

      const rid = runId || "";
      const liveIn = nodesRef.current.find((n) => n.type === "video_in");
      const liveOut = nodesRef.current.find((n) => n.type === "video_out");

      let nextEdges = (g.edges || []).map((e: any) => ({ ...e }));
      const recipeNodes = (g.nodes || []).map((n: any) => {
        const data = { ...(n.data || {}) };
        if (n.type === "depth") data.run_id = rid;
        if (n.type === "video_in") {
          data.run_id = rid || (liveIn?.data as any)?.run_id || "";
          if (liveIn?.data) {
            const ld = liveIn.data as any;
            if (ld.source_url) data.source_url = ld.source_url;
            if (ld.preview_gif_url) data.preview_gif_url = ld.preview_gif_url;
            if (ld.filename) data.filename = ld.filename;
            if (ld.frame_count) data.frame_count = ld.frame_count;
            if (ld.status) data.status = ld.status;
          }
        }
        if (n.type === "video_out" && liveOut?.data) {
          data.preview_width =
            (liveOut.data as any).preview_width ?? data.preview_width ?? 420;
        }
        return { ...n, data };
      });

      if (liveIn && liveIn.id !== "video_in") {
        const vi = recipeNodes.find((n: any) => n.type === "video_in");
        if (vi) {
          const oldId = vi.id;
          vi.id = liveIn.id;
          vi.position = toGraphPos(liveIn.position);
          nextEdges = nextEdges.map((e) => ({
            ...e,
            source: e.source === oldId ? liveIn.id : e.source,
            target: e.target === oldId ? liveIn.id : e.target,
          }));
        }
      }

      const graph: Graph = {
        version: 2,
        nodes: recipeNodes,
        edges: nextEdges,
      };
      const flow = toFlow(graph, meta);
      setNodes(flow.nodes);
      setEdges(flow.edges);
      setAppMode("kitchen");
      setStatus(`Loaded video setup · ${recipe.label}`);
      if (rid && inferStatus === "done") {
        scheduleEvaluate(flow.nodes, flow.edges, rid);
      }
    },
    [inferStatus, meta, runId, scheduleEvaluate, setEdges, setNodes]
  );

  const loadArenaKitchens = useCallback(async () => {
    const url = kitchenUrl.trim();
    if (!url) {
      setStatus("Paste an Are.na channel or block with Depth Ledger recipes");
      return;
    }
    setBusy(true);
    setStatus("Scanning Are.na for kitchens…");
    try {
      const res = await fetchArenaKitchens(url);
      const list = res.kitchens || [];
      setArenaKitchens(list);
      if (!list.length) {
        setKitchenPick("");
        setStatus("No Depth Ledger recipes found on that link");
        return;
      }
      const first = String(list[0].block_id ?? 0);
      setKitchenPick(first);
      applyArenaKitchen(list[0]);
      setStatus(
        list.length === 1
          ? `Loaded kitchen from Are.na · ${list[0].summary || list[0].title}`
          : `Found ${list.length} kitchens — applied newest; pick another in the dropdown`
      );
    } catch (e: any) {
      setStatus(`Are.na kitchen error: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }, [applyArenaKitchen, kitchenUrl]);

  const patchEdgeData = useCallback(
    (edgeId: string, patch: Partial<BlendEdgeData>) => {
      setEdges((eds) => {
        const next = eds.map((e) =>
          e.id === edgeId
            ? { ...e, data: { ...(e.data as object), ...patch } }
            : e
        );
        scheduleEvaluate(nodesRef.current, next, runId);
        return next;
      });
    },
    [runId, scheduleEvaluate, setEdges]
  );

  const deleteEdge = useCallback(
    (edgeId: string) => {
      const next = edgesRef.current.filter((e) => e.id !== edgeId);
      setEdges(next);
      if (selectedEdgeId === edgeId) setSelectedEdgeId(null);
      if (edgeModalId === edgeId) setEdgeModalId(null);
      scheduleEvaluate(nodesRef.current, next, runId);
    },
    [edgeModalId, runId, scheduleEvaluate, selectedEdgeId, setEdges]
  );

  const deleteNode = useCallback(
    (nodeId: string) => {
      const target = nodesRef.current.find((n) => n.id === nodeId);
      if (!target || PROTECTED_TYPES.has(target.type || "")) return;
      const nextNodes = nodesRef.current.filter((n) => n.id !== nodeId);
      const nextEdges = edgesRef.current.filter(
        (e) => e.source !== nodeId && e.target !== nodeId
      );
      setNodes(nextNodes);
      setEdges(nextEdges);
      if (selectedId === nodeId) setSelectedId(null);
      scheduleEvaluate(nextNodes, nextEdges, runId);
    },
    [runId, scheduleEvaluate, selectedId, setEdges, setNodes]
  );

  const openEdgeBlend = useCallback((edgeId: string) => {
    setEdgeModalId(edgeId);
    setSelectedEdgeId(edgeId);
    setEdges((eds) =>
      eds.map((e) => ({ ...e, selected: e.id === edgeId }))
    );
    setNodes((nds) => nds.map((n) => ({ ...n, selected: false })));
    setSelectedId(null);
  }, [setEdges, setNodes]);

  const addNode = useCallback(
    (
      type:
        | "look"
        | "scene"
        | "blend"
        | "relief"
        | "smooth"
        | "freq_fuse"
        | "arena"
        | "arena_out"
        | "video_in"
        | "video_out"
    ) => {
      const id = uid(type);
      const offset = nodesRef.current.length * 24 * LAYOUT_SCALE;
      let data: Record<string, unknown> = {};
      let position = { x: 360 + offset, y: 120 + offset };
      if (type === "look") {
        const preset = "ledger";
        const defaults = lookDefaults(meta, preset);
        data = {
          preset,
          cmap: "inherit",
          collapsed: false,
          lo_pct: defaults?.lo_pct ?? 2,
          hi_pct: defaults?.hi_pct ?? 98,
          gamma: defaults?.gamma ?? 1,
        };
      } else if (type === "scene") {
        data = {
          mode: "smart",
          protect: 0.22,
          feather: 0.35,
          clean: true,
          fringe_kill: true,
          far_remap: true,
          collapsed: false,
        };
      } else if (type === "smooth") {
        const d = meta?.smooth?.defaults;
        data = {
          mode: d?.mode ?? "ema",
          window_mode: d?.window_mode ?? "locked",
          window_lock: d?.window_lock ?? "first",
          spline_interval: d?.spline_interval ?? 30,
          lo_pct: d?.lo_pct ?? 2,
          hi_pct: d?.hi_pct ?? 98,
          temporal: d?.temporal ?? "ema",
          ema_alpha: d?.ema_alpha ?? 0.45,
          flow_prior: d?.flow_prior ?? 0.75,
          occ_thresh: d?.occ_thresh ?? 1.5,
          collapsed: false,
        };
        const depth = nodesRef.current.find((n) => n.type === "depth");
        position = depth
          ? { x: depth.position.x + 280 * LAYOUT_SCALE, y: depth.position.y }
          : { x: 320, y: 120 };
      } else if (type === "freq_fuse") {
        const d = meta?.freq_fuse?.defaults;
        data = {
          beta: d?.beta ?? 1,
          blur_sigma: d?.blur_sigma ?? 4,
          collapsed: false,
        };
        const depth = nodesRef.current.find((n) => n.type === "depth");
        position = depth
          ? { x: depth.position.x + 280 * LAYOUT_SCALE, y: depth.position.y + 40 * LAYOUT_SCALE }
          : { x: 320, y: 160 };
      } else if (type === "relief") {
        data = {
          gc_alpha: meta?.relief?.default_gc_alpha ?? 10,
          invert: meta?.relief?.default_invert ?? false,
          max_side: meta?.relief?.default_max_side ?? 512,
          collapsed: false,
        };
      } else if (type === "arena") {
        data = { url: "", status: "idle" };
        const depth = nodesRef.current.find((n) => n.type === "depth");
        position = depth
          ? { x: depth.position.x - 280 * LAYOUT_SCALE, y: depth.position.y }
          : { x: 40, y: 40 };
      } else if (type === "video_in") {
        data = { status: "idle" };
        const depth = nodesRef.current.find((n) => n.type === "depth");
        position = depth
          ? { x: depth.position.x - 300 * LAYOUT_SCALE, y: depth.position.y + 40 * LAYOUT_SCALE }
          : { x: 40, y: 80 };
      } else if (type === "video_out") {
        if (nodesRef.current.some((n) => n.type === "video_out")) {
          setStatus("Video Out already on the canvas");
          return;
        }
        data = { preview_width: 420 };
        const maxX = Math.max(
          0,
          ...nodesRef.current.map((n) => n.position.x)
        );
        position = { x: maxX + 280 * LAYOUT_SCALE, y: 40 };
      } else if (type === "arena_out") {
        data = {
          channel_id: "",
          title: "",
          status: "idle",
          batch_count: 5,
          wackiness: 0.45,
          max_fidelity: false,
        };
        const sink =
          nodesRef.current.find((n) => n.type === "video_out") ||
          nodesRef.current.find((n) => n.type === "output");
        position = sink
          ? { x: sink.position.x, y: sink.position.y + 280 * LAYOUT_SCALE }
          : { x: 1120, y: 320 };
      } else {
        data = { mode: "normal", opacity: 1 };
      }
      const node: Node = {
        id,
        type,
        position,
        data,
        selected: true,
        deletable: true,
      };
      const nextNodes = [
        ...nodesRef.current.map((n) => ({ ...n, selected: false })),
        node,
      ];
      let nextEdges = edgesRef.current;
      if (type === "arena" || type === "video_in") {
        const depth = nextNodes.find((n) => n.type === "depth");
        if (depth) {
          const filtered = nextEdges.filter(
            (e) =>
              !(
                e.target === depth.id &&
                (e.targetHandle ?? null) === "in"
              )
          );
          const byId = new Map(nextNodes.map((n) => [n.id, n]));
          const draft: Edge = {
            id: uid("e"),
            source: id,
            target: depth.id,
            sourceHandle: "out",
            targetHandle: "in",
            data: {},
          };
          nextEdges = addEdge(annotateEdge(draft, byId), filtered);
          setEdges(nextEdges);
        }
      }
      if (type === "arena_out") {
        const scene =
          nextNodes.find((n) => n.type === "relief") ||
          nextNodes.find((n) => n.type === "scene") ||
          nextNodes.find((n) => n.type === "look");
        if (scene) {
          const filtered = nextEdges.filter((e) => e.target !== id);
          const byId = new Map(nextNodes.map((n) => [n.id, n]));
          const draft: Edge = {
            id: uid("e"),
            source: scene.id,
            target: id,
            sourceHandle: "out",
            targetHandle: "in",
            data: {},
          };
          nextEdges = addEdge(annotateEdge(draft, byId), filtered);
          setEdges(nextEdges);
        }
      }
      if (type === "smooth" || type === "freq_fuse") {
        const depth = nextNodes.find((n) => n.type === "depth");
        const look = nextNodes.find((n) => n.type === "look");
        if (depth) {
          const byId = new Map(nextNodes.map((n) => [n.id, n]));
          const fromDepth = nextEdges.filter((e) => e.source === depth.id);
          const targetId = fromDepth[0]?.target || look?.id || null;
          let filtered = nextEdges.filter((e) => e.source !== depth.id);
          if (targetId) {
            filtered = filtered.filter(
              (e) => e.target !== targetId || e.source === id
            );
          }
          const toProc: Edge = {
            id: uid("e"),
            source: depth.id,
            target: id,
            sourceHandle: "out",
            targetHandle: "in",
            data: {},
          };
          nextEdges = addEdge(annotateEdge(toProc, byId), filtered);
          if (targetId) {
            const fromProc: Edge = {
              id: uid("e"),
              source: id,
              target: targetId,
              sourceHandle: "out",
              targetHandle: "in",
              data: {},
            };
            nextEdges = addEdge(annotateEdge(fromProc, byId), nextEdges);
          }
          setEdges(nextEdges);
        }
      }
      if (type === "video_out") {
        const plate =
          nextNodes.find((n) => n.type === "relief") ||
          nextNodes.find((n) => n.type === "scene") ||
          nextNodes.find((n) => n.type === "look");
        if (plate) {
          const filtered = nextEdges.filter(
            (e) =>
              !(
                e.target === id ||
                (nextNodes.find((n) => n.id === e.target)?.type === "output" &&
                  e.source === plate.id)
              )
          );
          const byId = new Map(nextNodes.map((n) => [n.id, n]));
          const draft: Edge = {
            id: uid("e"),
            source: plate.id,
            target: id,
            sourceHandle: "out",
            targetHandle: "in",
            data: {},
          };
          nextEdges = addEdge(annotateEdge(draft, byId), filtered);
          setEdges(nextEdges);
        }
      }
      setNodes(nextNodes);
      setSelectedId(id);
      setStatus(
        type === "arena"
          ? "Added Are.na — paste a link, then → Depth"
          : type === "video_in"
            ? "Added Video In — load a clip, arrange, then Infer"
            : type === "video_out"
              ? "Added Video Out — scrub / Export MP4 live here"
            : type === "arena_out"
            ? "Added Are.na Out — pick a channel and Save"
            : type === "relief"
              ? "Added Relief — wire a look/scene plate into it"
              : type === "smooth"
                ? "Added Smooth — sits after Depth to stabilize video"
                : type === "freq_fuse"
                  ? "Added Freq Fuse — stub video low-freq + Depth Pro edges"
                : `Added ${type} — wire an input, then click the connection to blend`
      );
      scheduleEvaluate(nextNodes, nextEdges, runId);
    },
    [meta, runId, scheduleEvaluate, setEdges, setNodes]
  );

  const applyInfer = (
    res: InferResponse,
    m: Meta | null,
    elapsedMs?: number
  ) => {
    setRunId(res.run_id);
    setOutputUrl(res.output_url || res.blend_url || null);
    setRgbUrl(res.rgb_url || null);
    setPreviewGifUrl(res.preview_gif_url || null);
    setFrameUrls(res.frame_urls || []);
    setFrameCount(Math.max(1, res.frame_count || 1));
    setReadyFrames(
      Math.max(
        0,
        res.ready_count ??
          (res.infer_status === "pending" ? 0 : res.frame_count ?? 1)
      )
    );
    setInferStatus(res.infer_status || "done");
    setFrameIndex(res.frame_index || 0);
    setSampleFps(res.sample_fps || 10);
    setExportUrl(null);
    setPlaying(false);
    applyModalBilling(res);
    const flow = toFlow(res.graph, m);
    // Stamp infer backend/device onto depth node for display
    flow.nodes = flow.nodes.map((n) =>
      n.type === "depth"
        ? {
            ...n,
            data: {
              ...n.data,
              run_id: res.run_id,
              infer_backend: res.infer_backend,
              infer_device: res.infer_device,
              modal_cost: res.modal_billing?.cost_usd,
            },
          }
        : n
    );
    setNodes(flow.nodes);
    setEdges(flow.edges);
    setEvalToken(Date.now());
    const costHint =
      res.modal_billing?.cost_usd != null
        ? ` · ${formatUsd(res.modal_billing.cost_usd)}`
        : "";
    const elapsedHint =
      elapsedMs != null ? ` · ${formatElapsed(elapsedMs)}` : "";
    const coldHint = res.modal_billing?.cold_start ? " · cold start" : "";
    setStatus(
      (res.frame_count || 1) > 1
        ? `Video run ${res.run_id} · ${res.frame_count} frames${elapsedHint}${coldHint}${costHint}`
        : `Run ${res.run_id}${
            res.infer_backend === "modal" ? " · modal" : ""
          }${elapsedHint}${coldHint}${costHint}`
    );
    listRuns().then(setRuns).catch(() => undefined);
    if (res.infer_status !== "pending") {
      scheduleEvaluate(flow.nodes, flow.edges, res.run_id);
    }
  };

  /** Keep kitchen + Are.na sources when feeding a new run from Are.na. */
  const mergeArenaInfer = useCallback(
    (
      res: InferResponse,
      m: Meta | null,
      arenaNodeId: string,
      arenaPatch: Record<string, unknown>
    ): { nodes: Node[]; edges: Edge[]; runId: string } => {
      setRunId(res.run_id);
      setOutputUrl(res.output_url || res.blend_url || null);
      setRgbUrl(res.rgb_url || null);
      setPreviewGifUrl(null);
      setFrameUrls([]);
      setFrameCount(Math.max(1, res.frame_count || 1));
      setReadyFrames(Math.max(1, res.ready_count ?? res.frame_count ?? 1));
      setInferStatus(res.infer_status || "done");
      setFrameIndex(res.frame_index || 0);
      setSampleFps(res.sample_fps || 10);
      setExportUrl(null);
      setPlaying(false);
      setEvalToken(Date.now());
      applyModalBilling(res);

      const prevNodes = nodesRef.current;
      const prevEdges = edgesRef.current;
      const hasKitchen = prevNodes.some((n) => n.type === "depth");

      let nextNodes: Node[];
      let nextEdges: Edge[];

      if (hasKitchen) {
        nextNodes = prevNodes.map((n) => {
          if (n.type === "depth") {
            return {
              ...n,
              data: {
                ...n.data,
                run_id: res.run_id,
                infer_backend: res.infer_backend,
                infer_device: res.infer_device,
                modal_cost: res.modal_billing?.cost_usd,
              },
            };
          }
          if (n.id === arenaNodeId) {
            return {
              ...n,
              data: mergeArenaNodeData(
                n.data as Record<string, unknown>,
                arenaPatch
              ),
            };
          }
          return n;
        });
        nextEdges = prevEdges;
      } else {
        const flow = toFlow(res.graph, m);
        const arenaPrev = prevNodes.find((n) => n.id === arenaNodeId);
        const depth = flow.nodes.find((n) => n.type === "depth");
        const arenaNode: Node = {
          id: arenaNodeId,
          type: "arena",
          position: arenaPrev?.position || {
            x: (depth?.position.x ?? 40) - 280 * LAYOUT_SCALE,
            y: depth?.position.y ?? 120,
          },
          data: mergeArenaNodeData(
            (arenaPrev?.data as Record<string, unknown>) || {},
            arenaPatch
          ),
          selected: false,
          deletable: true,
        };
        nextNodes = [
          arenaNode,
          ...flow.nodes.map((n) => ({ ...n, selected: false })),
        ];
        const byId = new Map(nextNodes.map((n) => [n.id, n]));
        const feed: Edge = annotateEdge(
          {
            id: uid("e"),
            source: arenaNodeId,
            target: depth?.id || "depth",
            sourceHandle: "out",
            targetHandle: "in",
            data: {},
          },
          byId
        );
        nextEdges = [feed, ...flow.edges];
      }

      const depthId = nextNodes.find((n) => n.type === "depth")?.id;
      if (
        depthId &&
        !nextEdges.some((e) => e.source === arenaNodeId && e.target === depthId)
      ) {
        const byId = new Map(nextNodes.map((n) => [n.id, n]));
        nextEdges = [
          annotateEdge(
            {
              id: uid("e"),
              source: arenaNodeId,
              target: depthId,
              sourceHandle: "out",
              targetHandle: "in",
              data: {},
            },
            byId
          ),
          ...nextEdges.filter(
            (e) =>
              !(e.target === depthId && (e.targetHandle ?? null) === "in")
          ),
        ];
      }

      setNodes(nextNodes);
      setEdges(nextEdges);
      nodesRef.current = nextNodes;
      edgesRef.current = nextEdges;
      listRuns().then(setRuns).catch(() => undefined);
      return { nodes: nextNodes, edges: nextEdges, runId: res.run_id };
    },
    [applyModalBilling, setEdges, setNodes]
  );

  const resolveArenaNode = useCallback(
    async (nodeId: string, mode: "auto" | "shuffle" = "auto") => {
      const node = nodesRef.current.find((n) => n.id === nodeId);
      const d = (node?.data || {}) as Record<string, unknown>;
      const url = String(d.channel_url || d.url || "").trim();
      if (!url) {
        setStatus("Paste an Are.na link first");
        return;
      }
      patchNodeUi(nodeId, { status: "resolving", error: null });
      setStatus(
        mode === "shuffle" ? "Shuffling Are.na channel…" : "Fetching Are.na…"
      );
      try {
        const resolved = await resolveArena(url, mode);
        const patch = arenaDataFromResolve(resolved);
        const prev = (nodesRef.current.find((n) => n.id === nodeId)?.data ||
          {}) as Record<string, unknown>;
        patchNodeUi(nodeId, mergeArenaNodeData(prev, patch));
        const blockTitle = resolved.block?.title || `#${resolved.block?.id}`;
        if (mode === "shuffle") {
          const fromCache = resolved.cached
            ? ` · cached · ${resolved.imageable_count} images`
            : "";
          setStatus(
            `Preview: “${blockTitle}”${fromCache} — click → Depth to process`
          );
        } else if (resolved.kind === "channel") {
          setStatus(
            `Channel “${resolved.channel?.title}” · ${resolved.imageable_count} images · click → Depth`
          );
        } else {
          setStatus(`Block #${resolved.block?.id} ready — click → Depth`);
        }
      } catch (e: any) {
        patchNodeUi(nodeId, { status: "error", error: e.message });
        setStatus(`Are.na error: ${e.message}`);
      }
    },
    [patchNodeUi]
  );

  const loadArenaIntoDepth = useCallback(
    async (nodeId: string, mode: "auto" | "shuffle" = "auto") => {
      const node = nodesRef.current.find((n) => n.id === nodeId);
      const d = (node?.data || {}) as Record<string, unknown>;
      const channelUrl = String(d.channel_url || "").trim();
      const url = String(d.url || "").trim();
      const blockUrl = String(
        (d.block as { arena_url?: string } | undefined)?.arena_url || ""
      ).trim();
      if (!url && !blockUrl && !channelUrl) {
        setStatus("Paste an Are.na link first");
        return;
      }
      // Use the previewed block when available so shuffle/fetch picks are stable.
      const inferUrl = blockUrl || channelUrl || url;
      const inferMode = blockUrl ? "auto" : mode;
      setBusy(true);
      setProgress("upload");
      patchNodeUi(nodeId, { status: "loading", error: null });
      const timer = startStatusTimer(
        setStatus,
        `Are.na → depth (${inferBackend})`
      );
      try {
        setProgress("infer");
        const res = await inferArena(
          inferUrl,
          modelKey,
          maxSize,
          inferMode,
          inferBackend
        );
        const elapsedMs = timer.stop();
        applyModalBilling(res);
        const prev = (nodesRef.current.find((n) => n.id === nodeId)?.data ||
          {}) as Record<string, unknown>;
        const arenaMeta = res.arena
          ? mergeArenaNodeData(prev, arenaDataFromResolve(res.arena))
          : mergeArenaNodeData(prev, {
              url: channelUrl || url,
              status: "ready" as const,
              error: null,
            });
        const { nodes: nextNodes, edges: nextEdges, runId: rid } =
          mergeArenaInfer(res, meta, nodeId, arenaMeta);
        const blockTitle =
          (arenaMeta as any).block?.title || rid;
        const coldHint = res.modal_billing?.cold_start ? " · cold start" : "";
        const costHint =
          res.modal_billing?.cost_usd != null
            ? ` · ${formatUsd(res.modal_billing.cost_usd)}`
            : "";
        setStatus(
          `Are.na → Depth · ${blockTitle} · ${formatElapsed(elapsedMs)}${coldHint}${costHint}`
        );
        scheduleEvaluate(nextNodes, nextEdges, rid);
      } catch (e: any) {
        timer.stop();
        patchNodeUi(nodeId, { status: "error", error: e.message });
        setStatus(`Are.na infer error: ${e.message}`);
      } finally {
        setBusy(false);
        setProgress(null);
      }
    },
    [applyModalBilling, inferBackend, maxSize, mergeArenaInfer, meta, modelKey, patchNodeUi, scheduleEvaluate]
  );

  const refreshArenaChannels = useCallback(async () => {
    try {
      const res = await fetchArenaChannels(24);
      setArenaChannels(res.channels || []);
      setArenaAuth(Boolean(res.authenticated));
    } catch (e: any) {
      setArenaAuth(false);
      setStatus(`Are.na channels: ${e.message}`);
    }
  }, []);

  const publishArenaOut = useCallback(
    async (nodeId: string) => {
      if (!runId) {
        setStatus("Load depth before saving to Are.na");
        return;
      }
      const node = nodesRef.current.find((n) => n.id === nodeId);
      const channelId = String((node?.data as any)?.channel_id || "").trim();
      const title = String((node?.data as any)?.title || "").trim();
      if (!channelId) {
        setStatus("Pick an Are.na channel first");
        return;
      }
      setBusy(true);
      patchNodeUi(nodeId, { status: "publishing", error: null });
      setStatus("Uploading to Are.na…");
      try {
        const res = await publishToArena(runId, channelId, title || undefined);
        patchNodeUi(nodeId, {
          status: "ready",
          error: null,
          last_block_id: res.block?.id,
          last_block_url: res.block?.arena_url,
        });
        const savedTitle = res.title || res.block?.title || res.run_label;
        setStatus(
          res.block?.arena_url
            ? `Saved · ${savedTitle || "block"} · ${res.block.arena_url}`
            : `Saved block #${res.block?.id}${savedTitle ? ` · ${savedTitle}` : ""}`
        );
        void refreshArenaChannels();
      } catch (e: any) {
        patchNodeUi(nodeId, { status: "error", error: e.message });
        setStatus(`Are.na save error: ${e.message}`);
      } finally {
        setBusy(false);
      }
    },
    [patchNodeUi, refreshArenaChannels, runId]
  );

  const runArenaBatch = useCallback(
    async (arenaOutId: string) => {
      const outNode = nodesRef.current.find((n) => n.id === arenaOutId);
      if (!outNode) return;

      const channelId = String((outNode.data as any)?.channel_id || "").trim();
      const baseTitle = String((outNode.data as any)?.title || "").trim();
      const count = Math.max(
        1,
        Math.min(200, Math.round(Number((outNode.data as any)?.batch_count ?? 5)))
      );
      const wackiness = Math.min(
        1,
        Math.max(0, Number((outNode.data as any)?.wackiness ?? 0.45))
      );
      const maxFidelity = Boolean((outNode.data as any)?.max_fidelity);
      const rollMaxSize = maxFidelity
        ? Math.max(...(meta?.max_sizes || [518, 756, 1008]))
        : maxSize;

      if (!channelId) {
        setStatus("Pick an Are.na channel before rolling");
        return;
      }

      const arenaIn = nodesRef.current.find((n) => n.type === "arena");
      const arenaUrl = String((arenaIn?.data as any)?.url || "").trim();
      if (!arenaUrl) {
        setStatus("Paste an Are.na channel/block on the Are.na in node first");
        return;
      }

      batchAbortRef.current = false;
      setBusy(true);
      setModalBatchBilling(null);
      patchNodeUi(arenaOutId, {
        status: "batching",
        error: null,
        batch_i: 0,
        batch_n: count,
        batch_progress: 0,
        batch_phase: "download",
        batch_message: "Starting…",
        batch_downloaded: 0,
        batch_inferred: 0,
        batch_cached: 0,
        batch_cooked: 0,
        batch_published: 0,
        batch_items: Array.from({ length: count }, (_, i) => ({
          index: i + 1,
          status: "queued",
        })),
      });

      try {
        const graph = fromFlow(nodesRef.current, edgesRef.current);
        const { job_id } = await startArenaRoll({
          arena_url: arenaUrl,
          count,
          model_key: modelKey,
          max_size: rollMaxSize,
          infer_backend: inferBackend,
          wackiness,
          max_fidelity: maxFidelity,
          channel_id: channelId,
          title: baseTitle,
          graph,
        });
        batchJobIdRef.current = job_id;
        setStatus(
          `Batch job ${job_id} · ${count} rolls on ${inferBackend}${
            maxFidelity ? " · max fidelity" : ""
          }…`
        );

        const job = await pollJob(
          job_id,
          (j) => {
            const i = Number(j.published ?? j.i ?? 0);
            const n = Number(j.n || count);
            patchNodeUi(arenaOutId, {
              status: "batching",
              batch_i: i,
              batch_n: n,
              batch_progress: j.progress ?? 0,
              batch_phase: j.phase || "pipeline",
              batch_message: j.message || "",
              batch_downloaded: j.downloaded ?? 0,
              batch_inferred: j.inferred ?? 0,
              batch_cached: j.cached ?? 0,
              batch_cooked: j.cooked ?? 0,
              batch_published: j.published ?? i,
              batch_items: j.items || [],
            });
            if (j.modal_billing) {
              setModalBatchBilling(j.modal_billing);
              if (j.modal_billing.cost_usd_so_far != null) {
                setModalLastCost(Number(j.modal_billing.cost_usd_so_far));
              }
            }
            if (j.modal_usage?.cost_usd != null) {
              setModalSessionCost(Number(j.modal_usage.cost_usd));
            }
            const bill =
              j.modal_billing?.cost_usd_so_far != null
                ? ` · ${formatUsd(Number(j.modal_billing.cost_usd_so_far))}`
                : "";
            setStatus(
              `${j.message || `Batch ${i}/${n}`}${bill}`
            );
            setProgress(
              j.phase === "download"
                ? "upload"
                : j.phase === "infer"
                  ? "infer"
                  : "evaluate"
            );
          },
          400,
          () => batchAbortRef.current
        );

        const result = (job.result || {}) as {
          saved?: number;
          last_block_url?: string;
          run_id?: string;
          modal_billing?: ModalBilling;
        };
        const saved = Number(result.saved || job.saved || 0);
        const lastUrl = result.last_block_url;
        if (result.run_id) {
          try {
            const loaded = await loadRun(result.run_id);
            applyInfer(loaded as InferResponse, meta);
          } catch {
            /* keep last kitchen state */
          }
        }
        if (result.modal_billing) setModalBatchBilling(result.modal_billing);
        if (job.modal_usage?.cost_usd != null) {
          setModalSessionCost(Number(job.modal_usage.cost_usd));
        }

        patchNodeUi(arenaOutId, {
          status: job.status === "error" ? "error" : "ready",
          error: job.status === "error" ? job.error || job.message : null,
          last_block_url: lastUrl,
          batch_i: saved,
          batch_n: count,
          batch_progress: 1,
          batch_phase: "done",
          batch_message: job.message || "",
          batch_downloaded: job.downloaded ?? count,
          batch_inferred: job.inferred ?? count,
          batch_cached: job.cached ?? 0,
          batch_cooked: job.cooked ?? saved,
          batch_published: job.published ?? saved,
          batch_items: job.items || [],
        });
        setStatus(
          job.status === "cancelled"
            ? `Batch stopped · saved ${saved}/${count}`
            : job.status === "error"
              ? `Batch error: ${job.error || job.message}`
              : `Batch done · saved ${saved} to Are.na`
        );
        void refreshArenaChannels();
      } catch (e: any) {
        patchNodeUi(arenaOutId, { status: "error", error: e.message });
        setStatus(`Batch error: ${e.message}`);
      } finally {
        setBusy(false);
        setProgress(null);
        batchAbortRef.current = false;
        batchJobIdRef.current = null;
        setModalBatchBilling(null);
      }
    },
    [
      applyInfer,
      inferBackend,
      maxSize,
      meta,
      modelKey,
      patchNodeUi,
      refreshArenaChannels,
    ]
  );

  const stopArenaBatch = useCallback(() => {
    batchAbortRef.current = true;
    const jid = batchJobIdRef.current;
    if (jid) void cancelJob(jid).catch(() => undefined);
    setStatus("Stopping batch…");
  }, []);

  const onGenerateRgb = async (file: File) => {
    setBusy(true);
    setProgress("upload");
    const modelLabel = modelKey.startsWith("promptda")
      ? "PromptDA"
      : "Depth Pro";
    const timer = startStatusTimer(
      setStatus,
      `Running ${modelLabel} on ${inferBackend}`
    );
    try {
      setProgress("infer");
      const res = await inferRgb(file, modelKey, maxSize, inferBackend);
      applyInfer(res, meta, timer.stop());
    } catch (e: any) {
      timer.stop();
      setStatus(`Infer error: ${e.message}`);
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  const onGenerateZip = async (file: File) => {
    setBusy(true);
    setProgress("upload");
    const timer = startStatusTimer(
      setStatus,
      `Running model on capture (${inferBackend})`
    );
    try {
      setProgress("infer");
      const res = await inferZip(file, modelKey, maxSize, inferBackend);
      applyInfer(res, meta, timer.stop());
    } catch (e: any) {
      timer.stop();
      setStatus(`Infer error: ${e.message}`);
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  const prepareVideoNode = useCallback(
    async (nodeId: string, file: File) => {
      setBusy(true);
      setPlaying(false);
      setProgress("upload");
      setFrameUrls([]);
      setExportUrl(null);
      patchNodeUi(nodeId, {
        status: "preparing",
        error: null,
        filename: file.name,
      });
      setStatus(`Preparing ${file.name}…`);
      try {
        const key = modelKey.startsWith("promptda") ? "depth-pro" : modelKey;
        if (key !== modelKey) setModelKey(key);
        const res = await prepareVideo(file, key, maxSize, inferBackend);
        setRunId(res.run_id);
        setRgbUrl(res.rgb_url || null);
        setPreviewGifUrl(res.preview_gif_url || null);
        setOutputUrl(res.preview_gif_url || null);
        setFrameCount(Math.max(1, res.frame_count || 1));
        setReadyFrames(0);
        setReadyIndices([]);
        setInferStatus("pending");
        setInferMode(null);
        if (res.frame_width) setFrameWidth(Number(res.frame_width));
        if (res.frame_height) setFrameHeight(Number(res.frame_height));
        setFrameIndex(0);
        setSampleFps(res.sample_fps || 10);
        setFrameUrls([]);
        setEvalToken(Date.now());

        const hasKitchen = nodesRef.current.some((n) => n.type === "depth");
        if (!hasKitchen) {
          const flow = toFlow(res.graph, meta);
          const patched = flow.nodes.map((n) => {
            if (n.type !== "video_in") return n;
            return {
              ...n,
              id: n.id === "video_in" ? nodeId : n.id,
              data: {
                ...(n.data as Record<string, unknown>),
                status: "pending",
                run_id: res.run_id,
                source_url: res.source_url,
                preview_gif_url: res.preview_gif_url,
                frame_count: res.frame_count,
                frame_width: res.frame_width,
                frame_height: res.frame_height,
                max_size: res.max_size ?? maxSize,
                filename: file.name,
                error: null,
              },
              deletable: true,
            };
          });
          // If prepare was triggered from a canvas video_in with a custom id,
          // remap edges that pointed at "video_in".
          const edges = flow.edges.map((e) =>
            e.source === "video_in" && nodeId !== "video_in"
              ? { ...e, source: nodeId }
              : e
          );
          const byId = new Map(patched.map((n) => [n.id, n]));
          setNodes(patched);
          setEdges(edges.map((e) => annotateEdge(e, byId)));
        } else {
          setNodes((nds) =>
            nds.map((n) => {
              if (n.type === "depth") {
                return {
                  ...n,
                  data: {
                    ...n.data,
                    run_id: res.run_id,
                    infer_backend: res.infer_backend,
                  },
                };
              }
              if (n.id === nodeId) {
                return {
                  ...n,
                  data: {
                    ...n.data,
                    status: "pending",
                    run_id: res.run_id,
                    source_url: res.source_url,
                    preview_gif_url: res.preview_gif_url,
                    frame_count: res.frame_count,
                    frame_width: res.frame_width,
                    frame_height: res.frame_height,
                    max_size: res.max_size ?? maxSize,
                    filename: file.name,
                    error: null,
                  },
                };
              }
              return n;
            })
          );
        }

        // Ensure Video In → Depth edge exists
        setEdges((eds) => {
          const depth = nodesRef.current.find((n) => n.type === "depth");
          if (!depth) return eds;
          if (eds.some((e) => e.source === nodeId && e.target === depth.id)) {
            return eds;
          }
          const filtered = eds.filter(
            (e) =>
              !(e.target === depth.id && (e.targetHandle ?? null) === "in")
          );
          const byId = new Map(nodesRef.current.map((n) => [n.id, n]));
          const draft: Edge = {
            id: uid("e"),
            source: nodeId,
            target: depth.id,
            sourceHandle: "out",
            targetHandle: "in",
            data: {},
          };
          return addEdge(annotateEdge(draft, byId), filtered);
        });
        setStatus(
          `Prepared ${res.frame_count} frames · arrange kitchen, then Infer`
        );
        listRuns().then(setRuns).catch(() => undefined);
      } catch (e: any) {
        patchNodeUi(nodeId, { status: "error", error: e.message });
        setStatus(`Prepare error: ${e.message}`);
      } finally {
        setBusy(false);
        setProgress(null);
      }
    },
    [inferBackend, maxSize, meta, modelKey, patchNodeUi, setEdges, setNodes]
  );

  const inferVideoRun = useCallback(
    async () => {
      const videoIn = nodesRef.current.find((n) => n.type === "video_in");
      const d = (videoIn?.data || {}) as Record<string, unknown>;
      const rid = String(d.run_id || runId || "");
      if (!rid) {
        setStatus("Load a video on Video In first");
        return;
      }
      setBusy(true);
      setPlaying(false);
      setProgress("infer");
      setVideoProgress(0);
      setFrameUrls([]);
      setReadyIndices([]);
      setFrameStatusExpanded(true);
      if (videoIn) {
        patchNodeUi(videoIn.id, { status: "inferring", error: null });
      }
      setInferStatus("running");
      setInferMode(
        inferBackend === "modal" ? "modal_parallel" : "local_sequential"
      );
      setStatus(
        inferBackend === "modal"
          ? "Modal · parallel Depth Pro across frames…"
          : "Local · sequential Depth Pro frame-by-frame…"
      );
      let openedPreview = false;
      try {
        const key = modelKey.startsWith("promptda") ? "depth-pro" : modelKey;
        if (key !== modelKey) setModelKey(key);
        const { job_id } = await startPreparedInfer(rid, {
          model_key: key,
          max_size: maxSize,
          infer_backend: inferBackend,
        });
        const res = await pollVideoJob(
          job_id,
          (job) => {
            setVideoProgress(job.progress ?? 0);
            const ready = job.ready_frames ?? job.result?.ready_count;
            const total = job.total_frames ?? job.result?.frame_count;
            if (ready != null) setReadyFrames(Math.max(0, Number(ready)));
            if (total != null) setFrameCount(Math.max(1, Number(total)));
            if (job.ready_indices) {
              setReadyIndices(job.ready_indices.map(Number));
            } else if (ready != null) {
              setReadyIndices(
                Array.from({ length: Number(ready) }, (_, i) => i)
              );
            }
            if (job.infer_mode) setInferMode(String(job.infer_mode));
            if (job.result?.infer_status) {
              setInferStatus(String(job.result.infer_status));
            } else if (job.status === "running") {
              setInferStatus("running");
            }
            if (job.result?.preview_gif_url) {
              setPreviewGifUrl(String(job.result.preview_gif_url));
            }
            if (job.result?.frame_width) {
              setFrameWidth(Number(job.result.frame_width));
            }
            if (job.result?.frame_height) {
              setFrameHeight(Number(job.result.frame_height));
            }
            setStatus(
              job.message ||
                (ready != null && total != null
                  ? `Depth ${ready}/${total}…`
                  : `Video ${Math.round((job.progress || 0) * 100)}%`)
            );
            if (job.modal_usage?.cost_usd != null) {
              setModalSessionCost(Number(job.modal_usage.cost_usd));
            }
          },
          (preview) => {
            openedPreview = true;
            setRunId(preview.run_id);
            setRgbUrl(preview.rgb_url || null);
            setOutputUrl(preview.output_url || preview.blend_url || null);
            setPreviewGifUrl(preview.preview_gif_url || previewGifUrl);
            setReadyFrames(Math.max(1, preview.ready_count ?? 1));
            setFrameCount(Math.max(1, preview.frame_count || 1));
            setInferStatus(preview.infer_status || "running");
            if (preview.ready_indices) {
              setReadyIndices(preview.ready_indices.map(Number));
            }
            if (preview.infer_mode) setInferMode(preview.infer_mode);
            if (preview.frame_width) setFrameWidth(preview.frame_width);
            if (preview.frame_height) setFrameHeight(preview.frame_height);
            setNodes((nds) =>
              nds.map((n) =>
                n.type === "depth"
                  ? {
                      ...n,
                      data: {
                        ...n.data,
                        run_id: preview.run_id,
                        infer_backend: preview.infer_backend,
                        infer_device: preview.infer_device,
                        modal_cost: preview.modal_billing?.cost_usd,
                      },
                    }
                  : n
              )
            );
            setBusy(false);
            setProgress(null);
            scheduleEvaluate(
              nodesRef.current,
              edgesRef.current,
              preview.run_id
            );
            setStatus(
              `Preview ready · inferring ${preview.ready_count ?? 1}/${preview.frame_count ?? "?"}…`
            );
          }
        );
        setRunId(res.run_id);
        setRgbUrl(res.rgb_url || null);
        setPreviewGifUrl(res.preview_gif_url || previewGifUrl);
        setFrameCount(Math.max(1, res.frame_count || 1));
        setReadyFrames(Math.max(1, res.ready_count ?? res.frame_count ?? 1));
        setReadyIndices(
          (res.ready_indices || Array.from(
            { length: res.frame_count || 1 },
            (_, i) => i
          )).map(Number)
        );
        if (res.infer_mode) setInferMode(res.infer_mode);
        if (res.frame_width) setFrameWidth(res.frame_width);
        if (res.frame_height) setFrameHeight(res.frame_height);
        setInferStatus("done");
        inferStatusRef.current = "done";
        readyFramesRef.current = Math.max(
          1,
          res.ready_count ?? res.frame_count ?? 1
        );
        frameCountRef.current = Math.max(1, res.frame_count || 1);
        applyModalBilling(res);
        setNodes((nds) =>
          nds.map((n) => {
            if (n.type === "depth") {
              return {
                ...n,
                data: {
                  ...n.data,
                  run_id: res.run_id,
                  infer_backend: res.infer_backend,
                  infer_device: res.infer_device,
                  modal_cost: res.modal_billing?.cost_usd,
                },
              };
            }
            if (n.type === "video_in") {
              return {
                ...n,
                data: {
                  ...n.data,
                  status: "done",
                  run_id: res.run_id,
                  source_url: res.source_url || d.source_url,
                  frame_count: res.frame_count,
                  frame_width: res.frame_width,
                  frame_height: res.frame_height,
                  error: null,
                },
              };
            }
            return n;
          })
        );
        setVideoProgress(null);
        scheduleEvaluate(nodesRef.current, edgesRef.current, res.run_id, {
          bake: true,
        });
        setStatus(`Video run ${res.run_id} · ${res.frame_count} frames ready`);
        listRuns().then(setRuns).catch(() => undefined);
        void openedPreview;
      } catch (e: any) {
        if (videoIn) {
          patchNodeUi(videoIn.id, { status: "error", error: e.message });
        }
        setStatus(`Video error: ${e.message}`);
        setVideoProgress(null);
        setInferStatus("pending");
      } finally {
        setBusy(false);
        setProgress(null);
      }
    },
    [
      applyModalBilling,
      inferBackend,
      maxSize,
      modelKey,
      patchNodeUi,
      previewGifUrl,
      runId,
      scheduleEvaluate,
      setNodes,
    ]
  );

  const scrubMax = Math.max(
    1,
    Math.min(Math.max(readyFrames, frameUrls.length || 0), frameCount)
  );

  /** Local Output scrub — swaps pre-baked frames only (no kitchen reload). */
  const scrubFrame = useCallback(
    (index: number) => {
      if (frameCount <= 1) return;
      const maxIdx = Math.max(0, scrubMax - 1);
      const i = Math.max(0, Math.min(maxIdx, index));
      setFrameIndex(i);
      if (frameUrls[i]) {
        setOutputUrl(frameUrls[i]);
      }
    },
    [frameCount, frameUrls, scrubMax]
  );

  /** Operator / legacy: reload kitchen onto a frame (still used by OperatorView). */
  const goToFrame = useCallback(
    async (index: number) => {
      if (!runId || frameCount <= 1) return;
      if (seekingRef.current) return;
      const maxIdx = Math.max(0, scrubMax - 1);
      const i = Math.max(0, Math.min(maxIdx, index));
      seekingRef.current = true;
      try {
        const res = await setRunFrame(
          runId,
          i,
          fromFlow(nodesRef.current, edgesRef.current)
        );
        setFrameIndex(res.frame_index);
        setOutputUrl(res.output_url);
        setRgbUrl(res.rgb_url);
        setNodePreviews(res.node_previews || {});
        setEvalToken(Date.now());
      } catch (e: any) {
        setStatus(`Frame error: ${e.message}`);
        setPlaying(false);
      } finally {
        seekingRef.current = false;
      }
    },
    [frameCount, runId, scrubMax]
  );

  useEffect(() => {
    if (!playing || scrubMax <= 1) {
      if (playTimer.current) {
        window.clearInterval(playTimer.current);
        playTimer.current = null;
      }
      return;
    }
    const ms = Math.max(40, Math.round(1000 / Math.max(sampleFps, 1)));
    playTimer.current = window.setInterval(() => {
      const next =
        frameIndexRef.current + 1 >= scrubMax
          ? 0
          : frameIndexRef.current + 1;
      scrubFrame(next);
    }, ms);
    return () => {
      if (playTimer.current) {
        window.clearInterval(playTimer.current);
        playTimer.current = null;
      }
    };
  }, [playing, scrubMax, sampleFps, scrubFrame]);

  const onExportVideo = useCallback(async () => {
    if (!runId || frameCount <= 1) return;
    setBusy(true);
    setProgress("evaluate");
    setStatus("Exporting MP4…");
    try {
      const res = await exportRunVideo(
        runId,
        fromFlow(nodesRef.current, edgesRef.current)
      );
      setExportUrl(res.export_url);
      setStatus(`Exported ${res.frame_count} frames @ ${res.fps.toFixed(1)} fps`);
    } catch (e: any) {
      setStatus(`Export error: ${e.message}`);
    } finally {
      setBusy(false);
      setProgress(null);
    }
  }, [frameCount, runId]);

  const isValidConnection = useCallback((connection: Connection | Edge) => {
    const { source, target, targetHandle } = connection;
    if (!source || !target || source === target) return false;
    const src = nodesRef.current.find((n) => n.id === source);
    const tgt = nodesRef.current.find((n) => n.id === target);
    if (!src || !tgt) return false;

    if (tgt.type === "output" || tgt.type === "video_out") {
      return (
        src.type !== "depth" &&
        src.type !== "smooth" &&
        src.type !== "freq_fuse" &&
        src.type !== "output" &&
        src.type !== "video_out" &&
        src.type !== "arena" &&
        src.type !== "arena_out" &&
        src.type !== "video_in"
      );
    }
    if (tgt.type === "blend") {
      return (
        (targetHandle === "base" || targetHandle === "overlay") &&
        src.type !== "depth" &&
        src.type !== "smooth" &&
        src.type !== "freq_fuse" &&
        src.type !== "output" &&
        src.type !== "video_out" &&
        src.type !== "arena" &&
        src.type !== "arena_out" &&
        src.type !== "video_in"
      );
    }
    if (tgt.type === "look" || tgt.type === "scene") {
      return (
        src.type === "depth" ||
        src.type === "smooth" ||
        src.type === "freq_fuse" ||
        src.type === "look" ||
        src.type === "scene" ||
        src.type === "blend" ||
        src.type === "relief"
      );
    }
    if (tgt.type === "smooth" || tgt.type === "freq_fuse") {
      return (
        src.type === "depth" ||
        src.type === "smooth" ||
        src.type === "freq_fuse"
      );
    }
    if (tgt.type === "relief") {
      return (
        src.type === "look" ||
        src.type === "scene" ||
        src.type === "blend" ||
        src.type === "relief"
      );
    }
    if (tgt.type === "depth") {
      return src.type === "arena" || src.type === "video_in";
    }
    if (tgt.type === "arena_out") {
      return (
        src.type === "look" ||
        src.type === "scene" ||
        src.type === "blend" ||
        src.type === "relief" ||
        src.type === "output" ||
        src.type === "video_out"
      );
    }
    return false;
  }, []);

  const onConnect = useCallback(
    (connection: Connection) => {
      if (!isValidConnection(connection)) return;
      setEdges((eds) => {
        const next = replaceTargetEdge(eds, connection, nodesRef.current);
        scheduleEvaluate(nodesRef.current, next, runId);
        return next;
      });
    },
    [isValidConnection, runId, scheduleEvaluate, setEdges]
  );

  const onSelectionChange = useCallback((p: OnSelectionChangeParams) => {
    setSelectedId(p.nodes[0]?.id ?? null);
    setSelectedEdgeId(p.edges[0]?.id ?? null);
  }, []);

  const patchSelected = (patch: Record<string, unknown>) => {
    if (!selectedNode) return;
    patchNodeData(selectedNode.id, patch);
  };

  const onNodesChangeWrapped = useCallback(
    (changes: NodeChange[]) => {
      const blocked = changes.filter(
        (c) =>
          c.type === "remove" &&
          PROTECTED_TYPES.has(
            nodesRef.current.find((n) => n.id === (c as any).id)?.type || ""
          )
      );
      const allowed =
        blocked.length > 0
          ? changes.filter((c) => !blocked.includes(c))
          : changes;
      const removals = allowed.filter((c) => c.type === "remove");
      onNodesChange(allowed);
      if (removals.length > 0) {
        window.setTimeout(() => {
          scheduleEvaluate(nodesRef.current, edgesRef.current, runId);
        }, 0);
      }
    },
    [onNodesChange, runId, scheduleEvaluate]
  );

  const onEdgesChangeWrapped = useCallback(
    (changes: EdgeChange[]) => {
      onEdgesChange(changes);
      const removed = changes.some((c) => c.type === "remove");
      if (removed) {
        window.setTimeout(() => {
          scheduleEvaluate(nodesRef.current, edgesRef.current, runId);
        }, 0);
      }
    },
    [onEdgesChange, runId, scheduleEvaluate]
  );

  const loadHistory = async (id: string) => {
    setBusy(true);
    setProgress("infer");
    setStatus(`Loading ${id}…`);
    try {
      const res = await loadRun(id);
      applyInfer(res as any, meta);
    } catch (e: any) {
      setStatus(`Load error: ${e.message}`);
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  const selectIngredient = (nodeId: string) => {
    setSelectedId(nodeId);
    setSelectedEdgeId(null);
    setNodes((nds) =>
      nds.map((n) => ({ ...n, selected: n.id === nodeId }))
    );
    setEdges((eds) => eds.map((e) => ({ ...e, selected: false })));
  };

  const inspector = () => {
    if (selectedEdgeId && !selectedNode) {
      const edge = edges.find((e) => e.id === selectedEdgeId);
      const d = (edge?.data || {}) as BlendEdgeData;
      return (
        <>
          <p className="muted">
            Connection selected (blue). Backspace deletes it, or open the blend
            modal.
          </p>
          <button type="button" onClick={() => openEdgeBlend(selectedEdgeId)}>
            Edit blend…
          </button>
          <button
            type="button"
            className="danger"
            style={{ marginLeft: "0.35rem" }}
            onClick={() => deleteEdge(selectedEdgeId)}
          >
            Disconnect
          </button>
          {d.blendable ? (
            <p className="muted" style={{ marginTop: "0.5rem" }}>
              {d.blend_mode || "normal"} @ {Number(d.opacity ?? 1).toFixed(2)}
            </p>
          ) : null}
        </>
      );
    }

    if (!selectedNode) {
      return (
        <p className="muted">
          Select a node (blue outline) or a wire. Click a connection pill to set
          blend. Backspace deletes the selection.
        </p>
      );
    }
    const t = selectedNode.type;
    const d = selectedNode.data as any;

    if (t === "look" && meta) {
      return (
        <p className="muted">
          Season this Look on the node (expand ▾). Blend mode lives on the
          incoming wire — click the connection pill.
        </p>
      );
    }

    if (t === "scene" && meta) {
      return (
        <>
          <div className="field">
            <label>Matte mode</label>
            <select
              value={d.mode || "smart"}
              onChange={(e) => patchSelected({ mode: e.target.value })}
            >
              {meta.scenery_modes.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Protect ({Number(d.protect ?? 0.22).toFixed(2)})</label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={Number(d.protect ?? 0.22)}
              onChange={(e) =>
                patchSelected({ protect: parseFloat(e.target.value) })
              }
            />
          </div>
          <div className="field">
            <label>Feather ({Number(d.feather ?? 0.35).toFixed(2)})</label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={Number(d.feather ?? 0.35)}
              onChange={(e) =>
                patchSelected({ feather: parseFloat(e.target.value) })
              }
            />
          </div>
          <div className="field">
            <label>
              <input
                type="checkbox"
                checked={Boolean(d.clean ?? true)}
                onChange={(e) => patchSelected({ clean: e.target.checked })}
              />{" "}
              Clean speckles
            </label>
          </div>
          <div className="field">
            <label>
              <input
                type="checkbox"
                checked={Boolean(d.fringe_kill ?? true)}
                onChange={(e) =>
                  patchSelected({ fringe_kill: e.target.checked })
                }
              />{" "}
              Fringe kill (white halo)
            </label>
          </div>
          <div className="field">
            <label>
              <input
                type="checkbox"
                checked={Boolean(d.far_remap ?? true)}
                onChange={(e) =>
                  patchSelected({ far_remap: e.target.checked })
                }
              />{" "}
              Far remap (skyline lift)
            </label>
          </div>
        </>
      );
    }

    if (t === "relief") {
      const sides = meta?.relief?.max_sides || [384, 512, 768, 1024];
      const gc = Number(d.gc_alpha ?? meta?.relief?.default_gc_alpha ?? 10);
      const maxSide = Number(
        d.max_side ?? meta?.relief?.default_max_side ?? 512
      );
      return (
        <>
          <p className="muted">
            Digital bas-relief (Weyrich) — juices the incoming cooked plate.
          </p>
          <div className="field">
            <label>Compression α ({gc.toFixed(1)})</label>
            <input
              type="range"
              min={1}
              max={40}
              step={0.5}
              value={gc}
              onChange={(e) =>
                patchSelected({ gc_alpha: parseFloat(e.target.value) })
              }
            />
          </div>
          <div className="field">
            <label>Solve size</label>
            <select
              value={maxSide}
              onChange={(e) =>
                patchSelected({ max_side: parseInt(e.target.value, 10) })
              }
            >
              {sides.map((s) => (
                <option key={s} value={s}>
                  {s}px
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>
              <input
                type="checkbox"
                checked={Boolean(
                  d.invert ?? meta?.relief?.default_invert ?? false
                )}
                onChange={(e) => patchSelected({ invert: e.target.checked })}
              />{" "}
              Invert (bright = far)
            </label>
          </div>
        </>
      );
    }

    if (t === "blend" && meta) {
      return (
        <p className="muted">
          Optional standalone mixer. Prefer blend on Look→Look / Look→Scene
          wires.
        </p>
      );
    }

    if (t === "depth") {
      return <p className="muted">Depth source for run {d.run_id || runId}</p>;
    }

    if (t === "output") {
      return <p className="muted">Final preview plate.</p>;
    }

    return <p className="muted">No controls for this node.</p>;
  };

  const kitchenValue = useMemo(
    () => ({
      meta,
      evalToken,
      nodePreviews,
      outputUrl,
      rgbUrl,
      progress,
      busy,
      modelKey,
      maxSize,
      setMaxSize,
      inferBackend,
      modalSessionCost,
      inferDevice,
      runId,
      frameIndex,
      frameCount,
      readyFrames,
      readyIndices,
      inferMode,
      frameWidth,
      frameHeight,
      sampleFps,
      inferStatus,
      previewGifUrl,
      frameUrls,
      exportUrl,
      frameStatusExpanded,
      setFrameStatusExpanded,
      playing,
      setPlaying,
      scrubFrame,
      exportVideo: onExportVideo,
      arenaChannels,
      arenaAuth,
      windowTune,
      patchNodeData,
      patchNodeUi,
      deleteNode,
      openEdgeBlend,
      openLightbox,
      startWindowTune,
      stopWindowTune,
      setWindowTuneMode,
      applyWindowPct,
      commitWindowTune,
      resolveArenaNode,
      loadArenaIntoDepth,
      prepareVideoNode,
      inferVideoRun,
      refreshArenaChannels,
      publishArenaOut,
      runArenaBatch,
      stopArenaBatch,
    }),
    [
      meta,
      evalToken,
      nodePreviews,
      outputUrl,
      rgbUrl,
      progress,
      busy,
      modelKey,
      maxSize,
      inferBackend,
      modalSessionCost,
      inferDevice,
      runId,
      frameIndex,
      frameCount,
      readyFrames,
      readyIndices,
      inferMode,
      frameWidth,
      frameHeight,
      sampleFps,
      inferStatus,
      previewGifUrl,
      frameUrls,
      exportUrl,
      frameStatusExpanded,
      playing,
      scrubFrame,
      onExportVideo,
      arenaChannels,
      arenaAuth,
      windowTune,
      patchNodeData,
      patchNodeUi,
      deleteNode,
      openEdgeBlend,
      openLightbox,
      startWindowTune,
      stopWindowTune,
      setWindowTuneMode,
      applyWindowPct,
      commitWindowTune,
      resolveArenaNode,
      loadArenaIntoDepth,
      prepareVideoNode,
      inferVideoRun,
      refreshArenaChannels,
      publishArenaOut,
      runArenaBatch,
      stopArenaBatch,
    ]
  );

  const modalData = (modalEdge?.data || {}) as BlendEdgeData;
  const modalBlendable = Boolean(modalData.blendable);
  const maxSizesList = meta?.max_sizes || [518, 756, 1008, 1536];
  const maxRes = Math.max(...maxSizesList);

  return (
    <KitchenContext.Provider value={kitchenValue}>
      <div className="app">
        <header className="topbar">
          <h1>Depth Ledger</h1>
          <div className="infer-toggle" title="Kitchen, Operator, Video, or a stills batch">
            <span className="infer-toggle-label">Mode</span>
            <div className="infer-seg">
              <button
                type="button"
                className={appMode === "kitchen" ? "active" : ""}
                onClick={() => setAppMode("kitchen")}
              >
                Kitchen
              </button>
              <button
                type="button"
                className={appMode === "operator" ? "active" : ""}
                onClick={() => setAppMode("operator")}
              >
                Operator
              </button>
              <button
                type="button"
                className={appMode === "video" ? "active" : ""}
                onClick={() => {
                  setAppMode("video");
                  if (modelKey.startsWith("promptda")) setModelKey("depth-pro");
                }}
              >
                Video
              </button>
              <button
                type="button"
                className={appMode === "batch" ? "active" : ""}
                onClick={() => {
                  setAppMode("batch");
                  if (modelKey.startsWith("promptda")) setModelKey("depth-pro");
                }}
              >
                Batch
              </button>
            </div>
          </div>
          <label>
            Model
            <select
              value={modelKey}
              onChange={(e) => setModelKey(e.target.value)}
              disabled={!meta}
            >
              {(meta?.models || []).map((m) => (
                <option key={m.key} value={m.key}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Max
            <select
              value={maxSize}
              onChange={(e) => setMaxSize(parseInt(e.target.value, 10))}
            >
              {maxSizesList.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          {appMode === "video" || appMode === "batch" ? (
            <button
              type="button"
              disabled={busy || maxSize === maxRes}
              title={`Use Depth Pro native max (${maxRes})`}
              onClick={() => setMaxSize(maxRes)}
            >
              Max res
            </button>
          ) : null}
          <div
            className="infer-toggle"
            title={
              meta?.modal_available
                ? "Run depth forward-pass locally or on Modal GPU"
                : "Modal unavailable — install modal + run: modal deploy modal_app/worker.py"
            }
          >
            <span className="infer-toggle-label">Infer</span>
            <div className="infer-seg">
              <button
                type="button"
                className={inferBackend === "local" ? "active" : ""}
                disabled={busy}
                onClick={() => setInferBackend("local")}
              >
                Local
              </button>
              <button
                type="button"
                className={inferBackend === "modal" ? "active" : ""}
                disabled={busy || !meta?.modal_available}
                onClick={() => setInferBackend("modal")}
              >
                Modal
              </button>
            </div>
          </div>
          {inferBackend === "modal" ? (
            <span
              className={`modal-ticker${
                progress === "infer" || busy ? " pulse" : ""
              }`}
              title={
                meta?.modal_pricing?.disclaimer ||
                "Estimate from GPU seconds × Modal list rate"
              }
            >
              {modalBatchBilling?.cost_usd_so_far != null
                ? `batch ${formatUsd(
                    Number(modalBatchBilling.cost_usd_so_far)
                  )}${
                    modalBatchBilling.rolls != null
                      ? ` · ${modalBatchBilling.rolls} rolls`
                      : ""
                  }`
                : modalLastCost != null
                  ? formatUsd(modalLastCost)
                  : progress === "infer"
                    ? "~$… computing"
                    : "~$—"}
              <span className="modal-ticker-session">
                {" "}
                · session {formatUsd(modalSessionCost)}
              </span>
            </span>
          ) : null}
          {appMode !== "video" && appMode !== "batch" ? (
            <>
          <button
            className="primary"
            disabled={busy}
            onClick={() => fileRgb.current?.click()}
          >
            Generate RGB
          </button>
          <button disabled={busy} onClick={() => fileZip.current?.click()}>
            Generate Zip
          </button>
          <button
            type="button"
            disabled={busy || !runId}
            onClick={() => void openBlender()}
            title="Export float EXR + near/far, then open DepthLedger_v1.blend with this run loaded"
          >
            Open in Blender
          </button>
            </>
          ) : null}
          <input
            ref={fileRgb}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onGenerateRgb(f);
              e.target.value = "";
            }}
          />
          <input
            ref={fileZip}
            type="file"
            accept=".zip"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onGenerateZip(f);
              e.target.value = "";
            }}
          />
          <label>
            History
            <select
              value=""
              disabled={appMode === "batch"}
              onChange={(e) => {
                const id = e.target.value;
                if (!id) return;
                if (appMode === "video") setFilmHistoryId(id);
                else if (appMode !== "batch") loadHistory(id);
              }}
            >
              <option value="">Recent runs…</option>
              {runs.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.label}
                </option>
              ))}
            </select>
          </label>
          {appMode === "kitchen" ? (
          <>
          <div
            className="arena-kitchen-load"
            title="Paste an Are.na channel or block that has Depth Ledger recipes embedded"
          >
            <input
              type="text"
              placeholder="Are.na channel / block…"
              value={kitchenUrl}
              disabled={busy || appMode !== "kitchen"}
              onChange={(e) => setKitchenUrl(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void loadArenaKitchens();
              }}
            />
            <button
              type="button"
              disabled={busy || appMode !== "kitchen" || !kitchenUrl.trim()}
              onClick={() => void loadArenaKitchens()}
            >
              Load kitchen
            </button>
            {arenaKitchens.length > 1 ? (
              <select
                value={kitchenPick}
                disabled={busy}
                onChange={(e) => {
                  const id = e.target.value;
                  setKitchenPick(id);
                  const entry = arenaKitchens.find(
                    (k) => String(k.block_id) === id
                  );
                  if (entry) applyArenaKitchen(entry);
                }}
              >
                {arenaKitchens.map((k) => (
                  <option key={String(k.block_id)} value={String(k.block_id)}>
                    {(k.run_name || k.title || `#${k.block_id}`).slice(0, 48)}
                    {k.summary ? ` · ${k.summary.slice(0, 40)}` : ""}
                  </option>
                ))}
              </select>
            ) : null}
          </div>
          <button
            disabled={busy || nodes.length === 0}
            onClick={() => saveAsDefault()}
            title={
              graphKind === "video"
                ? "Use this node layout for every new video generate"
                : "Use this node layout for every new generate"
            }
          >
            {graphKind === "video" ? "Save as video default" : "Save as default"}
          </button>
          {activeHasCustomDefault ? (
            <button
              disabled={busy}
              onClick={() => resetDefault()}
              title={
                graphKind === "video"
                  ? "Clear custom video kitchen"
                  : "Clear custom default kitchen"
              }
            >
              {graphKind === "video" ? "Reset video default" : "Reset default"}
            </button>
          ) : null}
          </>
          ) : null}
          <StatusStrip
            mode={appMode}
            runId={runId}
            lastEvaluateAt={lastEvaluateAt}
          />
          <span className="status">
            {videoProgress != null
              ? `${status} (${Math.round(videoProgress * 100)}%)`
              : status}
          </span>
        </header>

        {appMode === "video" ? (
          <VideoView
            meta={meta}
            modelKey={modelKey}
            maxSize={maxSize}
            setMaxSize={setMaxSize}
            inferBackend={inferBackend}
            historyRunId={filmHistoryId}
            onStatus={setStatus}
            onProgress={setVideoProgress}
            onBusy={setBusy}
          />
        ) : appMode === "batch" ? (
          <BatchView
            meta={meta}
            modelKey={modelKey}
            maxSize={maxSize}
            setMaxSize={setMaxSize}
            inferBackend={inferBackend}
            onStatus={setStatus}
            onProgress={setVideoProgress}
            onBusy={setBusy}
          />
        ) : appMode === "operator" ? (
          <OperatorView
            meta={meta}
            runId={runId}
            rgbUrl={rgbUrl}
            busy={busy}
            frameIndex={frameIndex}
            frameCount={frameCount}
            sampleFps={sampleFps}
            evalToken={evalToken}
            onGoToFrame={goToFrame}
            onStatus={setStatus}
          />
        ) : (
          <KitchenView
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChangeWrapped}
            onEdgesChange={onEdgesChangeWrapped}
            onConnect={onConnect}
            isValidConnection={isValidConnection}
            onSelectionChange={onSelectionChange}
            openEdgeBlend={openEdgeBlend}
            ingredientNodes={ingredientNodes}
            selectedId={selectedId}
            selectedNode={selectedNode}
            selectedEdgeId={selectedEdgeId}
            selectIngredient={selectIngredient}
            addNode={addNode}
            deleteNode={deleteNode}
            inspector={inspector}
            modalEdge={modalEdge}
            modalBlendable={modalBlendable}
            modalData={modalData}
            setEdgeModalId={setEdgeModalId}
            patchEdgeData={patchEdgeData}
            deleteEdge={deleteEdge}
            lightbox={lightbox}
            setLightbox={setLightbox}
            protectedTypes={PROTECTED_TYPES}
            meta={meta}
            videoRecipes={videoRecipes}
            applyVideoRecipe={applyVideoRecipe}
          />
        )}
      </div>
    </KitchenContext.Provider>
  );
}
