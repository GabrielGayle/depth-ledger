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
  bust,
  cancelJob,
  clearDefaultGraph,
  evaluateGraph,
  exportRunVideo,
  fetchArenaChannels,
  fetchDefaultGraph,
  fetchMeta,
  fetchModalUsage,
  inferArena,
  inferRgb,
  inferZip,
  listRuns,
  loadRun,
  pollJob,
  pollVideoJob,
  publishToArena,
  resolveArena,
  saveDefaultGraph,
  setRunFrame,
  startArenaRoll,
  startVideoInfer,
  type ArenaChannelOption,
  type Graph,
  type InferResponse,
  type Meta,
  type ModalBilling,
} from "./api";
import { type BlendEdgeData } from "./edges";
import { KitchenContext, clampWindowPct, lookDefaults } from "./kitchenContext";
import type { WindowTuneMode, WindowTuneState } from "./kitchenContext";
import { KitchenView } from "./KitchenView";
import { OperatorView } from "./operator/OperatorView";
import { arenaDataFromResolve, mergeArenaNodeData } from "./arenaUtils";

const PROTECTED_TYPES = new Set(["depth", "output"]);
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
    if (n.type === "output" && data.preview_width == null) {
      data.preview_width = 420;
    }
    return {
      id: n.id,
      type: n.type,
      position: n.position || { x: 0, y: 0 },
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
      position: n.position,
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
  const [status, setStatus] = useState("Booting…");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<ProgressPhase>(null);
  const [evalToken, setEvalToken] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [edgeModalId, setEdgeModalId] = useState<string | null>(null);
  const [runs, setRuns] = useState<{ run_id: string; label: string }[]>([]);
  const [frameCount, setFrameCount] = useState(1);
  const [frameIndex, setFrameIndex] = useState(0);
  const [sampleFps, setSampleFps] = useState(10);
  const [playing, setPlaying] = useState(false);
  const [exportUrl, setExportUrl] = useState<string | null>(null);
  const [videoProgress, setVideoProgress] = useState<number | null>(null);
  const [arenaChannels, setArenaChannels] = useState<ArenaChannelOption[]>([]);
  const [arenaAuth, setArenaAuth] = useState(false);
  const [windowTune, setWindowTune] = useState<WindowTuneState>(null);
  const [appMode, setAppMode] = useState<"kitchen" | "operator">("kitchen");

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const evalTimer = useRef<number | null>(null);
  const evalOverlayTimer = useRef<number | null>(null);
  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  const fileRgb = useRef<HTMLInputElement>(null);
  const fileZip = useRef<HTMLInputElement>(null);
  const fileVideo = useRef<HTMLInputElement>(null);
  const playTimer = useRef<number | null>(null);
  const frameIndexRef = useRef(0);
  const seekingRef = useRef(false);
  const batchAbortRef = useRef(false);
  const batchJobIdRef = useRef<string | null>(null);

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
    let cancelled = false;
    (async () => {
      try {
        const [m, d] = await Promise.all([fetchMeta(), fetchDefaultGraph()]);
        if (cancelled) return;
        setMeta(m);
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
        (n) => n.type === "look" || n.type === "scene" || n.type === "relief"
      ),
    [nodes]
  );

  const scheduleEvaluate = useCallback(
    (nextNodes: Node[], nextEdges: Edge[], rid: string | null) => {
      if (!rid) return;
      if (evalTimer.current) window.clearTimeout(evalTimer.current);
      if (evalOverlayTimer.current) window.clearTimeout(evalOverlayTimer.current);

      evalTimer.current = window.setTimeout(async () => {
        setStatus("Cooking graph…");
        evalOverlayTimer.current = window.setTimeout(() => {
          setProgress("evaluate");
        }, 300);
        try {
          const res = await evaluateGraph(rid, fromFlow(nextNodes, nextEdges));
          setOutputUrl(res.output_url);
          setNodePreviews(res.node_previews || {});
          setEvalToken(Date.now());
          setStatus(`Evaluated in ${res.elapsed_ms} ms`);
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
    setWindowTune({ nodeId, mode: "near", surface: "rgb" });
  }, []);

  const stopWindowTune = useCallback(() => {
    setWindowTune(null);
  }, []);

  const setWindowTuneMode = useCallback((mode: WindowTuneMode) => {
    setWindowTune((t) => (t ? { ...t, mode } : t));
  }, []);

  const setWindowTuneSurface = useCallback((surface: "rgb" | "look") => {
    setWindowTune((t) => (t ? { ...t, surface } : t));
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

  const saveAsDefault = useCallback(async () => {
    if (!nodesRef.current.length) {
      setStatus("Nothing to save");
      return;
    }
    try {
      await saveDefaultGraph(fromFlow(nodesRef.current, edgesRef.current));
      setHasCustomDefault(true);
      setStatus("Saved current graph as default kitchen");
    } catch (e: any) {
      setStatus(`Save default failed: ${e.message}`);
    }
  }, []);

  const resetDefault = useCallback(async () => {
    try {
      await clearDefaultGraph();
      setHasCustomDefault(false);
      const d = await fetchDefaultGraph();
      if (d.graph) {
        const flow = toFlow(d.graph, meta);
        setNodes(flow.nodes);
        setEdges(flow.edges);
      }
      setStatus("Restored built-in default kitchen");
    } catch (e: any) {
      setStatus(`Reset default failed: ${e.message}`);
    }
  }, [meta, setEdges, setNodes]);

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
    (type: "look" | "scene" | "blend" | "relief" | "arena" | "arena_out") => {
      const id = uid(type);
      const offset = nodesRef.current.length * 24;
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
          ? { x: depth.position.x - 280, y: depth.position.y }
          : { x: 40, y: 40 };
      } else if (type === "arena_out") {
        data = {
          channel_id: "",
          title: "",
          status: "idle",
          batch_count: 5,
          wackiness: 0.45,
        };
        const output = nodesRef.current.find((n) => n.type === "output");
        position = output
          ? { x: output.position.x, y: output.position.y + 280 }
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
      if (type === "arena") {
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
      setNodes(nextNodes);
      setSelectedId(id);
      setStatus(
        type === "arena"
          ? "Added Are.na — paste a link, then → Depth"
          : type === "arena_out"
            ? "Added Are.na Out — pick a channel and Save"
            : type === "relief"
              ? "Added Relief — wire a look/scene plate into it"
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
    setFrameCount(Math.max(1, res.frame_count || 1));
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
    scheduleEvaluate(flow.nodes, flow.edges, res.run_id);
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
      setFrameCount(Math.max(1, res.frame_count || 1));
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
            x: (depth?.position.x ?? 40) - 280,
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
        setStatus(
          res.block?.arena_url
            ? `Saved to Are.na · ${res.block.arena_url}`
            : `Saved block #${res.block?.id}`
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
        Math.min(40, Math.round(Number((outNode.data as any)?.batch_count ?? 5)))
      );
      const wackiness = Math.min(
        1,
        Math.max(0, Number((outNode.data as any)?.wackiness ?? 0.45))
      );

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
      });

      try {
        const graph = fromFlow(nodesRef.current, edgesRef.current);
        const { job_id } = await startArenaRoll({
          arena_url: arenaUrl,
          count,
          model_key: modelKey,
          max_size: maxSize,
          infer_backend: inferBackend,
          wackiness,
          channel_id: channelId,
          title: baseTitle,
          graph,
        });
        batchJobIdRef.current = job_id;
        setStatus(`Batch job ${job_id} · ${count} rolls on ${inferBackend}…`);

        const job = await pollJob(
          job_id,
          (j) => {
            const i = Number(j.i || 0);
            const n = Number(j.n || count);
            patchNodeUi(arenaOutId, {
              status: "batching",
              batch_i: i,
              batch_n: n,
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
          600,
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
          batch_i: count,
          batch_n: count,
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

  const onGenerateVideo = async (file: File) => {
    setBusy(true);
    setPlaying(false);
    setProgress("upload");
    setVideoProgress(0);
    setStatus("Uploading video…");
    try {
      setProgress("infer");
      setStatus(`Starting Depth Pro video on ${inferBackend}…`);
      // Force Depth Pro for RGB video
      const key = modelKey.startsWith("promptda") ? "depth-pro" : modelKey;
      if (key !== modelKey) setModelKey(key);
      const { job_id } = await startVideoInfer(file, key, maxSize, inferBackend);
      const res = await pollVideoJob(job_id, (job) => {
        setVideoProgress(job.progress ?? 0);
        setStatus(job.message || `Video ${Math.round((job.progress || 0) * 100)}%`);
        if (job.modal_usage?.cost_usd != null) {
          setModalSessionCost(Number(job.modal_usage.cost_usd));
        }
      });
      applyInfer(res, meta);
      setVideoProgress(null);
    } catch (e: any) {
      setStatus(`Video error: ${e.message}`);
      setVideoProgress(null);
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  const goToFrame = useCallback(
    async (index: number) => {
      if (!runId || frameCount <= 1) return;
      if (seekingRef.current) return;
      const i = Math.max(0, Math.min(frameCount - 1, index));
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
    [frameCount, runId]
  );

  useEffect(() => {
    if (!playing || frameCount <= 1) {
      if (playTimer.current) {
        window.clearInterval(playTimer.current);
        playTimer.current = null;
      }
      return;
    }
    const ms = Math.max(40, Math.round(1000 / Math.max(sampleFps, 1)));
    playTimer.current = window.setInterval(() => {
      const next =
        frameIndexRef.current + 1 >= frameCount
          ? 0
          : frameIndexRef.current + 1;
      void goToFrame(next);
    }, ms);
    return () => {
      if (playTimer.current) {
        window.clearInterval(playTimer.current);
        playTimer.current = null;
      }
    };
  }, [playing, frameCount, sampleFps, goToFrame]);

  const onExportVideo = async () => {
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
  };

  const isValidConnection = useCallback((connection: Connection | Edge) => {
    const { source, target, targetHandle } = connection;
    if (!source || !target || source === target) return false;
    const src = nodesRef.current.find((n) => n.id === source);
    const tgt = nodesRef.current.find((n) => n.id === target);
    if (!src || !tgt) return false;

    if (tgt.type === "output") {
      return (
        src.type !== "depth" &&
        src.type !== "output" &&
        src.type !== "arena" &&
        src.type !== "arena_out"
      );
    }
    if (tgt.type === "blend") {
      return (
        (targetHandle === "base" || targetHandle === "overlay") &&
        src.type !== "depth" &&
        src.type !== "output" &&
        src.type !== "arena" &&
        src.type !== "arena_out"
      );
    }
    if (tgt.type === "look" || tgt.type === "scene") {
      // Single input: depth or any image plate.
      return (
        src.type === "depth" ||
        src.type === "look" ||
        src.type === "scene" ||
        src.type === "blend" ||
        src.type === "relief"
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
      return src.type === "arena";
    }
    if (tgt.type === "arena_out") {
      return (
        src.type === "look" ||
        src.type === "scene" ||
        src.type === "blend" ||
        src.type === "relief" ||
        src.type === "output"
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
      inferBackend,
      modalSessionCost,
      inferDevice,
      runId,
      frameIndex,
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
      setWindowTuneSurface,
      applyWindowPct,
      commitWindowTune,
      resolveArenaNode,
      loadArenaIntoDepth,
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
      setWindowTuneSurface,
      applyWindowPct,
      commitWindowTune,
      resolveArenaNode,
      loadArenaIntoDepth,
      refreshArenaChannels,
      publishArenaOut,
      runArenaBatch,
      stopArenaBatch,
    ]
  );

  const modalData = (modalEdge?.data || {}) as BlendEdgeData;
  const modalBlendable = Boolean(modalData.blendable);

  return (
    <KitchenContext.Provider value={kitchenValue}>
      <div className="app">
        <header className="topbar">
          <h1>Depth Ledger</h1>
          <div className="infer-toggle" title="Kitchen node graph vs Operator synthesizer">
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
              {(meta?.max_sizes || [518, 756, 1008]).map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
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
            disabled={busy}
            onClick={() => fileVideo.current?.click()}
            title="RGB video → Depth Pro per frame (max ~60 frames @ ~10fps)"
          >
            Generate Video
          </button>
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
          <input
            ref={fileVideo}
            type="file"
            accept="video/mp4,video/quicktime,video/webm,video/*"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onGenerateVideo(f);
              e.target.value = "";
            }}
          />
          <label>
            History
            <select
              value=""
              onChange={(e) => {
                if (e.target.value) loadHistory(e.target.value);
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
          <button
            disabled={busy || nodes.length === 0 || appMode !== "kitchen"}
            onClick={() => saveAsDefault()}
            title="Use this node layout for every new generate"
          >
            Save as default
          </button>
          {hasCustomDefault && appMode === "kitchen" ? (
            <button
              disabled={busy}
              onClick={() => resetDefault()}
              title="Clear custom default kitchen"
            >
              Reset default
            </button>
          ) : null}
          <span className="status">
            {videoProgress != null
              ? `${status} (${Math.round(videoProgress * 100)}%)`
              : status}
          </span>
        </header>

        {appMode === "operator" ? (
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
            frameCount={frameCount}
            frameIndex={frameIndex}
            sampleFps={sampleFps}
            playing={playing}
            setPlaying={setPlaying}
            goToFrame={goToFrame}
            onExportVideo={onExportVideo}
            exportUrl={exportUrl}
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
          />
        )}
      </div>
    </KitchenContext.Provider>
  );
}
