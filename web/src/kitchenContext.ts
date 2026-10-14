import { createContext, useContext } from "react";
import type { ArenaChannelOption, LookMeta, Meta } from "./api";

export type WindowTuneMode = "near" | "far";

export type WindowTuneState = {
  nodeId: string;
  mode: WindowTuneMode;
} | null;

export type KitchenContextValue = {
  meta: Meta | null;
  evalToken: number;
  nodePreviews: Record<string, string>;
  outputUrl: string | null;
  rgbUrl: string | null;
  progress: null | "upload" | "infer" | "evaluate";
  busy: boolean;
  modelKey: string;
  maxSize: number;
  setMaxSize: (n: number) => void;
  inferBackend: "local" | "modal";
  modalSessionCost: number;
  inferDevice: string | null;
  runId: string | null;
  frameIndex: number;
  frameCount: number;
  readyFrames: number;
  readyIndices: number[];
  inferMode: string | null;
  frameWidth: number | null;
  frameHeight: number | null;
  sampleFps: number;
  inferStatus: string | null;
  previewGifUrl: string | null;
  frameUrls: string[];
  exportUrl: string | null;
  frameStatusExpanded: boolean;
  setFrameStatusExpanded: (v: boolean) => void;
  playing: boolean;
  setPlaying: (playing: boolean) => void;
  scrubFrame: (index: number) => void;
  exportVideo: () => Promise<void>;
  arenaChannels: ArenaChannelOption[];
  arenaAuth: boolean;
  windowTune: WindowTuneState;
  patchNodeData: (nodeId: string, patch: Record<string, unknown>) => void;
  /** UI-only node patches (preview size, etc.) — no graph re-eval. */
  patchNodeUi: (nodeId: string, patch: Record<string, unknown>) => void;
  deleteNode: (nodeId: string) => void;
  openEdgeBlend: (edgeId: string) => void;
  openLightbox: (url: string, title?: string) => void;
  startWindowTune: (nodeId: string) => void;
  stopWindowTune: () => void;
  setWindowTuneMode: (mode: WindowTuneMode) => void;
  applyWindowPct: (
    nodeId: string,
    field: "lo_pct" | "hi_pct",
    pct: number
  ) => void;
  commitWindowTune: () => Promise<void>;
  /** Resolve Are.na URL into node data (no depth run). */
  resolveArenaNode: (
    nodeId: string,
    mode?: "auto" | "shuffle"
  ) => Promise<void>;
  /** Download selected Are.na block and feed Depth (infer). */
  loadArenaIntoDepth: (
    nodeId: string,
    mode?: "auto" | "shuffle"
  ) => Promise<void>;
  /** Upload video into Video In node (prepare RGB + preview GIF). */
  prepareVideoNode: (nodeId: string, file: File) => Promise<void>;
  /** Run Depth Pro on the prepared video run (from Video Out). */
  inferVideoRun: () => Promise<void>;
  refreshArenaChannels: () => Promise<void>;
  publishArenaOut: (nodeId: string) => Promise<void>;
  runArenaBatch: (nodeId: string) => Promise<void>;
  stopArenaBatch: () => void;
};

export const KitchenContext = createContext<KitchenContextValue>({
  meta: null,
  evalToken: 0,
  nodePreviews: {},
  outputUrl: null,
  rgbUrl: null,
  progress: null,
  busy: false,
  modelKey: "depth-pro",
  maxSize: 756,
  setMaxSize: () => undefined,
  inferBackend: "local",
  modalSessionCost: 0,
  inferDevice: null,
  runId: null,
  frameIndex: 0,
  frameCount: 1,
  readyFrames: 1,
  readyIndices: [],
  inferMode: null,
  frameWidth: null,
  frameHeight: null,
  sampleFps: 10,
  inferStatus: null,
  previewGifUrl: null,
  frameUrls: [],
  exportUrl: null,
  frameStatusExpanded: false,
  setFrameStatusExpanded: () => undefined,
  playing: false,
  setPlaying: () => undefined,
  scrubFrame: () => undefined,
  exportVideo: async () => undefined,
  arenaChannels: [],
  arenaAuth: false,
  windowTune: null,
  patchNodeData: () => undefined,
  patchNodeUi: () => undefined,
  deleteNode: () => undefined,
  openEdgeBlend: () => undefined,
  openLightbox: () => undefined,
  startWindowTune: () => undefined,
  stopWindowTune: () => undefined,
  setWindowTuneMode: () => undefined,
  applyWindowPct: () => undefined,
  commitWindowTune: async () => undefined,
  resolveArenaNode: async () => undefined,
  loadArenaIntoDepth: async () => undefined,
  prepareVideoNode: async () => undefined,
  inferVideoRun: async () => undefined,
  refreshArenaChannels: async () => undefined,
  publishArenaOut: async () => undefined,
  runArenaBatch: async () => undefined,
  stopArenaBatch: () => undefined,
});

export function useKitchen() {
  return useContext(KitchenContext);
}

export function lookDefaults(
  meta: Meta | null,
  preset: string
): LookMeta | undefined {
  return meta?.looks.find((l) => l.key === preset);
}

export function clampWindowPct(
  field: "lo_pct" | "hi_pct",
  pct: number,
  currentLo: number,
  currentHi: number
): number {
  if (field === "lo_pct") {
    return Math.min(Math.max(pct, 1), Math.min(40, currentHi - 1));
  }
  return Math.max(Math.min(pct, 99), Math.max(20, currentLo + 1));
}
