import type { Edge, Node } from "@xyflow/react";
import type { Meta } from "./api";
import type { BlendEdgeData } from "./edges";
import { lookDefaults } from "./kitchenContext";

function clamp(n: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, n));
}

function rand() {
  return Math.random();
}

function pick<T>(arr: T[]): T {
  return arr[Math.floor(rand() * arr.length)]!;
}

function jitter(value: number, amount: number, lo: number, hi: number) {
  if (amount <= 0) return clamp(value, lo, hi);
  return clamp(value + (rand() * 2 - 1) * amount, lo, hi);
}

/**
 * Wackiness 0–1:
 *  - low: gentle nudges to windows / opacity
 *  - mid: occasional preset / blend swaps
 *  - high: wild windows, frequent swaps, chaos blends
 *
 * maxFidelity pins Relief solve size to the largest allowed side
 * (caller should also force infer max_size). Creative jitter stays on.
 */
export function randomizeKitchen(
  nodes: Node[],
  edges: Edge[],
  meta: Meta | null,
  wackiness: number,
  maxFidelity = false
): { nodes: Node[]; edges: Edge[] } {
  const w = clamp(wackiness, 0, 1);
  const looks = (meta?.looks || []).map((l) => l.key).filter(Boolean);
  const cmaps = meta?.cmaps || [
    "gray",
    "magma",
    "inferno",
    "viridis",
    "turbo",
    "photo",
    "wash",
  ];
  const blendModes = (meta?.blend_modes || []).map((m) => m.key);
  const sceneryModes = (meta?.scenery_modes || []).map((m) => m.key);
  const reliefSides = meta?.relief?.max_sides || [384, 512, 768, 1024];
  const maxReliefSide = Math.max(...reliefSides);

  const nextNodes = nodes.map((n) => {
    const d = { ...(n.data as Record<string, unknown>) };

    if (n.type === "look") {
      const swapChance = 0.08 + w * 0.72;
      if (looks.length && rand() < swapChance) {
        const key = pick(looks);
        const defaults = lookDefaults(meta, key);
        d.preset = key;
        d.lo_pct = defaults?.lo_pct ?? 2;
        d.hi_pct = defaults?.hi_pct ?? 98;
        d.gamma = defaults?.gamma ?? 1;
        d.cmap = "inherit";
      }

      const loJ = 1 + w * 14;
      const hiJ = 2 + w * 28;
      const gJ = 0.05 + w * 0.55;
      let lo = jitter(Number(d.lo_pct ?? 2), loJ, 1, 45);
      let hi = jitter(Number(d.hi_pct ?? 98), hiJ, 15, 99);
      if (hi <= lo + 3) hi = Math.min(99, lo + 8 + w * 20);
      d.lo_pct = Math.round(lo * 10) / 10;
      d.hi_pct = Math.round(hi * 10) / 10;
      d.gamma =
        Math.round(jitter(Number(d.gamma ?? 1), gJ, 0.35, 2.2) * 100) / 100;

      if (rand() < w * 0.55) {
        d.cmap = pick(cmaps);
      }
      const effectiveCmap =
        d.cmap && d.cmap !== "inherit"
          ? String(d.cmap)
          : lookDefaults(meta, String(d.preset || "ledger"))?.cmap || "gray";
      if (effectiveCmap === "wash") {
        d.blur_amount =
          Math.round(
            jitter(Number(d.blur_amount ?? 0.4), 0.05 + w * 0.35, 0, 1) * 100
          ) / 100;
      }
    }

    if (n.type === "scene") {
      if (sceneryModes.length && rand() < 0.12 + w * 0.7) {
        d.mode = pick(sceneryModes);
      }
      d.protect =
        Math.round(
          jitter(Number(d.protect ?? 0.22), 0.04 + w * 0.28, 0.02, 0.85) * 100
        ) / 100;
      d.feather =
        Math.round(
          jitter(Number(d.feather ?? 0.35), 0.05 + w * 0.35, 0.05, 0.95) * 100
        ) / 100;
      if (rand() < w * 0.4) d.clean = rand() > 0.35;
      if (rand() < w * 0.35) d.fringe_kill = rand() > 0.4;
      if (rand() < w * 0.35) d.far_remap = rand() > 0.35;
    }

    if (n.type === "relief") {
      d.gc_alpha =
        Math.round(
          jitter(Number(d.gc_alpha ?? 10), 1 + w * 12, 1, 40) * 10
        ) / 10;
      if (rand() < 0.1 + w * 0.45) d.invert = rand() > 0.55;
      if (maxFidelity) {
        d.max_side = maxReliefSide;
      } else if (rand() < 0.08 + w * 0.4) {
        d.max_side = pick(reliefSides);
      }
    }

    if (n.type === "smooth") {
      const windows = ["off", "locked", "spline"] as const;
      const temps = ["off", "ema", "flow"] as const;
      if (rand() < 0.12 + w * 0.5) d.window_mode = pick([...windows]);
      if (rand() < 0.12 + w * 0.5) d.temporal = pick([...temps]);
      d.ema_alpha =
        Math.round(
          jitter(Number(d.ema_alpha ?? 0.45), 0.05 + w * 0.25, 0.15, 0.95) * 100
        ) / 100;
      d.flow_prior =
        Math.round(
          jitter(Number(d.flow_prior ?? 0.75), 0.05 + w * 0.25, 0.2, 0.95) *
            100
        ) / 100;
      d.occ_thresh =
        Math.round(
          jitter(Number(d.occ_thresh ?? 1.5), 0.1 + w * 0.4, 0.5, 4) * 100
        ) / 100;
    }

    if (n.type === "freq_fuse") {
      d.beta =
        Math.round(
          jitter(Number(d.beta ?? 1), 0.05 + w * 0.35, 0.2, 1.8) * 100
        ) / 100;
      d.blur_sigma =
        Math.round(
          jitter(Number(d.blur_sigma ?? 4), 0.1 + w * 0.4, 1, 12) * 10
        ) / 10;
    }

    if (n.type === "blend") {
      if (blendModes.length && rand() < 0.15 + w * 0.75) {
        d.mode = pick(blendModes);
      }
      d.opacity =
        Math.round(
          jitter(Number(d.opacity ?? 1), 0.05 + w * 0.45, 0.15, 1) * 100
        ) / 100;
    }

    return { ...n, data: d };
  });

  const nextEdges = edges.map((e) => {
    const prev = (e.data || {}) as BlendEdgeData;
    if (prev.blendable === false) return e;
    // Only mutate blendable image→image wires
    const src = nextNodes.find((n) => n.id === e.source);
    const tgt = nextNodes.find((n) => n.id === e.target);
    const blendable =
      Boolean(src && ["look", "scene", "blend"].includes(src.type || "")) &&
      Boolean(tgt && (tgt.type === "look" || tgt.type === "scene"));
    if (!blendable && prev.blendable !== true) return e;

    const data: BlendEdgeData = {
      blend_mode: prev.blend_mode || "normal",
      opacity: prev.opacity ?? 1,
      blendable: true,
    };
    if (blendModes.length && rand() < 0.12 + w * 0.78) {
      data.blend_mode = pick(blendModes);
    }
    data.opacity =
      Math.round(
        jitter(Number(data.opacity ?? 1), 0.06 + w * 0.5, 0.12, 1) * 100
      ) / 100;
    return { ...e, data };
  });

  return { nodes: nextNodes, edges: nextEdges };
}
