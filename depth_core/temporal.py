"""Temporal depth / display-window helpers for video smooth nodes.

Composable methods: window lock/spline, EMA, flow-guided fuse (FB occlusion),
and frequency-domain fusion (video low-freq + Depth Pro high-freq).
Designed for per-frame kitchen eval + sequential export bake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

from depth_core.pipeline import depth_valid_mask

WINDOW_MODES = ("off", "locked", "spline")
TEMPORAL_MODES = ("off", "ema", "flow")
# Top-level Smooth presets (plug-and-play). "none" = full pass-through.
SMOOTH_MODES = ("none", "ema", "flow", "window")
WINDOW_LOCK_SOURCES = ("first", "median")

DEFAULT_WINDOW_MODE = "locked"
DEFAULT_TEMPORAL_MODE = "ema"
DEFAULT_SMOOTH_MODE = "ema"
DEFAULT_WINDOW_LOCK = "first"
DEFAULT_SPLINE_INTERVAL = 30
DEFAULT_LO_PCT = 2.0
DEFAULT_HI_PCT = 98.0
DEFAULT_EMA_ALPHA = 0.45  # weight on current frame
# Weight on warped prior in non-occluded regions (higher = more temporal stickiness).
DEFAULT_FLOW_PRIOR = 0.75
DEFAULT_OCC_THRESH = 1.5  # forward–backward consistency threshold (px)
# Legacy aliases kept for old graph JSON
DEFAULT_FLOW_WEIGHT = 0.25  # ≈ 1 - flow_prior (weight on current)
DEFAULT_MOTION_GATE = 0.0  # unused once FB occlusion is active

DEFAULT_FREQ_BETA = 1.0
DEFAULT_FREQ_BLUR_SIGMA = 4.0


@dataclass
class WindowRange:
    near_m: float
    far_m: float


def smooth_params_from_data(data: Optional[dict]) -> dict[str, Any]:
    """Normalize Smooth node data with defaults.

    Top-level ``mode`` is the plug-and-play control:
      none   — pass-through (no temporal, no window lock)
      ema    — EMA temporal (+ optional window)
      flow   — flow-guided temporal (+ optional window)
      window — display window only (no temporal blend)
    """
    d = dict(data or {})

    mode = str(d.get("mode") or "").lower().strip()
    if mode not in SMOOTH_MODES:
        # Infer from legacy temporal + window fields
        temporal_raw = str(d.get("temporal") or "").lower()
        window_raw = str(d.get("window_mode") or "").lower()
        if temporal_raw == "off" and window_raw == "off":
            mode = "none"
        elif temporal_raw == "flow":
            mode = "flow"
        elif temporal_raw == "off" and window_raw in ("locked", "spline"):
            mode = "window"
        elif temporal_raw in ("ema", ""):
            mode = DEFAULT_SMOOTH_MODE
        else:
            mode = DEFAULT_SMOOTH_MODE

    if mode == "none":
        window_mode = "off"
        temporal = "off"
    elif mode == "window":
        temporal = "off"
        window_mode = str(d.get("window_mode") or "locked").lower()
        if window_mode not in WINDOW_MODES or window_mode == "off":
            window_mode = "locked"
    elif mode == "flow":
        temporal = "flow"
        window_mode = str(d.get("window_mode") or DEFAULT_WINDOW_MODE).lower()
        if window_mode not in WINDOW_MODES:
            window_mode = DEFAULT_WINDOW_MODE
    else:  # ema
        temporal = "ema"
        window_mode = str(d.get("window_mode") or DEFAULT_WINDOW_MODE).lower()
        if window_mode not in WINDOW_MODES:
            window_mode = DEFAULT_WINDOW_MODE

    lock = str(d.get("window_lock") or DEFAULT_WINDOW_LOCK).lower()
    if lock not in WINDOW_LOCK_SOURCES:
        lock = DEFAULT_WINDOW_LOCK

    # Prefer flow_prior; migrate legacy flow_weight (was weight on current).
    if "flow_prior" in d and d.get("flow_prior") is not None:
        flow_prior = float(d.get("flow_prior"))
    elif "flow_weight" in d and d.get("flow_weight") is not None:
        flow_prior = 1.0 - float(d.get("flow_weight"))
    else:
        flow_prior = DEFAULT_FLOW_PRIOR

    return {
        "mode": mode,
        "window_mode": window_mode,
        "window_lock": lock,
        "spline_interval": int(
            np.clip(int(d.get("spline_interval", DEFAULT_SPLINE_INTERVAL)), 4, 120)
        ),
        "lo_pct": float(np.clip(float(d.get("lo_pct", DEFAULT_LO_PCT)), 0.5, 40.0)),
        "hi_pct": float(np.clip(float(d.get("hi_pct", DEFAULT_HI_PCT)), 20.0, 99.5)),
        "temporal": temporal,
        "ema_alpha": float(np.clip(float(d.get("ema_alpha", DEFAULT_EMA_ALPHA)), 0.05, 1.0)),
        "flow_prior": float(np.clip(flow_prior, 0.05, 0.95)),
        "occ_thresh": float(
            np.clip(float(d.get("occ_thresh", DEFAULT_OCC_THRESH)), 0.25, 8.0)
        ),
        # Kept for older UI / JSON; ignored by hardened flow fuse.
        "flow_weight": float(
            np.clip(1.0 - float(np.clip(flow_prior, 0.05, 0.95)), 0.05, 1.0)
        ),
        "motion_gate": float(
            np.clip(float(d.get("motion_gate", DEFAULT_MOTION_GATE)), 0.0, 1.0)
        ),
    }


def freq_fuse_params_from_data(data: Optional[dict]) -> dict[str, Any]:
    """Normalize Freq Fuse node data with defaults."""
    d = dict(data or {})
    return {
        "beta": float(np.clip(float(d.get("beta", DEFAULT_FREQ_BETA)), 0.0, 2.0)),
        "blur_sigma": float(
            np.clip(float(d.get("blur_sigma", DEFAULT_FREQ_BLUR_SIGMA)), 0.5, 32.0)
        ),
    }


def percentile_near_far(
    depth_m: np.ndarray,
    lo_pct: float = DEFAULT_LO_PCT,
    hi_pct: float = DEFAULT_HI_PCT,
) -> WindowRange:
    depth = np.asarray(depth_m, dtype=np.float64)
    valid = depth_valid_mask(depth)
    if not valid.any():
        return WindowRange(0.1, 10.0)
    near = float(np.percentile(depth[valid], lo_pct))
    far = float(np.percentile(depth[valid], hi_pct))
    if far <= near:
        far = near + 1e-3
    return WindowRange(near, far)


def smooth_min_max(
    min_vals: Sequence[float],
    max_vals: Sequence[float],
    interval: int = DEFAULT_SPLINE_INTERVAL,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cubic-spline smooth of per-frame min/max (PromptDA-style)."""
    from scipy.interpolate import CubicSpline

    n = len(min_vals)
    if n == 0:
        return np.array([]), np.array([])
    if n == 1:
        return np.array([float(min_vals[0])]), np.array([float(max_vals[0])])

    interval = max(1, int(interval))
    key_frames = list(range(0, n, interval))
    if key_frames[-1] != n - 1:
        key_frames.append(n - 1)
    key_idx = np.asarray(key_frames, dtype=np.float64)
    min_key = np.asarray([min_vals[i] for i in key_frames], dtype=np.float64)
    max_key = np.asarray([max_vals[i] for i in key_frames], dtype=np.float64)
    if len(key_frames) < 2:
        return min_key, max_key
    min_spline = CubicSpline(key_idx, min_key, bc_type="natural")
    max_spline = CubicSpline(key_idx, max_key, bc_type="natural")
    x = np.arange(n, dtype=np.float64)
    return min_spline(x), max_spline(x)


def compute_window_series(
    depths: Sequence[np.ndarray],
    lo_pct: float,
    hi_pct: float,
    *,
    window_mode: str = "locked",
    window_lock: str = "first",
    spline_interval: int = DEFAULT_SPLINE_INTERVAL,
) -> list[WindowRange]:
    """Per-frame near/far for a clip under the chosen window mode."""
    raw = [percentile_near_far(d, lo_pct, hi_pct) for d in depths]
    if not raw:
        return []
    if window_mode == "off":
        return raw
    if window_mode == "locked":
        if window_lock == "median":
            near = float(np.median([r.near_m for r in raw]))
            far = float(np.median([r.far_m for r in raw]))
        else:
            near, far = raw[0].near_m, raw[0].far_m
        if far <= near:
            far = near + 1e-3
        locked = WindowRange(near, far)
        return [locked for _ in raw]
    # spline
    mins, maxs = smooth_min_max(
        [r.near_m for r in raw],
        [r.far_m for r in raw],
        spline_interval,
    )
    out: list[WindowRange] = []
    for a, b in zip(mins, maxs):
        near = float(a)
        far = float(max(b, near + 1e-3))
        out.append(WindowRange(near, far))
    return out


def _to_inverse(depth_m: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 1e-4)
    inv = np.zeros_like(depth, dtype=np.float32)
    if valid.any():
        inv[valid] = (1.0 / np.clip(depth[valid], 1e-4, None)).astype(np.float32)
        fill = float(np.median(inv[valid]))
        inv[~valid] = fill
    return inv


def _from_inverse(inv: np.ndarray, ref_depth: np.ndarray) -> np.ndarray:
    inv = np.asarray(inv, dtype=np.float32)
    out = np.asarray(ref_depth, dtype=np.float32).copy()
    valid = np.isfinite(inv) & (inv > 1e-8)
    out[valid] = (1.0 / inv[valid]).astype(np.float32)
    return out


def ema_blend_depth(
    current: np.ndarray,
    previous: np.ndarray,
    alpha: float = DEFAULT_EMA_ALPHA,
    *,
    space: str = "inverse",
) -> np.ndarray:
    """Blend current depth with previous. alpha = weight on current."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    cur = np.asarray(current, dtype=np.float32)
    prev = np.asarray(previous, dtype=np.float32)
    if cur.shape != prev.shape:
        prev = cv2.resize(prev, (cur.shape[1], cur.shape[0]), interpolation=cv2.INTER_LINEAR)
    if space == "inverse":
        inv = alpha * _to_inverse(cur) + (1.0 - alpha) * _to_inverse(prev)
        return _from_inverse(inv, cur)
    return (alpha * cur + (1.0 - alpha) * prev).astype(np.float32)


def _rgb_to_gray_u8(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _farneback(g0: np.ndarray, g1: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def _occlusion_mask(
    flow_fw: np.ndarray,
    flow_bw: np.ndarray,
    thresh: float,
) -> np.ndarray:
    """True where forward–backward flow is consistent (non-occluded)."""
    h, w = flow_fw.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    # Sample backward flow at forward-warped locations.
    map_x = (grid_x + flow_fw[..., 0]).astype(np.float32)
    map_y = (grid_y + flow_fw[..., 1]).astype(np.float32)
    bw_x = cv2.remap(
        flow_bw[..., 0], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    bw_y = cv2.remap(
        flow_bw[..., 1], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    # Cycle consistency: fw(p) + bw(p+fw) ≈ 0
    dx = flow_fw[..., 0] + bw_x
    dy = flow_fw[..., 1] + bw_y
    err = np.sqrt(dx * dx + dy * dy)
    inside = (map_x >= 0) & (map_x < w - 1) & (map_y >= 0) & (map_y < h - 1)
    return (err <= float(thresh)) & inside


def flow_fuse_depth(
    curr_rgb: np.ndarray,
    prev_rgb: np.ndarray,
    curr_depth: np.ndarray,
    prev_depth: np.ndarray,
    *,
    flow_prior: float = DEFAULT_FLOW_PRIOR,
    occ_thresh: float = DEFAULT_OCC_THRESH,
    weight: Optional[float] = None,
    motion_gate: float = DEFAULT_MOTION_GATE,
) -> np.ndarray:
    """Warp previous depth toward current via Farneback + FB occlusion mask.

    Non-occluded: blend warped prior with current (flow_prior on warped).
    Occluded / disoccluded: use current depth only.
    """
    del motion_gate  # retained for call-site compat; FB mask replaces motion gate
    cur_d = np.asarray(curr_depth, dtype=np.float32)
    prev_d = np.asarray(prev_depth, dtype=np.float32)
    h, w = cur_d.shape[:2]
    if prev_d.shape[:2] != (h, w):
        prev_d = cv2.resize(prev_d, (w, h), interpolation=cv2.INTER_LINEAR)

    g0 = _rgb_to_gray_u8(prev_rgb)
    g1 = _rgb_to_gray_u8(curr_rgb)
    if g0.shape[:2] != (h, w):
        g0 = cv2.resize(g0, (w, h), interpolation=cv2.INTER_AREA)
    if g1.shape[:2] != (h, w):
        g1 = cv2.resize(g1, (w, h), interpolation=cv2.INTER_AREA)

    flow_fw = _farneback(g0, g1)  # prev → curr
    flow_bw = _farneback(g1, g0)  # curr → prev
    consistent = _occlusion_mask(flow_fw, flow_bw, occ_thresh)

    # Backward warp: sample prev depth at curr pixel using fw flow (prev→curr).
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x - flow_fw[..., 0]).astype(np.float32)
    map_y = (grid_y - flow_fw[..., 1]).astype(np.float32)
    warped = cv2.remap(
        prev_d,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_valid = (
        (map_x >= 0)
        & (map_x < w - 1)
        & (map_y >= 0)
        & (map_y < h - 1)
        & np.isfinite(warped)
        & (warped > 1e-4)
    )
    use_prior = consistent & warped_valid

    if weight is not None:
        # Legacy: weight was on current → prior = 1 - weight
        prior = float(np.clip(1.0 - float(weight), 0.05, 0.95))
    else:
        prior = float(np.clip(flow_prior, 0.05, 0.95))

    inv_cur = _to_inverse(cur_d)
    inv_warp = _to_inverse(warped)
    inv = inv_cur.copy()
    inv[use_prior] = (
        prior * inv_warp[use_prior] + (1.0 - prior) * inv_cur[use_prior]
    ).astype(np.float32)
    return _from_inverse(inv, cur_d)


def affine_align_inverse(
    src_depth: np.ndarray,
    ref_depth: np.ndarray,
) -> np.ndarray:
    """Affine-align src to ref in inverse-depth space: inv_src' = a * inv_src + b."""
    src = np.asarray(src_depth, dtype=np.float32)
    ref = np.asarray(ref_depth, dtype=np.float32)
    if src.shape[:2] != ref.shape[:2]:
        src = cv2.resize(src, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)
    inv_s = _to_inverse(src)
    inv_r = _to_inverse(ref)
    valid = depth_valid_mask(ref) & depth_valid_mask(src) & np.isfinite(inv_s) & np.isfinite(inv_r)
    if valid.sum() < 64:
        return src
    x = inv_s[valid].astype(np.float64)
    y = inv_r[valid].astype(np.float64)
    # least squares [a, b] for y ≈ a x + b
    A = np.column_stack([x, np.ones_like(x)])
    try:
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b = float(coef[0]), float(coef[1])
    except Exception:
        return src
    if not np.isfinite(a) or abs(a) < 1e-8:
        return src
    inv_aligned = (a * inv_s + b).astype(np.float32)
    return _from_inverse(inv_aligned, ref)


def freq_fuse_depth(
    pro_depth: np.ndarray,
    video_depth: np.ndarray,
    *,
    beta: float = DEFAULT_FREQ_BETA,
    blur_sigma: float = DEFAULT_FREQ_BLUR_SIGMA,
) -> np.ndarray:
    """Low-freq from aligned video depth + high-freq from Depth Pro.

    D_final = Blur(D_vid_aligned) + beta * (D_pro - Blur(D_pro))
    Work in inverse-depth for stabler metric blending, then convert back.
    """
    pro = np.asarray(pro_depth, dtype=np.float32)
    vid = np.asarray(video_depth, dtype=np.float32)
    if vid.shape[:2] != pro.shape[:2]:
        vid = cv2.resize(vid, (pro.shape[1], pro.shape[0]), interpolation=cv2.INTER_LINEAR)

    aligned = affine_align_inverse(vid, pro)
    inv_pro = _to_inverse(pro)
    inv_vid = _to_inverse(aligned)

    sigma = float(max(blur_sigma, 0.5))
    k = max(3, int(round(sigma * 4)) | 1)
    inv_pro_blur = cv2.GaussianBlur(inv_pro, (k, k), sigma)
    inv_vid_blur = cv2.GaussianBlur(inv_vid, (k, k), sigma)
    inv_high = inv_pro - inv_pro_blur
    inv_final = inv_vid_blur + float(beta) * inv_high
    return _from_inverse(inv_final.astype(np.float32), pro)


def apply_temporal_depth(
    curr_depth: np.ndarray,
    prev_depth: Optional[np.ndarray],
    *,
    temporal: str,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
    flow_prior: float = DEFAULT_FLOW_PRIOR,
    occ_thresh: float = DEFAULT_OCC_THRESH,
    flow_weight: Optional[float] = None,
    motion_gate: float = DEFAULT_MOTION_GATE,
    curr_rgb: Optional[np.ndarray] = None,
    prev_rgb: Optional[np.ndarray] = None,
    prev_smoothed: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply temporal mode for one frame. Prefer prev_smoothed for sequential bake."""
    if temporal == "off" or prev_depth is None:
        return np.asarray(curr_depth, dtype=np.float32)
    hist = prev_smoothed if prev_smoothed is not None else prev_depth
    if temporal == "ema":
        return ema_blend_depth(curr_depth, hist, ema_alpha)
    if temporal == "flow":
        if curr_rgb is None or prev_rgb is None:
            return ema_blend_depth(curr_depth, hist, ema_alpha)
        prior = flow_prior
        if flow_weight is not None and flow_prior == DEFAULT_FLOW_PRIOR:
            # Only use legacy weight if caller didn't set flow_prior explicitly
            # via smooth_params (which always sets both). Prefer flow_prior.
            prior = flow_prior
        return flow_fuse_depth(
            curr_rgb,
            prev_rgb,
            curr_depth,
            hist,
            flow_prior=prior,
            occ_thresh=occ_thresh,
            motion_gate=motion_gate,
        )
    return np.asarray(curr_depth, dtype=np.float32)


def smooth_meta() -> dict[str, Any]:
    """UI/meta payload for Smooth node defaults."""
    return {
        "modes": [
            {"key": "none", "label": "None (pass-through)"},
            {"key": "ema", "label": "EMA temporal"},
            {"key": "flow", "label": "Flow-guided"},
            {"key": "window", "label": "Window only"},
        ],
        "window_modes": [
            {"key": "off", "label": "Per-frame (raw)"},
            {"key": "locked", "label": "Locked window"},
            {"key": "spline", "label": "Spline-smoothed window"},
        ],
        "temporal_modes": [
            {"key": "off", "label": "Off"},
            {"key": "ema", "label": "EMA (light)"},
            {"key": "flow", "label": "Flow-guided"},
        ],
        "window_locks": [
            {"key": "first", "label": "First frame"},
            {"key": "median", "label": "Median of clip"},
        ],
        "defaults": {
            "mode": DEFAULT_SMOOTH_MODE,
            "window_mode": DEFAULT_WINDOW_MODE,
            "window_lock": DEFAULT_WINDOW_LOCK,
            "spline_interval": DEFAULT_SPLINE_INTERVAL,
            "lo_pct": DEFAULT_LO_PCT,
            "hi_pct": DEFAULT_HI_PCT,
            "temporal": DEFAULT_TEMPORAL_MODE,
            "ema_alpha": DEFAULT_EMA_ALPHA,
            "flow_prior": DEFAULT_FLOW_PRIOR,
            "occ_thresh": DEFAULT_OCC_THRESH,
            "flow_weight": DEFAULT_FLOW_WEIGHT,
            "motion_gate": DEFAULT_MOTION_GATE,
        },
    }


def freq_fuse_meta() -> dict[str, Any]:
    """UI/meta payload for Freq Fuse node defaults."""
    return {
        "defaults": {
            "beta": DEFAULT_FREQ_BETA,
            "blur_sigma": DEFAULT_FREQ_BLUR_SIGMA,
        },
        "backend_label": "stub EMA (swap for ChronoDepth later)",
    }
