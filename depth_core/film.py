"""Flicker-free Depth Pro video stack — independent of kitchen / operator graphs.

Three locked rules:
  1. Anchor focallength_px on frame 0, pass the same f_px into every later frame.
  2. Fixed global visualization bounds in meters (never per-frame 0–255 normalize).
  3. Temporal EMA on metric/inverse depth after scale is anchored.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import imageio
import numpy as np

from depth_core.pipeline import (
    _infer_depth_pro_focal,
    _norm_to_rgb,
    depth_valid_mask,
    get_backend,
    is_promptda,
    normalize_model_key,
)
from depth_core.temporal import ema_blend_depth
from depth_core.video import (
    VIDEO_MAX_FRAMES,
    VIDEO_TARGET_FPS,
    export_frames_to_mp4,
    load_frame_arrays,
    load_frames_meta,
    prepare_video,
    reextract_rgb_frames,
    save_frames_meta,
)

FILM_JSON = "film.json"
FILM_LIVE = "film_live.png"
DEFAULT_PREVIEW_FRAMES = 8
DEFAULT_EMA_ALPHA = 0.45
DEFAULT_VIS_MIN_M = 0.5
DEFAULT_VIS_MAX_M = 10.0
DEFAULT_GAMMA = 1.0
DEFAULT_CMAP = "gray"
FILM_CMAPS = ("gray", "magma", "inferno", "viridis", "turbo")

ProgressCb = Optional[Callable[[float, str], None]]
PartialCb = Optional[Callable[[int, int, Optional[list[int]]], None]]


def _frame_dir(out_dir: Path, index: int) -> Path:
    return Path(out_dir) / "frames" / f"{index:06d}"


def film_meta() -> dict[str, Any]:
    return {
        "preview_frames": DEFAULT_PREVIEW_FRAMES,
        "ema_alpha": DEFAULT_EMA_ALPHA,
        "vis_min_m": DEFAULT_VIS_MIN_M,
        "vis_max_m": DEFAULT_VIS_MAX_M,
        "gamma": DEFAULT_GAMMA,
        "cmap": DEFAULT_CMAP,
        "cmaps": list(FILM_CMAPS),
        "anchor_focal": True,
        "ema_enabled": True,
        "video_max_frames": VIDEO_MAX_FRAMES,
        "video_target_fps": VIDEO_TARGET_FPS,
    }


def default_params() -> dict[str, Any]:
    return {
        "pipeline": "film",
        "anchor_focal": True,
        "focallength_px": None,
        "focallength_override": None,
        "inferred_with_f_px": None,
        "vis_min_m": DEFAULT_VIS_MIN_M,
        "vis_max_m": DEFAULT_VIS_MAX_M,
        "suggested_vis_min_m": None,
        "suggested_vis_max_m": None,
        "bounds_locked": False,
        "ema_enabled": True,
        "ema_alpha": DEFAULT_EMA_ALPHA,
        "gamma": DEFAULT_GAMMA,
        "cmap": DEFAULT_CMAP,
        "preview_frames": DEFAULT_PREVIEW_FRAMES,
        "phase": "idle",
        "preview_ready": 0,
    }


def normalize_params(data: Optional[dict] = None) -> dict[str, Any]:
    d = default_params()
    src = dict(data or {})
    d.update({k: src[k] for k in src if k in d or k in ("pipeline",)})
    d["pipeline"] = "film"
    d["anchor_focal"] = bool(d.get("anchor_focal", True))
    d["ema_enabled"] = bool(d.get("ema_enabled", True))
    d["bounds_locked"] = bool(d.get("bounds_locked", False))
    d["ema_alpha"] = float(np.clip(float(d.get("ema_alpha") or DEFAULT_EMA_ALPHA), 0.05, 1.0))
    d["gamma"] = float(np.clip(float(d.get("gamma") or DEFAULT_GAMMA), 0.3, 2.4))
    cmap = str(d.get("cmap") or DEFAULT_CMAP).lower()
    d["cmap"] = cmap if cmap in FILM_CMAPS else DEFAULT_CMAP
    d["preview_frames"] = int(np.clip(int(d.get("preview_frames") or DEFAULT_PREVIEW_FRAMES), 2, 24))
    d["vis_min_m"] = float(np.clip(float(d.get("vis_min_m") or DEFAULT_VIS_MIN_M), 0.05, 80.0))
    d["vis_max_m"] = float(np.clip(float(d.get("vis_max_m") or DEFAULT_VIS_MAX_M), 0.2, 200.0))
    if d["vis_max_m"] <= d["vis_min_m"]:
        d["vis_max_m"] = d["vis_min_m"] + 0.05
    for key in ("focallength_px", "focallength_override", "inferred_with_f_px"):
        val = d.get(key)
        if val is None or val == "":
            d[key] = None
        else:
            try:
                fv = float(val)
                d[key] = fv if np.isfinite(fv) and fv > 1.0 else None
            except (TypeError, ValueError):
                d[key] = None
    for key in ("suggested_vis_min_m", "suggested_vis_max_m"):
        val = d.get(key)
        if val is None or val == "":
            d[key] = None
        else:
            try:
                d[key] = float(val)
            except (TypeError, ValueError):
                d[key] = None
    phase = str(d.get("phase") or "idle")
    if phase not in ("idle", "pending", "preview", "running", "done"):
        phase = "idle"
    d["phase"] = phase
    d["preview_ready"] = int(d.get("preview_ready") or 0)
    return d


def load_params(out_dir: Path | str) -> dict[str, Any]:
    path = Path(out_dir) / FILM_JSON
    if not path.is_file():
        return default_params()
    try:
        return normalize_params(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return default_params()


def save_params(out_dir: Path | str, params: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(out_dir)
    params = normalize_params(params)
    payload = dict(params)
    payload["saved_at"] = datetime.now(timezone.utc).isoformat()
    (out_dir / FILM_JSON).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return params


def suggest_vis_bounds(
    depths: list[np.ndarray],
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
) -> tuple[float, float]:
    chunks: list[np.ndarray] = []
    for depth in depths:
        arr = np.asarray(depth, dtype=np.float64)
        valid = depth_valid_mask(arr)
        if valid.any():
            chunks.append(arr[valid])
    if not chunks:
        return DEFAULT_VIS_MIN_M, DEFAULT_VIS_MAX_M
    pooled = np.concatenate(chunks)
    near = float(np.percentile(pooled, lo_pct))
    far = float(np.percentile(pooled, hi_pct))
    if far <= near:
        far = near + 0.05
    return near, far


def render_fixed_depth(
    depth_m: np.ndarray,
    vis_min_m: float,
    vis_max_m: float,
    *,
    gamma: float = 1.0,
    cmap: str = "gray",
) -> np.ndarray:
    """Inverse-depth grayscale/cmap using a clip-global meter window."""
    depth = np.asarray(depth_m, dtype=np.float64)
    near = float(min(vis_min_m, vis_max_m))
    far = float(max(vis_min_m, vis_max_m))
    if far <= near:
        far = near + 1e-3
    valid = np.isfinite(depth) & (depth > 1e-4)
    inv = np.zeros_like(depth, dtype=np.float64)
    if valid.any():
        inv[valid] = 1.0 / np.clip(depth[valid], near, far)
    lo, hi = 1.0 / far, 1.0 / near
    span = max(hi - lo, 1e-12)
    norm = np.zeros_like(depth, dtype=np.float64)
    norm[valid] = np.clip((inv[valid] - lo) / span, 0.0, 1.0)
    if gamma != 1.0:
        norm = np.power(np.clip(norm, 0.0, 1.0), float(gamma))
    cmap_key = cmap if cmap in FILM_CMAPS else DEFAULT_CMAP
    return _norm_to_rgb(norm.astype(np.float32), cmap_key)


def _progress(cb: ProgressCb, p: float, msg: str) -> None:
    if cb:
        cb(p, msg)


def prepare_film(
    video_file: str | Path,
    model_key: str = "depth-pro",
    max_size: int = 756,
    *,
    target_fps: float = VIDEO_TARGET_FPS,
    max_frames: int = VIDEO_MAX_FRAMES,
    infer_backend: Optional[str] = None,
    on_progress: ProgressCb = None,
) -> dict[str, Any]:
    state = prepare_video(
        video_file,
        model_key,
        max_size,
        target_fps=target_fps,
        max_frames=max_frames,
        infer_backend=infer_backend,
        on_progress=on_progress,
    )
    out_dir = Path(state["out_dir"])
    params = default_params()
    params["phase"] = "pending"
    save_params(out_dir, params)
    meta = load_frames_meta(out_dir) or {}
    meta["pipeline"] = "film"
    save_frames_meta(out_dir, meta)
    state["pipeline"] = "film"
    state["film"] = params
    return state


def _write_frame_depth(
    out_dir: Path, index: int, color: np.ndarray, depth: np.ndarray
) -> None:
    fd = _frame_dir(out_dir, index)
    fd.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(fd / "rgb.jpg", color)
    np.savez_compressed(fd / "depth.npz", depth=np.asarray(depth, dtype=np.float32))


def _has_depth(out_dir: Path, index: int) -> bool:
    return (_frame_dir(out_dir, index) / "depth.npz").is_file()


def _rgb_path(out_dir: Path, index: int) -> Path:
    return _frame_dir(out_dir, index) / "rgb.jpg"


def _align_max_size(
    out_dir: Path,
    params: dict[str, Any],
    max_sz: int,
    on_progress: ProgressCb = None,
) -> tuple[dict[str, Any], bool]:
    """Re-extract RGB if the longest-edge cap changed.

    Pixel focallength_px is resolution-dependent, so an auto-anchor is cleared
    and re-estimated on frame 0. An explicit override is scaled by the actual
    frame-width change.
    """
    meta = load_frames_meta(out_dir) or {}
    prepared = int(meta.get("max_size") or max_sz)
    if prepared == int(max_sz):
        return params, False
    source = out_dir / str(meta.get("source_file") or "source.mp4")
    if not source.is_file():
        raise ValueError(
            f"Cannot re-extract at max {max_sz}: source video missing under {out_dir}"
        )
    old_w = meta.get("frame_width")
    _progress(on_progress, 0.02, f"Max side {prepared}→{max_sz}; re-extracting…")
    _n, fw, _fh = reextract_rgb_frames(
        out_dir, max_size=int(max_sz), on_progress=on_progress
    )
    if params.get("focallength_override") and old_w and fw:
        scale = float(fw) / float(old_w)
        if np.isfinite(scale) and scale > 0:
            params["focallength_override"] = float(params["focallength_override"]) * scale
    else:
        params["focallength_px"] = None
        params["inferred_with_f_px"] = None
    return params, True


def infer_indices(
    out_dir: Path,
    indices: list[int],
    *,
    params: dict[str, Any],
    model_key: str,
    max_size: int,
    infer_backend: Optional[str],
    force: bool = False,
    on_progress: ProgressCb = None,
    on_partial: PartialCb = None,
) -> tuple[Optional[float], list[int], Optional[dict[str, Any]]]:
    """Infer Depth Pro on selected indices with optional locked f_px.

    Frame 0 (or the first index that needs an estimate) runs first so later
    frames can share that exact focallength_px.
    """
    from depth_core.infer_backend import resolve_backend

    key = normalize_model_key(model_key)
    if is_promptda(key):
        raise ValueError("Film video uses Depth Pro only.")
    backend_used = resolve_backend(infer_backend)
    out_dir = Path(out_dir)
    override = params.get("focallength_override")
    locked: Optional[float] = (
        float(override)
        if override
        else (None if params.get("anchor_focal", True) else None)
    )
    if locked is None and params.get("anchor_focal", True):
        locked = params.get("focallength_px")
        if locked is not None:
            locked = float(locked)

    need: list[int] = []
    for i in indices:
        if force or not _has_depth(out_dir, i):
            need.append(i)
    ready_existing = [i for i in indices if i not in need and _has_depth(out_dir, i)]
    billing = None

    if not need:
        return locked, sorted(set(ready_existing)), None

    def publish(ready_idx: list[int], total_need: int) -> None:
        if on_partial:
            on_partial(len(ready_idx), max(total_need, 1), sorted(ready_idx))

    if backend_used == "modal":
        from depth_core.infer_modal import forward_rgb, forward_rgb_batch_as_completed

        estimate_first: Optional[int] = None
        if (
            locked is None
            and bool(params.get("anchor_focal", True))
            and override is None
        ):
            if 0 in need or _rgb_path(out_dir, 0).is_file():
                estimate_first = 0
            else:
                estimate_first = need[0]

        done: set[int] = set(ready_existing)
        n_need = len(need)

        if estimate_first is not None:
            _progress(
                on_progress,
                0.04,
                f"Frame {estimate_first + 1} · estimating focallength_px…",
            )
            path = _rgb_path(out_dir, estimate_first)
            res = forward_rgb(
                path, key, int(max_size), f_px=None, estimate_focal=True
            )
            _write_frame_depth(out_dir, estimate_first, res.color, res.depth)
            locked = res.focallength_px
            if locked is None:
                raise RuntimeError(
                    "Frame 0 did not return focallength_px; cannot lock scale "
                    "for parallel Modal frames."
                )
            params["focallength_px"] = float(locked)
            save_params(out_dir, params)
            if res.modal_billing:
                billing = dict(res.modal_billing)
            done.add(estimate_first)
            publish(sorted(done), n_need)
            need = [i for i in need if i != estimate_first]

        rest_bytes = [_rgb_path(out_dir, i).read_bytes() for i in need]
        if rest_bytes:
            fl_label = f"{locked:.1f}" if locked is not None else "unlocked"
            _progress(
                on_progress,
                0.12,
                f"Modal parallel · {len(need)} frames · f_px {fl_label}…",
            )
            index_of = {k: idx for k, idx in enumerate(need)}
            for k, res in forward_rgb_batch_as_completed(
                rest_bytes,
                key,
                int(max_size),
                f_px=locked,
                estimate_focal=False,
            ):
                i = index_of[k]
                _write_frame_depth(out_dir, i, res.color, res.depth)
                if res.focallength_px and locked is None:
                    locked = res.focallength_px
                if res.modal_billing:
                    if billing is None:
                        billing = dict(res.modal_billing)
                    else:
                        billing["gpu_seconds"] = float(billing.get("gpu_seconds") or 0) + float(
                            res.modal_billing.get("gpu_seconds") or 0
                        )
                        billing["cost_usd"] = float(billing.get("cost_usd") or 0) + float(
                            res.modal_billing.get("cost_usd") or 0
                        )
                        billing["calls"] = int(billing.get("calls") or 1) + 1
                done.add(i)
                ready_n = len(done)
                _progress(
                    on_progress,
                    0.12 + 0.8 * (ready_n / max(n_need + len(ready_existing), 1)),
                    f"Modal parallel · depth {ready_n}/{n_need + len(ready_existing)}"
                    + (f" · f_px {locked:.1f}" if locked else ""),
                )
                publish(sorted(done), n_need)
        return locked, sorted(done), billing

    backend = get_backend(key)
    done = set(ready_existing)
    n_need = len(need)
    for k, i in enumerate(need):
        _progress(
            on_progress,
            0.05 + 0.85 * (k / max(n_need, 1)),
            f"Depth frame {i + 1} · f_px "
            + ("estimating" if locked is None else f"{locked:.1f}"),
        )
        use_f = locked
        if not params.get("anchor_focal", True) and override is None:
            use_f = None
        color, depth, fl = _infer_depth_pro_focal(
            backend, _rgb_path(out_dir, i), f_px=use_f
        )
        _write_frame_depth(out_dir, i, color, depth)
        if locked is None and params.get("anchor_focal", True):
            locked = fl
        elif override:
            locked = float(override)
        done.add(i)
        publish(sorted(done), n_need)
    return locked, sorted(done), None


def render_vis(
    out_dir: Path | str,
    params: Optional[dict[str, Any]] = None,
    indices: Optional[list[int]] = None,
    *,
    on_progress: ProgressCb = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    params = normalize_params(params or load_params(out_dir))
    meta = load_frames_meta(out_dir) or {}
    n = int(meta.get("frame_count") or 0)
    if n < 1:
        raise ValueError("No frames to render")
    if indices is None:
        indices = [i for i in range(n) if _has_depth(out_dir, i)]
    if not indices:
        raise ValueError("No depth frames ready to visualize")

    prev_smooth: Optional[np.ndarray] = None
    last_vis: Optional[np.ndarray] = None
    # EMA must run in index order from the start of the clip.
    ordered = sorted(indices)
    full_prefix = list(range(ordered[0], ordered[-1] + 1))
    apply_idx = [i for i in full_prefix if _has_depth(out_dir, i)]
    want = set(ordered)

    for k, i in enumerate(apply_idx):
        _progress(on_progress, k / max(len(apply_idx), 1), f"Vis frame {i + 1}…")
        _rgb, depth = load_frame_arrays(out_dir, i)
        if params["ema_enabled"] and prev_smooth is not None:
            smooth = ema_blend_depth(depth, prev_smooth, float(params["ema_alpha"]))
        else:
            smooth = np.asarray(depth, dtype=np.float32)
        prev_smooth = smooth
        if i not in want:
            continue
        vis = render_fixed_depth(
            smooth,
            float(params["vis_min_m"]),
            float(params["vis_max_m"]),
            gamma=float(params["gamma"]),
            cmap=str(params["cmap"]),
        )
        imageio.imwrite(_frame_dir(out_dir, i) / "vis.png", vis)
        last_vis = vis

    if last_vis is not None:
        imageio.imwrite(out_dir / FILM_LIVE, last_vis)
        # Keep frame 0 vis as the live plate when available.
        zpath = _frame_dir(out_dir, 0) / "vis.png"
        if zpath.is_file():
            imageio.imwrite(out_dir / FILM_LIVE, imageio.imread(zpath))
    save_params(out_dir, params)
    return params


def sample_depth(out_dir: Path | str, index: int, u: float, v: float) -> dict[str, Any]:
    from depth_core.pipeline import depth_percentile_at

    _rgb, depth = load_frame_arrays(out_dir, int(index))
    sample = depth_percentile_at(depth, u, v)
    sample["frame_index"] = int(index)
    return sample


def _update_ready_meta(
    out_dir: Path,
    ready_indices: list[int],
    status: str,
    *,
    infer_backend: Optional[str] = None,
    max_size: Optional[int] = None,
) -> None:
    from depth_core.infer_backend import resolve_backend

    meta = load_frames_meta(out_dir) or {}
    meta["pipeline"] = "film"
    meta["ready_indices"] = list(ready_indices)
    meta["ready_count"] = len(ready_indices)
    meta["infer_status"] = status
    backend = resolve_backend(infer_backend or meta.get("infer_backend"))
    meta["infer_backend"] = backend
    meta["infer_mode"] = (
        "modal_parallel" if backend == "modal" else "local_sequential"
    )
    if max_size is not None:
        meta["max_size"] = int(max_size)
    save_frames_meta(out_dir, meta)


def run_preview(
    out_dir: Path | str,
    params: Optional[dict[str, Any]] = None,
    *,
    model_key: Optional[str] = None,
    max_size: Optional[int] = None,
    infer_backend: Optional[str] = None,
    on_progress: ProgressCb = None,
    on_partial: PartialCb = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    meta = load_frames_meta(out_dir)
    if not meta:
        raise ValueError(f"No frames_meta.json under {out_dir}")
    params = normalize_params({**(load_params(out_dir)), **(params or {})})
    n = int(meta.get("frame_count") or 0)
    preview_n = min(int(params["preview_frames"]), n)
    indices = list(range(preview_n))
    key = str(model_key or meta.get("model_key") or "depth-pro")
    max_sz = int(max_size or meta.get("max_size") or 756)
    params, size_changed = _align_max_size(out_dir, params, max_sz, on_progress)
    meta = load_frames_meta(out_dir) or meta
    n = int(meta.get("frame_count") or n)
    preview_n = min(int(params["preview_frames"]), n)
    indices = list(range(preview_n))

    override_changed = False
    inferred = params.get("inferred_with_f_px")
    want_f = params.get("focallength_override") or (
        params.get("focallength_px") if params.get("anchor_focal") else None
    )
    if inferred is not None and want_f is not None and abs(float(inferred) - float(want_f)) > 0.5:
        override_changed = True

    params["phase"] = "running"
    save_params(out_dir, params)
    _update_ready_meta(
        out_dir, [], "running", infer_backend=infer_backend, max_size=max_sz
    )

    locked, ready, billing = infer_indices(
        out_dir,
        indices,
        params=params,
        model_key=key,
        max_size=max_sz,
        infer_backend=infer_backend,
        force=override_changed or size_changed,
        on_progress=on_progress,
        on_partial=on_partial,
    )
    depths = [load_frame_arrays(out_dir, i)[1] for i in ready]
    smin, smax = suggest_vis_bounds(depths)
    params["suggested_vis_min_m"] = smin
    params["suggested_vis_max_m"] = smax
    if not params.get("bounds_locked"):
        params["vis_min_m"] = smin
        params["vis_max_m"] = smax
    if locked is not None:
        params["focallength_px"] = float(locked)
        params["inferred_with_f_px"] = float(locked)
    params["preview_ready"] = len(ready)
    params["phase"] = "preview"
    save_params(out_dir, params)
    _update_ready_meta(
        out_dir, ready, "preview", infer_backend=infer_backend, max_size=max_sz
    )
    _progress(on_progress, 0.92, "Rendering preview vis…")
    render_vis(out_dir, params, ready)
    _progress(on_progress, 1.0, "Preview ready")
    if billing:
        params["modal_billing"] = billing
    return params


def run_full(
    out_dir: Path | str,
    params: Optional[dict[str, Any]] = None,
    *,
    model_key: Optional[str] = None,
    max_size: Optional[int] = None,
    infer_backend: Optional[str] = None,
    on_progress: ProgressCb = None,
    on_partial: PartialCb = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    meta = load_frames_meta(out_dir)
    if not meta:
        raise ValueError(f"No frames_meta.json under {out_dir}")
    params = normalize_params({**(load_params(out_dir)), **(params or {})})
    n = int(meta.get("frame_count") or 0)
    indices = list(range(n))
    key = str(model_key or meta.get("model_key") or "depth-pro")
    max_sz = int(max_size or meta.get("max_size") or 756)
    params, size_changed = _align_max_size(out_dir, params, max_sz, on_progress)
    meta = load_frames_meta(out_dir) or meta
    n = int(meta.get("frame_count") or n)
    indices = list(range(n))

    inferred = params.get("inferred_with_f_px")
    want_f = params.get("focallength_override") or params.get("focallength_px")
    force = bool(size_changed)
    if inferred is not None and want_f is not None and abs(float(inferred) - float(want_f)) > 0.5:
        force = True

    params["phase"] = "running"
    save_params(out_dir, params)
    _update_ready_meta(
        out_dir,
        [i for i in indices if _has_depth(out_dir, i)],
        "running",
        infer_backend=infer_backend,
        max_size=max_sz,
    )

    locked, ready, billing = infer_indices(
        out_dir,
        indices,
        params=params,
        model_key=key,
        max_size=max_sz,
        infer_backend=infer_backend,
        force=force,
        on_progress=on_progress,
        on_partial=on_partial,
    )
    if locked is not None:
        params["focallength_px"] = float(locked)
        params["inferred_with_f_px"] = float(locked)
    # Refresh suggested bounds from the full clip (do not overwrite locked user window).
    depths = [load_frame_arrays(out_dir, i)[1] for i in ready]
    smin, smax = suggest_vis_bounds(depths)
    params["suggested_vis_min_m"] = smin
    params["suggested_vis_max_m"] = smax
    params["preview_ready"] = len(ready)
    params["phase"] = "done"
    save_params(out_dir, params)
    _update_ready_meta(
        out_dir, ready, "done", infer_backend=infer_backend, max_size=max_sz
    )
    _progress(on_progress, 0.9, "Rendering locked vis…")
    render_vis(out_dir, params, ready)
    _progress(on_progress, 1.0, "Done")
    if billing:
        params["modal_billing"] = billing
    return params


def export_mp4(out_dir: Path | str, fps: Optional[float] = None) -> Path:
    out_dir = Path(out_dir)
    meta = load_frames_meta(out_dir) or {}
    n = int(meta.get("frame_count") or 0)
    frames = [_frame_dir(out_dir, i) / "vis.png" for i in range(n)]
    missing = [p for p in frames if not p.is_file()]
    if missing:
        raise ValueError(
            f"Missing vis frames ({len(missing)}). Infer and render before export."
        )
    export_dir = out_dir / "_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(frames):
        dest = export_dir / f"{i:06d}.png"
        dest.write_bytes(src.read_bytes())
    sample_fps = float(fps or meta.get("sample_fps") or VIDEO_TARGET_FPS)
    return export_frames_to_mp4(
        [export_dir / f"{i:06d}.png" for i in range(n)],
        export_dir / "film.mp4",
        sample_fps,
    )


def clip_info(out_dir: Path | str) -> dict[str, Any]:
    out_dir = Path(out_dir)
    meta = load_frames_meta(out_dir) or {}
    params = load_params(out_dir)
    n = int(meta.get("frame_count") or 0)
    ready = [i for i in range(n) if _has_depth(out_dir, i)]
    vis_ready = [
        i for i in range(n) if (_frame_dir(out_dir, i) / "vis.png").is_file()
    ]
    return {
        "pipeline": "film",
        "run_id": out_dir.name,
        "kind": "video",
        "frame_count": n,
        "ready_count": len(ready),
        "ready_indices": ready,
        "vis_ready": vis_ready,
        "infer_status": meta.get("infer_status") or params.get("phase") or "pending",
        "sample_fps": float(meta.get("sample_fps") or VIDEO_TARGET_FPS),
        "source_fps": meta.get("source_fps"),
        "duration_s": meta.get("duration_s"),
        "max_size": meta.get("max_size"),
        "frame_width": meta.get("frame_width"),
        "frame_height": meta.get("frame_height"),
        "model_key": meta.get("model_key"),
        "preview_gif": meta.get("preview_gif"),
        "source_file": meta.get("source_file"),
        "infer_mode": meta.get("infer_mode"),
        "infer_backend": meta.get("infer_backend"),
        "params": params,
    }
