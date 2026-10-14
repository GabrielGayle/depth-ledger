"""Stub video-depth stream for frequency fusion testing.

Writes temporally EMA'd Depth Pro maps as frames/*/depth_video.npz so
freq_fuse can exercise align + Laplacian blend without ChronoDepth/DepthCrafter.
Swap build_stub_video_depth for a real video-depth backend later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from depth_core.temporal import ema_blend_depth
from depth_core.video import frames_meta_path, load_frame_arrays, load_frames_meta, save_frames_meta

STUB_BACKEND = "stub_ema"
STUB_EMA_ALPHA = 0.35  # weight on current Pro frame when building video stream
DEPTH_VIDEO_NAME = "depth_video.npz"


def _frame_dir(out_dir: Path, index: int) -> Path:
    return Path(out_dir) / "frames" / f"{index:06d}"


def depth_video_path(out_dir: Path | str, index: int) -> Path:
    return _frame_dir(Path(out_dir), index) / DEPTH_VIDEO_NAME


def video_depth_ready(out_dir: Path | str, frame_count: int) -> bool:
    """True if all ready frames have depth_video.npz."""
    out = Path(out_dir)
    meta = load_frames_meta(out) or {}
    n = int(meta.get("ready_count") or frame_count or 0)
    if n < 1:
        return False
    for i in range(n):
        if not depth_video_path(out, i).is_file():
            return False
    return True


def load_video_depth(out_dir: Path | str, index: int) -> np.ndarray:
    path = depth_video_path(out_dir, index)
    if not path.is_file():
        raise FileNotFoundError(f"Missing video depth frame {index}: {path}")
    return np.load(path)["depth"].astype(np.float32)


def build_stub_video_depth(
    out_dir: Path | str,
    *,
    ema_alpha: float = STUB_EMA_ALPHA,
    force: bool = False,
) -> dict[str, Any]:
    """Build stub video-depth from Depth Pro frames via temporal EMA.

    Returns metadata written into frames_meta.json under video_depth.
    """
    out = Path(out_dir)
    meta = load_frames_meta(out) or {}
    ready = int(meta.get("ready_count") or meta.get("frame_count") or 0)
    if ready < 1:
        raise ValueError("No Depth Pro frames ready for stub video depth")

    existing = meta.get("video_depth") if isinstance(meta.get("video_depth"), dict) else {}
    if (
        not force
        and existing.get("backend") == STUB_BACKEND
        and int(existing.get("frame_count") or 0) >= ready
        and video_depth_ready(out, ready)
    ):
        return dict(existing)

    prev: Optional[np.ndarray] = None
    for i in range(ready):
        _rgb, depth = load_frame_arrays(out, i)
        depth = depth.astype(np.float32)
        if prev is None:
            smoothed = depth
        else:
            smoothed = ema_blend_depth(depth, prev, float(ema_alpha))
        dest = depth_video_path(out, i)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, depth=smoothed.astype(np.float32))
        prev = smoothed

    info: dict[str, Any] = {
        "backend": STUB_BACKEND,
        "label": "stub EMA of Depth Pro (placeholder for ChronoDepth)",
        "ema_alpha": float(ema_alpha),
        "frame_count": ready,
    }
    meta["video_depth"] = info
    save_frames_meta(out, meta)
    # Touch a tiny sidecar for debugging
    (out / "video_depth_stub.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    return info


def ensure_stub_video_depth(out_dir: Path | str, frame_count: int) -> dict[str, Any]:
    """Lazy-build stub if missing; return video_depth meta."""
    out = Path(out_dir)
    meta = load_frames_meta(out) or {}
    ready = int(meta.get("ready_count") or frame_count or 0)
    if ready < 1:
        raise ValueError("No frames ready for video depth stub")
    if video_depth_ready(out, ready):
        vd = meta.get("video_depth") if isinstance(meta.get("video_depth"), dict) else {}
        if vd:
            return dict(vd)
        return {"backend": STUB_BACKEND, "frame_count": ready}
    return build_stub_video_depth(out)
