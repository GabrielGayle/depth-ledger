"""Evaluate Operator patches against cached depth runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import imageio
import numpy as np

from depth_core.operator import (
    default_operator_patch,
    depth_to_vector_field,
    maybe_send_ndi,
    normalize_patch,
    operator_meta,
    render_operator,
    write_spread_ply,
)
from depth_core.pipeline import _downscale_thumb
from depth_core.video import export_frames_to_mp4
from server.cache import get_run, set_frame, update_run

OSC_THUMB_MAX = 128
_DEFAULT_PATCH_PATH = (
    Path(__file__).resolve().parent.parent / "outputs" / "_default_operator.json"
)


def load_default_operator_patch() -> Optional[dict[str, Any]]:
    if not _DEFAULT_PATCH_PATH.is_file():
        return None
    try:
        data = json.loads(_DEFAULT_PATCH_PATH.read_text(encoding="utf-8"))
        return normalize_patch(data if isinstance(data, dict) else {})
    except Exception:
        return None


def save_default_operator_patch(patch: dict[str, Any]) -> Path:
    _DEFAULT_PATCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_patch(patch)
    # Don't persist ephemeral time cursor
    normalized["t"] = None
    _DEFAULT_PATCH_PATH.write_text(
        json.dumps(normalized, indent=2), encoding="utf-8"
    )
    return _DEFAULT_PATCH_PATH


def clear_default_operator_patch() -> bool:
    if _DEFAULT_PATCH_PATH.is_file():
        _DEFAULT_PATCH_PATH.unlink()
        return True
    return False


def get_operator_default() -> dict[str, Any]:
    custom = load_default_operator_patch()
    return {
        "has_custom": custom is not None,
        "patch": custom if custom is not None else default_operator_patch(),
    }


def evaluate_operator(run_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    """Cook Operator patch → operator_live.png + osc thumbs + optional voices."""
    t0 = time.perf_counter()
    state = get_run(run_id)
    if state is None:
        raise KeyError(f"Run not found: {run_id}")

    depth = state["depth"]
    photo = state.get("rgb")
    out_dir = Path(state["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    result = render_operator(depth, patch, photo=photo)
    norm_patch = result["patch"]
    master = result["rgb"]

    live_name = "operator_live.png"
    imageio.imwrite(out_dir / live_name, master)

    osc_previews: dict[str, str] = {}
    for k, rgb in result["osc_rgbs"].items():
        thumb = _downscale_thumb(rgb, OSC_THUMB_MAX)
        name = f"operator_osc_{k}_thumb.png"
        imageio.imwrite(out_dir / name, thumb)
        osc_previews[k] = f"/api/runs/{run_id}/files/{name}"

    voices_out: dict[str, Optional[str]] = {"plate": f"/api/runs/{run_id}/files/{live_name}"}
    enabled = set((norm_patch.get("global") or {}).get("voices") or ["plate"])

    if "vector" in enabled:
        dx, dy, viz = depth_to_vector_field(depth)
        np.savez_compressed(out_dir / "operator_vector.npz", dx=dx, dy=dy)
        imageio.imwrite(out_dir / "operator_vector.png", viz)
        voices_out["vector"] = f"/api/runs/{run_id}/files/operator_vector.png"
        voices_out["vector_npz"] = f"/api/runs/{run_id}/files/operator_vector.npz"
    else:
        voices_out["vector"] = None
        voices_out["vector_npz"] = None

    if "ply" in enabled:
        color = photo if photo is not None else master
        if color.shape[:2] != depth.shape[:2]:
            # resize color to depth
            import cv2

            color = cv2.resize(
                color, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA
            )
        ply_name = "operator_pointcloud.ply"
        write_spread_ply(
            depth, color, str(out_dir / ply_name), spread=float(result["spread"])
        )
        voices_out["ply"] = f"/api/runs/{run_id}/files/{ply_name}"
    else:
        voices_out["ply"] = None

    ndi_sent = False
    if "ndi" in enabled:
        ndi_sent = maybe_send_ndi(master)
    voices_out["ndi"] = "sent" if ndi_sent else None

    # Persist patch on run for export / stream
    update_run(
        run_id,
        operator_patch=norm_patch,
        operator_live=str(out_dir / live_name),
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "run_id": run_id,
        "output_url": f"/api/runs/{run_id}/files/{live_name}",
        "elapsed_ms": elapsed_ms,
        "osc_previews": osc_previews,
        "voices": voices_out,
        "patch": norm_patch,
        "spread": result["spread"],
        "ndi_sent": ndi_sent,
    }


def export_operator_video(run_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    """Bake Operator across video frames with time-varying t."""
    state = get_run(run_id)
    if state is None:
        raise KeyError(f"Run not found: {run_id}")
    frame_count = int(state.get("frame_count") or 1)
    if frame_count <= 1:
        raise ValueError("Export requires a multi-frame video run")

    out_dir = Path(state["out_dir"])
    frames_dir = out_dir / "operator_export_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    base = normalize_patch(patch)
    saved_index = int(state.get("frame_index") or 0)

    try:
        for i in range(frame_count):
            set_frame(run_id, i)
            t = 0.0 if frame_count <= 1 else i / (frame_count - 1)
            frame_patch = dict(base)
            frame_patch["t"] = t
            result = evaluate_operator(run_id, frame_patch)
            src = out_dir / "operator_live.png"
            dest = frames_dir / f"{i:06d}.png"
            dest.write_bytes(src.read_bytes())
            _ = result
    finally:
        set_frame(run_id, saved_index)
        # restore plate for saved frame
        restore = dict(base)
        n = max(frame_count - 1, 1)
        restore["t"] = saved_index / n
        evaluate_operator(run_id, restore)

    sample_fps = float(state.get("sample_fps") or 10.0)
    mp4_name = "operator_export.mp4"
    mp4_path = out_dir / mp4_name
    frame_pngs = [frames_dir / f"{i:06d}.png" for i in range(frame_count)]
    export_frames_to_mp4(frame_pngs, mp4_path, fps=sample_fps)
    return {
        "run_id": run_id,
        "export_url": f"/api/runs/{run_id}/files/{mp4_name}",
        "frame_count": frame_count,
    }


def operator_stream_jpeg(run_id: str) -> Optional[bytes]:
    """Latest operator_live.png bytes for MJPEG consumers."""
    state = get_run(run_id)
    if state is None:
        return None
    path = Path(state["out_dir"]) / "operator_live.png"
    if not path.is_file():
        return None
    return path.read_bytes()
