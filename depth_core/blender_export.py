"""Export metric depth as float EXR + near/far JSON for Blender GN/Shader."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

# OpenCV builds ship with EXR disabled until this env flag is set.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

from depth_core.pipeline import (
    LOOK_PRESETS,
    _prepare_inv_smooth,
    _resolve_hi_pct,
    _resolve_lo_pct,
    _subject_hi_pct,
    depth_valid_mask,
)


def compute_near_far(
    depth_m: np.ndarray,
    *,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
    near_m_clip: float = 0.1,
    far_m_clip: float = 250.0,
    near_override: Optional[float] = None,
    far_override: Optional[float] = None,
    cliff: Optional[float] = None,
) -> dict[str, Any]:
    """Percentile / override window matching `_visualize_inverse_depth`."""
    depth = np.asarray(depth_m, dtype=np.float64)
    valid = depth_valid_mask(depth)
    if cliff is None and valid.any():
        cliff = float(_subject_hi_pct(depth))
    if near_override is not None and far_override is not None:
        near = float(near_override)
        far = float(max(far_override, near + 1e-3))
    elif valid.any():
        near = float(np.percentile(depth[valid], lo_pct))
        far = float(np.percentile(depth[valid], hi_pct))
    else:
        near, far = float(near_m_clip), float(far_m_clip)
    near = float(np.clip(near, near_m_clip, far_m_clip - 1e-3))
    far = float(np.clip(max(far, near + 1e-3), near + 1e-3, far_m_clip))
    return {
        "near_m": near,
        "far_m": far,
        "lo_pct": float(lo_pct),
        "hi_pct": float(hi_pct),
        "cliff": float(cliff) if cliff is not None else None,
        "inv_bilateral": True,
        "unit": "meters",
    }


def near_far_for_preset(
    depth_m: np.ndarray,
    preset: str,
    *,
    cliff: Optional[float] = None,
    near_override: Optional[float] = None,
    far_override: Optional[float] = None,
) -> dict[str, Any]:
    cfg = LOOK_PRESETS.get(preset) or LOOK_PRESETS["ledger"]
    if cliff is None:
        cliff = float(_subject_hi_pct(depth_m))
    lo = _resolve_lo_pct(cfg, cliff)
    hi = _resolve_hi_pct(cfg, cliff)
    meta = compute_near_far(
        depth_m,
        lo_pct=lo,
        hi_pct=hi,
        near_override=near_override,
        far_override=far_override,
        cliff=cliff,
    )
    meta["preset"] = preset
    meta["gamma"] = float(cfg.get("gamma", 1.0))
    meta["grade"] = cfg.get("grade")
    meta["cmap"] = cfg.get("cmap", "gray")
    return meta


def write_float_exr(path: Path | str, array: np.ndarray) -> Path:
    """Write single-channel float32 EXR (OpenCV)."""
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        # OpenCV EXR expects HxWxC; use 3 identical channels for Blender Image.
        out = np.dstack([arr, arr, arr])
    elif arr.ndim == 3 and arr.shape[2] == 1:
        out = np.repeat(arr, 3, axis=2)
    else:
        out = arr.astype(np.float32)
    ok = cv2.imwrite(
        str(path),
        out,
        [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT],
    )
    if not ok:
        raise RuntimeError(f"Failed to write EXR: {path}")
    return path


def export_blender_bundle(
    out_dir: Path | str,
    run_id: str,
    depth_m: np.ndarray,
    *,
    rgb: Optional[np.ndarray] = None,
    cliff: Optional[float] = None,
    preset: str = "ledger",
    near_override: Optional[float] = None,
    far_override: Optional[float] = None,
    write_inv: bool = True,
    write_presets: bool = True,
) -> dict[str, Any]:
    """Write `{run}_depth.exr`, `{run}_inv_smooth.exr`, `{run}_near_far.json`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    depth = np.asarray(depth_m, dtype=np.float32)

    depth_path = write_float_exr(out_dir / f"{run_id}_depth.exr", depth)
    _, inv = _prepare_inv_smooth(depth)
    inv_path = None
    if write_inv:
        inv_path = write_float_exr(out_dir / f"{run_id}_inv_smooth.exr", inv)

    if cliff is None:
        cliff = float(_subject_hi_pct(depth))

    near_far = near_far_for_preset(
        depth,
        preset,
        cliff=cliff,
        near_override=near_override,
        far_override=far_override,
    )
    near_far["run_id"] = run_id

    presets: dict[str, Any] = {}
    if write_presets:
        for key in LOOK_PRESETS:
            if key == "scenery":
                continue
            presets[key] = near_far_for_preset(
                depth,
                key,
                cliff=cliff,
                near_override=near_override,
                far_override=far_override,
            )
        near_far["presets"] = presets

    nf_path = out_dir / f"{run_id}_near_far.json"
    nf_path.write_text(json.dumps(near_far, indent=2), encoding="utf-8")

    rgb_path = out_dir / f"{run_id}_rgb.jpg"
    if not rgb_path.is_file():
        alt = out_dir / "rgb.jpg"
        rgb_path = alt if alt.is_file() else rgb_path
    result: dict[str, Any] = {
        "run_id": run_id,
        "depth_exr": str(depth_path),
        "inv_smooth_exr": str(inv_path) if inv_path else None,
        "near_far_json": str(nf_path),
        "near_far": near_far,
        "rgb": str(rgb_path) if rgb_path.is_file() else None,
        "cliff": cliff,
    }
    return result


def export_matte_exr(
    out_dir: Path | str,
    run_id: str,
    matte: np.ndarray,
) -> Path:
    """Bake a soft/smart scenery matte as float EXR for Blender DL_Matte hybrid."""
    out_dir = Path(out_dir)
    arr = np.asarray(matte, dtype=np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return write_float_exr(out_dir / f"{run_id}_matte.exr", arr)


def export_relief_png(
    out_dir: Path | str,
    run_id: str,
    plate: np.ndarray,
    *,
    gc_alpha: float = 10.0,
    invert: bool = False,
    max_side: int = 512,
) -> Path:
    from depth_core.relief import render_bas_relief

    out_dir = Path(out_dir)
    relief = render_bas_relief(
        plate, gc_alpha=gc_alpha, invert=invert, max_side=max_side
    )
    path = out_dir / f"{run_id}_relief.png"
    import imageio

    imageio.imwrite(path, relief)
    return path


def export_npz(
    out_dir: Path | str,
    run_id: str,
    depth_m: np.ndarray,
    *,
    rgb: Optional[np.ndarray] = None,
    output: Optional[np.ndarray] = None,
    cliff: Optional[float] = None,
    preset: str = "ledger",
) -> Path:
    """Write `{run}_export.npz`: float depth (m), inv_smooth, valid mask, plates, window."""
    out_dir = Path(out_dir)
    depth = np.asarray(depth_m, dtype=np.float32)
    _, inv = _prepare_inv_smooth(depth)
    if cliff is None:
        cliff = float(_subject_hi_pct(depth))
    nf = near_far_for_preset(depth, preset, cliff=cliff)
    arrays: dict[str, np.ndarray] = {
        "depth": depth,
        "inv_smooth": np.asarray(inv, dtype=np.float32),
        "valid": depth_valid_mask(depth),
        "near_m": np.float32(nf["near_m"]),
        "far_m": np.float32(nf["far_m"]),
        "cliff": np.float32(cliff),
    }
    if rgb is not None:
        arrays["rgb"] = np.asarray(rgb, dtype=np.uint8)
    if output is not None:
        arrays["output"] = np.asarray(output, dtype=np.uint8)
    path = out_dir / f"{run_id}_export.npz"
    np.savez_compressed(path, **arrays)
    return path


def export_frame_sequence_meta(
    out_dir: Path | str,
    run_id: str,
    frame_windows: list[dict[str, Any]],
) -> Path:
    """Write prebaked temporal near/far sequence for Blender Image Sequence."""
    out_dir = Path(out_dir)
    path = out_dir / f"{run_id}_near_far_sequence.json"
    payload = {
        "run_id": run_id,
        "frame_count": len(frame_windows),
        "frames": frame_windows,
        "note": "API-prebaked windows; feed matching inv EXR sequence into DL_Look",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
