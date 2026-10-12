"""Camera intrinsics loading and scaling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

STRAY_REF_MAX_DIMENSION = 1920.0


def load_camera_matrix_csv(path: Path | str) -> np.ndarray:
    """Load a 3×3 pinhole matrix from Stray Scanner ``camera_matrix.csv``."""
    matrix = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected 3×3 camera matrix, got {matrix.shape}")
    return matrix


def scale_stray_camera_matrix(
    ixt: np.ndarray,
    output_height: int,
    output_width: int,
    ref_max_dimension: float = STRAY_REF_MAX_DIMENSION,
) -> np.ndarray:
    """Scale Stray-style intrinsics to match inference output resolution."""
    k = np.asarray(ixt, dtype=np.float64).copy()
    now_max = max(int(output_height), int(output_width))
    scale = ref_max_dimension / float(now_max)
    k[:2, :] /= scale
    return k


def intrinsics_dict_from_matrix(
    ixt: np.ndarray,
    width: int,
    height: int,
    focal_estimate_source: str,
    depth_m: np.ndarray,
) -> dict:
    k = np.asarray(ixt, dtype=np.float64)
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    valid = np.isfinite(depth_m) & (depth_m > 1e-4)
    if valid.any():
        vals = depth_m[valid]
        near_m = float(np.percentile(vals, 2))
        far_m = float(np.percentile(vals, 98))
    else:
        near_m, far_m = 0.05, 10.0
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "f_px": fx,
        "width": int(width),
        "height": int(height),
        "near_m": near_m,
        "far_m": far_m,
        "world_units": "meters",
        "depth_units": "millimeters",
        "focal_estimate_source": focal_estimate_source,
    }
