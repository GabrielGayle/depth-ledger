"""Lift a DepthPaper bundle to a single-image 3D Gaussian splat (.ply + .spz)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from depth_ledger.bundle import CameraIntrinsics, read_depthpaper_bundle
from depth_ledger.geometry import (
    pixel_footprint_m,
    suppress_depth_discontinuities,
    unproject_depth,
)

try:
    import spz
except ImportError:  # pragma: no cover
    spz = None


@dataclass
class LiftOptions:
    stride: int = 2
    depth_min_m: float = 0.05
    depth_max_m: float = 80.0
    scale_relative_to_footprint: float = 0.75
    opacity: float = 0.92
    discontinuity_threshold: float = 0.05
    coordinate_system: str = "RDF"


def _inverse_sigmoid(x: float) -> float:
    x = float(np.clip(x, 1e-6, 1.0 - 1e-6))
    return math.log(x / (1.0 - x))


def _rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    c0 = 0.28209479177387814
    return (rgb.astype(np.float32) - 0.5) / c0


def lift_rgbd_to_gaussian_cloud(
    color: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    options: Optional[LiftOptions] = None,
):
    """Build an spz.GaussianCloud from aligned RGB + metric depth."""
    if spz is None:
        raise ImportError(
            "The `spz` package is required for lift. "
            "Install with: pip install 'git+https://github.com/nianticlabs/spz.git'"
        )

    opts = options or LiftOptions()
    stride = max(1, int(opts.stride))
    depth = depth_m.astype(np.float32)
    rgb = color.astype(np.uint8)

    if depth.shape[:2] != rgb.shape[:2]:
        raise ValueError("RGB and depth spatial sizes must match")

    depth = suppress_depth_discontinuities(
        depth,
        relative_threshold=float(opts.discontinuity_threshold),
    )

    depth_sub = depth[::stride, ::stride]
    rgb_sub = rgb[::stride, ::stride]
    ixt = intrinsics.as_matrix().copy()
    ixt[0, 0] /= stride
    ixt[1, 1] /= stride
    ixt[0, 2] /= stride
    ixt[1, 2] /= stride
    fx_eff = float(ixt[0, 0])

    points, colors = unproject_depth(
        depth_sub,
        ixt=ixt,
        depth_min=opts.depth_min_m,
        depth_max=opts.depth_max_m,
        color=rgb_sub,
        ret_pcd=False,
    )
    if points.shape[0] == 0:
        raise ValueError("No valid depth samples after filtering; adjust depth limits.")

    n = points.shape[0]
    z = np.clip(points[:, 2], opts.depth_min_m, None)
    footprint = pixel_footprint_m(z, fx_eff)
    rel = float(np.clip(opts.scale_relative_to_footprint, 0.5, 1.0))
    radius = np.clip(footprint * rel, 1e-4, 2.0)
    log_scale = np.log(radius).astype(np.float32)

    sh_dc = _rgb_to_sh_dc(colors)
    alpha = np.full(n, _inverse_sigmoid(opts.opacity), dtype=np.float32)
    rot = np.zeros((n, 4), dtype=np.float32)
    rot[:, 3] = 1.0

    cloud = spz.GaussianCloud()
    cloud.sh_degree = 0
    cloud.antialiased = False
    cloud.positions = np.ascontiguousarray(points.astype(np.float32).reshape(-1))
    cloud.scales = np.ascontiguousarray(
        np.tile(log_scale[:, None], (1, 3)).astype(np.float32).reshape(-1)
    )
    cloud.rotations = np.ascontiguousarray(rot.reshape(-1))
    cloud.alphas = np.ascontiguousarray(alpha)
    cloud.colors = np.ascontiguousarray(sh_dc.reshape(-1))
    cloud.sh = np.ascontiguousarray(np.array([], dtype=np.float32))
    return cloud


def _pack_options_from_lift(opts: LiftOptions) -> "spz.PackOptions":
    pack = spz.PackOptions()
    coord = getattr(spz, opts.coordinate_system, spz.RDF)
    pack.from_coord = coord
    return pack


def write_gaussian_splat(
    cloud,
    ply_path: Path,
    spz_path: Path,
    options: Optional[LiftOptions] = None,
) -> tuple[Path, Path]:
    opts = options or LiftOptions()
    pack = _pack_options_from_lift(opts)
    ply_path = Path(ply_path)
    spz_path = Path(spz_path)
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    if not spz.save_splat_to_ply(cloud, pack, str(ply_path)):
        raise RuntimeError(f"Failed to write PLY: {ply_path}")
    if not spz.save_spz(cloud, pack, str(spz_path)):
        raise RuntimeError(f"Failed to write SPZ: {spz_path}")
    return ply_path, spz_path


def lift_bundle(
    bundle_dir: Path,
    ply_path: Path,
    spz_path: Path,
    options: Optional[LiftOptions] = None,
) -> tuple[Path, Path]:
    rgb, depth_m, cam = read_depthpaper_bundle(bundle_dir)
    cloud = lift_rgbd_to_gaussian_cloud(rgb, depth_m, cam, options)
    return write_gaussian_splat(cloud, ply_path, spz_path, options)
