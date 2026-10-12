"""Geometry helpers (no torch)."""

from __future__ import annotations

import numpy as np


def unproject_depth(
    depth,
    ixt,
    depth_min=0.01,
    depth_max=None,
    color=None,
    ext=None,
    conf=None,
    ret_pcd=False,
    clip_box=None,
):
    """Unproject a depth map to 3D points (OpenCV camera, +Z forward)."""
    height, width = depth.shape
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    zz = depth.reshape(-1)
    mask = np.ones_like(xx, dtype=np.bool_)
    if depth_min is not None:
        mask &= zz >= depth_min
    if depth_max is not None:
        mask &= zz <= depth_max
    if conf is not None:
        mask &= conf.reshape(-1) == 2
    xx = xx[mask]
    yy = yy[mask]
    zz = zz[mask]
    pcd = np.stack([xx, yy, np.ones_like(xx)], axis=1)
    pcd = pcd * zz[:, None]
    pcd = np.dot(pcd, np.linalg.inv(ixt).T)
    if ext is not None:
        pcd = np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1)
        pcd = np.dot(pcd, np.linalg.inv(ext).T)
    new_mask = np.ones_like(pcd[:, 0]).astype(np.bool_)
    if clip_box is not None:
        assert len(clip_box) == 6
        for i, val in enumerate(clip_box):
            if val is None:
                continue
            if i == 0:
                new_mask &= pcd[:, 0] <= val
            elif i == 1:
                new_mask &= pcd[:, 1] <= val
            elif i == 2:
                new_mask &= pcd[:, 2] <= val
            elif i == 3:
                new_mask &= pcd[:, 0] >= val
            elif i == 4:
                new_mask &= pcd[:, 1] >= val
            elif i == 5:
                new_mask &= pcd[:, 2] >= val
    if color is not None:
        if color.dtype == np.uint8:
            color = color.astype(np.float32) / 255.0
        if ret_pcd:
            try:
                import open3d as o3d
            except ImportError as exc:
                raise ImportError("open3d is required when ret_pcd=True") from exc
            points = pcd
            out = o3d.geometry.PointCloud()
            out.points = o3d.utility.Vector3dVector(points[:, :3][new_mask])
            out.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3)[mask][new_mask])
            return out
        return pcd[:, :3][new_mask], color.reshape(-1, 3)[mask][new_mask]
    if ret_pcd:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError("open3d is required when ret_pcd=True") from exc
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(pcd[:, :3][new_mask])
        return out
    return pcd[:, :3][new_mask]


def pixel_footprint_m(depth_m: np.ndarray, fx: float) -> np.ndarray:
    """Approximate world-space pixel width at each depth (metres)."""
    z = np.asarray(depth_m, dtype=np.float64)
    return z / max(float(fx), 1e-6)


def suppress_depth_discontinuities(
    depth_m: np.ndarray,
    *,
    relative_threshold: float,
    absolute_threshold_m: float = 0.03,
) -> np.ndarray:
    """Zero depth at pixels that jump sharply vs neighbours (rubber-sheet edges)."""
    if relative_threshold <= 0:
        return depth_m.astype(np.float32, copy=True)

    depth = depth_m.astype(np.float32, copy=True)
    valid = np.isfinite(depth) & (depth > 1e-4)
    if not valid.any():
        return depth

    jump = np.zeros_like(depth, dtype=np.float32)
    for shifted, axis in (
        (np.roll(depth, -1, axis=1), 0),
        (np.roll(depth, 1, axis=1), 0),
        (np.roll(depth, -1, axis=0), 1),
        (np.roll(depth, 1, axis=0), 1),
    ):
        denom = np.maximum(np.maximum(np.abs(depth), np.abs(shifted)), 1e-4)
        rel = np.abs(depth - shifted) / denom
        jump = np.maximum(jump, rel)

    abs_jump = np.zeros_like(depth, dtype=np.float32)
    for shifted in (
        np.roll(depth, -1, axis=1),
        np.roll(depth, 1, axis=1),
        np.roll(depth, -1, axis=0),
        np.roll(depth, 1, axis=0),
    ):
        abs_jump = np.maximum(abs_jump, np.abs(depth - shifted))

    bad = valid & (
        (jump > float(relative_threshold))
        | (abs_jump > float(absolute_threshold_m))
    )
    depth[bad] = 0.0
    return depth
