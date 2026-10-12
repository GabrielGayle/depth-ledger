"""DepthPaper bundle read/write (viewer interchange format)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import imageio
import numpy as np

from depth_ledger.intrinsics import intrinsics_dict_from_matrix, scale_stray_camera_matrix
from depth_ledger.paths import (
    BUNDLE_CAMERA_NAME,
    BUNDLE_DEPTH_NAME,
    BUNDLE_FORMAT_VERSION,
    BUNDLE_RGB_NAME,
)
from depth_ledger.types import MetricDepthResult


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    focal_estimate_source: str
    f_px: float
    near_m: float
    far_m: float
    world_units: str = "meters"
    format_version: str = BUNDLE_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "f_px": self.f_px,
            "near_m": self.near_m,
            "far_m": self.far_m,
            "world_units": self.world_units,
            "depth_units": "millimeters",
            "focal_estimate_source": self.focal_estimate_source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraIntrinsics":
        fx = float(data["fx"])
        return cls(
            fx=fx,
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            width=int(data["width"]),
            height=int(data["height"]),
            focal_estimate_source=str(data.get("focal_estimate_source", "unknown")),
            f_px=float(data.get("f_px", fx)),
            near_m=float(data.get("near_m", 0.05)),
            far_m=float(data.get("far_m", 80.0)),
            world_units=str(data.get("world_units", "meters")),
            format_version=str(data.get("format_version", BUNDLE_FORMAT_VERSION)),
        )

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def _depth_range_m(depth_m: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(depth_m) & (depth_m > 1e-4)
    if not valid.any():
        return 0.05, 10.0
    vals = depth_m[valid]
    return float(np.percentile(vals, 2)), float(np.percentile(vals, 98))


def intrinsics_from_depth_result(
    result: MetricDepthResult,
    camera_matrix: Optional[np.ndarray] = None,
) -> CameraIntrinsics:
    h, w = result.depth_m.shape
    near_m, far_m = _depth_range_m(result.depth_m)

    if camera_matrix is not None:
        k = scale_stray_camera_matrix(camera_matrix, h, w)
        fields = intrinsics_dict_from_matrix(
            k, w, h, result.focal_estimate_source, result.depth_m
        )
        return CameraIntrinsics(
            fx=fields["fx"],
            fy=fields["fy"],
            cx=fields["cx"],
            cy=fields["cy"],
            width=w,
            height=h,
            focal_estimate_source=fields["focal_estimate_source"],
            f_px=fields["f_px"],
            near_m=fields["near_m"],
            far_m=fields["far_m"],
        )

    f = float(result.focal_length_px)
    return CameraIntrinsics(
        fx=f,
        fy=f,
        cx=(w - 1) * 0.5,
        cy=(h - 1) * 0.5,
        width=w,
        height=h,
        focal_estimate_source=result.focal_estimate_source,
        f_px=f,
        near_m=near_m,
        far_m=far_m,
    )


def save_depth_mm(depth_m: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_mm = np.clip(depth_m.astype(np.float64) * 1000.0, 0, 65535).astype(np.uint16)
    imageio.imwrite(path, depth_mm)
    return path


def load_depth_mm(path: Path) -> np.ndarray:
    depth_mm = imageio.imread(path)
    return depth_mm.astype(np.float32) / 1000.0


def write_depthpaper_bundle(
    out_dir: Path,
    result: MetricDepthResult,
    *,
    intrinsics: Optional[CameraIntrinsics] = None,
    camera_matrix: Optional[np.ndarray] = None,
) -> dict[str, Path]:
    """Write rgb.jpg, depth_mm.png, and camera.json into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam = intrinsics or intrinsics_from_depth_result(result, camera_matrix=camera_matrix)

    rgb_path = out_dir / BUNDLE_RGB_NAME
    depth_path = out_dir / BUNDLE_DEPTH_NAME
    camera_path = out_dir / BUNDLE_CAMERA_NAME

    imageio.imwrite(rgb_path, result.color)
    save_depth_mm(result.depth_m, depth_path)
    camera_path.write_text(json.dumps(cam.to_dict(), indent=2) + "\n", encoding="utf-8")

    return {"rgb": rgb_path, "depth_mm": depth_path, "camera": camera_path}


def read_depthpaper_bundle(bundle_dir: Path) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
    bundle_dir = Path(bundle_dir)
    rgb = imageio.imread(bundle_dir / BUNDLE_RGB_NAME)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    depth_m = load_depth_mm(bundle_dir / BUNDLE_DEPTH_NAME)
    cam = CameraIntrinsics.from_dict(
        json.loads((bundle_dir / BUNDLE_CAMERA_NAME).read_text(encoding="utf-8"))
    )
    return rgb, depth_m, cam


def bundle_from_image(
    image_path: Path,
    out_dir: Path,
    *,
    model_key: str,
    max_size: int,
    prompt_depth_path: Optional[Path] = None,
    camera_matrix: Optional[np.ndarray] = None,
) -> dict[str, Path]:
    from depth_ledger.models import is_promptda, run_metric_depth

    key = model_key
    if is_promptda(key) and camera_matrix is None:
        raise ValueError(
            "PromptDA bundles require --camera-matrix (Stray camera_matrix.csv)."
        )

    result = run_metric_depth(
        Path(image_path),
        model_key=key,
        max_size=max_size,
        prompt_depth_path=prompt_depth_path,
        camera_matrix=camera_matrix,
    )
    return write_depthpaper_bundle(
        out_dir, result, camera_matrix=camera_matrix if is_promptda(key) else None
    )
