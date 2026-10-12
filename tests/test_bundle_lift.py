"""Tests for DepthPaper bundle I/O and splat lift (no GPU weights required)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import imageio
import numpy as np
import pytest

from depth_ledger.bundle import (
    CameraIntrinsics,
    load_depth_mm,
    read_depthpaper_bundle,
    save_depth_mm,
    write_depthpaper_bundle,
)
from depth_ledger.geometry import pixel_footprint_m
from depth_ledger.paths import BUNDLE_CAMERA_NAME, BUNDLE_DEPTH_NAME, BUNDLE_RGB_NAME
from depth_ledger.types import MetricDepthResult


def _synthetic_result(h: int = 64, w: int = 80) -> MetricDepthResult:
    yy, xx = np.mgrid[0:h, 0:w]
    depth = (1.0 + 0.01 * xx).astype(np.float32)
    color = np.stack(
        [
            (xx % 256).astype(np.uint8),
            (yy % 256).astype(np.uint8),
            np.full((h, w), 128, dtype=np.uint8),
        ],
        axis=-1,
    )
    return MetricDepthResult(
        color=color,
        depth_m=depth,
        model_key="depth-pro",
        focal_length_px=float(w),
        focal_estimate_source="default_pinhole",
    )


def test_depthpaper_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "bundle"
        result = _synthetic_result()
        write_depthpaper_bundle(out, result)
        assert (out / BUNDLE_RGB_NAME).is_file()
        assert (out / BUNDLE_DEPTH_NAME).is_file()
        assert (out / BUNDLE_CAMERA_NAME).is_file()

        rgb, depth_m, cam = read_depthpaper_bundle(out)
        assert rgb.shape[:2] == depth_m.shape
        assert cam.width == result.depth_m.shape[1]
        assert cam.height == result.depth_m.shape[0]
        np.testing.assert_allclose(depth_m, result.depth_m, atol=0.002)


def test_depth_mm_png_is_uint16_millimetres():
    depth_m = np.array([[0.5, 1.0], [2.0, 0.0]], dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "d.png"
        save_depth_mm(depth_m, path)
        raw = imageio.imread(path)
        assert raw.dtype == np.uint16
        restored = load_depth_mm(path)
        assert restored[0, 0] == pytest.approx(0.5)
        assert restored[0, 1] == pytest.approx(1.0)


def test_camera_json_schema_fields():
    cam = CameraIntrinsics(
        fx=100.0,
        fy=100.0,
        cx=40.0,
        cy=30.0,
        width=80,
        height=60,
        focal_estimate_source="exif",
        f_px=100.0,
        near_m=0.5,
        far_m=5.0,
    )
    data = cam.to_dict()
    assert data["format_version"] == "1.0"
    assert data["depth_units"] == "millimeters"
    assert data["world_units"] == "meters"
    assert data["f_px"] == 100.0
    assert set(data) >= {
        "fx",
        "fy",
        "cx",
        "cy",
        "width",
        "height",
        "focal_estimate_source",
        "near_m",
        "far_m",
    }


def test_lift_writes_ply_and_spz():
    spz = pytest.importorskip("spz")
    from depth_ledger.lift import lift_bundle

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "bundle"
        write_depthpaper_bundle(bundle, _synthetic_result())
        ply_path = bundle / "splat.ply"
        spz_path = bundle / "splat.spz"
        lift_bundle(bundle, ply_path, spz_path)

        assert ply_path.stat().st_size > 100
        assert spz_path.stat().st_size > 50

        with open(ply_path, "rb") as f:
            assert f.read(3) == b"ply"
        header = ply_path.read_text(encoding="latin-1").split("end_header")[0]
        assert "format binary_little_endian 1.0" in header
        assert "element vertex" in header

        loaded = spz.load_spz(str(spz_path), spz.UnpackOptions())
        assert loaded.num_points > 0


def test_ply_vertex_count_matches_header():
    spz = pytest.importorskip("spz")
    from depth_ledger.lift import LiftOptions, lift_bundle

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "bundle"
        write_depthpaper_bundle(bundle, _synthetic_result(h=32, w=32))
        ply_path = bundle / "splat.ply"
        spz_path = bundle / "splat.spz"
        lift_bundle(
            bundle,
            ply_path,
            spz_path,
            options=LiftOptions(stride=2, discontinuity_threshold=0.0),
        )

        text = ply_path.read_text(encoding="latin-1", errors="replace")
        n_declared = int(
            next(line for line in text.splitlines() if line.startswith("element vertex ")).split()[-1]
        )
        reloaded = spz.load_splat_from_ply(str(ply_path), spz.UnpackOptions())
        assert reloaded.num_points == n_declared


def test_splat_scale_within_twice_pixel_footprint():
    spz = pytest.importorskip("spz")
    from depth_ledger.bundle import CameraIntrinsics
    from depth_ledger.lift import LiftOptions, lift_rgbd_to_gaussian_cloud

    h, w = 40, 40
    depth_m = np.full((h, w), 2.0, dtype=np.float32)
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[..., :] = 128
    cam = CameraIntrinsics(
        fx=80.0,
        fy=80.0,
        cx=19.5,
        cy=19.5,
        width=w,
        height=h,
        focal_estimate_source="default_pinhole",
        f_px=80.0,
        near_m=1.0,
        far_m=3.0,
    )
    cloud = lift_rgbd_to_gaussian_cloud(
        color,
        depth_m,
        cam,
        LiftOptions(stride=1, discontinuity_threshold=0.0, scale_relative_to_footprint=0.75),
    )
    linear_scale = np.exp(cloud.scales.reshape(-1, 3)[:, 0])
    footprint = pixel_footprint_m(2.0, 80.0)
    assert np.all(linear_scale >= footprint * 0.5)
    assert np.all(linear_scale <= footprint * 2.0)


def test_import_bundle_without_torch():
    import subprocess
    import sys

    subprocess.check_call(
        [
            sys.executable,
            "-c",
            "import depth_ledger.bundle; import depth_ledger.lift; import depth_ledger.geometry",
        ]
    )
