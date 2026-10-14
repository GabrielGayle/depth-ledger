"""Modal GPU worker — depth forward-pass only.

Deploy:
  modal deploy modal_app/worker.py
"""

from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import modal

APP_NAME = "depth-ledger"
GPU_TYPE = "T4"
CHECKPOINT_URL = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
CHECKPOINT_VOL_PATH = "/checkpoints"

# Image: torch + project deps + DepthPro/PromptDA sources.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "imageio>=2.33.1",
        "Pillow>=10.1.0,<11.0",
        "imageio-ffmpeg",
        "einops",
        "tqdm",
        "transformers",
        "termcolor",
        "numpy>=1.26.0,<2",
        "opencv-python-headless>=4.9.0",
        "scipy",
        "matplotlib",
        "h5py",
        "huggingface_hub",
        "timm",
        "pillow_heif>=0.13.0,<1.0",
        "trimesh",
    )
    .env({"PROMPTDA_DEVICE": "cuda"})
    .add_local_dir("DepthPro", remote_path="/root/DepthPro", copy=True)
    .add_local_dir("PromptDA", remote_path="/root/PromptDA", copy=True)
    .add_local_dir("depth_core", remote_path="/root/depth_core", copy=True)
    .run_commands(
        "pip install -e /root/DepthPro",
        "pip install -e /root/PromptDA",
    )
)

checkpoints = modal.Volume.from_name(
    "depth-ledger-checkpoints", create_if_missing=True
)

app = modal.App(APP_NAME, image=image)


def _encode_depth(depth) -> bytes:
    import numpy as np

    buf = io.BytesIO()
    np.savez_compressed(buf, depth=np.asarray(depth, dtype=np.float32))
    return buf.getvalue()


def _encode_rgb(color) -> bytes:
    import imageio

    buf = io.BytesIO()
    imageio.imwrite(buf, color, format="JPEG", quality=92)
    return buf.getvalue()


def _ensure_checkpoint() -> Path:
    dest = Path(CHECKPOINT_VOL_PATH) / "depth_pro.pt"
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Depth Pro checkpoint → {dest}…")
    import urllib.request

    tmp = dest.with_suffix(".pt.partial")
    urllib.request.urlretrieve(CHECKPOINT_URL, tmp)
    tmp.rename(dest)
    try:
        checkpoints.commit()
    except Exception:
        pass
    return dest


@app.cls(
    gpu=GPU_TYPE,
    timeout=600,
    scaledown_window=300,
    max_containers=16,
    volumes={CHECKPOINT_VOL_PATH: checkpoints},
)
class DepthWorker:
    @modal.enter()
    def load(self):
        os.environ["PROMPTDA_DEVICE"] = "cuda"
        # Point Depth Pro at the volume checkpoint.
        ckpt = _ensure_checkpoint()
        os.environ.setdefault("DEPTH_PRO_CHECKPOINT", str(ckpt))

        import sys

        sys.path.insert(0, "/root")
        # Override depth_core checkpoint path for Modal volume.
        import depth_core.pipeline as pipe

        pipe.CHECKPOINT_PATH = ckpt
        self._pipe = pipe
        self._backends: dict[str, Any] = {}
        self._cold = True
        print("DepthWorker ready")

    def _get_backend(self, model_key: str):
        key = self._pipe.normalize_model_key(model_key)
        if key not in self._backends:
            self._backends[key] = self._pipe.get_backend(key)
        return self._backends[key]

    def _billing(self, t0: float) -> dict[str, Any]:
        cold = self._cold
        self._cold = False
        return {
            "gpu_type": GPU_TYPE,
            "gpu_seconds": round(time.perf_counter() - t0, 4),
            "cold_start": cold,
            "device": "cuda",
        }

    @modal.method()
    def ping(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        # Touch Depth Pro so first real call is warm.
        self._get_backend("depth-pro")
        out = self._billing(t0)
        out["ok"] = True
        return out

    @modal.method()
    def infer_rgb(
        self,
        rgb_bytes: bytes,
        model_key: str = "depth-pro",
        max_size: int = 756,
        f_px: Optional[float] = None,
        estimate_focal: bool = False,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        key = self._pipe.normalize_model_key(model_key)
        if self._pipe.is_promptda(key):
            raise ValueError(
                "infer_rgb is Depth Pro only. Use infer_pair for PromptDA."
            )
        backend = self._get_backend(key)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(rgb_bytes)
            path = Path(tmp.name)
        try:
            # Always use the focal path so EXIF cannot silently re-scale
            # a video JPEG sequence. Pass f_px to lock metric scale; omit
            # it (or estimate_focal=True) to estimate on this frame.
            use_f = f_px if f_px is not None else None
            color, depth, focallength_px = self._pipe._infer_depth_pro_focal(
                backend, path, f_px=use_f, max_size=int(max_size)
            )
        finally:
            path.unlink(missing_ok=True)
        out = self._billing(t0)
        out.update(
            {
                "depth": _encode_depth(depth),
                "rgb": _encode_rgb(color),
                "model_key": backend.key,
                "prompt_depth": None,
                "focallength_px": focallength_px,
            }
        )
        return out

    @modal.method()
    def infer_pair(
        self,
        rgb_bytes: bytes,
        depth_bytes: bytes,
        model_key: str = "promptda-small",
        max_size: int = 756,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        key = self._pipe.normalize_model_key(model_key)
        if not self._pipe.is_promptda(key):
            raise ValueError("infer_pair requires a PromptDA model_key")
        backend = self._get_backend(key)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_rgb:
            tmp_rgb.write(rgb_bytes)
            rgb_path = Path(tmp_rgb.name)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_d:
            tmp_d.write(depth_bytes)
            depth_path = Path(tmp_d.name)
        try:
            color, depth, prompt_np = self._pipe._infer_promptda(
                backend, rgb_path, depth_path, int(max_size)
            )
        finally:
            rgb_path.unlink(missing_ok=True)
            depth_path.unlink(missing_ok=True)
        out = self._billing(t0)
        out.update(
            {
                "depth": _encode_depth(depth),
                "rgb": _encode_rgb(color),
                "prompt_depth": _encode_depth(prompt_np),
                "model_key": backend.key,
            }
        )
        return out

    @modal.method()
    def infer_frames(
        self,
        frame_bytes_list: list[bytes],
        model_key: str = "depth-pro",
        max_size: int = 756,
        f_px: Optional[float] = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        key = self._pipe.normalize_model_key(model_key)
        if self._pipe.is_promptda(key):
            raise ValueError("Video frames use Depth Pro only")
        backend = self._get_backend(key)
        frames: list[dict[str, Any]] = []
        locked = f_px
        for rgb_bytes in frame_bytes_list:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(rgb_bytes)
                path = Path(tmp.name)
            try:
                color, depth, focallength_px = self._pipe._infer_depth_pro_focal(
                    backend, path, f_px=locked, max_size=int(max_size)
                )
            finally:
                path.unlink(missing_ok=True)
            if locked is None and focallength_px:
                locked = focallength_px
            frames.append(
                {
                    "depth": _encode_depth(depth),
                    "rgb": _encode_rgb(color),
                    "focallength_px": focallength_px,
                }
            )
        out = self._billing(t0)
        out["frames"] = frames
        out["model_key"] = backend.key
        out["focallength_px"] = locked
        return out
