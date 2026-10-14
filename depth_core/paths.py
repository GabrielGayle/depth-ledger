"""Project paths and output helpers."""
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_RGB = ROOT / "PromptDA" / "assets" / "example_images" / "image.jpg"
EXAMPLE_DEPTH = ROOT / "PromptDA" / "assets" / "example_images" / "arkit_depth.png"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = ROOT / "checkpoints" / "depth_pro.pt"

MAX_SIZE = int(os.environ.get("PROMPTDA_MAX_SIZE", "756"))
THUMB_MAX = 384


def save_depth_mm(depth_m: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    imageio.imwrite(path, depth_mm)
    return path


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_")
    return (stem[:80] or "image").lower()


def make_run_dir(source_path: Path, model_key: str) -> Path:
    from depth_core.models import normalize_model_key

    stem = safe_stem(source_path.name)
    model = normalize_model_key(model_key).replace("_", "-")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{stem}_{model}_{ts}"
    out = OUTPUT_DIR / base
    if out.exists():
        out = OUTPUT_DIR / f"{base}_{uuid.uuid4().hex[:4]}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def downscale_thumb(img: np.ndarray, max_side: int = THUMB_MAX) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
