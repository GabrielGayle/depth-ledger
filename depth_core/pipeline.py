"""Depth Ledger core — models, looks, blend, scenery, infer (no UI)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import imageio
import matplotlib
import numpy as np
import torch
import trimesh

from promptda.promptda import PromptDA
from promptda.utils.depth_utils import unproject_depth, visualize_depth
from promptda.utils.io_wrapper import load_depth, load_image

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_RGB = ROOT / "PromptDA" / "assets" / "example_images" / "image.jpg"
EXAMPLE_DEPTH = ROOT / "PromptDA" / "assets" / "example_images" / "arkit_depth.png"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = ROOT / "checkpoints" / "depth_pro.pt"

# --- Model registry ---------------------------------------------------------

MODEL_CHOICES = [
    ("Depth Pro (RGB only)", "depth-pro"),
    ("PromptDA small", "promptda-small"),
    ("PromptDA large", "promptda-large"),
]
MODEL_KEYS = {k for _, k in MODEL_CHOICES}

PROMPTDA_IDS = {
    "promptda-small": ("depth-anything/prompt-depth-anything-vits", "vits"),
    "promptda-large": ("depth-anything/prompt-depth-anything-vitl", "vitl"),
}

# Map legacy env values and short names.
_LEGACY_MODEL = {
    "small": "promptda-small",
    "large": "promptda-large",
    "promptda": "promptda-small",
    "depthpro": "depth-pro",
    "depth_pro": "depth-pro",
}


def normalize_model_key(raw: str) -> str:
    key = (raw or "").strip().lower()
    key = _LEGACY_MODEL.get(key, key)
    return key if key in MODEL_KEYS else "depth-pro"


def is_promptda(model_key: str) -> bool:
    return normalize_model_key(model_key) in PROMPTDA_IDS


def pick_device(model_key: str) -> str:
    """PromptDA defaults to CPU (MPS fragile); Depth Pro prefers MPS on Apple Silicon."""
    forced = os.environ.get("PROMPTDA_DEVICE", "").strip().lower()
    if forced in {"cpu", "mps", "cuda"}:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if normalize_model_key(model_key) == "depth-pro" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


MAX_SIZE = int(os.environ.get("PROMPTDA_MAX_SIZE", "756"))
DEFAULT_MODEL = normalize_model_key(os.environ.get("PROMPTDA_MODEL", "depth-pro"))


@dataclass
class LoadedBackend:
    key: str
    device: str
    kind: str  # "promptda" | "depth-pro"
    model: Any
    transform: Any = None  # Depth Pro only


_backend: Optional[LoadedBackend] = None


def get_backend(model_key: str = DEFAULT_MODEL) -> LoadedBackend:
    global _backend
    key = normalize_model_key(model_key)
    device = pick_device(key)

    if _backend is not None and _backend.key == key and _backend.device == device:
        return _backend

    if key in PROMPTDA_IDS:
        repo_id, encoder = PROMPTDA_IDS[key]
        print(f"Loading PromptDA ({key}) on {device} from {repo_id}…")
        model = (
            PromptDA.from_pretrained(repo_id, model_kwargs={"encoder": encoder})
            .to(device)
            .eval()
        )
        _backend = LoadedBackend(key=key, device=device, kind="promptda", model=model)
        return _backend

    # Depth Pro
    import depth_pro
    from depth_pro.depth_pro import DepthProConfig, DEFAULT_MONODEPTH_CONFIG_DICT

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Depth Pro checkpoint missing at {CHECKPOINT_PATH}. "
            "Run: bash DepthPro/get_pretrained_models.sh  "
            "(or curl the weights into checkpoints/depth_pro.pt)."
        )

    print(f"Loading Depth Pro on {device} from {CHECKPOINT_PATH}…")
    precision = torch.half if device in {"mps", "cuda"} else torch.float32
    config = DepthProConfig(
        patch_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.patch_encoder_preset,
        image_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.image_encoder_preset,
        decoder_features=DEFAULT_MONODEPTH_CONFIG_DICT.decoder_features,
        checkpoint_uri=str(CHECKPOINT_PATH),
        use_fov_head=DEFAULT_MONODEPTH_CONFIG_DICT.use_fov_head,
        fov_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.fov_encoder_preset,
    )
    model, transform = depth_pro.create_model_and_transforms(
        config=config,
        device=torch.device(device),
        precision=precision,
    )
    model.eval()
    _backend = LoadedBackend(
        key=key, device=device, kind="depth-pro", model=model, transform=transform
    )
    return _backend


def _tensor_to_rgb(image: torch.Tensor) -> np.ndarray:
    arr = image[0].detach().cpu().permute(1, 2, 0).numpy()
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def _save_depth_mm(depth_m: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Clip — uint16 wrap would corrupt depths > 65.535 m.
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    imageio.imwrite(path, depth_mm)
    return path


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_")
    return (stem[:80] or "image").lower()


def _make_run_dir(source_path: Path, model_key: str) -> Path:
    """outputs/{stem}_{model}_{YYYYMMDD-HHMMSS}/ — collision-safe."""
    stem = _safe_stem(source_path.name)
    model = normalize_model_key(model_key).replace("_", "-")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{stem}_{model}_{ts}"
    out = OUTPUT_DIR / base
    if out.exists():
        out = OUTPUT_DIR / f"{base}_{uuid.uuid4().hex[:4]}"
    out.mkdir(parents=True, exist_ok=True)
    return out



def _float01_to_u8_dither(norm: np.ndarray) -> np.ndarray:
    """Ordered dither float [0,1] → uint8 so smooth depth ramps don't posterize."""
    bayer = (
        np.array(
            [
                [0, 48, 12, 60, 3, 51, 15, 63],
                [32, 16, 44, 28, 35, 19, 47, 31],
                [8, 56, 4, 52, 11, 59, 7, 55],
                [40, 24, 36, 20, 43, 27, 39, 23],
                [2, 50, 14, 62, 1, 49, 13, 61],
                [34, 18, 46, 30, 33, 17, 45, 29],
                [10, 58, 6, 54, 9, 57, 5, 53],
                [42, 26, 38, 22, 41, 25, 37, 21],
            ],
            dtype=np.float32,
        )
        / 64.0
        - 0.5
    )
    h, w = norm.shape
    tiles = np.tile(bayer, (h // 8 + 1, w // 8 + 1))[:h, :w]
    return np.clip(norm * 255.0 + tiles, 0, 255).astype(np.uint8)


def _apply_grade(norm: np.ndarray, grade: Optional[str]) -> np.ndarray:
    """Remap normalized inverse-depth [0,1] for artistic / facial detail."""
    if not grade or grade == "none":
        return norm
    x = norm.astype(np.float32)
    if grade == "expand_near":
        # Face usually sits in the top of the range after near-field windowing.
        # Rebase from that band so micro-depth (sockets, nose) uses the full 0–1.
        live = x[x > 0.02]
        if live.size > 64:
            lo = float(np.percentile(live, 45))
            hi = float(np.percentile(live, 99.5))
        else:
            lo, hi = 0.55, 1.0
        if hi > lo + 1e-4:
            x = ((x - lo) / (hi - lo)).clip(0.0, 1.0)
        x = np.power(x, 1.25)
    elif grade == "s_curve":
        x = 0.5 * (np.sin(np.pi * (x - 0.5)) + 1.0)
    elif grade == "lift_shadows":
        x = 1.0 - np.power(1.0 - x, 1.35)
    elif grade == "sockets":
        # Rebase near band first, then multi-scale local contrast for eyes/nose.
        live = x[x > 0.02]
        if live.size > 64:
            lo = float(np.percentile(live, 40))
            hi = float(np.percentile(live, 99.5))
            if hi > lo + 1e-4:
                x = ((x - lo) / (hi - lo)).clip(0.0, 1.0)
        h, w = x.shape
        detail = np.zeros_like(x)
        for frac, gain in ((110.0, 1.1), (50.0, 1.35), (22.0, 1.0)):
            sigma = max(h, w) / frac
            blur = cv2.GaussianBlur(x, (0, 0), sigmaX=sigma)
            detail += gain * (x - blur)
        x = np.clip(x + detail, 0.0, 1.0)
        x = np.power(x.clip(0, 1), 1.2)
        lo, hi = float(np.percentile(x, 1)), float(np.percentile(x, 99.5))
        if hi > lo:
            x = ((x - lo) / (hi - lo)).clip(0, 1)
    elif grade == "scenery":
        # Mild stretch + lift far/dark buildings, then multi-scale facade detail.
        live = x[x > 0.008]
        if live.size > 64:
            lo = float(np.percentile(live, 2))
            hi = float(np.percentile(live, 98))
            if hi > lo + 1e-4:
                x = ((x - lo) / (hi - lo)).clip(0.0, 1.0)
        # Lift far (dark) tones so distant skyline isn't crushed to black.
        x = 1.0 - np.power(1.0 - x.clip(0, 1), 1.45)
        h, w = x.shape
        detail = np.zeros_like(x)
        for frac, gain in ((100.0, 0.75), (45.0, 1.05), (20.0, 0.85)):
            sigma = max(h, w) / frac
            blur = cv2.GaussianBlur(x, (0, 0), sigmaX=sigma)
            detail += gain * (x - blur)
        x = np.clip(0.6 * x + 0.4 * (x + detail), 0.0, 1.0)
        x = np.power(x.clip(0, 1), 0.95)
    return x.astype(np.float64)


def _subject_hi_pct(depth_m: np.ndarray, default: float = 30.0) -> float:
    """Find depth percentile just before the first subject→background cliff."""
    depth = np.asarray(depth_m, dtype=np.float64)
    # Keep far sentinels — excluding them shifts percentiles and hides the cliff.
    valid = np.isfinite(depth) & (depth > 1e-4)
    if not valid.any():
        return default
    ps = np.linspace(10.0, 55.0, 46)
    vals = np.percentile(depth[valid], ps)
    abs_jump = np.diff(vals)
    rel = abs_jump / np.maximum(vals[:-1], 1e-3)
    # First cliff: ≥50cm absolute jump or ≥50% relative.
    hits = np.where((abs_jump >= 0.5) | (rel >= 0.5))[0]
    if len(hits):
        idx = int(hits[0])
        return float(max(ps[idx] - 0.5, 12.0))
    hits = np.where((abs_jump >= 0.25) | (rel >= 0.25))[0]
    if len(hits):
        idx = int(hits[0])
        return float(max(ps[idx] - 0.5, 12.0))
    return default


DEFAULT_WASH_BLUR = 0.4


def _prepare_photo_rgb(
    photo: Optional[np.ndarray],
    shape_hw: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Resize / coerce source photo to HxWx3 float32 in [0, 255]."""
    if photo is None:
        return None
    rgb = np.asarray(photo)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    elif rgb.shape[-1] >= 4:
        rgb = rgb[..., :3]
    h, w = shape_hw
    if rgb.shape[0] != h or rgb.shape[1] != w:
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return rgb.astype(np.float32)


def _photo_tint_norm(norm: np.ndarray, photo: Optional[np.ndarray]) -> np.ndarray:
    """Color the depth preview with the source image at each pixel.

    Depth stays the intensity map; RGB comes from the original photo so hues
    sit in their true positions.
    """
    rgb = _prepare_photo_rgb(photo, norm.shape[:2])
    if rgb is None:
        u8 = _float01_to_u8_dither(norm.astype(np.float32))
        return np.stack([u8, u8, u8], axis=-1)
    weight = np.clip(norm.astype(np.float32), 0.0, 1.0)
    out = rgb * weight[..., None]
    return (out + 0.5).clip(0.0, 255.0).astype(np.uint8)


def _wash_lut_from_photo(
    photo: np.ndarray,
    blur_amount: float,
) -> np.ndarray:
    """Build a 256-entry RGB LUT from a blurred source image.

    Pixels are averaged into luminance bins so bright depth maps to the
    image's light colors and dark depth to its shadows. Blur washes local
    detail into a smoother gradient.
    """
    blur = float(np.clip(blur_amount, 0.0, 1.0))
    h, w = photo.shape[:2]
    # 0 → barely soften; 1 → ~1/5 of the long edge (big palette wash).
    sigma = 0.8 + blur * max(h, w) / 5.0
    blurred = cv2.GaussianBlur(photo, (0, 0), sigmaX=sigma)

    r = blurred[..., 0]
    g = blurred[..., 1]
    b = blurred[..., 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    bins = np.clip((lum + 0.5).astype(np.int32), 0, 255)

    lut = np.zeros((256, 3), dtype=np.float64)
    counts = np.zeros(256, dtype=np.float64)
    flat_bins = bins.ravel()
    np.add.at(lut[:, 0], flat_bins, r.ravel())
    np.add.at(lut[:, 1], flat_bins, g.ravel())
    np.add.at(lut[:, 2], flat_bins, b.ravel())
    np.add.at(counts, flat_bins, 1.0)

    nonempty = counts > 0
    if not nonempty.any():
        gray = np.linspace(0.0, 255.0, 256, dtype=np.float64)
        return np.stack([gray, gray, gray], axis=-1).astype(np.uint8)

    lut[nonempty] /= counts[nonempty, None]
    xs = np.arange(256, dtype=np.float64)
    filled = xs[nonempty]
    for c in range(3):
        lut[:, c] = np.interp(xs, filled, lut[nonempty, c])

    # Extra 1D smooth on the LUT so higher blur = softer gradient stops.
    if blur > 0.02:
        k = int(1 + blur * 64) | 1
        lut = (
            cv2.GaussianBlur(
                lut.reshape(256, 1, 3).astype(np.float32),
                (1, k),
                0,
            ).reshape(256, 3)
        )
    return lut.clip(0.0, 255.0).astype(np.uint8)


def _photo_wash_norm(
    norm: np.ndarray,
    photo: Optional[np.ndarray],
    blur_amount: float = DEFAULT_WASH_BLUR,
) -> np.ndarray:
    """Depth → color via a gradient colormap sampled from the blurred photo."""
    rgb = _prepare_photo_rgb(photo, norm.shape[:2])
    if rgb is None:
        u8 = _float01_to_u8_dither(norm.astype(np.float32))
        return np.stack([u8, u8, u8], axis=-1)
    lut = _wash_lut_from_photo(rgb, blur_amount)
    u8 = _float01_to_u8_dither(norm.astype(np.float32))
    return lut[u8]


def _norm_to_rgb(
    norm: np.ndarray,
    cmap: str,
    photo: Optional[np.ndarray] = None,
    blur_amount: float = DEFAULT_WASH_BLUR,
) -> np.ndarray:
    if cmap in (None, "gray", "grey"):
        u8 = _float01_to_u8_dither(norm.astype(np.float32))
        return np.stack([u8, u8, u8], axis=-1)
    if cmap == "photo":
        return _photo_tint_norm(norm, photo)
    if cmap == "wash":
        return _photo_wash_norm(norm, photo, blur_amount=blur_amount)
    u8 = _float01_to_u8_dither(norm.astype(np.float32))
    colored = matplotlib.colormaps[cmap](u8.astype(np.float64) / 255.0)[..., :3]
    return (colored * 255.0 + 0.5).astype(np.uint8)


def _prepare_inv_smooth(depth_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Shared inverse-depth + bilateral (once per generate)."""
    depth = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 1e-4)
    inv = np.zeros_like(depth, dtype=np.float32)
    if valid.any():
        inv[valid] = (1.0 / np.clip(depth[valid], 1e-4, None)).astype(np.float32)
        fill = float(np.median(inv[valid]))
        inv[~valid] = fill
        inv = cv2.bilateralFilter(inv, d=0, sigmaColor=0.025, sigmaSpace=6)
    return depth, inv


def depth_valid_mask(depth_m: np.ndarray) -> np.ndarray:
    """Valid pixels for percentile windowing (matches _visualize_inverse_depth)."""
    depth = np.asarray(depth_m, dtype=np.float64)
    return np.isfinite(depth) & (depth > 1e-4) & (depth < 5e3)


def depth_percentile_at(
    depth_m: np.ndarray,
    u: float,
    v: float,
) -> dict[str, float | int]:
    """Map normalized image coords to depth (m) and scene percentile rank."""
    depth = np.asarray(depth_m, dtype=np.float64)
    h, w = depth.shape[:2]
    u_clamped = float(np.clip(u, 0.0, 1.0))
    v_clamped = float(np.clip(v, 0.0, 1.0))
    px = int(round(u_clamped * max(w - 1, 0)))
    py = int(round(v_clamped * max(h - 1, 0)))
    valid = depth_valid_mask(depth)
    if not valid.any():
        raise ValueError("No valid depth in scene")
    if not valid[py, px]:
        raise ValueError("Clicked pixel has no valid depth")
    depth_at = float(depth[py, px])
    valid_depths = depth[valid]
    pct = float(100.0 * np.mean(valid_depths <= depth_at))
    return {
        "depth_m": depth_at,
        "pct": pct,
        "u": u_clamped,
        "v": v_clamped,
        "px": px,
        "py": py,
        "w": w,
        "h": h,
    }


def render_linear_inverse_turbo(
    depth_m: np.ndarray,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
) -> np.ndarray:
    """Linear inverse-depth grayscale plate for Tune picking (not the tent look)."""
    depth, inv_smooth = _prepare_inv_smooth(depth_m)
    valid = depth_valid_mask(depth)
    h, w = depth.shape[:2]
    if not valid.any():
        return np.zeros((h, w, 3), dtype=np.uint8)
    inv = inv_smooth.astype(np.float64)
    lo = float(np.percentile(inv[valid], lo_pct))
    hi = float(np.percentile(inv[valid], hi_pct))
    span = max(hi - lo, 1e-12)
    norm = np.clip((inv - lo) / span, 0.0, 1.0)
    norm[~valid] = 0.0
    return _norm_to_rgb(norm, "gray")


def _visualize_inverse_depth(
    depth_m: np.ndarray,
    near_m: float = 0.1,
    far_m: float = 250.0,
    cmap: str = "gray",
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
    gamma: float = 0.85,
    grade: Optional[str] = None,
    inv_smooth: Optional[np.ndarray] = None,
    mask_below_pct: Optional[float] = None,
    mask_below_m: Optional[float] = None,
    near_override: Optional[float] = None,
    far_override: Optional[float] = None,
    photo: Optional[np.ndarray] = None,
    blur_amount: float = DEFAULT_WASH_BLUR,
) -> np.ndarray:
    """Scene-adaptive inverse depth preview.

    Peak white at lo_pct (near window edge); falloff to black both nearer and
    farther. Park Near on the hips → hips white, knees (closer) black.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    # Ignore Depth Pro far sentinels (~1e4) when picking display percentiles.
    valid = np.isfinite(depth) & (depth > 1e-4) & (depth < 5e3)
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    if near_override is not None and far_override is not None:
        near = float(near_override)
        far = float(max(far_override, near + 1e-3))
    else:
        near = float(np.percentile(depth[valid], lo_pct))
        far = float(np.percentile(depth[valid], hi_pct))
    near = float(np.clip(near, near_m, far_m - 1e-3))
    far = float(np.clip(max(far, near + 1e-3), near + 1e-3, far_m))

    if inv_smooth is None:
        _, inv_smooth = _prepare_inv_smooth(depth)

    # Inverse-depth tent: peak at 1/near, black at 1/far and mirrored nearer.
    lo, peak = 1.0 / far, 1.0 / near
    span = max(peak - lo, 1e-12)
    inv = inv_smooth.astype(np.float64)
    norm = np.zeros_like(depth, dtype=np.float64)
    show = np.isfinite(depth) & (depth > 1e-4)
    inv_s = inv[show]
    # Far side of peak (farther / lower inv): lo → peak maps 0 → 1
    far_side = inv_s <= peak
    near_side = ~far_side
    out = np.zeros_like(inv_s)
    out[far_side] = (inv_s[far_side] - lo) / span
    # Nearer than window: fall back to black over the same inv span
    out[near_side] = 1.0 - (inv_s[near_side] - peak) / span
    norm[show] = out.clip(0.0, 1.0)
    if mask_below_m is not None:
        norm[depth < float(mask_below_m)] = 0.0
    elif mask_below_pct is not None:
        # Punch out foreground (subject) so only mid/far scenery remains.
        thr = float(np.percentile(depth[valid], float(mask_below_pct)))
        norm[depth < thr] = 0.0
    if gamma != 1.0:
        norm = np.power(norm, gamma)
    norm = _apply_grade(norm, grade)
    return _norm_to_rgb(norm, cmap, photo=photo, blur_amount=blur_amount)


# Artistic Looks — one metric depth, many previews (no re-infer).
LOOK_PRESETS = {
    "ledger": {
        "label": "Ledger gray",
        "cmap": "gray",
        "lo_pct": 2.0,
        "hi_pct": 98.0,
        "gamma": 0.85,
        "grade": None,
    },
    "magma": {
        "label": "Magma",
        "cmap": "magma",
        "lo_pct": 2.0,
        "hi_pct": 98.0,
        "gamma": 0.85,
        "grade": None,
    },
    "wash": {
        # Gradient colormap sampled from the source photo (blurred).
        "label": "Wash",
        "cmap": "wash",
        "lo_pct": 2.0,
        "hi_pct": 98.0,
        "gamma": 0.9,
        "grade": None,
        "blur_amount": DEFAULT_WASH_BLUR,
    },
    "punchy": {
        "label": "Punchy near",
        "cmap": "gray",
        "lo_pct": 5.0,
        "hi_pct": 90.0,
        "gamma": 0.7,
        "grade": None,
    },
    "soft": {
        "label": "Soft far",
        "cmap": "gray",
        "lo_pct": 1.0,
        "hi_pct": 99.0,
        "gamma": 1.1,
        "grade": None,
    },
    "portrait": {
        # Near-field only — auto-clips at subject→bg cliff, opens crushed face whites.
        "label": "Portrait",
        "cmap": "gray",
        "lo_pct": 1.0,
        "hi_pct": "auto",
        "hi_pct_fallback": 36.0,
        "gamma": 1.05,
        "grade": "expand_near",
    },
    "sockets": {
        # Tighter near window + multi-scale local contrast for eyes/nose.
        "label": "Sockets",
        "cmap": "gray",
        "lo_pct": 1.0,
        "hi_pct": "auto",
        "hi_pct_fallback": 30.0,
        "hi_pct_scale": 0.88,
        "gamma": 1.0,
        "grade": "sockets",
    },
    "scenery": {
        # Far field only — subject punched out; buildings / skyline graded.
        "label": "Scenery",
        "cmap": "gray",
        "lo_pct": "auto_bg",
        "hi_pct": 92.0,
        "gamma": 0.95,
        "grade": "scenery",
        "mask_subject": True,
    },
}
LOOK_CHOICES = [(v["label"], k) for k, v in LOOK_PRESETS.items()]
DEFAULT_LOOK_A = "ledger"
DEFAULT_LOOK_B = "sockets"
DEFAULT_LOOK_C = "scenery"

SCENERY_MATTE_MODES = {
    "smart": "Smart (depth edge)",
    "soft": "Soft depth gate",
    "protect": "Protect (dilate)",
}
SCENERY_MATTE_CHOICES = [(label, key) for key, label in SCENERY_MATTE_MODES.items()]
DEFAULT_SCENERY_MATTE = "smart"
DEFAULT_SCENERY_PROTECT = 0.22
DEFAULT_SCENERY_FEATHER = 0.35
DEFAULT_SCENERY_CLEAN = True

THUMB_MAX = 384

BLEND_MODES = {
    "normal": "Normal",
    "multiply": "Multiply",
    "screen": "Screen",
    "overlay": "Overlay",
    "soft_light": "Soft Light",
    "hard_light": "Hard Light",
    "darken": "Darken",
    "lighten": "Lighten",
    "difference": "Difference",
}
BLEND_MODE_CHOICES = [(label, key) for key, label in BLEND_MODES.items()]
DEFAULT_BLEND_MODE = "normal"

# --- Run recipe / history (mixer + scenery "ingredients") -------------------

HISTORY_LIMIT = 40
RECIPE_FILENAME = "recipe.json"


def _default_recipe() -> dict:
    return {
        "model_key": DEFAULT_MODEL,
        "max_size": MAX_SIZE if MAX_SIZE in (518, 756, 1008) else 756,
        "infer_backend": "local",
        "look_a": DEFAULT_LOOK_A,
        "look_b": DEFAULT_LOOK_B,
        "look_c": DEFAULT_LOOK_C,
        "fade_b": 0.0,
        "fade_c": 0.0,
        "blend_mode_b": DEFAULT_BLEND_MODE,
        "blend_mode_c": "screen",
        "scenery_mode": DEFAULT_SCENERY_MATTE,
        "scenery_protect": DEFAULT_SCENERY_PROTECT,
        "scenery_feather": DEFAULT_SCENERY_FEATHER,
        "scenery_clean": DEFAULT_SCENERY_CLEAN,
    }


def _recipe_path(out_dir: Path | str) -> Path:
    return Path(out_dir) / RECIPE_FILENAME


def build_recipe(
    model_key: str,
    max_size: int,
    look_a: str,
    look_b: str,
    look_c: str,
    fade_b: float,
    fade_c: float,
    blend_mode_b: str,
    blend_mode_c: str,
    scenery_mode: str,
    scenery_protect: float,
    scenery_feather: float,
    scenery_clean: bool,
) -> dict:
    return {
        "model_key": normalize_model_key(model_key),
        "max_size": int(max_size),
        "look_a": look_a,
        "look_b": look_b,
        "look_c": look_c,
        "fade_b": float(fade_b),
        "fade_c": float(fade_c),
        "blend_mode_b": blend_mode_b,
        "blend_mode_c": blend_mode_c,
        "scenery_mode": scenery_mode,
        "scenery_protect": float(scenery_protect),
        "scenery_feather": float(scenery_feather),
        "scenery_clean": bool(scenery_clean),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_recipe(out_dir: Path | str, recipe: dict) -> Path:
    path = _recipe_path(out_dir)
    payload = dict(recipe)
    payload["out_dir"] = str(Path(out_dir).resolve())
    payload["run_name"] = Path(out_dir).name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_recipe(out_dir: Path | str) -> Optional[dict]:
    path = _recipe_path(out_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    merged = _default_recipe()
    merged.update({k: data[k] for k in merged if k in data})
    # Preserve graph / version fields for node-editor recipes.
    if "graph" in data:
        merged["graph"] = data["graph"]
    if "version" in data:
        merged["version"] = data["version"]
    if "arena" in data:
        merged["arena"] = data["arena"]
    merged["out_dir"] = str(Path(out_dir).resolve())
    merged["run_name"] = Path(out_dir).name
    if "saved_at" in data:
        merged["saved_at"] = data["saved_at"]
    return merged


def _look_label(key: str) -> str:
    preset = LOOK_PRESETS.get(key)
    return preset["label"] if preset else key


def _blend_label(key: str) -> str:
    return BLEND_MODES.get(key, key)


def _scenery_label(key: str) -> str:
    return SCENERY_MATTE_MODES.get(key, key)


def format_recipe_md(recipe: Optional[dict], run_name: str = "") -> str:
    if recipe is None:
        name = run_name or "—"
        return (
            f"**{name}**\n\n"
            "_No recipe saved for this run (predates history)._"
        )
    name = recipe.get("run_name") or run_name or "—"
    clean = "on" if recipe.get("scenery_clean") else "off"
    lines = [
        f"**{name}**",
        "",
        f"- **Model** — `{recipe['model_key']}` · max side `{recipe['max_size']}`",
        (
            f"- **Looks** — A `{_look_label(recipe['look_a'])}` · "
            f"B `{_look_label(recipe['look_b'])}` · "
            f"C `{_look_label(recipe['look_c'])}`"
        ),
        (
            f"- **Blend B** — {_blend_label(recipe['blend_mode_b'])} @ "
            f"{recipe['fade_b']:.2f}"
        ),
        (
            f"- **Blend C** — {_blend_label(recipe['blend_mode_c'])} @ "
            f"{recipe['fade_c']:.2f}"
        ),
        (
            f"- **Scenery** — {_scenery_label(recipe['scenery_mode'])} · "
            f"protect `{recipe['scenery_protect']:.2f}` · "
            f"feather `{recipe['scenery_feather']:.2f}` · clean `{clean}`"
        ),
    ]
    if recipe.get("saved_at"):
        lines.append(f"- **Saved** — `{recipe['saved_at']}`")
    return "\n".join(lines)


def list_recent_runs(limit: int = HISTORY_LIMIT) -> list[tuple[str, str]]:
    """Return [(dropdown_label, out_dir_str), ...] newest first."""
    if not OUTPUT_DIR.is_dir():
        return []
    dirs = [
        p
        for p in OUTPUT_DIR.iterdir()
        if p.is_dir() and not p.name.startswith("_") and not p.name.startswith(".")
    ]

    def sort_key(p: Path):
        recipe = _recipe_path(p)
        try:
            mtime = recipe.stat().st_mtime if recipe.is_file() else p.stat().st_mtime
        except OSError:
            mtime = 0.0
        return mtime

    dirs.sort(key=sort_key, reverse=True)
    choices: list[tuple[str, str]] = []
    for p in dirs[:limit]:
        has_recipe = _recipe_path(p).is_file()
        mark = "" if has_recipe else " · no recipe"
        choices.append((f"{p.name}{mark}", str(p.resolve())))
    return choices




def resize_max_side(
    img: np.ndarray, max_size: Optional[int]
) -> Tuple[np.ndarray, float]:
    """Downscale so the longest edge is <= max_size. Returns (image, scale)."""
    if not max_size:
        return img, 1.0
    h, w = img.shape[:2]
    m = max(h, w)
    cap = int(max_size)
    if cap <= 0 or m <= cap:
        return img, 1.0
    scale = cap / float(m)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), scale


def _downscale_thumb(img: np.ndarray, max_side: int = THUMB_MAX) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _resolve_hi_pct(cfg: dict, cliff: float) -> float:
    hi = cfg["hi_pct"]
    if hi != "auto":
        return float(hi)
    fb = float(cfg.get("hi_pct_fallback", 28.0))
    scale = float(cfg.get("hi_pct_scale", 1.0))
    resolved = cliff * scale
    # Cap to fallback so we stay in the subject band even if cliff is soft.
    return float(np.clip(resolved, 12.0, fb))


def _resolve_lo_pct(cfg: dict, cliff: float) -> float:
    lo = cfg["lo_pct"]
    if lo == "auto_bg":
        # Start just after the subject→background cliff.
        offset = float(cfg.get("lo_pct_offset", 1.5))
        return float(np.clip(cliff + offset, 15.0, 70.0))
    return float(lo)


def _scenery_depth_window(
    depth_m: np.ndarray, cliff_pct: float
) -> Tuple[float, float, float]:
    """Return (near_m, far_m, subject_cut_m) for far-field / building scenery."""
    depth = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 1e-4)
    if not valid.any():
        return 2.0, 80.0, 2.0
    cliff_d = float(np.percentile(depth[valid], cliff_pct))
    subject_cut = cliff_d * 1.12
    band = depth[valid & (depth >= subject_cut) & (depth < 5e3)]
    if band.size < 64:
        return subject_cut, min(subject_cut + 60.0, 200.0), subject_cut
    near = float(np.percentile(band, 5))
    # Pull skyline / distant facades into range (was ~p72 → crushed to black).
    far = float(np.percentile(band, 92))
    if far <= near + 0.5:
        far = near + 10.0
    return near, far, subject_cut


def _largest_near_components(seed: np.ndarray, min_frac: float = 0.004) -> np.ndarray:
    """Keep the main subject blob(s); drop tiny near-depth speckles."""
    h, w = seed.shape
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        seed.astype(np.uint8), connectivity=8
    )
    if n_labels <= 1:
        return seed.astype(np.uint8)
    min_area = max(32, int(min_frac * h * w))
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)]
    areas.sort(key=lambda t: t[1], reverse=True)
    keep = np.zeros_like(seed, dtype=np.uint8)
    for i, area in areas:
        if i == areas[0][0] or area >= min_area:
            keep[labels == i] = 1
        if area < min_area and i != areas[0][0]:
            break
    return keep


def _subject_matte(
    depth_m: np.ndarray,
    cut_m: float,
    near_m: float,
    protect: float = DEFAULT_SCENERY_PROTECT,
    feather: float = DEFAULT_SCENERY_FEATHER,
    mode: str = DEFAULT_SCENERY_MATTE,
) -> np.ndarray:
    """Subject punch-out matte in [0,1]. Higher protect = kill more aggressively."""
    depth = np.asarray(depth_m, dtype=np.float64)
    h, w = depth.shape
    protect = float(np.clip(protect, 0.0, 1.0))
    feather = float(np.clip(feather, 0.0, 1.0))
    valid = np.isfinite(depth) & (depth > 1e-4)

    seed = (valid & (depth < cut_m)).astype(np.uint8)
    seed = _largest_near_components(seed)

    if mode == "soft":
        span = max(near_m - cut_m, 1e-3)
        t = ((depth - cut_m) / span).astype(np.float32)
        power = 0.45 + 1.6 * protect
        gate_kill = (1.0 - np.clip(t, 0.0, 1.0)) ** power
        gate_kill = np.where(valid, gate_kill, 0.0).astype(np.float32)
        gate_kill = np.maximum(gate_kill, seed.astype(np.float32))
        if feather > 0.01:
            sigma = 1.0 + feather * max(h, w) / 250.0
            gate_kill = cv2.GaussianBlur(gate_kill, (0, 0), sigmaX=sigma)
        return np.clip(gate_kill, 0.0, 1.0)

    base = max(h, w)
    if mode == "protect":
        rad = max(1, int(round(1 + protect * base / 75.0)))
        iters = 1 + int(round(protect * 2))
    else:
        rad = max(1, int(round(1 + protect * base / 140.0)))
        iters = 1 if protect < 0.55 else 2
    if rad % 2 == 0:
        rad += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad, rad))
    grown = cv2.dilate(seed, kernel, iterations=iters)

    if mode == "smart" and protect > 0.02:
        depth_f = np.where(valid, depth, near_m).astype(np.float32)
        gx = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        g_thr = float(np.percentile(grad[valid], 70 - 25 * protect))
        edge = (grad >= max(g_thr, 0.05)).astype(np.uint8)
        allowed = np.maximum(seed, edge)
        cur = seed.copy()
        steps = 2 + int(protect * 10)
        for _ in range(steps):
            cur = cv2.dilate(cur, kernel, iterations=1)
            cur = np.minimum(cur, allowed)
        grown = np.maximum(grown, cur)

    soft = grown.astype(np.float32)
    if feather > 0.01:
        sigma = 1.0 + feather * (2.0 + rad)
        soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=sigma)
    return np.clip(soft, 0.0, 1.0)


def _render_scenery_look(
    depth_m: np.ndarray,
    cliff_pct: float,
    protect: float = DEFAULT_SCENERY_PROTECT,
    feather: float = DEFAULT_SCENERY_FEATHER,
    mode: str = DEFAULT_SCENERY_MATTE,
    clean_specks: bool = DEFAULT_SCENERY_CLEAN,
    fringe_kill: bool = True,
    far_remap: bool = True,
) -> np.ndarray:
    """Far-field look: raw inv, adjustable subject matte, building grade.

    fringe_kill: zero mid-depth halo pixels that light up as white rings.
    far_remap: log-stretch the darker/far half so skyline isn't crushed.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    near, far, cut = _scenery_depth_window(depth, cliff_pct)
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 1e-4) & (depth < 5e3)

    inv = np.zeros_like(depth, dtype=np.float64)
    inv[valid] = 1.0 / np.clip(depth[valid], 1e-4, None)

    lo, hi = 1.0 / far, 1.0 / near
    norm = np.zeros_like(depth, dtype=np.float64)
    inv_c = np.clip(inv, lo, hi)
    norm[valid] = ((inv_c[valid] - lo) / (hi - lo)).clip(0.0, 1.0)

    matte = _subject_matte(
        depth, cut, near, protect=protect, feather=feather, mode=mode
    )
    if protect > 0.35 and mode != "soft":
        span = max(near - cut, 1e-3)
        mid = ((near - depth) / span).clip(0.0, 1.0).astype(np.float32)
        mid = mid * (protect - 0.35) / 0.65
        matte = np.maximum(matte, mid * (depth < near))

    keep = 1.0 - np.clip(matte, 0.0, 1.0)

    # Kill white silhouette fringe: mid-depths between subject cut and near band
    # that still map near the bright end of the scenery window.
    if fringe_kill:
        fringe = (
            valid
            & (depth >= cut)
            & (depth < near * 1.05)
            & (keep > 0.15)
            & (norm > 0.55)
        )
        keep = np.where(fringe, 0.0, keep)

    norm = norm * keep

    if far_remap:
        # Log-lift the far (dark) half without blowing the near building.
        dark = norm < 0.55
        x = norm.copy()
        band = x[dark & (x > 1e-4)]
        if band.size > 64:
            x[dark] = np.power(np.clip(x[dark], 0, 1), 0.72)
            norm = np.where(dark, x, norm)

    graded = _apply_grade(norm, "scenery") * keep

    if clean_specks:
        live = (graded > 0.02).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            live, connectivity=8
        )
        # Keep small-but-real facade islands; only drop hair-tip dust.
        min_area = max(24, int((0.0004 + 0.001 * protect) * h * w))
        keep_cc = np.zeros_like(live)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep_cc[labels == i] = 1
        graded = graded * keep_cc.astype(np.float64)
    return _norm_to_rgb(graded, "gray")


def render_looks(
    depth_m: np.ndarray,
    scenery_protect: float = DEFAULT_SCENERY_PROTECT,
    scenery_feather: float = DEFAULT_SCENERY_FEATHER,
    scenery_mode: str = DEFAULT_SCENERY_MATTE,
    scenery_clean: bool = DEFAULT_SCENERY_CLEAN,
    photo: Optional[np.ndarray] = None,
) -> dict:
    """Render all Looks presets from one metric depth map (bilateral once)."""
    _, inv_smooth = _prepare_inv_smooth(depth_m)
    cliff = _subject_hi_pct(depth_m)
    mode = scenery_mode if scenery_mode in SCENERY_MATTE_MODES else DEFAULT_SCENERY_MATTE
    out = {}
    for key, cfg in LOOK_PRESETS.items():
        if cfg.get("mask_subject"):
            out[key] = _render_scenery_look(
                depth_m,
                cliff,
                protect=scenery_protect,
                feather=scenery_feather,
                mode=mode,
                clean_specks=scenery_clean,
                fringe_kill=True,
                far_remap=True,
            )
        else:
            cmap = cfg["cmap"]
            out[key] = _visualize_inverse_depth(
                depth_m,
                cmap=cmap,
                lo_pct=_resolve_lo_pct(cfg, cliff),
                hi_pct=_resolve_hi_pct(cfg, cliff),
                gamma=cfg["gamma"],
                grade=cfg.get("grade"),
                inv_smooth=inv_smooth,
                photo=photo if cmap in ("photo", "wash") else None,
                blur_amount=float(cfg.get("blur_amount", DEFAULT_WASH_BLUR)),
            )
    return out


def _blend_mode_fn(base: np.ndarray, blend: np.ndarray, mode: str) -> np.ndarray:
    """Photoshop-style blend of two float images in [0, 1]."""
    a, b = base, blend
    if mode == "multiply":
        return a * b
    if mode == "screen":
        return 1.0 - (1.0 - a) * (1.0 - b)
    if mode == "overlay":
        return np.where(a < 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
    if mode == "soft_light":
        # Pegtop soft-light
        return (1.0 - 2.0 * b) * a * a + 2.0 * b * a
    if mode == "hard_light":
        return np.where(b < 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
    if mode == "darken":
        return np.minimum(a, b)
    if mode == "lighten":
        return np.maximum(a, b)
    if mode == "difference":
        return np.abs(a - b)
    # normal
    return b


def blend_looks(
    look_a: np.ndarray,
    look_b: np.ndarray,
    t: float,
    mode: str = "normal",
) -> np.ndarray:
    """Fade Look B over Look A with a blend mode. t=0 → A, t=1 → full mode result.

    Matches Photoshop: result = (1-opacity)*A + opacity*blendMode(A, B).
    """
    t = float(np.clip(t, 0.0, 1.0))
    a = look_a.astype(np.float32) / 255.0
    b = look_b.astype(np.float32) / 255.0
    mixed = _blend_mode_fn(a, b, mode if mode in BLEND_MODES else "normal")
    out = (1.0 - t) * a + t * mixed
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def blend_looks_triple(
    look_a: np.ndarray,
    look_b: np.ndarray,
    look_c: np.ndarray,
    t_b: float,
    t_c: float,
    mode_b: str = "normal",
    mode_c: str = "normal",
) -> np.ndarray:
    """Layer B then C over A (Photoshop stack)."""
    mid = blend_looks(look_a, look_b, t_b, mode_b)
    return blend_looks(mid, look_c, t_c, mode_c)


def _colorize_depth(depth_m: np.ndarray, backend_kind: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (default vis, optional shared minmax for prompt alignment)."""
    vis = _visualize_inverse_depth(depth_m, cmap="gray")
    if backend_kind == "promptda":
        _, dmin, dmax = visualize_depth(depth_m, ret_minmax=True)
        return vis, (dmin, dmax)
    return vis, None



def _extract_stray_first_frame(
    zip_path: Path, work_dir: Path
) -> Tuple[Path, Optional[Path], Optional[np.ndarray]]:
    """Extract Stray Scanner zip → (rgb_path, prompt_depth_path or None, camera_matrix or None)."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(work_dir)

    candidates = [p for p in work_dir.iterdir() if p.is_dir()]
    scene_dir = candidates[0] if len(candidates) == 1 else work_dir

    rgb_mp4 = scene_dir / "rgb.mp4"
    depth_png: Optional[Path] = scene_dir / "depth" / "000000.png"
    if depth_png is not None and not depth_png.exists():
        depth_dir = scene_dir / "depth"
        depth_files = sorted(depth_dir.glob("*.png")) if depth_dir.exists() else []
        depth_png = depth_files[0] if depth_files else None

    rgb_dir = scene_dir / "rgb"
    rgb_dir.mkdir(exist_ok=True)
    rgb_path = rgb_dir / "000000.jpg"

    if not rgb_path.exists():
        if not rgb_mp4.exists():
            raise FileNotFoundError(f"rgb.mp4 not found in {scene_dir}")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(rgb_mp4),
                    "-start_number",
                    "0",
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(rgb_path),
                ],
                check=True,
                capture_output=True,
            )
        else:
            cap = cv2.VideoCapture(str(rgb_mp4))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError("Failed to read first frame from rgb.mp4")
            cv2.imwrite(str(rgb_path), frame)

    ixt = None
    ixt_path = scene_dir / "camera_matrix.csv"
    if ixt_path.exists():
        ixt = np.loadtxt(ixt_path, delimiter=",")

    return rgb_path, depth_png, ixt


def _infer_promptda(
    backend: LoadedBackend,
    rgb_path: Path,
    prompt_depth_path: Path,
    max_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = backend.device
    image = load_image(str(rgb_path), max_size=max_size).to(device)
    prompt_depth = load_depth(str(prompt_depth_path)).to(device)

    with torch.no_grad():
        depth_t = backend.model.predict(image, prompt_depth)

    depth = depth_t[0, 0].detach().cpu().numpy()
    color = _tensor_to_rgb(image)
    prompt_np = prompt_depth[0, 0].detach().cpu().numpy()
    return color, depth, prompt_np


def _depth_pro_color(image_np: np.ndarray, depth: np.ndarray) -> np.ndarray:
    if image_np.dtype != np.uint8:
        color = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
    else:
        color = image_np.copy()
    if color.shape[0] != depth.shape[0] or color.shape[1] != depth.shape[1]:
        color = cv2.resize(
            color, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA
        )
    return color


def _infer_depth_pro_focal(
    backend: LoadedBackend,
    rgb_path: Path,
    f_px: Optional[float] = None,
    max_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Depth Pro infer that can lock metric scale via focallength_px.

    When ``f_px`` is None the model estimates focal length (use this on frame 0).
    EXIF is ignored so a video JPEG sequence cannot silently re-scale per frame.
    """
    from depth_pro import load_rgb

    image_np, _, _exif_f = load_rgb(rgb_path)
    image_np, scale = resize_max_side(image_np, max_size)
    if f_px is not None and scale != 1.0 and np.isfinite(float(f_px)):
        f_px = float(f_px) * scale
    image_t = backend.transform(image_np)
    infer_f: Any = None
    if f_px is not None and np.isfinite(float(f_px)) and float(f_px) > 1.0:
        infer_f = torch.tensor(
            float(f_px), device=image_t.device, dtype=torch.float32
        )

    with torch.no_grad():
        prediction = backend.model.infer(image_t, f_px=infer_f)

    depth = prediction["depth"].detach().cpu().numpy().squeeze()
    fl_t = prediction.get("focallength_px")
    if fl_t is None:
        fl = float(f_px or 0.0)
    elif hasattr(fl_t, "detach"):
        fl = float(fl_t.detach().cpu().reshape(-1)[0].item())
    else:
        fl = float(fl_t)
    return _depth_pro_color(image_np, depth), depth, fl


def _infer_depth_pro(
    backend: LoadedBackend,
    rgb_path: Path,
    max_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    from depth_pro import load_rgb

    image_np, _, f_px = load_rgb(rgb_path)
    image_np, scale = resize_max_side(image_np, max_size)
    if f_px is not None and scale != 1.0:
        try:
            f_px = float(f_px) * scale
        except (TypeError, ValueError):
            pass
    image_t = backend.transform(image_np)

    with torch.no_grad():
        prediction = backend.model.infer(image_t, f_px=f_px)

    depth = prediction["depth"].detach().cpu().numpy().squeeze()
    return _depth_pro_color(image_np, depth), depth


def _forward_pass_local(
    rgb_path: Path,
    prompt_depth_path: Optional[Path],
    model_key: str,
    max_size: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str, str, Optional[dict]]:
    """Local neural forward pass. Returns color, depth, prompt_np, key, device, billing."""
    backend = get_backend(model_key)
    if backend.kind == "promptda":
        if prompt_depth_path is None:
            raise ValueError(
                "PromptDA requires a LiDAR prompt depth PNG (uint16 mm). "
                "Switch to Depth Pro for RGB-only, or provide prompt depth."
            )
        color, depth, prompt_np = _infer_promptda(
            backend, rgb_path, prompt_depth_path, max_size
        )
    else:
        color, depth = _infer_depth_pro(backend, rgb_path, max_size=max_size)
        prompt_np = None
    return color, depth, prompt_np, backend.key, backend.device, None


def _forward_pass(
    rgb_path: Path,
    prompt_depth_path: Optional[Path],
    model_key: str,
    max_size: int,
    infer_backend: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str, str, str, Optional[dict]]:
    """Dispatch local or Modal forward pass.

    Returns (color, depth, prompt_np, model_key, device, backend_used, modal_billing).
    """
    from depth_core.infer_backend import resolve_backend

    backend_used = resolve_backend(infer_backend)
    if backend_used == "modal":
        from depth_core.infer_modal import forward_pair, forward_rgb

        key = normalize_model_key(model_key)
        if is_promptda(key):
            if prompt_depth_path is None:
                raise ValueError(
                    "PromptDA requires a LiDAR prompt depth PNG (uint16 mm)."
                )
            result = forward_pair(rgb_path, prompt_depth_path, key, int(max_size))
        else:
            result = forward_rgb(rgb_path, key, int(max_size))
        return (
            result.color,
            result.depth,
            result.prompt_np,
            result.model_key,
            result.device,
            "modal",
            result.modal_billing,
        )

    color, depth, prompt_np, key, device, _ = _forward_pass_local(
        rgb_path, prompt_depth_path, model_key, max_size
    )
    return color, depth, prompt_np, key, device, "local", None


def finalize_from_arrays(
    color: np.ndarray,
    depth: np.ndarray,
    prompt_np: Optional[np.ndarray],
    model_key: str,
    max_size: int,
    *,
    source_name: Path,
    ixt: Optional[np.ndarray] = None,
    infer_backend: str = "local",
    infer_device: str = "cpu",
    modal_billing: Optional[dict] = None,
    cache_key: Optional[str] = None,
    arena_block_id: Optional[Any] = None,
) -> dict:
    """Write slim run files (rgb + depth). Looks are rendered on demand by the kitchen."""
    from depth_core.store import materialize_run_arrays

    key = normalize_model_key(model_key)
    depth = np.asarray(depth, dtype=np.float32)
    color = np.asarray(color)
    cliff = _subject_hi_pct(depth)

    vis_prompt = None
    if prompt_np is not None:
        kind = "promptda" if is_promptda(key) else "depth-pro"
        vis_depth, minmax = _colorize_depth(depth, kind)
        if minmax is not None:
            dmin, dmax = minmax
            vis_prompt = visualize_depth(prompt_np, depth_min=dmin, depth_max=dmax)
            vis_prompt = cv2.resize(
                vis_prompt,
                (vis_depth.shape[1], vis_depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

    out_dir = _make_run_dir(source_name, key)
    materialize_run_arrays(
        out_dir,
        color,
        depth,
        cache_key=cache_key,
        cache_meta={
            "model_key": key,
            "max_size": int(max_size) if max_size is not None else MAX_SIZE,
            "arena_block_id": arena_block_id,
        },
    )

    ply_path: Optional[str] = None
    if ixt is not None:
        now_max = max(color.shape[0], color.shape[1])
        scale = 1920.0 / now_max
        ixt = ixt.copy()
        ixt[:2] = ixt[:2] / scale
        points, colors = unproject_depth(depth, ixt=ixt, color=color, ret_pcd=False)
        pcd = trimesh.PointCloud(vertices=points, colors=colors)
        ply_file = out_dir / f"{out_dir.name}_pointcloud.ply"
        pcd.export(ply_file)
        ply_path = str(ply_file)

    state = {
        "rgb": color,
        "depth": depth.astype(np.float32),
        "cliff": cliff,
        "looks": {},
        "vis_prompt": vis_prompt,
        "blend": None,
        "blend_path": None,
        "metric_path": None,
        "look_paths": {},
        "out_dir": str(out_dir),
        "ply_path": ply_path,
        "model_key": key,
        "max_size": int(max_size) if max_size is not None else MAX_SIZE,
        "infer_backend": infer_backend,
        "infer_device": infer_device,
        "cache_key": cache_key,
        "look_a": DEFAULT_LOOK_A,
        "look_b": DEFAULT_LOOK_B,
        "look_c": DEFAULT_LOOK_C,
        "fade_b": 0.0,
        "fade_c": 0.0,
        "blend_mode_b": DEFAULT_BLEND_MODE,
        "blend_mode_c": "screen",
        "scenery_protect": DEFAULT_SCENERY_PROTECT,
        "scenery_feather": DEFAULT_SCENERY_FEATHER,
        "scenery_mode": DEFAULT_SCENERY_MATTE,
        "scenery_clean": DEFAULT_SCENERY_CLEAN,
    }
    if modal_billing:
        state["modal_billing"] = modal_billing
    return state


def _run_inference(
    rgb_path: Path,
    prompt_depth_path: Optional[Path],
    model_key: str,
    max_size: int,
    ixt: Optional[np.ndarray] = None,
    source_name: Optional[Path] = None,
    infer_backend: Optional[str] = None,
    arena_block_id: Optional[Any] = None,
    cache_key: Optional[str] = None,
) -> dict:
    """Run model once (or hit the infer cache); write slim rgb + depth."""
    from depth_core.store import (
        infer_cache_key,
        load_infer_cache,
        release_torch_memory,
    )

    ck = cache_key
    if ck is None:
        try:
            ck = infer_cache_key(
                model_key,
                int(max_size),
                arena_block_id=arena_block_id,
                image_path=rgb_path,
            )
        except ValueError:
            ck = None

    if ck and prompt_depth_path is None:
        hit = load_infer_cache(ck)
        if hit is not None:
            color, depth = hit
            return finalize_from_arrays(
                color,
                depth,
                None,
                model_key,
                int(max_size),
                source_name=source_name or rgb_path,
                ixt=ixt,
                infer_backend="cache",
                infer_device="cache",
                cache_key=ck,
                arena_block_id=arena_block_id,
            )

    color, depth, prompt_np, key, device, backend_used, billing = _forward_pass(
        rgb_path,
        prompt_depth_path,
        model_key,
        max_size,
        infer_backend=infer_backend,
    )
    release_torch_memory()
    return finalize_from_arrays(
        color,
        depth,
        prompt_np,
        key,
        int(max_size),
        source_name=source_name or rgb_path,
        ixt=ixt,
        infer_backend=backend_used,
        infer_device=device,
        modal_billing=billing,
        cache_key=ck,
        arena_block_id=arena_block_id,
    )


def run_zip(
    zip_file, model_key: str, max_size: int, infer_backend: Optional[str] = None
) -> dict:
    if zip_file is None:
        raise ValueError("Upload a Stray Scanner zip file.")
    zip_path = Path(zip_file if isinstance(zip_file, str) else zip_file.name)
    key = normalize_model_key(model_key)
    with tempfile.TemporaryDirectory(prefix="depth_") as tmp:
        rgb_path, depth_path, ixt = _extract_stray_first_frame(zip_path, Path(tmp))
        if is_promptda(key) and depth_path is None:
            raise ValueError(
                "No prompt depth found in zip. PromptDA needs depth/*.png; "
                "use Depth Pro for RGB-only."
            )
        return _run_inference(
            rgb_path,
            depth_path if is_promptda(key) else None,
            key,
            int(max_size),
            ixt=ixt,
            source_name=zip_path,
            infer_backend=infer_backend,
        )


def run_pair(
    rgb_file,
    depth_file,
    model_key: str,
    max_size: int,
    infer_backend: Optional[str] = None,
) -> dict:
    key = normalize_model_key(model_key)
    if not is_promptda(key):
        raise ValueError(
            "RGB + prompt depth is for PromptDA. Switch model to PromptDA, "
            "or use the RGB image tab with Depth Pro."
        )
    if rgb_file is None or depth_file is None:
        raise ValueError("Provide both an RGB image and a prompt depth PNG (uint16 mm).")
    rgb_path = Path(rgb_file if isinstance(rgb_file, str) else rgb_file.name)
    depth_path = Path(depth_file if isinstance(depth_file, str) else depth_file.name)
    return _run_inference(
        rgb_path,
        depth_path,
        key,
        int(max_size),
        ixt=None,
        source_name=rgb_path,
        infer_backend=infer_backend,
    )


def run_rgb(
    rgb_file,
    model_key: str,
    max_size: int,
    infer_backend: Optional[str] = None,
    arena_block_id: Optional[Any] = None,
    cache_key: Optional[str] = None,
) -> dict:
    key = normalize_model_key(model_key)
    if rgb_file is None:
        raise ValueError("Upload an RGB image.")
    if is_promptda(key):
        raise ValueError(
            "PromptDA needs LiDAR prompt depth. Use the RGB + prompt depth tab, "
            "or switch to Depth Pro for a single photo."
        )
    rgb_path = Path(rgb_file if isinstance(rgb_file, str) else rgb_file.name)
    return _run_inference(
        rgb_path,
        None,
        key,
        int(max_size),
        ixt=None,
        source_name=rgb_path,
        infer_backend=infer_backend,
        arena_block_id=arena_block_id,
        cache_key=cache_key,
    )


def update_blend(
    state: Optional[dict],
    look_a: str,
    look_b: str,
    look_c: str,
    fade_b: float,
    fade_c: float,
    mode_b: str = DEFAULT_BLEND_MODE,
    mode_c: str = DEFAULT_BLEND_MODE,
):
    """Recompute triple blend from cached looks (no model)."""
    if state is None or "looks" not in state:
        return None, None, state
    looks = state["looks"]
    a = looks[look_a] if look_a in looks else looks[DEFAULT_LOOK_A]
    b = looks[look_b] if look_b in looks else looks[DEFAULT_LOOK_B]
    c = looks[look_c] if look_c in looks else looks[DEFAULT_LOOK_C]
    blend = blend_looks_triple(a, b, c, fade_b, fade_c, mode_b, mode_c)
    out_dir = Path(state["out_dir"])
    blend_path = out_dir / "blend_live.png"
    imageio.imwrite(blend_path, blend)
    state = dict(state)
    state["blend"] = blend
    state["blend_path"] = str(blend_path)
    state["look_a"] = look_a
    state["look_b"] = look_b
    state["look_c"] = look_c
    state["fade_b"] = float(fade_b)
    state["fade_c"] = float(fade_c)
    state["blend_mode_b"] = mode_b
    state["blend_mode_c"] = mode_c
    _persist_state_recipe(state)
    return blend, str(blend_path), state


def _persist_state_recipe(state: dict) -> None:
    """Write recipe.json for the active run from state + last known settings."""
    out_dir = state.get("out_dir")
    if not out_dir:
        return
    existing = load_recipe(out_dir) or _default_recipe()
    recipe = {
        **existing,
        "look_a": state.get("look_a", existing["look_a"]),
        "look_b": state.get("look_b", existing["look_b"]),
        "look_c": state.get("look_c", existing["look_c"]),
        "fade_b": float(state.get("fade_b", existing["fade_b"])),
        "fade_c": float(state.get("fade_c", existing["fade_c"])),
        "blend_mode_b": state.get("blend_mode_b", existing["blend_mode_b"]),
        "blend_mode_c": state.get("blend_mode_c", existing["blend_mode_c"]),
        "scenery_mode": state.get("scenery_mode", existing["scenery_mode"]),
        "scenery_protect": float(
            state.get("scenery_protect", existing["scenery_protect"])
        ),
        "scenery_feather": float(
            state.get("scenery_feather", existing["scenery_feather"])
        ),
        "scenery_clean": bool(state.get("scenery_clean", existing["scenery_clean"])),
        "model_key": state.get("model_key", existing["model_key"]),
        "max_size": int(state.get("max_size", existing["max_size"])),
    }
    arena = state.get("arena") or existing.get("arena")
    if arena:
        recipe["arena"] = arena
    save_recipe(out_dir, recipe)


def update_scenery(
    state: Optional[dict],
    mode: str,
    protect: float,
    feather: float,
    clean: bool,
    look_a: str,
    look_b: str,
    look_c: str,
    fade_b: float,
    fade_c: float,
    mode_b: str,
    mode_c: str,
):
    """Re-render Scenery from cached depth with new matte settings; refresh blend."""
    if state is None or "depth" not in state:
        return None, None, None, state
    depth = state["depth"]
    cliff = float(state.get("cliff", _subject_hi_pct(depth)))
    mode_key = mode if mode in SCENERY_MATTE_MODES else DEFAULT_SCENERY_MATTE
    scenery = _render_scenery_look(
        depth,
        cliff,
        protect=float(protect),
        feather=float(feather),
        mode=mode_key,
        clean_specks=bool(clean),
    )
    state = dict(state)
    looks = dict(state["looks"])
    looks["scenery"] = scenery
    state["looks"] = looks
    state["scenery_protect"] = float(protect)
    state["scenery_feather"] = float(feather)
    state["scenery_mode"] = mode_key
    state["scenery_clean"] = bool(clean)

    out_dir = Path(state["out_dir"])
    imageio.imwrite(out_dir / "look_scenery_live.png", scenery)

    blend, blend_path, state = update_blend(
        state, look_a, look_b, look_c, fade_b, fade_c, mode_b, mode_c
    )
    return _downscale_thumb(scenery), blend, blend_path, state

