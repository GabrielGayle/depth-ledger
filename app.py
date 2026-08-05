"""Local Gradio UI for PromptDA and Apple Depth Pro.

PromptDA: RGB + LiDAR prompt depth (Stray Scanner zip, or image/depth pair).
Depth Pro: RGB only (single photo).
"""

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
import gradio as gr
import imageio
import matplotlib
import numpy as np
import torch
import trimesh

from promptda.promptda import PromptDA
from promptda.utils.depth_utils import unproject_depth, visualize_depth
from promptda.utils.io_wrapper import load_depth, load_image

try:
    from gradio_imageslider import ImageSlider
except Exception:  # ImportError on missing pkg; TypeError on Python < 3.10
    ImageSlider = None


def _patch_gradio_bool_schema() -> None:
    """Gradio 4.44 crashes when JSON Schema additionalProperties is a bool (pydantic 2.11+)."""
    try:
        import gradio_client.utils as gu
    except ImportError:
        return
    original = gu.get_type

    def get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return original(schema)

    gu.get_type = get_type


_patch_gradio_bool_schema()

ROOT = Path(__file__).resolve().parent
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


def _norm_to_rgb(norm: np.ndarray, cmap: str) -> np.ndarray:
    if cmap in (None, "gray", "grey"):
        u8 = _float01_to_u8_dither(norm.astype(np.float32))
        return np.stack([u8, u8, u8], axis=-1)
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
) -> np.ndarray:
    """Scene-adaptive inverse depth preview (near = bright)."""
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

    lo, hi = 1.0 / far, 1.0 / near
    norm = np.zeros_like(depth, dtype=np.float64)
    inv_c = np.clip(inv_smooth.astype(np.float64), lo, hi)
    # Use full valid mask for display (include far as black)
    show = np.isfinite(depth) & (depth > 1e-4)
    norm[show] = ((inv_c[show] - lo) / (hi - lo)).clip(0.0, 1.0)
    if mask_below_m is not None:
        norm[depth < float(mask_below_m)] = 0.0
    elif mask_below_pct is not None:
        # Punch out foreground (subject) so only mid/far scenery remains.
        thr = float(np.percentile(depth[valid], float(mask_below_pct)))
        norm[depth < thr] = 0.0
    if gamma != 1.0:
        norm = np.power(norm, gamma)
    norm = _apply_grade(norm, grade)
    return _norm_to_rgb(norm, cmap)


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


def history_dropdown_update(selected: Optional[str] = None):
    choices = list_recent_runs()
    if not choices:
        return gr.update(choices=[], value=None)
    values = {v for _, v in choices}
    value = selected if selected in values else choices[0][1]
    return gr.update(choices=choices, value=value)



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
) -> np.ndarray:
    """Far-field look: raw inv, adjustable subject matte, building grade."""
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
    norm = norm * keep
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
            )
        else:
            out[key] = _visualize_inverse_depth(
                depth_m,
                cmap=cfg["cmap"],
                lo_pct=_resolve_lo_pct(cfg, cliff),
                hi_pct=_resolve_hi_pct(cfg, cliff),
                gamma=cfg["gamma"],
                grade=cfg.get("grade"),
                inv_smooth=inv_smooth,
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


def _infer_depth_pro(
    backend: LoadedBackend, rgb_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    from depth_pro import load_rgb

    image_np, _, f_px = load_rgb(rgb_path)
    image_t = backend.transform(image_np)

    with torch.no_grad():
        prediction = backend.model.infer(image_t, f_px=f_px)

    depth = prediction["depth"].detach().cpu().numpy().squeeze()
    # load_rgb returns HxWx3 uint8 (or float); normalize to uint8 RGB for viz/PLY.
    if image_np.dtype != np.uint8:
        color = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
    else:
        color = image_np.copy()
    # Depth Pro resizes internally; match color to depth spatial size for PLY/viz.
    if color.shape[0] != depth.shape[0] or color.shape[1] != depth.shape[1]:
        color = cv2.resize(
            color, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA
        )
    return color, depth


def _run_inference(
    rgb_path: Path,
    prompt_depth_path: Optional[Path],
    model_key: str,
    max_size: int,
    ixt: Optional[np.ndarray] = None,
    source_name: Optional[Path] = None,
) -> dict:
    """Run model once; return RGB, metric paths, prompt viz, and all Looks."""
    backend = get_backend(model_key)

    if backend.kind == "promptda":
        if prompt_depth_path is None:
            raise gr.Error(
                "PromptDA requires a LiDAR prompt depth PNG (uint16 mm). "
                "Switch to Depth Pro for RGB-only, or provide prompt depth."
            )
        color, depth, prompt_np = _infer_promptda(
            backend, rgb_path, prompt_depth_path, max_size
        )
    else:
        color, depth = _infer_depth_pro(backend, rgb_path)
        prompt_np = None

    looks = render_looks(depth)
    cliff = _subject_hi_pct(depth)
    vis_depth = looks[DEFAULT_LOOK_A]

    vis_prompt = None
    if prompt_np is not None:
        _, minmax = _colorize_depth(depth, backend.kind)
        if minmax is not None:
            dmin, dmax = minmax
            vis_prompt = visualize_depth(prompt_np, depth_min=dmin, depth_max=dmax)
            vis_prompt = cv2.resize(
                vis_prompt,
                (vis_depth.shape[1], vis_depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

    label_src = source_name or rgb_path
    out_dir = _make_run_dir(label_src, backend.key)
    run_prefix = out_dir.name
    metric_path = _save_depth_mm(depth, out_dir / f"{run_prefix}_depth_mm.png")
    np.savez_compressed(out_dir / f"{run_prefix}_depth.npz", depth=depth.astype(np.float32))
    imageio.imwrite(out_dir / f"{run_prefix}_rgb.jpg", color)

    look_paths = {}
    for key, img in looks.items():
        p = out_dir / f"{run_prefix}_look_{key}.png"
        imageio.imwrite(p, img)
        look_paths[key] = str(p)

    blend = blend_looks_triple(
        looks[DEFAULT_LOOK_A],
        looks[DEFAULT_LOOK_B],
        looks[DEFAULT_LOOK_C],
        0.0,
        0.0,
        DEFAULT_BLEND_MODE,
        DEFAULT_BLEND_MODE,
    )
    blend_path = out_dir / f"{run_prefix}_blend.png"
    imageio.imwrite(blend_path, blend)

    ply_path: Optional[str] = None
    if ixt is not None:
        now_max = max(color.shape[0], color.shape[1])
        scale = 1920.0 / now_max
        ixt = ixt.copy()
        ixt[:2] = ixt[:2] / scale
        points, colors = unproject_depth(depth, ixt=ixt, color=color, ret_pcd=False)
        pcd = trimesh.PointCloud(vertices=points, colors=colors)
        ply_file = out_dir / f"{run_prefix}_pointcloud.ply"
        pcd.export(ply_file)
        ply_path = str(ply_file)

    return {
        "rgb": color,
        "depth": depth.astype(np.float32),
        "cliff": cliff,
        "looks": looks,
        "vis_prompt": vis_prompt,
        "blend": blend,
        "blend_path": str(blend_path),
        "metric_path": str(metric_path),
        "look_paths": look_paths,
        "out_dir": str(out_dir),
        "ply_path": ply_path,
        "model_key": backend.key,
        "max_size": int(max_size) if max_size is not None else MAX_SIZE,
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


def run_zip(zip_file, model_key: str, max_size: int) -> dict:
    if zip_file is None:
        raise gr.Error("Upload a Stray Scanner zip file.")
    zip_path = Path(zip_file if isinstance(zip_file, str) else zip_file.name)
    key = normalize_model_key(model_key)
    with tempfile.TemporaryDirectory(prefix="depth_") as tmp:
        rgb_path, depth_path, ixt = _extract_stray_first_frame(zip_path, Path(tmp))
        if is_promptda(key) and depth_path is None:
            raise gr.Error(
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
        )


def run_pair(rgb_file, depth_file, model_key: str, max_size: int) -> dict:
    key = normalize_model_key(model_key)
    if not is_promptda(key):
        raise gr.Error(
            "RGB + prompt depth is for PromptDA. Switch model to PromptDA, "
            "or use the RGB image tab with Depth Pro."
        )
    if rgb_file is None or depth_file is None:
        raise gr.Error("Provide both an RGB image and a prompt depth PNG (uint16 mm).")
    rgb_path = Path(rgb_file if isinstance(rgb_file, str) else rgb_file.name)
    depth_path = Path(depth_file if isinstance(depth_file, str) else depth_file.name)
    return _run_inference(
        rgb_path, depth_path, key, int(max_size), ixt=None, source_name=rgb_path
    )


def run_rgb(rgb_file, model_key: str, max_size: int) -> dict:
    key = normalize_model_key(model_key)
    if rgb_file is None:
        raise gr.Error("Upload an RGB image.")
    if is_promptda(key):
        raise gr.Error(
            "PromptDA needs LiDAR prompt depth. Use the RGB + prompt depth tab, "
            "or switch to Depth Pro for a single photo."
        )
    rgb_path = Path(rgb_file if isinstance(rgb_file, str) else rgb_file.name)
    return _run_inference(
        rgb_path, None, key, int(max_size), ixt=None, source_name=rgb_path
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



# --- Brand CSS (Wondrous Isles) ---------------------------------------------

BRAND_CSS = """
:root {
  --bg-color: #f4f4f4;
  --line-color: #333333;
  --line-color-light: #aaaaaa;
  --next-color: #0044ff;
  --past-color: #666666;
  --ui-font: 'Courier New', Courier, monospace;
}

.gradio-container {
  font-family: var(--ui-font) !important;
  background: var(--bg-color) !important;
  color: var(--line-color) !important;
  max-width: 1100px !important;
}

.gradio-container, .gradio-container * {
  border-radius: 0 !important;
}

/* Soften Gradio's default blue / rounded chrome */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .svelte-1f354aw,
.gradio-container .wrap {
  border-radius: 0 !important;
}

.gradio-container .block {
  border: 1px solid var(--line-color) !important;
  box-shadow: 6px 6px 0 rgba(0,0,0,0.2) !important;
  background: var(--bg-color) !important;
}

.gradio-container label,
.gradio-container .label-wrap span,
.gradio-container .tab-nav button,
.gradio-container button {
  font-family: var(--ui-font) !important;
  text-transform: uppercase !important;
  font-weight: normal !important;
  letter-spacing: 0 !important;
}

.gradio-container label,
.gradio-container .label-wrap span {
  font-size: 0.65rem !important;
  color: var(--line-color) !important;
}

/* Slider / accordion heads — Gradio paints these light-on-light without this */
.gradio-container .head,
.gradio-container .slider .head,
.gradio-container .block .head,
.gradio-container span[data-testid="block-info"],
.gradio-container .block-title,
.gradio-container .block-label,
.gradio-container .block-label span {
  color: var(--line-color) !important;
  opacity: 1 !important;
}

.gradio-container h1,
.gradio-container .prose h1 {
  font-family: var(--ui-font) !important;
  font-size: 1rem !important;
  font-weight: normal !important;
  text-transform: uppercase !important;
  margin-bottom: 8px !important;
  color: var(--line-color) !important;
}

.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .markdown,
.gradio-container .md {
  font-family: var(--ui-font) !important;
  font-size: 0.7rem !important;
  color: var(--line-color) !important;
  opacity: 1;
}

.gradio-container .prose a {
  color: var(--next-color) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .wrap-inner {
  font-family: var(--ui-font) !important;
  font-size: 0.7rem !important;
  border: 1px solid var(--line-color) !important;
  border-radius: 0 !important;
  background: var(--bg-color) !important;
  color: var(--line-color) !important;
  box-shadow: none !important;
}

/* Inner dropdown chrome — no second border (avoids double frame) */
.gradio-container .secondary-wrap,
.gradio-container .wrap-inner .secondary-wrap {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  outline: none !important;
}

.gradio-container .tab-nav {
  border-bottom: 1px solid var(--line-color) !important;
  gap: 0 !important;
}

.gradio-container .tab-nav button {
  font-size: 0.65rem !important;
  padding: 5px 12px !important;
  border: 1px solid var(--line-color) !important;
  border-bottom: none !important;
  background: var(--bg-color) !important;
  color: var(--line-color) !important;
  box-shadow: none !important;
}

.gradio-container .tab-nav button.selected {
  background: var(--line-color) !important;
  color: var(--bg-color) !important;
}

.gradio-container button.primary,
.gradio-container button[variant="primary"],
.gradio-container .primary {
  background: var(--line-color) !important;
  color: var(--bg-color) !important;
  border: 1px solid var(--line-color) !important;
  font-size: 0.65rem !important;
  padding: 5px 12px !important;
  box-shadow: 2px 2px 0 rgba(0,0,0,0.2) !important;
}

.gradio-container button.primary:hover,
.gradio-container .primary:hover {
  background: var(--bg-color) !important;
  color: var(--line-color) !important;
}

.gradio-container button.secondary,
.gradio-container button:not(.primary) {
  background: var(--bg-color) !important;
  color: var(--line-color) !important;
  border: 1px solid var(--line-color) !important;
  font-size: 0.65rem !important;
  padding: 5px 12px !important;
  box-shadow: none !important;
}

.gradio-container button.secondary:hover {
  background: var(--line-color) !important;
  color: var(--bg-color) !important;
}

.gradio-container footer,
.gradio-container .footer {
  display: none !important;
}

#model-help {
  font-size: 0.65rem !important;
  text-transform: none !important;
  opacity: 0.75;
  font-style: italic;
}

/* --- Progress / loader (keep tickers visible; hide HF logo only) --- */
:root {
  --loader-color: #333333;
  --color-accent: #333333;
  --color-accent-soft: rgba(51, 51, 51, 0.12);
}

/* Hide HuggingFace logo spinner only — do not blanket-hide all SVGs */
.wrap.loading > svg.svelte-*,
.loading > svg[viewBox="0 0 100 100"],
svg.subdued {
  display: none !important;
}

/* Hard-edged ink spinner fallback */
.wrap.loading::before,
.loading::before {
  content: "";
  display: block;
  width: 22px;
  height: 22px;
  margin: 0 auto 10px;
  border: 2px solid var(--line-color);
  border-top-color: transparent;
  border-radius: 0;
  animation: ledger-spin 0.7s linear infinite;
}

@keyframes ledger-spin {
  to { transform: rotate(360deg); }
}

.generating {
  border: 1px solid var(--line-color) !important;
  background: var(--bg-color) !important;
  animation: none !important;
  opacity: 1 !important;
}

/* Progress tickers — force readable ink on paper */
.progress-text,
.meta-text,
.meta-text-center,
.eta-text,
.toast-body,
.toast-title,
.pending,
.status {
  font-family: var(--ui-font) !important;
  font-size: 0.65rem !important;
  text-transform: uppercase !important;
  color: var(--line-color) !important;
  opacity: 1 !important;
  visibility: visible !important;
}

.progress-bar,
.eta-bar,
.progress-level,
.progress-bar-wrap .progress-bar {
  display: block !important;
  visibility: visible !important;
  opacity: 1 !important;
  background: var(--line-color) !important;
  min-height: 4px !important;
}

.progress-bar-wrap {
  display: block !important;
  visibility: visible !important;
  background: var(--line-color-light) !important;
  border: 1px solid var(--line-color) !important;
  opacity: 1 !important;
}

/* Dropdown menus / portals — dark text on light */
ul.options,
ul[role="listbox"],
.dropdown ul,
.svelte-select-list,
div[class*="options"] {
  background: #f4f4f4 !important;
  color: #333333 !important;
  border: 1px solid #333333 !important;
}

ul.options li,
ul[role="listbox"] li,
.svelte-select-list .item,
div[class*="options"] > * {
  color: #333333 !important;
  background: #f4f4f4 !important;
}

ul.options li:hover,
ul.options li.selected,
ul[role="listbox"] [aria-selected="true"],
.svelte-select-list .item:hover {
  background: #333333 !important;
  color: #f4f4f4 !important;
}

/* Sliders / checkboxes */
input[type="range"] {
  accent-color: #333333 !important;
}
.gradio-container .icon-wrap,
.gradio-container .empty,
.gradio-container .upload-container,
.gradio-container .or,
.gradio-container .upload-text,
.gradio-container .wrap.center.full {
  color: #333333 !important;
  opacity: 0.85 !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: #666666 !important;
  opacity: 1 !important;
}
.gradio-container .checkbox-label,
.gradio-container span[data-testid="checkbox"] + span {
  color: #333333 !important;
}
.gradio-container .file-preview,
.gradio-container .file,
.gradio-container .filename,
.gradio-container .file-name {
  color: #333333 !important;
  background: #f4f4f4 !important;
}

/* Image toolbar (fullscreen / download) — white icons on dark chips */
.gradio-container .image-container .icon-button,
.gradio-container .image-container .icon-button button,
.gradio-container .image-container button.svelte-rk35yg,
.gradio-container .image-container button[aria-label],
.gradio-container .image-frame button,
.gradio-container button[title="View in full screen"],
.gradio-container button[title="Download"],
.gradio-container button[aria-label="View in full screen"],
.gradio-container button[aria-label="Download"],
.gradio-container button[aria-label="Share"] {
  background: #333333 !important;
  color: #f4f4f4 !important;
  border: 1px solid #333333 !important;
  box-shadow: none !important;
  padding: 4px 6px !important;
  min-width: 28px !important;
  min-height: 24px !important;
  text-transform: none !important;
  opacity: 1 !important;
}
.gradio-container .image-container .icon-button button svg,
.gradio-container .image-container button svg,
.gradio-container .image-frame button svg,
.gradio-container button[aria-label="View in full screen"] svg,
.gradio-container button[aria-label="Download"] svg {
  display: block !important;
  width: 14px !important;
  height: 14px !important;
  fill: #f4f4f4 !important;
  stroke: #f4f4f4 !important;
  color: #f4f4f4 !important;
  opacity: 1 !important;
  visibility: visible !important;
}

/* Upload / webcam / clipboard source icons — keep glyphs visible */
.gradio-container .source-selection,
.gradio-container span.source-selection {
  display: inline-flex !important;
  align-items: center !important;
  gap: 4px !important;
  overflow: visible !important;
}
.gradio-container .source-selection button,
.gradio-container .source-selection .icon,
.gradio-container span.source-selection button {
  background: transparent !important;
  color: #333333 !important;
  border: 1px solid transparent !important;
  box-shadow: none !important;
  padding: 4px !important;
  min-width: 26px !important;
  min-height: 22px !important;
  overflow: visible !important;
  text-transform: none !important;
  opacity: 1 !important;
}
.gradio-container .source-selection button.selected,
.gradio-container .source-selection .icon.selected {
  background: #333333 !important;
  color: #f4f4f4 !important;
  border-color: #333333 !important;
}
.gradio-container .source-selection button svg,
.gradio-container .source-selection .icon svg {
  display: block !important;
  width: 16px !important;
  height: 16px !important;
  max-width: none !important;
  max-height: none !important;
  fill: currentColor !important;
  stroke: currentColor !important;
  color: inherit !important;
  opacity: 1 !important;
  visibility: visible !important;
}

#recipe-panel {
  border: 1px solid var(--line-color) !important;
  box-shadow: 6px 6px 0 rgba(0,0,0,0.2) !important;
  background: var(--bg-color) !important;
  padding: 10px 12px !important;
}
#recipe-panel .prose,
#recipe-panel .markdown,
#recipe-detail {
  font-size: 0.7rem !important;
  text-transform: none !important;
  color: var(--line-color) !important;
}

/* Looks thumbs — inverted rainbow labels (violet→red) over gray depth */
:root {
  --look-ledger: #6b2d9b;
  --look-magma: #3d5ae8;
  --look-punchy: #1a9fd4;
  --look-soft: #1f9a4a;
  --look-portrait: #c9a000;
  --look-sockets: #e07010;
  --look-scenery: #d62828;
}

.gradio-container .look-thumb {
  box-shadow: 4px 4px 0 rgba(0,0,0,0.18) !important;
}
.gradio-container .look-thumb .block-label,
.gradio-container .look-thumb span.block-info,
.gradio-container .look-thumb .label-wrap,
.gradio-container .look-thumb label,
.gradio-container .look-thumb .block-title {
  font-size: 0.7rem !important;
  font-weight: bold !important;
  letter-spacing: 0.02em !important;
  padding: 3px 8px !important;
  opacity: 1 !important;
  border: 1px solid #111111 !important;
  text-shadow: none !important;
}

.gradio-container .look-ledger {
  border-color: var(--look-ledger) !important;
  border-width: 2px !important;
}
.gradio-container .look-ledger .block-label,
.gradio-container .look-ledger span.block-info,
.gradio-container .look-ledger .label-wrap,
.gradio-container .look-ledger label,
.gradio-container .look-ledger .block-title {
  background: var(--look-ledger) !important;
  color: #ffffff !important;
  border-color: var(--look-ledger) !important;
}

.gradio-container .look-magma {
  border-color: var(--look-magma) !important;
  border-width: 2px !important;
}
.gradio-container .look-magma .block-label,
.gradio-container .look-magma span.block-info,
.gradio-container .look-magma .label-wrap,
.gradio-container .look-magma label,
.gradio-container .look-magma .block-title {
  background: var(--look-magma) !important;
  color: #ffffff !important;
  border-color: var(--look-magma) !important;
}

.gradio-container .look-punchy {
  border-color: var(--look-punchy) !important;
  border-width: 2px !important;
}
.gradio-container .look-punchy .block-label,
.gradio-container .look-punchy span.block-info,
.gradio-container .look-punchy .label-wrap,
.gradio-container .look-punchy label,
.gradio-container .look-punchy .block-title {
  background: var(--look-punchy) !important;
  color: #111111 !important;
  border-color: var(--look-punchy) !important;
}

.gradio-container .look-soft {
  border-color: var(--look-soft) !important;
  border-width: 2px !important;
}
.gradio-container .look-soft .block-label,
.gradio-container .look-soft span.block-info,
.gradio-container .look-soft .label-wrap,
.gradio-container .look-soft label,
.gradio-container .look-soft .block-title {
  background: var(--look-soft) !important;
  color: #ffffff !important;
  border-color: var(--look-soft) !important;
}

.gradio-container .look-portrait {
  border-color: var(--look-portrait) !important;
  border-width: 2px !important;
}
.gradio-container .look-portrait .block-label,
.gradio-container .look-portrait span.block-info,
.gradio-container .look-portrait .label-wrap,
.gradio-container .look-portrait label,
.gradio-container .look-portrait .block-title {
  background: var(--look-portrait) !important;
  color: #111111 !important;
  border-color: var(--look-portrait) !important;
}

.gradio-container .look-sockets {
  border-color: var(--look-sockets) !important;
  border-width: 2px !important;
}
.gradio-container .look-sockets .block-label,
.gradio-container .look-sockets span.block-info,
.gradio-container .look-sockets .label-wrap,
.gradio-container .look-sockets label,
.gradio-container .look-sockets .block-title {
  background: var(--look-sockets) !important;
  color: #111111 !important;
  border-color: var(--look-sockets) !important;
}

.gradio-container .look-scenery {
  border-color: var(--look-scenery) !important;
  border-width: 2px !important;
}
.gradio-container .look-scenery .block-label,
.gradio-container .look-scenery span.block-info,
.gradio-container .look-scenery .label-wrap,
.gradio-container .look-scenery label,
.gradio-container .look-scenery .block-title {
  background: var(--look-scenery) !important;
  color: #ffffff !important;
  border-color: var(--look-scenery) !important;
}
"""


def _model_help(model_key: str) -> str:
    key = normalize_model_key(model_key)
    if key == "depth-pro":
        return (
            "Depth Pro — RGB only. Drop a single photo. "
            "Internal resolution 1536×1536; MPS preferred on Apple Silicon."
        )
    return (
        "PromptDA — needs RGB + LiDAR prompt depth (Stray Scanner zip or PNG pair). "
        "Defaults to CPU on Apple Silicon."
    )


def _max_size_interactive(model_key: str):
    return gr.update(interactive=is_promptda(model_key))


DESCRIPTION = f"""
# Depth Ledger

Metric depth from a photo — [Apple Depth Pro](https://github.com/apple/ml-depth-pro)
or [PromptDA](https://promptda.github.io/) (RGB + LiDAR).

Default model: **{DEFAULT_MODEL}**

**Looks** — one infer, seven graded previews (**Portrait** / **Sockets** for faces,
**Scenery** for far buildings with an adjustable subject matte). Stack Look B then C
over A with Photoshop-style blend modes. **Download blend** exports the result.
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Depth Ledger", css=BRAND_CSS) as demo:
        gr.Markdown(DESCRIPTION)
        looks_state = gr.State(None)

        with gr.Row():
            model_key = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=DEFAULT_MODEL,
                label="Model",
            )
            max_size = gr.Dropdown(
                choices=[518, 756, 1008],
                value=MAX_SIZE if MAX_SIZE in (518, 756, 1008) else 756,
                label="Max image side (PromptDA)",
                interactive=is_promptda(DEFAULT_MODEL),
            )

        model_help = gr.Markdown(_model_help(DEFAULT_MODEL), elem_id="model-help")
        model_key.change(_model_help, inputs=[model_key], outputs=[model_help])
        model_key.change(_max_size_interactive, inputs=[model_key], outputs=[max_size])

        with gr.Tabs():
            with gr.Tab("RGB image"):
                rgb_only_in = gr.Image(type="filepath", label="RGB image")
                rgb_btn = gr.Button("Generate depth", variant="primary")
                if EXAMPLE_RGB.exists():
                    gr.Examples(
                        examples=[[str(EXAMPLE_RGB)]],
                        inputs=[rgb_only_in],
                        label="Bundled example",
                    )

            with gr.Tab("Stray Scanner zip"):
                zip_in = gr.File(label="Stray Scanner capture (.zip)", file_types=[".zip"])
                zip_btn = gr.Button("Generate depth", variant="primary")

            with gr.Tab("RGB + prompt depth"):
                rgb_in = gr.Image(type="filepath", label="RGB image")
                depth_in = gr.Image(type="filepath", label="Prompt depth PNG (uint16, mm)")
                pair_btn = gr.Button("Generate depth", variant="primary")
                if EXAMPLE_RGB.exists() and EXAMPLE_DEPTH.exists():
                    gr.Examples(
                        examples=[[str(EXAMPLE_RGB), str(EXAMPLE_DEPTH)]],
                        inputs=[rgb_in, depth_in],
                        label="Bundled example",
                    )

        with gr.Row():
            out_rgb = gr.Image(type="numpy", label="Source RGB", format="png")
            out_blend = gr.Image(
                type="numpy", label="Blend preview (near = bright)", format="png"
            )
            out_prompt = gr.Image(
                type="numpy",
                label="LiDAR prompt (PromptDA only)",
                format="png",
                visible=False,
            )

        gr.Markdown("### Mixer — A ← B ← C")
        with gr.Row():
            look_a = gr.Dropdown(
                choices=LOOK_CHOICES, value=DEFAULT_LOOK_A, label="Look A (base)"
            )
            look_b = gr.Dropdown(
                choices=LOOK_CHOICES, value=DEFAULT_LOOK_B, label="Look B"
            )
            look_c = gr.Dropdown(
                choices=LOOK_CHOICES, value=DEFAULT_LOOK_C, label="Look C"
            )
        with gr.Row():
            blend_mode_b = gr.Dropdown(
                choices=BLEND_MODE_CHOICES,
                value=DEFAULT_BLEND_MODE,
                label="Mode B",
            )
            fade_b = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.01,
                label="Opacity B",
            )
            blend_mode_c = gr.Dropdown(
                choices=BLEND_MODE_CHOICES,
                value="screen",
                label="Mode C",
            )
            fade_c = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.01,
                label="Opacity C",
            )

        gr.Markdown("### Looks")
        with gr.Row():
            thumb_ledger = gr.Image(
                type="numpy",
                label="Ledger gray",
                format="png",
                elem_classes=["look-thumb", "look-ledger"],
            )
            thumb_magma = gr.Image(
                type="numpy",
                label="Magma",
                format="png",
                elem_classes=["look-thumb", "look-magma"],
            )
            thumb_punchy = gr.Image(
                type="numpy",
                label="Punchy near",
                format="png",
                elem_classes=["look-thumb", "look-punchy"],
            )
            thumb_soft = gr.Image(
                type="numpy",
                label="Soft far",
                format="png",
                elem_classes=["look-thumb", "look-soft"],
            )
        with gr.Row():
            thumb_portrait = gr.Image(
                type="numpy",
                label="Portrait",
                format="png",
                elem_classes=["look-thumb", "look-portrait"],
            )
            thumb_sockets = gr.Image(
                type="numpy",
                label="Sockets",
                format="png",
                elem_classes=["look-thumb", "look-sockets"],
            )
            thumb_scenery = gr.Image(
                type="numpy",
                label="Scenery",
                format="png",
                elem_classes=["look-thumb", "look-scenery"],
            )

        gr.Markdown("### Scenery matte")
        with gr.Row():
            scenery_mode = gr.Dropdown(
                choices=SCENERY_MATTE_CHOICES,
                value=DEFAULT_SCENERY_MATTE,
                label="Matte mode",
            )
            scenery_protect = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=DEFAULT_SCENERY_PROTECT,
                step=0.01,
                label="Protect (0 = recover buildings, 1 = heavy cut)",
            )
            scenery_feather = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=DEFAULT_SCENERY_FEATHER,
                step=0.01,
                label="Feather",
            )
            scenery_clean = gr.Checkbox(
                value=DEFAULT_SCENERY_CLEAN,
                label="Clean edge speckles",
            )

        with gr.Group(elem_id="recipe-panel"):
            gr.Markdown("### Run history — ingredients")
            _hist_choices = list_recent_runs()
            _hist_value = _hist_choices[0][1] if _hist_choices else None
            with gr.Row():
                history_dd = gr.Dropdown(
                    choices=_hist_choices,
                    value=_hist_value,
                    label="Recent runs",
                    interactive=True,
                )
                history_refresh = gr.Button("Refresh", variant="secondary")
                history_reapply = gr.Button("Reapply settings", variant="primary")
            recipe_detail = gr.Markdown(
                format_recipe_md(
                    load_recipe(_hist_value) if _hist_value else None,
                    Path(_hist_value).name if _hist_value else "",
                ),
                elem_id="recipe-detail",
            )

        with gr.Row():
            out_blend_file = gr.File(label="Download blend preview (PNG)")
            out_metric_file = gr.File(label="Download metric depth (uint16 mm, optional)")
            out_ply = gr.File(label="Download point cloud (.ply, optional)")

        def _pack(result: dict, recipe: dict):
            looks = result["looks"]
            show_prompt = result["vis_prompt"] is not None
            state = dict(result)
            state.update(
                {
                    "model_key": recipe["model_key"],
                    "max_size": recipe["max_size"],
                    "look_a": recipe["look_a"],
                    "look_b": recipe["look_b"],
                    "look_c": recipe["look_c"],
                    "fade_b": recipe["fade_b"],
                    "fade_c": recipe["fade_c"],
                    "blend_mode_b": recipe["blend_mode_b"],
                    "blend_mode_c": recipe["blend_mode_c"],
                    "scenery_mode": recipe["scenery_mode"],
                    "scenery_protect": recipe["scenery_protect"],
                    "scenery_feather": recipe["scenery_feather"],
                    "scenery_clean": recipe["scenery_clean"],
                }
            )
            save_recipe(state["out_dir"], recipe)

            # Apply live mixer/scenery from the UI so the first frame matches ingredients.
            scenery_thumb, blend, blend_path, state = update_scenery(
                state,
                recipe["scenery_mode"],
                recipe["scenery_protect"],
                recipe["scenery_feather"],
                recipe["scenery_clean"],
                recipe["look_a"],
                recipe["look_b"],
                recipe["look_c"],
                recipe["fade_b"],
                recipe["fade_c"],
                recipe["blend_mode_b"],
                recipe["blend_mode_c"],
            )
            hist = history_dropdown_update(str(Path(state["out_dir"]).resolve()))
            detail = format_recipe_md(load_recipe(state["out_dir"]))
            return (
                state,
                state["rgb"],
                blend,
                gr.update(value=state["vis_prompt"], visible=show_prompt),
                _downscale_thumb(looks["ledger"]),
                _downscale_thumb(looks["magma"]),
                _downscale_thumb(looks["punchy"]),
                _downscale_thumb(looks["soft"]),
                _downscale_thumb(looks["portrait"]),
                _downscale_thumb(looks["sockets"]),
                scenery_thumb,
                blend_path,
                state["metric_path"],
                state["ply_path"],
                hist,
                detail,
            )

        def _recipe_from_ui(m, s, a, b, c, tb, tc, mb, mc, sm, sp, sf, sc):
            return build_recipe(m, s, a, b, c, tb, tc, mb, mc, sm, sp, sf, sc)

        recipe_inputs = [
            model_key,
            max_size,
            look_a,
            look_b,
            look_c,
            fade_b,
            fade_c,
            blend_mode_b,
            blend_mode_c,
            scenery_mode,
            scenery_protect,
            scenery_feather,
            scenery_clean,
        ]

        gen_outputs = [
            looks_state,
            out_rgb,
            out_blend,
            out_prompt,
            thumb_ledger,
            thumb_magma,
            thumb_punchy,
            thumb_soft,
            thumb_portrait,
            thumb_sockets,
            thumb_scenery,
            out_blend_file,
            out_metric_file,
            out_ply,
            history_dd,
            recipe_detail,
        ]

        def _gen_rgb(r, m, s, a, b, c, tb, tc, mb, mc, sm, sp, sf, sc):
            recipe = _recipe_from_ui(m, s, a, b, c, tb, tc, mb, mc, sm, sp, sf, sc)
            return _pack(run_rgb(r, m, s), recipe)

        def _gen_zip(z, m, s, a, b, c, tb, tc, mb, mc, sm, sp, sf, sc):
            recipe = _recipe_from_ui(m, s, a, b, c, tb, tc, mb, mc, sm, sp, sf, sc)
            return _pack(run_zip(z, m, s), recipe)

        def _gen_pair(r, d, m, s, a, b, c, tb, tc, mb, mc, sm, sp, sf, sc):
            recipe = _recipe_from_ui(m, s, a, b, c, tb, tc, mb, mc, sm, sp, sf, sc)
            return _pack(run_pair(r, d, m, s), recipe)

        rgb_btn.click(
            _gen_rgb,
            inputs=[rgb_only_in, *recipe_inputs],
            outputs=gen_outputs,
            show_progress="full",
        )
        zip_btn.click(
            _gen_zip,
            inputs=[zip_in, *recipe_inputs],
            outputs=gen_outputs,
            show_progress="full",
        )
        pair_btn.click(
            _gen_pair,
            inputs=[rgb_in, depth_in, *recipe_inputs],
            outputs=gen_outputs,
            show_progress="full",
        )

        def _on_fade(state, a, b, c, tb, tc, mb, mc):
            blend, path, new_state = update_blend(state, a, b, c, tb, tc, mb, mc)
            detail = (
                format_recipe_md(load_recipe(new_state["out_dir"]))
                if new_state and new_state.get("out_dir")
                else gr.update()
            )
            return blend, path, new_state, detail

        blend_ctrls = (
            look_a,
            look_b,
            look_c,
            fade_b,
            fade_c,
            blend_mode_b,
            blend_mode_c,
        )
        for ctrl in blend_ctrls:
            ctrl.change(
                _on_fade,
                inputs=[looks_state, *blend_ctrls],
                outputs=[out_blend, out_blend_file, looks_state, recipe_detail],
            )

        def _on_scenery(state, mode, protect, feather, clean, a, b, c, tb, tc, mb, mc):
            thumb, blend, path, new_state = update_scenery(
                state, mode, protect, feather, clean, a, b, c, tb, tc, mb, mc
            )
            detail = (
                format_recipe_md(load_recipe(new_state["out_dir"]))
                if new_state and new_state.get("out_dir")
                else gr.update()
            )
            return thumb, blend, path, new_state, detail

        scenery_ctrls = (scenery_mode, scenery_protect, scenery_feather, scenery_clean)
        for ctrl in scenery_ctrls:
            ctrl.change(
                _on_scenery,
                inputs=[looks_state, *scenery_ctrls, *blend_ctrls],
                outputs=[
                    thumb_scenery,
                    out_blend,
                    out_blend_file,
                    looks_state,
                    recipe_detail,
                ],
            )

        def _on_history_select(run_dir: Optional[str]):
            if not run_dir:
                return "_Select a run to inspect its recipe._"
            return format_recipe_md(load_recipe(run_dir), Path(run_dir).name)

        def _on_history_refresh(selected: Optional[str]):
            choices = list_recent_runs()
            values = {v for _, v in choices}
            pick = selected if selected in values else (choices[0][1] if choices else None)
            detail = (
                format_recipe_md(load_recipe(pick), Path(pick).name)
                if pick
                else "_No runs in outputs/ yet._"
            )
            return history_dropdown_update(pick), detail

        def _on_reapply(run_dir: Optional[str], state: Optional[dict]):
            n_ctrl = 13
            idle_ctrls = tuple(gr.update() for _ in range(n_ctrl))
            idle_views = (gr.update(), gr.update(), gr.update())

            if not run_dir:
                return (
                    *idle_ctrls,
                    *idle_views,
                    state,
                    "_Select a run with a saved recipe first._",
                    gr.update(),
                )

            recipe = load_recipe(run_dir)
            if recipe is None:
                return (
                    *idle_ctrls,
                    *idle_views,
                    state,
                    format_recipe_md(None, Path(run_dir).name),
                    gr.update(),
                )

            ctrl_updates = (
                gr.update(value=recipe["model_key"]),
                gr.update(
                    value=recipe["max_size"],
                    interactive=is_promptda(recipe["model_key"]),
                ),
                gr.update(value=recipe["look_a"]),
                gr.update(value=recipe["look_b"]),
                gr.update(value=recipe["look_c"]),
                gr.update(value=recipe["fade_b"]),
                gr.update(value=recipe["fade_c"]),
                gr.update(value=recipe["blend_mode_b"]),
                gr.update(value=recipe["blend_mode_c"]),
                gr.update(value=recipe["scenery_mode"]),
                gr.update(value=recipe["scenery_protect"]),
                gr.update(value=recipe["scenery_feather"]),
                gr.update(value=recipe["scenery_clean"]),
            )
            detail = format_recipe_md(recipe)
            help_txt = _model_help(recipe["model_key"])

            if state is None or "depth" not in state:
                return (
                    *ctrl_updates,
                    *idle_views,
                    state,
                    detail,
                    help_txt,
                )

            thumb, blend, path, new_state = update_scenery(
                state,
                recipe["scenery_mode"],
                recipe["scenery_protect"],
                recipe["scenery_feather"],
                recipe["scenery_clean"],
                recipe["look_a"],
                recipe["look_b"],
                recipe["look_c"],
                recipe["fade_b"],
                recipe["fade_c"],
                recipe["blend_mode_b"],
                recipe["blend_mode_c"],
            )
            # Keep recipe on the *current* run folder, not the history source.
            if new_state and new_state.get("out_dir"):
                save_recipe(
                    new_state["out_dir"],
                    build_recipe(
                        recipe["model_key"],
                        recipe["max_size"],
                        recipe["look_a"],
                        recipe["look_b"],
                        recipe["look_c"],
                        recipe["fade_b"],
                        recipe["fade_c"],
                        recipe["blend_mode_b"],
                        recipe["blend_mode_c"],
                        recipe["scenery_mode"],
                        recipe["scenery_protect"],
                        recipe["scenery_feather"],
                        recipe["scenery_clean"],
                    ),
                )
                detail = format_recipe_md(load_recipe(new_state["out_dir"]))
            return (
                *ctrl_updates,
                thumb,
                blend,
                path,
                new_state,
                detail,
                help_txt,
            )

        history_dd.change(
            _on_history_select,
            inputs=[history_dd],
            outputs=[recipe_detail],
        )
        history_refresh.click(
            _on_history_refresh,
            inputs=[history_dd],
            outputs=[history_dd, recipe_detail],
        )
        history_reapply.click(
            _on_reapply,
            inputs=[history_dd, looks_state],
            outputs=[
                model_key,
                max_size,
                look_a,
                look_b,
                look_c,
                fade_b,
                fade_c,
                blend_mode_b,
                blend_mode_c,
                scenery_mode,
                scenery_protect,
                scenery_feather,
                scenery_clean,
                thumb_scenery,
                out_blend,
                out_blend_file,
                looks_state,
                recipe_detail,
                model_help,
            ],
        )

    return demo


if __name__ == "__main__":
    # Warm-load default model so the first click is faster.
    try:
        get_backend(DEFAULT_MODEL)
    except FileNotFoundError as exc:
        print(f"Warning: {exc}")
    ui = build_ui()
    ui.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
        show_error=True,
        inbrowser=False,
    )
