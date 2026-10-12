"""Metric depth inference (Depth Pro / PromptDA)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import cv2
import numpy as np
import torch

from promptda.promptda import PromptDA
from promptda.utils.io_wrapper import load_depth, load_image

from depth_ledger.paths import CHECKPOINT_PATH

MODEL_CHOICES: list[tuple[str, str]] = [
    ("Depth Pro (RGB only)", "depth-pro"),
    ("PromptDA small", "promptda-small"),
    ("PromptDA large", "promptda-large"),
]
MODEL_KEYS = {k for _, k in MODEL_CHOICES}

PROMPTDA_IDS = {
    "promptda-small": ("depth-anything/prompt-depth-anything-vits", "vits"),
    "promptda-large": ("depth-anything/prompt-depth-anything-vitl", "vitl"),
}

_LEGACY_MODEL = {
    "small": "promptda-small",
    "large": "promptda-large",
    "promptda": "promptda-small",
    "depthpro": "depth-pro",
    "depth_pro": "depth-pro",
}

FocalEstimateSource = Literal[
    "exif",
    "depth_pro_fov",
    "promptda_intrinsics",
    "default_pinhole",
]


def normalize_model_key(raw: str) -> str:
    key = (raw or "").strip().lower()
    key = _LEGACY_MODEL.get(key, key)
    return key if key in MODEL_KEYS else "depth-pro"


def is_promptda(model_key: str) -> bool:
    return normalize_model_key(model_key) in PROMPTDA_IDS


def pick_device(model_key: str) -> str:
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
    transform: Any = None


@dataclass
class MetricDepthResult:
    """Single-frame metric depth in metres."""

    color: np.ndarray  # HxWx3 uint8 RGB
    depth_m: np.ndarray  # HxW float32 metres
    model_key: str
    focal_length_px: float
    focal_estimate_source: FocalEstimateSource
    prompt_depth_m: Optional[np.ndarray] = None


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

    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT, DepthProConfig

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Depth Pro checkpoint missing at {CHECKPOINT_PATH}. "
            "Run: bash DepthPro/get_pretrained_models.sh "
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


def infer_promptda(
    backend: LoadedBackend,
    rgb_path: Path,
    prompt_depth_path: Path,
    max_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, FocalEstimateSource]:
    device = backend.device
    image = load_image(str(rgb_path), max_size=max_size).to(device)
    prompt_depth = load_depth(str(prompt_depth_path)).to(device)

    with torch.no_grad():
        depth_t = backend.model.predict(image, prompt_depth)

    depth = depth_t[0, 0].detach().cpu().numpy()
    color = _tensor_to_rgb(image)
    prompt_np = prompt_depth[0, 0].detach().cpu().numpy()
    h, w = depth.shape
    focal = float(max(h, w))
    return color, depth.astype(np.float32), prompt_np, focal, "promptda_intrinsics"


def infer_depth_pro(
    backend: LoadedBackend, rgb_path: Path
) -> Tuple[np.ndarray, np.ndarray, float, FocalEstimateSource]:
    from depth_pro import load_rgb

    image_np, _, f_px_exif = load_rgb(rgb_path)
    image_t = backend.transform(image_np)

    with torch.no_grad():
        prediction = backend.model.infer(image_t, f_px=f_px_exif)

    depth = prediction["depth"].detach().cpu().numpy().squeeze()
    if image_np.dtype != np.uint8:
        color = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
    else:
        color = image_np.copy()
    if color.shape[0] != depth.shape[0] or color.shape[1] != depth.shape[1]:
        color = cv2.resize(
            color, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA
        )

    if f_px_exif is not None:
        focal = float(f_px_exif)
        source: FocalEstimateSource = "exif"
    elif prediction["focallength_px"] is not None:
        focal = float(prediction["focallength_px"].detach().cpu().item())
        source = "depth_pro_fov"
    else:
        h, w = depth.shape
        focal = float(max(h, w))
        source = "default_pinhole"

    return color, depth.astype(np.float32), focal, source


def run_metric_depth(
    rgb_path: Path,
    model_key: str = DEFAULT_MODEL,
    max_size: int = MAX_SIZE,
    prompt_depth_path: Optional[Path] = None,
) -> MetricDepthResult:
    """Run Depth Pro or PromptDA and return aligned RGB + metric depth."""
    key = normalize_model_key(model_key)
    backend = get_backend(key)

    if backend.kind == "promptda":
        if prompt_depth_path is None:
            raise ValueError(
                "PromptDA requires a LiDAR prompt depth PNG (uint16 mm). "
                "Use depth-pro for RGB-only images."
            )
        color, depth, prompt_np, focal, source = infer_promptda(
            backend, rgb_path, prompt_depth_path, int(max_size)
        )
        return MetricDepthResult(
            color=color,
            depth_m=depth,
            model_key=backend.key,
            focal_length_px=focal,
            focal_estimate_source=source,
            prompt_depth_m=prompt_np,
        )

    color, depth, focal, source = infer_depth_pro(backend, rgb_path)
    return MetricDepthResult(
        color=color,
        depth_m=depth,
        model_key=backend.key,
        focal_length_px=focal,
        focal_estimate_source=source,
    )
