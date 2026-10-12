"""Shared types without heavy ML dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

FocalEstimateSource = Literal[
    "exif",
    "depth_pro_fov",
    "camera_matrix",
    "default_pinhole",
]


@dataclass
class MetricDepthResult:
    """Single-frame metric depth in metres."""

    color: np.ndarray  # HxWx3 uint8 RGB
    depth_m: np.ndarray  # HxW float32 metres
    model_key: str
    focal_length_px: float
    focal_estimate_source: FocalEstimateSource
    prompt_depth_m: Optional[np.ndarray] = None
