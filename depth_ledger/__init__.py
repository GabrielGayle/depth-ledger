"""Depth Ledger — metric depth inference and DepthPaper interchange."""

from depth_ledger.bundle import (
    CameraIntrinsics,
    read_depthpaper_bundle,
    write_depthpaper_bundle,
)
from depth_ledger.lift import lift_bundle
from depth_ledger.models import (
    DEFAULT_MODEL,
    get_backend,
    normalize_model_key,
    run_metric_depth,
)
from depth_ledger.types import MetricDepthResult

__all__ = [
    "DEFAULT_MODEL",
    "CameraIntrinsics",
    "MetricDepthResult",
    "get_backend",
    "lift_bundle",
    "normalize_model_key",
    "read_depthpaper_bundle",
    "run_metric_depth",
    "write_depthpaper_bundle",
]


def bundle_from_image(*args, **kwargs):
    from depth_ledger.bundle import bundle_from_image as _bundle_from_image

    return _bundle_from_image(*args, **kwargs)
