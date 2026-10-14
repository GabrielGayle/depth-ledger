"""IO helpers: paths, depth save, Stray zip extract."""
from depth_core.pipeline import (  # noqa: F401
    ROOT,
    EXAMPLE_RGB,
    EXAMPLE_DEPTH,
    OUTPUT_DIR,
    CHECKPOINT_PATH,
    THUMB_MAX,
    _save_depth_mm,
    _safe_stem,
    _make_run_dir,
    _downscale_thumb,
    _extract_stray_first_frame,
)
