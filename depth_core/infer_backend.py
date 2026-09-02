"""Local vs Modal inference backend resolution."""

from __future__ import annotations

import os
from typing import Literal, Optional

InferBackend = Literal["local", "modal"]

DEFAULT_INFER_BACKEND: InferBackend = (
    "modal"
    if os.environ.get("INFER_BACKEND", "local").strip().lower() == "modal"
    else "local"
)


def resolve_backend(override: Optional[str] = None) -> InferBackend:
    raw = (override or os.environ.get("INFER_BACKEND", "local")).strip().lower()
    return "modal" if raw == "modal" else "local"


def modal_app_name() -> str:
    return os.environ.get("MODAL_APP_NAME", "depth-ledger").strip() or "depth-ledger"


def modal_ready() -> bool:
    """True when Modal SDK + credentials look usable (app may still be undeployed)."""
    try:
        from depth_core.infer_modal import modal_available

        return modal_available()
    except Exception:
        return False


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def local_infer_slots() -> int:
    """Concurrent local neural forwards. GPU/MPS stay serialized (one model)."""
    forced = (os.environ.get("DEPTH_LEDGER_INFER_WORKERS") or "").strip()
    if forced:
        return _env_int("DEPTH_LEDGER_INFER_WORKERS", 1)
    return 1


def local_cook_slots() -> int:
    """CPU graph-eval + Are.na publish workers."""
    n = os.cpu_count() or 4
    return _env_int("DEPTH_LEDGER_COOK_WORKERS", max(2, min(8, n)))


def local_download_slots() -> int:
    return _env_int("DEPTH_LEDGER_DOWNLOAD_WORKERS", 8)


def configure_thread_pools() -> None:
    """Avoid N cook threads × N OpenCV/BLAS threads blowing RAM.

    Never import torch here if it is not already loaded — initializing MPS
    off the main thread can abort the process on macOS.
    """
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass
    import sys

    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return
    try:
        cpu = os.cpu_count() or 4
        torch_mod.set_num_threads(max(1, min(4, cpu)))
    except Exception:
        pass
