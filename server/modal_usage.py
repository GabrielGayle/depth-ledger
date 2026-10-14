"""In-memory Modal usage accumulator for session cost ticker."""

from __future__ import annotations

import threading
from typing import Any

_LOCK = threading.Lock()
_USAGE: dict[str, Any] = {
    "calls": 0,
    "gpu_seconds": 0.0,
    "cost_usd": 0.0,
}


def record_modal_usage(billing: dict[str, Any]) -> dict[str, Any]:
    with _LOCK:
        _USAGE["calls"] = int(_USAGE["calls"]) + 1
        _USAGE["gpu_seconds"] = round(
            float(_USAGE["gpu_seconds"]) + float(billing.get("gpu_seconds") or 0),
            4,
        )
        _USAGE["cost_usd"] = round(
            float(_USAGE["cost_usd"]) + float(billing.get("cost_usd") or 0),
            6,
        )
        return dict(_USAGE)


def get_modal_usage() -> dict[str, Any]:
    with _LOCK:
        return dict(_USAGE)


def reset_modal_usage() -> dict[str, Any]:
    with _LOCK:
        _USAGE["calls"] = 0
        _USAGE["gpu_seconds"] = 0.0
        _USAGE["cost_usd"] = 0.0
        return dict(_USAGE)
