"""Rough Modal GPU cost estimates (list rates, not exact billing)."""

from __future__ import annotations

import os
from typing import Any

# Modal list prices — USD per GPU-second (see modal.com/pricing).
_DEFAULT_GPU_RATES: dict[str, float] = {
    "T4": 0.000164,
    "A10": 0.000306,
    "L4": 0.000222,
}

DISCLAIMER = (
    "Estimate from GPU seconds × Modal list rate. Actual bill may differ "
    "(CPU/RAM, idle container, free credits)."
)


def gpu_rates() -> dict[str, float]:
    rates = dict(_DEFAULT_GPU_RATES)
    for key in _DEFAULT_GPU_RATES:
        env_key = f"MODAL_GPU_RATE_{key}"
        raw = os.environ.get(env_key, "").strip()
        if raw:
            try:
                rates[key] = float(raw)
            except ValueError:
                pass
    return rates


def estimate_modal_cost(gpu_type: str, gpu_seconds: float) -> float:
    rates = gpu_rates()
    rate = rates.get(gpu_type, rates["T4"])
    return round(max(0.0, float(gpu_seconds)) * rate, 6)


def billing_dict(
    gpu_type: str,
    gpu_seconds: float,
    *,
    cold_start: bool = False,
) -> dict[str, Any]:
    cost = estimate_modal_cost(gpu_type, gpu_seconds)
    return {
        "gpu_type": gpu_type,
        "gpu_seconds": round(float(gpu_seconds), 4),
        "cost_usd": cost,
        "cold_start": bool(cold_start),
    }


def pricing_meta() -> dict[str, Any]:
    return {
        "gpu_rates": gpu_rates(),
        "disclaimer": DISCLAIMER,
    }
