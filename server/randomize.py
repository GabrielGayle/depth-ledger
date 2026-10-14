"""Server-side kitchen randomization (mirrors web/src/randomize.ts)."""

from __future__ import annotations

import copy
import random
from typing import Any, Optional


def _clamp(n: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, n))


def _jitter(value: float, amount: float, lo: float, hi: float) -> float:
    if amount <= 0:
        return _clamp(value, lo, hi)
    return _clamp(value + (random.random() * 2 - 1) * amount, lo, hi)


def _pick(arr: list):
    return arr[int(random.random() * len(arr))]


def randomize_kitchen(
    graph: dict[str, Any],
    *,
    wackiness: float = 0.45,
    max_fidelity: bool = False,
    look_keys: Optional[list[str]] = None,
    cmaps: Optional[list[str]] = None,
    blend_modes: Optional[list[str]] = None,
    scenery_modes: Optional[list[str]] = None,
    look_defaults: Optional[dict[str, dict]] = None,
) -> dict[str, Any]:
    """Return a deep-copied graph with looks/scene/edges jittered by wackiness.

    When ``max_fidelity`` is True, Relief solve size is pinned to the largest
    allowed side (caller should also force infer max_size). Creative jitter
    is unchanged.
    """
    from depth_core.pipeline import (
        BLEND_MODES,
        LOOK_PRESETS,
        SCENERY_MATTE_MODES,
    )
    from depth_core.relief import RELIEF_MAX_SIDE_CHOICES

    w = _clamp(float(wackiness), 0.0, 1.0)
    fidelity = bool(max_fidelity)
    relief_sides = list(RELIEF_MAX_SIDE_CHOICES)
    max_relief_side = max(relief_sides)
    looks = look_keys or [k for k in LOOK_PRESETS if k != "scenery"]
    cmap_list = cmaps or ["gray", "magma", "inferno", "viridis", "turbo", "photo", "wash"]
    blends = blend_modes or list(BLEND_MODES.keys())
    scenes = scenery_modes or list(SCENERY_MATTE_MODES.keys())
    defaults = look_defaults or {
        k: {
            "lo_pct": float(v.get("hi_pct_fallback") or v.get("lo_pct") or 2)
            if v.get("lo_pct") == "auto_bg"
            else float(v["lo_pct"])
            if isinstance(v.get("lo_pct"), (int, float))
            else 2.0,
            "hi_pct": float(v.get("hi_pct_fallback", 36))
            if v.get("hi_pct") == "auto"
            else float(v["hi_pct"])
            if isinstance(v.get("hi_pct"), (int, float))
            else 98.0,
            "gamma": float(v.get("gamma", 1.0)),
            "cmap": v.get("cmap") or "gray",
            "blur_amount": float(v["blur_amount"])
            if v.get("blur_amount") is not None
            else None,
        }
        for k, v in LOOK_PRESETS.items()
        if k != "scenery"
    }
    # Fix lo_pct for normal presets
    for k, v in LOOK_PRESETS.items():
        if k == "scenery":
            continue
        lo = v.get("lo_pct")
        hi = v.get("hi_pct")
        defaults[k] = {
            "lo_pct": float(lo)
            if isinstance(lo, (int, float))
            else float(defaults.get(k, {}).get("lo_pct", 2)),
            "hi_pct": float(hi)
            if isinstance(hi, (int, float))
            else float(v.get("hi_pct_fallback", 36)),
            "gamma": float(v.get("gamma", 1.0)),
            "cmap": v.get("cmap") or "gray",
            "blur_amount": float(v["blur_amount"])
            if v.get("blur_amount") is not None
            else None,
        }

    out = copy.deepcopy(graph)
    nodes = out.get("nodes") or []
    edges = out.get("edges") or []
    by_id = {n["id"]: n for n in nodes}

    for n in nodes:
        d = dict(n.get("data") or {})
        ntype = n.get("type")

        if ntype == "look":
            swap_chance = 0.08 + w * 0.72
            if looks and random.random() < swap_chance:
                key = _pick(looks)
                d["preset"] = key
                defn = defaults.get(key) or {}
                d["lo_pct"] = defn.get("lo_pct", 2)
                d["hi_pct"] = defn.get("hi_pct", 98)
                d["gamma"] = defn.get("gamma", 1)
                d["cmap"] = "inherit"

            lo_j = 1 + w * 14
            hi_j = 2 + w * 28
            g_j = 0.05 + w * 0.55
            lo = _jitter(float(d.get("lo_pct", 2)), lo_j, 1, 45)
            hi = _jitter(float(d.get("hi_pct", 98)), hi_j, 15, 99)
            if hi <= lo + 3:
                hi = min(99.0, lo + 8 + w * 20)
            d["lo_pct"] = round(lo * 10) / 10
            d["hi_pct"] = round(hi * 10) / 10
            d["gamma"] = round(_jitter(float(d.get("gamma", 1)), g_j, 0.35, 2.2) * 100) / 100
            if random.random() < w * 0.55:
                d["cmap"] = _pick(cmap_list)
            cmap = d.get("cmap") or "inherit"
            preset_key = d.get("preset") or "ledger"
            effective = (
                cmap
                if cmap not in (None, "", "inherit")
                else (defaults.get(preset_key) or {}).get("cmap") or "gray"
            )
            if effective == "wash":
                base_blur = float(
                    d.get("blur_amount")
                    if d.get("blur_amount") is not None
                    else (defaults.get(preset_key) or {}).get("blur_amount", 0.4)
                )
                d["blur_amount"] = (
                    round(_jitter(base_blur, 0.05 + w * 0.35, 0.0, 1.0) * 100) / 100
                )
            n["data"] = d

        if ntype == "scene":
            if scenes and random.random() < 0.12 + w * 0.7:
                d["mode"] = _pick(scenes)
            d["protect"] = (
                round(
                    _jitter(float(d.get("protect", 0.22)), 0.04 + w * 0.28, 0.02, 0.85)
                    * 100
                )
                / 100
            )
            d["feather"] = (
                round(
                    _jitter(float(d.get("feather", 0.35)), 0.05 + w * 0.35, 0.05, 0.95)
                    * 100
                )
                / 100
            )
            if random.random() < w * 0.4:
                d["clean"] = random.random() > 0.35
            if random.random() < w * 0.35:
                d["fringe_kill"] = random.random() > 0.4
            if random.random() < w * 0.35:
                d["far_remap"] = random.random() > 0.35
            n["data"] = d

        if ntype == "relief":
            d["gc_alpha"] = (
                round(
                    _jitter(float(d.get("gc_alpha", 10)), 1 + w * 12, 1, 40) * 10
                )
                / 10
            )
            if random.random() < 0.1 + w * 0.45:
                d["invert"] = random.random() > 0.55
            if fidelity:
                d["max_side"] = max_relief_side
            elif random.random() < 0.08 + w * 0.4:
                d["max_side"] = _pick(relief_sides)
            n["data"] = d

        if ntype == "blend":
            if blends and random.random() < 0.15 + w * 0.75:
                d["mode"] = _pick(blends)
            d["opacity"] = (
                round(_jitter(float(d.get("opacity", 1)), 0.05 + w * 0.45, 0.15, 1) * 100)
                / 100
            )
            n["data"] = d

    for e in edges:
        prev = dict(e.get("data") or {})
        if prev.get("blendable") is False:
            continue
        src = by_id.get(e.get("source") or "")
        tgt = by_id.get(e.get("target") or "")
        blendable = bool(
            src
            and src.get("type") in ("look", "scene", "blend")
            and tgt
            and tgt.get("type") in ("look", "scene")
        )
        if not blendable and prev.get("blendable") is not True:
            continue
        data = {
            "blend_mode": prev.get("blend_mode") or "normal",
            "opacity": prev.get("opacity", 1),
            "blendable": True,
        }
        if blends and random.random() < 0.12 + w * 0.78:
            data["blend_mode"] = _pick(blends)
        data["opacity"] = (
            round(
                _jitter(float(data.get("opacity") or 1), 0.06 + w * 0.5, 0.12, 1) * 100
            )
            / 100
        )
        e["data"] = data

    out["nodes"] = nodes
    out["edges"] = edges
    return out
