"""Ableton-style Operator synthesizer for depth looks.

Oscillators A–D are Look instances (preset + window + gamma + cmap + level)
rendered from a single metric depth plate, then routed through 11 FM-style
algorithms, filter/shaper, and optional ADSR/LFO modulation.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

import cv2
import numpy as np

from depth_core.pipeline import (
    DEFAULT_LOOK_A,
    DEFAULT_LOOK_B,
    DEFAULT_LOOK_C,
    DEFAULT_WASH_BLUR,
    LOOK_PRESETS,
    _apply_grade,
    _prepare_inv_smooth,
    _resolve_hi_pct,
    _resolve_lo_pct,
    _subject_hi_pct,
    _visualize_inverse_depth,
    blend_looks,
)

OSC_IDS = ("A", "B", "C", "D")

ALGORITHM_META = [
    {"id": 1, "label": "D→C→B→A", "desc": "Full stack FM"},
    {"id": 2, "label": "(D+C)→B→A", "desc": "Dual mod into B→A"},
    {"id": 3, "label": "D→C→B→A*", "desc": "Stack with C→B feed"},
    {"id": 4, "label": "D→C→(B∥A)", "desc": "Dual carriers"},
    {"id": 5, "label": "(D∥C)→(B∥A)", "desc": "Parallel pairs"},
    {"id": 6, "label": "D→(C∥B∥A)", "desc": "One mod, three carriers"},
    {"id": 7, "label": "(D→C→B)∥A", "desc": "Stack + parallel A"},
    {"id": 8, "label": "(D→C)∥(B→A)", "desc": "Dual sub-chains"},
    {"id": 9, "label": "(D→C∥B)→A", "desc": "Hybrid into A"},
    {"id": 10, "label": "D∥(C→B→A)", "desc": "D additive + stack"},
    {"id": 11, "label": "A∥B∥C∥D", "desc": "Additive looks"},
]

FILTER_TYPES = ("off", "lowpass", "highpass", "formant")
SHAPER_CURVES = ("linear", "s_curve", "quantize", "expand_near")
LFO_SHAPES = ("sine", "triangle", "saw", "square")
MOD_DESTS = (
    "level_A",
    "level_B",
    "level_C",
    "level_D",
    "filter_cutoff",
    "lo_pct_A",
    "hi_pct_A",
    "spread",
    "drive",
)
VOICE_KEYS = ("plate", "ply", "vector", "ndi")

DEFAULT_OSC = {
    "A": {
        "on": True,
        "level": 1.0,
        "preset": DEFAULT_LOOK_A,
        "lo_pct": 2.0,
        "hi_pct": 98.0,
        "gamma": 0.85,
        "cmap": "inherit",
    },
    "B": {
        "on": True,
        "level": 0.55,
        "preset": DEFAULT_LOOK_B,
        "lo_pct": 1.0,
        "hi_pct": 30.0,
        "gamma": 1.0,
        "cmap": "inherit",
    },
    "C": {
        "on": False,
        "level": 0.4,
        "preset": DEFAULT_LOOK_C if DEFAULT_LOOK_C in LOOK_PRESETS else "soft",
        "lo_pct": 40.0,
        "hi_pct": 92.0,
        "gamma": 0.95,
        "cmap": "inherit",
    },
    "D": {
        "on": False,
        "level": 0.35,
        "preset": "punchy",
        "lo_pct": 5.0,
        "hi_pct": 90.0,
        "gamma": 0.7,
        "cmap": "inherit",
    },
}


def default_operator_patch() -> dict[str, Any]:
    return {
        "version": 1,
        "algorithm": 1,
        "oscillators": {k: dict(v) for k, v in DEFAULT_OSC.items()},
        "filter": {"type": "off", "cutoff": 0.55, "drive": 0.0},
        "shaper": {"curve": "linear", "amount": 0.0},
        "envelope": {
            "attack": 0.05,
            "decay": 0.2,
            "sustain": 0.75,
            "release": 0.3,
            "amount": 0.0,
            "dest": ["level_A"],
        },
        "lfo": {
            "rate": 0.25,
            "shape": "sine",
            "amount": 0.0,
            "sync": False,
            "dest": ["filter_cutoff"],
        },
        "global": {
            "time": 1.0,
            "spread": 1.0,
            "voices": ["plate"],
        },
        "t": None,
    }


def operator_meta() -> dict[str, Any]:
    return {
        "algorithms": ALGORITHM_META,
        "filter_types": list(FILTER_TYPES),
        "shaper_curves": list(SHAPER_CURVES),
        "lfo_shapes": list(LFO_SHAPES),
        "mod_dests": list(MOD_DESTS),
        "voices": list(VOICE_KEYS),
        "ndi_available": bool(os.environ.get("OPERATOR_NDI")),
    }


def _luminance(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32) / 255.0
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def _wrap01(x: np.ndarray) -> np.ndarray:
    """Fold into [0, 1] like a soft FM wrap."""
    return np.abs(((x + 1.0) % 2.0) - 1.0).astype(np.float32)


def _fm(carrier: np.ndarray, modulator: np.ndarray, index: float) -> np.ndarray:
    idx = float(np.clip(index, 0.0, 2.0))
    return _wrap01(carrier + idx * (modulator - 0.5))


def _adsr(t: float, attack: float, decay: float, sustain: float, release: float) -> float:
    """One-shot ADSR over normalized t in [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0))
    a = max(float(attack), 1e-4)
    d = max(float(decay), 1e-4)
    s = float(np.clip(sustain, 0.0, 1.0))
    r = max(float(release), 1e-4)
    # Scale segments to fill [0,1]
    total = a + d + r + 0.25  # sustain dwell share
    a_n, d_n, r_n = a / total, d / total, r / total
    s_n = 1.0 - a_n - d_n - r_n
    if t < a_n:
        return t / a_n
    t2 = t - a_n
    if t2 < d_n:
        return 1.0 + (s - 1.0) * (t2 / d_n)
    t3 = t2 - d_n
    if t3 < s_n:
        return s
    t4 = t3 - s_n
    if t4 >= r_n:
        return 0.0
    return s * (1.0 - t4 / r_n)


def _lfo_value_fixed(t: float, rate: float, shape: str, time_scale: float) -> float:
    phase = (float(t) * max(float(rate), 0.01) * float(time_scale) * 4.0) % 1.0
    if shape == "triangle":
        return (1.0 - abs(2.0 * phase - 1.0)) * 2.0 - 1.0
    if shape == "saw":
        return phase * 2.0 - 1.0
    if shape == "square":
        return 1.0 if phase < 0.5 else -1.0
    return math.sin(phase * 2.0 * math.pi)


def _apply_mod_matrix(
    patch: dict[str, Any],
    t: Optional[float],
) -> dict[str, float]:
    """Return additive deltas keyed by mod dest."""
    deltas: dict[str, float] = {d: 0.0 for d in MOD_DESTS}
    if t is None:
        return deltas
    g = patch.get("global") or {}
    time_scale = float(g.get("time", 1.0) or 1.0)

    env = patch.get("envelope") or {}
    env_amt = float(env.get("amount", 0.0) or 0.0)
    if env_amt != 0.0:
        env_v = _adsr(
            float(t),
            float(env.get("attack", 0.05)),
            float(env.get("decay", 0.2)),
            float(env.get("sustain", 0.75)),
            float(env.get("release", 0.3)),
        )
        for dest in env.get("dest") or []:
            if dest in deltas:
                deltas[dest] += env_amt * (env_v - 0.5) * 2.0

    lfo = patch.get("lfo") or {}
    lfo_amt = float(lfo.get("amount", 0.0) or 0.0)
    if lfo_amt != 0.0:
        shape = lfo.get("shape") or "sine"
        if shape not in LFO_SHAPES:
            shape = "sine"
        lfo_v = _lfo_value_fixed(
            float(t), float(lfo.get("rate", 0.25)), shape, time_scale
        )
        for dest in lfo.get("dest") or []:
            if dest in deltas:
                deltas[dest] += lfo_amt * lfo_v
    return deltas


def render_oscillator(
    depth: np.ndarray,
    osc: dict[str, Any],
    cliff: float,
    inv_smooth: Optional[np.ndarray],
    photo: Optional[np.ndarray] = None,
    lo_delta: float = 0.0,
    hi_delta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb_u8, luminance_float01)."""
    preset = osc.get("preset") or DEFAULT_LOOK_A
    if preset not in LOOK_PRESETS or preset == "scenery":
        preset = DEFAULT_LOOK_A
    cfg = dict(LOOK_PRESETS[preset])
    cmap_override = osc.get("cmap")
    cmap = (
        cmap_override
        if cmap_override not in (None, "", "inherit")
        else cfg["cmap"]
    )
    lo = float(osc.get("lo_pct", _resolve_lo_pct(cfg, cliff))) + lo_delta
    hi = float(osc.get("hi_pct", _resolve_hi_pct(cfg, cliff))) + hi_delta
    lo = float(np.clip(lo, 0.5, 98.0))
    hi = float(np.clip(max(hi, lo + 1.0), 1.5, 99.5))
    gamma = float(osc.get("gamma", cfg["gamma"]))
    if inv_smooth is None:
        _, inv_smooth = _prepare_inv_smooth(depth)
    rgb = _visualize_inverse_depth(
        depth,
        cmap=cmap,
        lo_pct=lo,
        hi_pct=hi,
        gamma=gamma,
        grade=cfg.get("grade"),
        inv_smooth=inv_smooth,
        photo=photo if cmap in ("photo", "wash") else None,
        blur_amount=float(
            osc["blur_amount"]
            if osc.get("blur_amount") is not None
            else cfg.get("blur_amount", DEFAULT_WASH_BLUR)
        ),
    )
    return rgb, _luminance(rgb)


def _sum_carriers(
    fields: list[tuple[np.ndarray, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted sum of (rgb, lum, weight) carriers → (rgb_u8, lum)."""
    if not fields:
        raise ValueError("No carrier oscillators enabled")
    w_sum = sum(max(w, 0.0) for _, _, w in fields) or 1.0
    acc_rgb = None
    acc_lum = None
    for rgb, lum, w in fields:
        ww = max(float(w), 0.0) / w_sum
        f = rgb.astype(np.float32) * ww
        acc_rgb = f if acc_rgb is None else acc_rgb + f
        acc_lum = lum * ww if acc_lum is None else acc_lum + lum * ww
    out = np.clip(acc_rgb, 0, 255).astype(np.uint8)
    return out, np.clip(acc_lum, 0, 1).astype(np.float32)


def run_algorithm(
    algo: int,
    rgbs: dict[str, np.ndarray],
    lums: dict[str, np.ndarray],
    levels: dict[str, float],
    ons: dict[str, bool],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply FM algorithm; return master (rgb, lum)."""

    def L(k: str) -> np.ndarray:
        return lums[k]

    def R(k: str) -> np.ndarray:
        return rgbs[k]

    def lv(k: str) -> float:
        return float(levels.get(k, 0.0)) if ons.get(k, False) else 0.0

    algo = int(np.clip(algo, 1, 11))

    # Helpers producing modulated luminance then tinting with carrier RGB
    def tint(lum: np.ndarray, carrier_key: str) -> np.ndarray:
        base = R(carrier_key).astype(np.float32) / 255.0
        # Preserve chroma-ish by scaling luminance toward target
        cur = _luminance((base * 255).astype(np.uint8)) + 1e-6
        scale = (lum / cur)[..., None]
        return np.clip(base * scale * 255.0, 0, 255).astype(np.uint8)

    def stack_chain(keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """keys[0] is deepest modulator, keys[-1] is carrier."""
        active = [k for k in keys if lv(k) > 0]
        if not active:
            # fallback silent
            z = np.zeros_like(L("A"))
            return np.zeros_like(R("A")), z
        carrier = active[-1]
        field = L(carrier)
        for mod in reversed(active[:-1]):
            field = _fm(field, L(mod), lv(mod))
        field = np.clip(field * max(lv(carrier), 0.01), 0, 1)
        return tint(field, carrier), field

    if algo == 1:
        return stack_chain(["D", "C", "B", "A"])

    if algo == 2:
        # (D+C) → B → A
        if lv("D") > 0 or lv("C") > 0:
            parts = []
            if lv("D") > 0:
                parts.append((L("D"), lv("D")))
            if lv("C") > 0:
                parts.append((L("C"), lv("C")))
            w = sum(p[1] for p in parts) or 1.0
            mod = sum(p[0] * (p[1] / w) for p in parts)
        else:
            mod = L("B")
        field = L("B")
        if lv("B") > 0:
            field = _fm(field, mod, max(lv("D"), lv("C")))
        field = _fm(L("A"), field, lv("B")) if lv("A") > 0 else field
        field = np.clip(field * max(lv("A"), 0.01), 0, 1)
        return tint(field, "A"), field

    if algo == 3:
        return stack_chain(["D", "C", "B", "A"])

    if algo == 4:
        # D→C→(B∥A)
        mid = L("C")
        if lv("D") > 0:
            mid = _fm(mid, L("D"), lv("D"))
        carriers = []
        for k in ("B", "A"):
            if lv(k) <= 0:
                continue
            f = _fm(L(k), mid, lv("C"))
            f = np.clip(f * lv(k), 0, 1)
            carriers.append((tint(f, k), f, lv(k)))
        if not carriers:
            return stack_chain(["D", "C", "B", "A"])
        return _sum_carriers(carriers)

    if algo == 5:
        # (D∥C) → (B∥A)  — mods sum into each carrier
        mods = []
        for k in ("D", "C"):
            if lv(k) > 0:
                mods.append((L(k), lv(k)))
        if mods:
            w = sum(m[1] for m in mods) or 1.0
            mod = sum(m[0] * (m[1] / w) for m in mods)
            mod_idx = max(m[1] for m in mods)
        else:
            mod = np.zeros_like(L("A"))
            mod_idx = 0.0
        carriers = []
        for k in ("B", "A"):
            if lv(k) <= 0:
                continue
            f = _fm(L(k), mod, mod_idx)
            f = np.clip(f * lv(k), 0, 1)
            carriers.append((tint(f, k), f, lv(k)))
        if not carriers:
            return R("A"), L("A")
        return _sum_carriers(carriers)

    if algo == 6:
        # D → (C∥B∥A)
        carriers = []
        for k in ("C", "B", "A"):
            if lv(k) <= 0:
                continue
            f = _fm(L(k), L("D"), lv("D")) if lv("D") > 0 else L(k)
            f = np.clip(f * lv(k), 0, 1)
            carriers.append((tint(f, k), f, lv(k)))
        if not carriers:
            return R("A"), L("A")
        return _sum_carriers(carriers)

    if algo == 7:
        # (D→C→B) ∥ A
        stack_rgb, stack_lum = stack_chain(["D", "C", "B"])
        parts = []
        if lv("B") > 0 or lv("C") > 0 or lv("D") > 0:
            parts.append((stack_rgb, stack_lum, max(lv("B"), 0.01)))
        if lv("A") > 0:
            parts.append((R("A"), L("A"), lv("A")))
        if not parts:
            return R("A"), L("A")
        return _sum_carriers(parts)

    if algo == 8:
        # (D→C) ∥ (B→A)
        left_rgb, left_lum = stack_chain(["D", "C"])
        right_rgb, right_lum = stack_chain(["B", "A"])
        parts = []
        if lv("C") > 0 or lv("D") > 0:
            parts.append((left_rgb, left_lum, max(lv("C"), 0.01)))
        if lv("A") > 0 or lv("B") > 0:
            parts.append((right_rgb, right_lum, max(lv("A"), 0.01)))
        if not parts:
            return R("A"), L("A")
        return _sum_carriers(parts)

    if algo == 9:
        # (D→C∥B) → A
        mods = []
        if lv("D") > 0 and lv("C") > 0:
            c_field = _fm(L("C"), L("D"), lv("D"))
            mods.append((c_field, lv("C")))
        elif lv("C") > 0:
            mods.append((L("C"), lv("C")))
        if lv("B") > 0:
            mods.append((L("B"), lv("B")))
        if mods:
            w = sum(m[1] for m in mods) or 1.0
            mod = sum(m[0] * (m[1] / w) for m in mods)
            idx = max(m[1] for m in mods)
        else:
            mod = L("A")
            idx = 0.0
        field = _fm(L("A"), mod, idx) if lv("A") > 0 else mod
        field = np.clip(field * max(lv("A"), 0.01), 0, 1)
        return tint(field, "A"), field

    if algo == 10:
        # D ∥ (C→B→A)
        stack_rgb, stack_lum = stack_chain(["C", "B", "A"])
        parts = []
        if lv("A") > 0 or lv("B") > 0 or lv("C") > 0:
            parts.append((stack_rgb, stack_lum, max(lv("A"), 0.01)))
        if lv("D") > 0:
            parts.append((R("D"), L("D"), lv("D")))
        if not parts:
            return R("A"), L("A")
        return _sum_carriers(parts)

    # algo 11 — pure additive
    parts = []
    for k in OSC_IDS:
        if lv(k) > 0:
            parts.append((R(k), L(k), lv(k)))
    if not parts:
        # at least A
        return R("A"), L("A")
    return _sum_carriers(parts)


def apply_filter_shaper(
    rgb: np.ndarray,
    lum: np.ndarray,
    filt: dict[str, Any],
    shaper: dict[str, Any],
    cutoff_delta: float = 0.0,
    drive_delta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    ftype = (filt or {}).get("type") or "off"
    if ftype not in FILTER_TYPES:
        ftype = "off"
    cutoff = float(np.clip(float((filt or {}).get("cutoff", 0.55)) + cutoff_delta, 0.02, 1.0))
    drive = float(np.clip(float((filt or {}).get("drive", 0.0)) + drive_delta, 0.0, 1.0))

    out_lum = lum.astype(np.float32)
    ksize = max(1, int(round((1.0 - cutoff) * 24)) | 1)  # odd
    if ftype == "lowpass" and ksize >= 3:
        out_lum = cv2.GaussianBlur(out_lum, (ksize, ksize), 0)
    elif ftype in ("highpass", "formant") and ksize >= 3:
        blur = cv2.GaussianBlur(out_lum, (ksize, ksize), 0)
        out_lum = np.clip(out_lum - blur + 0.5, 0, 1)
        if ftype == "formant":
            edges = cv2.Sobel(out_lum, cv2.CV_32F, 1, 0, ksize=3)
            edges = np.abs(edges)
            edges = edges / (edges.max() + 1e-6)
            out_lum = np.clip(out_lum * 0.7 + edges * 0.3, 0, 1)

    if drive > 0:
        # soft clip / contrast
        x = out_lum * 2.0 - 1.0
        x = np.tanh(x * (1.0 + drive * 3.0))
        out_lum = (x * 0.5 + 0.5).astype(np.float32)

    curve = (shaper or {}).get("curve") or "linear"
    amount = float((shaper or {}).get("amount", 0.0) or 0.0)
    if curve == "s_curve" and amount > 0:
        out_lum = _apply_grade(out_lum, "s_curve")
        out_lum = (1.0 - amount) * lum + amount * out_lum
    elif curve == "expand_near" and amount > 0:
        graded = _apply_grade(out_lum, "expand_near")
        out_lum = (1.0 - amount) * out_lum + amount * graded
    elif curve == "quantize" and amount > 0:
        steps = int(np.clip(2 + amount * 14, 2, 16))
        out_lum = np.round(out_lum * (steps - 1)) / (steps - 1)

    out_lum = np.clip(out_lum, 0, 1).astype(np.float32)
    # Remap RGB by luminance ratio
    base = rgb.astype(np.float32) / 255.0
    cur = _luminance(rgb) + 1e-6
    scale = (out_lum / cur)[..., None]
    out_rgb = np.clip(base * scale * 255.0, 0, 255).astype(np.uint8)
    return out_rgb, out_lum


def normalize_patch(raw: Optional[dict[str, Any]]) -> dict[str, Any]:
    patch = default_operator_patch()
    if not raw:
        return patch
    patch["version"] = int(raw.get("version", 1))
    patch["algorithm"] = int(np.clip(int(raw.get("algorithm", 1)), 1, 11))
    patch["t"] = raw.get("t", None)
    if patch["t"] is not None:
        patch["t"] = float(np.clip(float(patch["t"]), 0.0, 1.0))

    osc_in = raw.get("oscillators") or {}
    for k in OSC_IDS:
        base = dict(DEFAULT_OSC[k])
        src = osc_in.get(k) or {}
        base.update({kk: src[kk] for kk in src if kk in base or kk in (
            "on", "level", "preset", "lo_pct", "hi_pct", "gamma", "cmap", "blur_amount"
        )})
        base["on"] = bool(base.get("on", True))
        base["level"] = float(np.clip(float(base.get("level", 1.0)), 0.0, 1.0))
        preset = base.get("preset") or DEFAULT_LOOK_A
        if preset not in LOOK_PRESETS or preset == "scenery":
            preset = DEFAULT_LOOK_A
        base["preset"] = preset
        patch["oscillators"][k] = base

    filt = dict(patch["filter"])
    filt.update(raw.get("filter") or {})
    if filt.get("type") not in FILTER_TYPES:
        filt["type"] = "off"
    patch["filter"] = filt

    sh = dict(patch["shaper"])
    sh.update(raw.get("shaper") or {})
    if sh.get("curve") not in SHAPER_CURVES:
        sh["curve"] = "linear"
    patch["shaper"] = sh

    env = dict(patch["envelope"])
    env.update(raw.get("envelope") or {})
    dest = env.get("dest") or ["level_A"]
    if isinstance(dest, str):
        dest = [dest]
    env["dest"] = [d for d in dest if d in MOD_DESTS] or ["level_A"]
    patch["envelope"] = env

    lfo = dict(patch["lfo"])
    lfo.update(raw.get("lfo") or {})
    if lfo.get("shape") not in LFO_SHAPES:
        lfo["shape"] = "sine"
    dest = lfo.get("dest") or ["filter_cutoff"]
    if isinstance(dest, str):
        dest = [dest]
    lfo["dest"] = [d for d in dest if d in MOD_DESTS] or ["filter_cutoff"]
    patch["lfo"] = lfo

    glob = dict(patch["global"])
    glob.update(raw.get("global") or {})
    voices = glob.get("voices") or ["plate"]
    if isinstance(voices, str):
        voices = [voices]
    glob["voices"] = [v for v in voices if v in VOICE_KEYS] or ["plate"]
    glob["time"] = float(np.clip(float(glob.get("time", 1.0)), 0.05, 4.0))
    glob["spread"] = float(np.clip(float(glob.get("spread", 1.0)), 0.05, 4.0))
    patch["global"] = glob
    return patch


def render_operator(
    depth: np.ndarray,
    patch: dict[str, Any],
    photo: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Render Operator patch → master plate + per-osc RGB + luminances."""
    patch = normalize_patch(patch)
    cliff = float(_subject_hi_pct(depth))
    _, inv_smooth = _prepare_inv_smooth(depth)
    deltas = _apply_mod_matrix(patch, patch.get("t"))

    levels = {}
    ons = {}
    rgbs: dict[str, np.ndarray] = {}
    lums: dict[str, np.ndarray] = {}

    for k in OSC_IDS:
        osc = dict(patch["oscillators"][k])
        ons[k] = bool(osc.get("on", True))
        level = float(osc.get("level", 1.0)) + deltas.get(f"level_{k}", 0.0)
        levels[k] = float(np.clip(level, 0.0, 1.5))
        lo_d = deltas.get(f"lo_pct_{k}", 0.0) * 20.0
        hi_d = deltas.get(f"hi_pct_{k}", 0.0) * 20.0
        rgb, lum = render_oscillator(
            depth, osc, cliff, inv_smooth, photo=photo, lo_delta=lo_d, hi_delta=hi_d
        )
        rgbs[k] = rgb
        lums[k] = lum
        if not ons[k]:
            levels[k] = 0.0

    # Ensure at least one carrier has signal
    if all(levels[k] <= 0 for k in OSC_IDS):
        ons["A"] = True
        levels["A"] = 1.0

    master_rgb, master_lum = run_algorithm(
        int(patch["algorithm"]), rgbs, lums, levels, ons
    )
    master_rgb, master_lum = apply_filter_shaper(
        master_rgb,
        master_lum,
        patch["filter"],
        patch["shaper"],
        cutoff_delta=deltas.get("filter_cutoff", 0.0) * 0.35,
        drive_delta=deltas.get("drive", 0.0) * 0.5,
    )

    spread = float(patch["global"]["spread"]) + deltas.get("spread", 0.0)
    spread = float(np.clip(spread, 0.05, 4.0))

    return {
        "patch": patch,
        "rgb": master_rgb,
        "lum": master_lum,
        "osc_rgbs": rgbs,
        "osc_lums": lums,
        "levels": levels,
        "spread": spread,
        "cliff": cliff,
    }


def depth_to_vector_field(
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dx, dy, viz_rgb) float32 gradients + RGB viz."""
    d = depth.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    filled = d.copy()
    if valid.any():
        med = float(np.median(d[valid]))
        filled[~valid] = med
    dx = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(dx * dx + dy * dy)
    mag_n = mag / (mag.max() + 1e-6)
    ang = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)
    hsv = np.zeros((*d.shape, 3), dtype=np.float32)
    hsv[..., 0] = ang * 180.0  # OpenCV H: 0-180
    hsv[..., 1] = 1.0
    hsv[..., 2] = mag_n
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    viz = (np.clip(bgr[..., ::-1], 0, 1) * 255).astype(np.uint8)
    return dx, dy, viz


def write_spread_ply(
    depth: np.ndarray,
    color: np.ndarray,
    out_path: str,
    spread: float = 1.0,
) -> str:
    """Simple orthographic-ish point cloud from depth + RGB (no intrinsics required)."""
    h, w = depth.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    z = depth.astype(np.float32) * float(spread)
    valid = np.isfinite(z) & (z > 0)
    # Normalize XY to roughly match Z scale
    fx = float(max(h, w))
    x = (xx.astype(np.float32) - w * 0.5) / fx * z
    y = (yy.astype(np.float32) - h * 0.5) / fx * z
    pts = np.stack([x[valid], -y[valid], z[valid]], axis=-1)
    cols = color.reshape(-1, 3)[valid.reshape(-1)]
    # Subsample for manageable files
    n = pts.shape[0]
    if n > 250_000:
        idx = np.linspace(0, n - 1, 250_000).astype(np.int64)
        pts = pts[idx]
        cols = cols[idx]
    try:
        import trimesh

        pcd = trimesh.PointCloud(vertices=pts, colors=cols)
        pcd.export(out_path)
    except Exception:
        # ASCII PLY fallback
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(pts, cols):
                f.write(
                    f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[0])} {int(c[1])} {int(c[2])}\n"
                )
    return out_path


def maybe_send_ndi(rgb: np.ndarray) -> bool:
    """Optional NDI send when OPERATOR_NDI=1. Returns True if sent."""
    if not os.environ.get("OPERATOR_NDI"):
        return False
    try:
        # Best-effort; package may not be installed
        import NDIlib as ndi  # type: ignore

        if not getattr(maybe_send_ndi, "_init", False):
            if not ndi.initialize():
                return False
            send_settings = ndi.SendCreate()
            send_settings.ndi_name = "DepthLedgerOperator"
            maybe_send_ndi._send = ndi.send_create(send_settings)  # type: ignore
            maybe_send_ndi._init = True  # type: ignore
        send = getattr(maybe_send_ndi, "_send", None)
        if send is None:
            return False
        frame = ndi.VideoFrameV2()
        # NDI expects BGRA
        bgra = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)
        frame.data = bgra
        frame.xres = bgra.shape[1]
        frame.yres = bgra.shape[0]
        frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRA
        ndi.send_send_video_v2(send, frame)
        return True
    except Exception:
        return False
