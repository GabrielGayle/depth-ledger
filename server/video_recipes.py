"""Built-in video kitchen recipes (EMA / Flow / Freq Fuse)."""

from __future__ import annotations

from typing import Any, Optional

from depth_core.pipeline import (
    DEFAULT_LOOK_A,
    DEFAULT_LOOK_B,
    DEFAULT_SCENERY_CLEAN,
    DEFAULT_SCENERY_FEATHER,
    DEFAULT_SCENERY_MATTE,
    DEFAULT_SCENERY_PROTECT,
)
from depth_core.temporal import (
    DEFAULT_EMA_ALPHA,
    DEFAULT_FLOW_PRIOR,
    DEFAULT_FREQ_BETA,
    DEFAULT_FREQ_BLUR_SIGMA,
    DEFAULT_OCC_THRESH,
)


def _look_nodes() -> list[dict[str, Any]]:
    return [
        {
            "id": "look_a",
            "type": "look",
            "position": {"x": 880, "y": 120},
            "data": {
                "preset": DEFAULT_LOOK_A,
                "cmap": "inherit",
                "collapsed": True,
                "lo_pct": 2.0,
                "hi_pct": 98.0,
                "gamma": 0.85,
            },
        },
        {
            "id": "look_b",
            "type": "look",
            "position": {"x": 1160, "y": 120},
            "data": {
                "preset": DEFAULT_LOOK_B,
                "cmap": "inherit",
                "collapsed": True,
                "lo_pct": 1.0,
                "hi_pct": 30.0,
                "gamma": 1.0,
            },
        },
        {
            "id": "scene",
            "type": "scene",
            "position": {"x": 1440, "y": 120},
            "data": {
                "mode": DEFAULT_SCENERY_MATTE,
                "protect": DEFAULT_SCENERY_PROTECT,
                "feather": DEFAULT_SCENERY_FEATHER,
                "clean": DEFAULT_SCENERY_CLEAN,
                "fringe_kill": True,
                "far_remap": True,
                "collapsed": True,
            },
        },
        {
            "id": "video_out",
            "type": "video_out",
            "position": {"x": 1720, "y": 40},
            "data": {"preview_width": 420},
        },
    ]


def _look_edges(source_id: str) -> list[dict[str, Any]]:
    return [
        {"id": f"e_{source_id}_a", "source": source_id, "target": "look_a"},
        {
            "id": "e_a_b",
            "source": "look_a",
            "target": "look_b",
            "data": {"blend_mode": "soft_light", "opacity": 0.55},
        },
        {
            "id": "e_b_s",
            "source": "look_b",
            "target": "scene",
            "data": {"blend_mode": "screen", "opacity": 0.45},
        },
        {"id": "e_s_vo", "source": "scene", "target": "video_out"},
    ]


def _base_io(run_id: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "video_in",
            "type": "video_in",
            "position": {"x": 40, "y": 100},
            "data": {"status": "idle", "run_id": run_id or ""},
        },
        {
            "id": "depth",
            "type": "depth",
            "position": {"x": 320, "y": 120},
            "data": {"run_id": run_id or ""},
        },
    ]


def _recipe_ema(run_id: str) -> dict[str, Any]:
    nodes = _base_io(run_id) + [
        {
            "id": "smooth",
            "type": "smooth",
            "position": {"x": 600, "y": 120},
            "data": {
                "mode": "ema",
                "window_mode": "locked",
                "window_lock": "first",
                "spline_interval": 30,
                "lo_pct": 2.0,
                "hi_pct": 98.0,
                "temporal": "ema",
                "ema_alpha": DEFAULT_EMA_ALPHA,
                "flow_prior": DEFAULT_FLOW_PRIOR,
                "occ_thresh": DEFAULT_OCC_THRESH,
                "collapsed": True,
            },
        },
        *_look_nodes(),
    ]
    # Shift look positions already set; edges from smooth
    edges = [
        {"id": "e_vi_d", "source": "video_in", "target": "depth"},
        {"id": "e_d_sm", "source": "depth", "target": "smooth"},
        *_look_edges("smooth"),
    ]
    return {"version": 2, "nodes": nodes, "edges": edges}


def _recipe_none(run_id: str) -> dict[str, Any]:
    """Same graph topology; Smooth is pass-through (no temporal / window)."""
    nodes = _base_io(run_id) + [
        {
            "id": "smooth",
            "type": "smooth",
            "position": {"x": 600, "y": 120},
            "data": {
                "mode": "none",
                "window_mode": "off",
                "window_lock": "first",
                "spline_interval": 30,
                "lo_pct": 2.0,
                "hi_pct": 98.0,
                "temporal": "off",
                "ema_alpha": DEFAULT_EMA_ALPHA,
                "flow_prior": DEFAULT_FLOW_PRIOR,
                "occ_thresh": DEFAULT_OCC_THRESH,
                "collapsed": True,
            },
        },
        *_look_nodes(),
    ]
    edges = [
        {"id": "e_vi_d", "source": "video_in", "target": "depth"},
        {"id": "e_d_sm", "source": "depth", "target": "smooth"},
        *_look_edges("smooth"),
    ]
    return {"version": 2, "nodes": nodes, "edges": edges}


def _recipe_flow(run_id: str) -> dict[str, Any]:
    nodes = _base_io(run_id) + [
        {
            "id": "smooth",
            "type": "smooth",
            "position": {"x": 600, "y": 120},
            "data": {
                "mode": "flow",
                "window_mode": "locked",
                "window_lock": "first",
                "spline_interval": 30,
                "lo_pct": 2.0,
                "hi_pct": 98.0,
                "temporal": "flow",
                "ema_alpha": DEFAULT_EMA_ALPHA,
                "flow_prior": DEFAULT_FLOW_PRIOR,
                "occ_thresh": DEFAULT_OCC_THRESH,
                "collapsed": True,
            },
        },
        *_look_nodes(),
    ]
    edges = [
        {"id": "e_vi_d", "source": "video_in", "target": "depth"},
        {"id": "e_d_sm", "source": "depth", "target": "smooth"},
        *_look_edges("smooth"),
    ]
    return {"version": 2, "nodes": nodes, "edges": edges}


def _recipe_freq(run_id: str) -> dict[str, Any]:
    nodes = _base_io(run_id) + [
        {
            "id": "freq_fuse",
            "type": "freq_fuse",
            "position": {"x": 600, "y": 120},
            "data": {
                "beta": DEFAULT_FREQ_BETA,
                "blur_sigma": DEFAULT_FREQ_BLUR_SIGMA,
                "collapsed": True,
            },
        },
        *_look_nodes(),
    ]
    edges = [
        {"id": "e_vi_d", "source": "video_in", "target": "depth"},
        {"id": "e_d_ff", "source": "depth", "target": "freq_fuse"},
        *_look_edges("freq_fuse"),
    ]
    return {"version": 2, "nodes": nodes, "edges": edges}


VIDEO_RECIPES: list[dict[str, Any]] = [
    {
        "id": "video-none",
        "label": "No smoothing",
        "summary": "Video In → Depth → Smooth(none) → Looks → Video Out",
    },
    {
        "id": "video-ema",
        "label": "EMA temporal",
        "summary": "Video In → Depth → Smooth(EMA) → Looks → Video Out",
    },
    {
        "id": "video-flow",
        "label": "Flow-guided",
        "summary": "Video In → Depth → Smooth(flow + occlusion) → Looks → Video Out",
    },
    {
        "id": "video-freq",
        "label": "Freq fuse (stub)",
        "summary": "Video In → Depth → Freq Fuse (stub EMA + Pro edges) → Looks → Video Out",
    },
]

_BUILDERS = {
    "video-none": _recipe_none,
    "video-ema": _recipe_ema,
    "video-flow": _recipe_flow,
    "video-freq": _recipe_freq,
}


def list_video_recipes() -> list[dict[str, Any]]:
    return [
        {
            "id": r["id"],
            "label": r["label"],
            "summary": r["summary"],
            "graph": builtin_video_recipe_graph(r["id"], ""),
        }
        for r in VIDEO_RECIPES
    ]


def builtin_video_recipe_graph(recipe_id: str, run_id: str = "") -> dict[str, Any]:
    builder = _BUILDERS.get(recipe_id) or _recipe_ema
    return builder(run_id)


def get_video_recipe(recipe_id: str) -> Optional[dict[str, Any]]:
    for r in VIDEO_RECIPES:
        if r["id"] == recipe_id:
            return {
                "id": r["id"],
                "label": r["label"],
                "summary": r["summary"],
                "graph": builtin_video_recipe_graph(r["id"], ""),
            }
    return None
