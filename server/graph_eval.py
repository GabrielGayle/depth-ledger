"""Graph evaluation for Depth Ledger node recipes (v2)."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional, Tuple

import imageio
import numpy as np

from depth_core.pipeline import (
    BLEND_MODES,
    DEFAULT_BLEND_MODE,
    DEFAULT_LOOK_A,
    DEFAULT_LOOK_B,
    DEFAULT_SCENERY_CLEAN,
    DEFAULT_SCENERY_FEATHER,
    DEFAULT_SCENERY_MATTE,
    DEFAULT_SCENERY_PROTECT,
    DEFAULT_WASH_BLUR,
    LOOK_PRESETS,
    SCENERY_MATTE_MODES,
    _downscale_thumb,
    _prepare_inv_smooth,
    _render_scenery_look,
    _resolve_hi_pct,
    _resolve_lo_pct,
    _subject_hi_pct,
    _visualize_inverse_depth,
    blend_looks,
    render_linear_inverse_turbo,
)
from depth_core.relief import (
    DEFAULT_RELIEF_GC_ALPHA,
    DEFAULT_RELIEF_INVERT,
    DEFAULT_RELIEF_MAX_SIDE,
    RELIEF_MAX_SIDE_CHOICES,
    relief_params_from_data,
    render_bas_relief,
)
from depth_core.temporal import (
    WindowRange,
    apply_temporal_depth,
    compute_window_series,
    freq_fuse_depth,
    freq_fuse_params_from_data,
    percentile_near_far,
    smooth_params_from_data,
)
from server.cache import get_run, update_run

# Low-res thumbs for Look / Scene / Blend / Relief / Smooth node previews.
NODE_THUMB_MAX = 128

# User-saved kitchen layout (positions, edges, blend settings) — not run-specific.
_DEFAULT_GRAPH_PATH = Path(__file__).resolve().parent.parent / "outputs" / "_default_graph.json"
_DEFAULT_VIDEO_GRAPH_PATH = (
    Path(__file__).resolve().parent.parent / "outputs" / "_default_video_graph.json"
)

# Depth-like sources: no plate for blend compositing.
_DEPTH_LIKE = frozenset({"depth", "arena", "arena_out", "smooth", "freq_fuse", "video_in"})
# Processors that rewrite metric depth for downstream looks.
_DEPTH_PROCESSORS = frozenset({"smooth", "freq_fuse"})
# Plate sinks (still Output + video Video Out).
_SINK_TYPES = frozenset({"output", "video_out"})


def _builtin_default_graph(run_id: str) -> dict[str, Any]:
    """Are.na → Depth → Ledger → Soft Light Sockets → Screen Scene → Relief → Output.

    Blank-ready: works with an empty run_id so the kitchen can boot before
    any image is loaded. Blend mode/opacity live on edges between image nodes.
    """
    return {
        "version": 2,
        "nodes": [
            {
                "id": "arena",
                "type": "arena",
                "position": {"x": 40, "y": 100},
                "data": {"url": "", "status": "idle"},
            },
            {
                "id": "depth",
                "type": "depth",
                "position": {"x": 320, "y": 120},
                "data": {"run_id": run_id or ""},
            },
            {
                "id": "look_a",
                "type": "look",
                "position": {"x": 560, "y": 120},
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
                "position": {"x": 840, "y": 120},
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
                "position": {"x": 1120, "y": 120},
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
                "id": "relief",
                "type": "relief",
                "position": {"x": 1400, "y": 120},
                "data": {
                    "gc_alpha": DEFAULT_RELIEF_GC_ALPHA,
                    "invert": DEFAULT_RELIEF_INVERT,
                    "max_side": DEFAULT_RELIEF_MAX_SIDE,
                    "collapsed": True,
                },
            },
            {
                "id": "output",
                "type": "output",
                "position": {"x": 1680, "y": 40},
                "data": {"preview_width": 420},
            },
            {
                "id": "arena_out",
                "type": "arena_out",
                "position": {"x": 1680, "y": 320},
                "data": {
                    "channel_id": "",
                    "title": "",
                    "status": "idle",
                    "batch_count": 5,
                    "wackiness": 0.45,
                    "max_fidelity": False,
                },
            },
        ],
        "edges": [
            {
                "id": "e_ar_d",
                "source": "arena",
                "target": "depth",
                "sourceHandle": "out",
                "targetHandle": "in",
            },
            {"id": "e_d_a", "source": "depth", "target": "look_a"},
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
            {"id": "e_s_r", "source": "scene", "target": "relief"},
            {"id": "e_r_o", "source": "relief", "target": "output"},
            {
                "id": "e_r_ao",
                "source": "relief",
                "target": "arena_out",
            },
        ],
    }


def _builtin_default_video_graph(run_id: str) -> dict[str, Any]:
    """Video In → Depth → Smooth(EMA) → Looks → Scene → Video Out."""
    from server.video_recipes import builtin_video_recipe_graph

    return builtin_video_recipe_graph("video-ema", run_id)


_VIDEO_IN_EPHEMERAL = (
    "run_id",
    "source_url",
    "preview_gif_url",
    "status",
    "filename",
    "frame_count",
    "error",
)


def graph_to_template(graph: dict[str, Any]) -> dict[str, Any]:
    """Strip run-specific fields so a graph can be reused as the kitchen default."""
    nodes = []
    for n in graph.get("nodes") or []:
        data = dict(n.get("data") or {})
        if n.get("type") == "depth":
            data.pop("run_id", None)
        if n.get("type") == "video_in":
            for key in _VIDEO_IN_EPHEMERAL:
                data.pop(key, None)
        if n.get("type") == "freq_fuse":
            data.pop("status", None)
            data.pop("backend_label", None)
        nodes.append(
            {
                "id": n["id"],
                "type": n.get("type"),
                "position": n.get("position") or {"x": 0, "y": 0},
                "data": data,
            }
        )
    edges = []
    for e in graph.get("edges") or []:
        edge: dict[str, Any] = {
            "id": e["id"],
            "source": e["source"],
            "target": e["target"],
        }
        if e.get("sourceHandle") is not None:
            edge["sourceHandle"] = e["sourceHandle"]
        if e.get("targetHandle") is not None:
            edge["targetHandle"] = e["targetHandle"]
        if e.get("data"):
            edge["data"] = dict(e["data"])
        edges.append(edge)
    return {"version": int(graph.get("version") or 2), "nodes": nodes, "edges": edges}


def template_to_graph(template: dict[str, Any], run_id: str) -> dict[str, Any]:
    nodes = []
    for n in template.get("nodes") or []:
        data = dict(n.get("data") or {})
        if n.get("type") == "depth":
            data["run_id"] = run_id
        if n.get("type") == "video_in" and run_id:
            data["run_id"] = run_id
        nodes.append(
            {
                "id": n["id"],
                "type": n.get("type"),
                "position": n.get("position") or {"x": 0, "y": 0},
                "data": data,
            }
        )
    return {
        "version": int(template.get("version") or 2),
        "nodes": nodes,
        "edges": list(template.get("edges") or []),
    }


def load_default_template(kind: str = "still") -> Optional[dict[str, Any]]:
    path = _DEFAULT_VIDEO_GRAPH_PATH if kind == "video" else _DEFAULT_GRAPH_PATH
    if not path.is_file():
        return None
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "nodes" not in data:
            return None
        return data
    except Exception:
        return None


def save_default_template(graph: dict[str, Any], kind: str = "still") -> Path:
    import json
    from datetime import datetime, timezone

    template = graph_to_template(graph)
    template["saved_at"] = datetime.now(timezone.utc).isoformat()
    template["kind"] = kind
    path = _DEFAULT_VIDEO_GRAPH_PATH if kind == "video" else _DEFAULT_GRAPH_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    return path


def clear_default_template(kind: str = "still") -> bool:
    path = _DEFAULT_VIDEO_GRAPH_PATH if kind == "video" else _DEFAULT_GRAPH_PATH
    if path.is_file():
        path.unlink()
        return True
    return False


def default_graph(run_id: str, kind: str = "still") -> dict[str, Any]:
    """User-saved kitchen default if present, else built-in stack (still or video)."""
    template = load_default_template(kind)
    if template is not None:
        return template_to_graph(template, run_id)
    if kind == "video":
        return _builtin_default_video_graph(run_id)
    return _builtin_default_graph(run_id)


def _topo(nodes: list[dict], edges: list[dict]) -> list[str]:
    ids = {n["id"] for n in nodes}
    indeg = {i: 0 for i in ids}
    adj: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        s, t = e["source"], e["target"]
        if s in ids and t in ids:
            adj[s].append(t)
            indeg[t] += 1
    q = deque([i for i, d in indeg.items() if d == 0])
    order: list[str] = []
    while q:
        n = q.popleft()
        order.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    if len(order) != len(ids):
        raise ValueError("Graph has a cycle or disconnected dependency issue")
    return order


def _incoming(edges: list[dict], node_id: str) -> list[dict]:
    return [e for e in edges if e["target"] == node_id]


def _look_meta_entry(key: str, cfg: dict) -> dict:
    lo = cfg.get("lo_pct", 2.0)
    hi = cfg.get("hi_pct", 98.0)
    if lo == "auto_bg":
        lo_seed = 15.0
    else:
        lo_seed = float(lo)
    if hi == "auto":
        hi_seed = float(cfg.get("hi_pct_fallback", 36.0))
    else:
        hi_seed = float(hi)
    entry = {
        "key": key,
        "label": cfg["label"],
        "cmap": cfg.get("cmap") or "gray",
        "lo_pct": lo_seed,
        "hi_pct": hi_seed,
        "gamma": float(cfg.get("gamma", 1.0)),
        "grade": cfg.get("grade"),
    }
    if cfg.get("cmap") == "wash" or "blur_amount" in cfg:
        entry["blur_amount"] = float(cfg.get("blur_amount", DEFAULT_WASH_BLUR))
    return entry


def _incoming_plate(
    edges: list[dict],
    node_id: str,
    outputs: dict[str, np.ndarray],
    nodes: dict[str, dict],
) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """First image plate into this node + the edge that carried it.

    Depth / Smooth sources don't produce a plate — Look/Scene still render
    from (optionally smoothed) run depth. Legacy `base` handle edges accepted.
    """
    for e in _incoming(edges, node_id):
        handle = e.get("targetHandle")
        if handle in ("overlay",):
            continue
        src_id = e["source"]
        src = nodes.get(src_id) or {}
        if src.get("type") in _DEPTH_LIKE:
            continue
        img = outputs.get(src_id)
        if img is not None:
            return img, e
    return None, None


def _upstream_depth_processor_id(
    edges: list[dict],
    node_id: str,
    nodes: dict[str, dict],
) -> Optional[str]:
    """Walk single-input chain toward sources; return nearest Smooth / Freq Fuse id."""
    seen: set[str] = set()
    frontier = [node_id]
    while frontier:
        cur = frontier.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        for e in _incoming(edges, cur):
            src_id = e["source"]
            src = nodes.get(src_id) or {}
            if src.get("type") in _DEPTH_PROCESSORS:
                return src_id
            if src.get("type") in ("look", "scene", "blend", "relief", "smooth", "freq_fuse"):
                frontier.append(src_id)
    return None


def _upstream_smooth_id(
    edges: list[dict],
    node_id: str,
    nodes: dict[str, dict],
) -> Optional[str]:
    """Alias for nearest depth processor (smooth or freq_fuse)."""
    return _upstream_depth_processor_id(edges, node_id, nodes)


def _exec_look(
    depth: np.ndarray,
    cliff: float,
    inv_smooth: Optional[np.ndarray],
    data: dict,
    looks_cache: dict,
    photo: Optional[np.ndarray] = None,
    near_override: Optional[float] = None,
    far_override: Optional[float] = None,
) -> np.ndarray:
    preset = data.get("preset") or DEFAULT_LOOK_A
    cmap_override = data.get("cmap")

    if preset == "scenery":
        return looks_cache.get("scenery") or _render_scenery_look(depth, cliff)

    cfg = dict(LOOK_PRESETS.get(preset, LOOK_PRESETS[DEFAULT_LOOK_A]))
    cmap = (
        cmap_override
        if cmap_override not in (None, "", "inherit")
        else cfg["cmap"]
    )

    if data.get("lo_pct") is not None:
        lo_pct = float(data["lo_pct"])
    else:
        lo_pct = _resolve_lo_pct(cfg, cliff)
    if data.get("hi_pct") is not None:
        hi_pct = float(data["hi_pct"])
    else:
        hi_pct = _resolve_hi_pct(cfg, cliff)
    gamma = float(data.get("gamma", cfg["gamma"]))
    blur_amount = float(
        data["blur_amount"]
        if data.get("blur_amount") is not None
        else cfg.get("blur_amount", DEFAULT_WASH_BLUR)
    )

    seasoning_keys = {
        "lo_pct",
        "hi_pct",
        "gamma",
        "cmap",
        "blur_amount",
        "collapsed",
        "preset",
        "blend_mode",
        "opacity",
    }
    extras = {k: v for k, v in data.items() if k not in seasoning_keys}
    has_window_override = data.get("lo_pct") is not None or data.get("hi_pct") is not None
    has_gamma_override = "gamma" in data and data["gamma"] is not None
    has_cmap_override = cmap_override not in (None, "", "inherit")
    has_blur_override = data.get("blur_amount") is not None
    has_meter_override = near_override is not None and far_override is not None
    # Photo / wash always need a live re-render (depends on source RGB).
    # Meter overrides (from Smooth) also skip the preset cache.
    photo_cmaps = ("photo", "wash")
    if (
        preset in looks_cache
        and not extras
        and not has_window_override
        and not has_gamma_override
        and not has_cmap_override
        and not has_blur_override
        and not has_meter_override
        and cmap not in photo_cmaps
    ):
        return looks_cache[preset]

    if inv_smooth is None:
        _, inv_smooth = _prepare_inv_smooth(depth)
    return _visualize_inverse_depth(
        depth,
        cmap=cmap,
        lo_pct=lo_pct,
        hi_pct=hi_pct,
        gamma=gamma,
        grade=cfg.get("grade"),
        inv_smooth=inv_smooth,
        photo=photo if cmap in photo_cmaps else None,
        blur_amount=blur_amount,
        near_override=near_override,
        far_override=far_override,
    )


def _exec_scene(depth: np.ndarray, cliff: float, data: dict) -> np.ndarray:
    mode = data.get("mode", DEFAULT_SCENERY_MATTE)
    if mode not in SCENERY_MATTE_MODES:
        mode = DEFAULT_SCENERY_MATTE
    return _render_scenery_look(
        depth,
        cliff,
        protect=float(data.get("protect", DEFAULT_SCENERY_PROTECT)),
        feather=float(data.get("feather", DEFAULT_SCENERY_FEATHER)),
        mode=mode,
        clean_specks=bool(data.get("clean", DEFAULT_SCENERY_CLEAN)),
        fringe_kill=bool(data.get("fringe_kill", True)),
        far_remap=bool(data.get("far_remap", True)),
    )


def _exec_blend(base: np.ndarray, overlay: np.ndarray, data: dict) -> np.ndarray:
    mode = data.get("mode") or data.get("blend_mode") or DEFAULT_BLEND_MODE
    if mode not in BLEND_MODES:
        mode = DEFAULT_BLEND_MODE
    opacity = float(data.get("opacity", 1.0))
    return blend_looks(base, overlay, opacity, mode)


def _exec_relief(plate: np.ndarray, data: dict) -> np.ndarray:
    """Image→image bas-relief juice of an incoming look / scene plate."""
    params = relief_params_from_data(data)
    return render_bas_relief(plate, **params)


def _composite_from_edge(
    rendered: np.ndarray,
    base: Optional[np.ndarray],
    edge: Optional[dict],
) -> np.ndarray:
    """Composite rendered look onto incoming plate using *edge* blend params."""
    if base is None:
        return rendered
    edata = dict((edge or {}).get("data") or {})
    # Legacy: blend lived on the target node.
    return _exec_blend(base, rendered, edata)


def _write_node_thumb(out_dir: Path, node_id: str, img: np.ndarray) -> str:
    thumb = _downscale_thumb(img, NODE_THUMB_MAX)
    path = out_dir / f"node_{node_id}_thumb.png"
    imageio.imwrite(path, thumb)
    return path.name


def _window_cache_key(params: dict[str, Any], frame_count: int) -> tuple:
    return (
        params["window_mode"],
        params["window_lock"],
        params["spline_interval"],
        params["lo_pct"],
        params["hi_pct"],
        int(frame_count),
    )


def _ensure_window_series(
    state: dict[str, Any],
    params: dict[str, Any],
) -> Optional[list[WindowRange]]:
    """Cache per-frame near/far for locked/spline modes on video runs."""
    window_mode = params["window_mode"]
    if window_mode == "off":
        return None
    frame_count = int(state.get("frame_count") or 1)
    ready_count = int(state.get("ready_count") or frame_count)
    kind = state.get("kind") or "still"
    if kind != "video" or frame_count <= 1:
        return None

    # Only use frames that exist; recompute as ready_count grows (except locked/first).
    use_n = max(1, min(ready_count, frame_count))
    key = _window_cache_key(params, frame_count) + (use_n, params["window_lock"])
    cached = state.get("_smooth_window_cache")
    if (
        isinstance(cached, dict)
        and cached.get("key") == key
        and isinstance(cached.get("ranges"), list)
        and len(cached["ranges"]) >= use_n
    ):
        ranges = cached["ranges"]
        # Pad locked ranges out to frame_count for scrub indices beyond cache build.
        if len(ranges) < frame_count and params["window_mode"] == "locked":
            pad = ranges[-1]
            ranges = list(ranges) + [pad] * (frame_count - len(ranges))
        return ranges[:frame_count] if len(ranges) >= frame_count else ranges

    from depth_core.video import load_frame_arrays

    out_dir = Path(state["out_dir"])
    depths = []
    for i in range(use_n):
        _rgb, depth = load_frame_arrays(out_dir, i)
        depths.append(depth)
    ranges = compute_window_series(
        depths,
        params["lo_pct"],
        params["hi_pct"],
        window_mode=window_mode,
        window_lock=params["window_lock"],
        spline_interval=params["spline_interval"],
    )
    # For locked mode, extend the lock across the full planned clip length.
    if params["window_mode"] == "locked" and ranges:
        locked = ranges[0]
        ranges = [locked for _ in range(frame_count)]
    update_run(
        state["run_id"], _smooth_window_cache={"key": key, "ranges": ranges}
    )
    return ranges


def _load_prev_frame(
    state: dict[str, Any],
    frame_index: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if frame_index <= 0:
        return None, None
    kind = state.get("kind") or "still"
    count = int(state.get("frame_count") or 1)
    if kind != "video" or count <= 1:
        return None, None
    from depth_core.video import load_frame_arrays

    try:
        return load_frame_arrays(Path(state["out_dir"]), frame_index - 1)
    except FileNotFoundError:
        return None, None


def _exec_smooth(
    state: dict[str, Any],
    data: dict,
    *,
    temporal_state: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Produce smoothed depth + optional locked near/far for downstream looks.

    mode=none is a true pass-through: depth unchanged, no near/far overrides.
    """
    params = smooth_params_from_data(data)
    depth = np.asarray(state["depth"], dtype=np.float32)
    photo = state.get("rgb")
    frame_index = int(state.get("frame_index") or 0)

    # Plug-and-play bypass: leave graph wired, change nothing.
    if params.get("mode") == "none":
        cliff = float(_subject_hi_pct(depth))
        _, inv = _prepare_inv_smooth(depth)
        preview = _visualize_inverse_depth(
            depth,
            cmap="gray",
            lo_pct=params["lo_pct"],
            hi_pct=params["hi_pct"],
            gamma=0.85,
            inv_smooth=inv,
        )
        if temporal_state is not None:
            temporal_state["prev_smoothed"] = depth
        return {
            "depth": depth,
            "inv_smooth": inv,
            "cliff": cliff,
            "near_override": None,
            "far_override": None,
            "preview": preview,
            "params": params,
        }

    prev_rgb, prev_depth = _load_prev_frame(state, frame_index)
    prev_smoothed = None
    if temporal_state is not None:
        prev_smoothed = temporal_state.get("prev_smoothed")

    work_depth = apply_temporal_depth(
        depth,
        prev_depth,
        temporal=params["temporal"],
        ema_alpha=params["ema_alpha"],
        flow_prior=params["flow_prior"],
        occ_thresh=params["occ_thresh"],
        flow_weight=params.get("flow_weight"),
        motion_gate=params.get("motion_gate", 0.0),
        curr_rgb=photo,
        prev_rgb=prev_rgb,
        prev_smoothed=prev_smoothed,
    )

    near_override: Optional[float] = None
    far_override: Optional[float] = None
    ranges = _ensure_window_series(state, params)
    if ranges is not None and 0 <= frame_index < len(ranges):
        near_override = ranges[frame_index].near_m
        far_override = ranges[frame_index].far_m
    elif params["window_mode"] == "locked":
        # Still or single-frame: lock to this frame's percentiles.
        wr = percentile_near_far(work_depth, params["lo_pct"], params["hi_pct"])
        near_override, far_override = wr.near_m, wr.far_m

    cliff = float(_subject_hi_pct(work_depth))
    _, inv = _prepare_inv_smooth(work_depth)
    preview = _visualize_inverse_depth(
        work_depth,
        cmap="gray",
        lo_pct=params["lo_pct"],
        hi_pct=params["hi_pct"],
        gamma=0.85,
        inv_smooth=inv,
        near_override=near_override,
        far_override=far_override,
    )

    if temporal_state is not None:
        temporal_state["prev_smoothed"] = work_depth

    return {
        "depth": work_depth,
        "inv_smooth": inv,
        "cliff": cliff,
        "near_override": near_override,
        "far_override": far_override,
        "preview": preview,
        "params": params,
    }


def _exec_freq_fuse(
    state: dict[str, Any],
    data: dict,
) -> dict[str, Any]:
    """Fuse stub/video depth low-freq with Depth Pro high-freq."""
    from depth_core.video_depth_stub import (
        ensure_stub_video_depth,
        load_video_depth,
    )

    params = freq_fuse_params_from_data(data)
    depth = np.asarray(state["depth"], dtype=np.float32)
    frame_index = int(state.get("frame_index") or 0)
    out_dir = Path(state["out_dir"])
    frame_count = int(state.get("frame_count") or 1)
    kind = state.get("kind") or "still"

    if kind == "video" and frame_count > 1:
        from depth_core.video_depth_stub import build_stub_video_depth

        try:
            info = ensure_stub_video_depth(out_dir, frame_count)
            vid = load_video_depth(out_dir, frame_index)
        except FileNotFoundError:
            info = build_stub_video_depth(out_dir, force=True)
            vid = load_video_depth(out_dir, frame_index)
        work_depth = freq_fuse_depth(
            depth,
            vid,
            beta=params["beta"],
            blur_sigma=params["blur_sigma"],
        )
        backend_label = str(info.get("label") or info.get("backend") or "stub")
    else:
        # Still: no-op (Pro only)
        work_depth = depth
        backend_label = "still (pass-through)"

    cliff = float(_subject_hi_pct(work_depth))
    _, inv = _prepare_inv_smooth(work_depth)
    preview = _visualize_inverse_depth(
        work_depth,
        cmap="gray",
        lo_pct=2.0,
        hi_pct=98.0,
        gamma=0.85,
        inv_smooth=inv,
    )
    return {
        "depth": work_depth,
        "inv_smooth": inv,
        "cliff": cliff,
        "near_override": None,
        "far_override": None,
        "preview": preview,
        "params": params,
        "backend_label": backend_label,
    }


def evaluate_graph(
    run_id: str,
    graph: dict[str, Any],
    *,
    temporal_state: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Evaluate node graph against cached depth; write blend_live.png + low-res thumbs.

    temporal_state: optional mutable dict for sequential video export EMA/flow
    (keeps prev_smoothed across frames). Scrub without it uses raw previous frame.
    """
    t0 = time.perf_counter()
    state = get_run(run_id)
    if state is None:
        raise KeyError(f"Unknown run_id: {run_id}")

    depth = state.get("depth")
    if depth is None:
        raise ValueError("Run has no depth yet — Infer the video first")
    photo = state.get("rgb")
    cliff = float(state.get("cliff") or _subject_hi_pct(depth))
    looks_cache = dict(state.get("looks") or {})
    # Render only the looks the graph actually uses — never precompute every preset.

    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    edges = graph.get("edges", [])
    order = _topo(list(nodes.values()), edges)

    outputs: dict[str, np.ndarray] = {}
    smooth_ctx: dict[str, dict[str, Any]] = {}
    _, inv_smooth = _prepare_inv_smooth(depth)
    out_dir = Path(state["out_dir"])
    thumb_names: dict[str, str] = {}

    for nid in order:
        node = nodes[nid]
        ntype = node.get("type")
        data = dict(node.get("data") or {})

        if ntype == "depth":
            continue

        # Source-only nodes (Are.na feed, etc.) — no plate output.
        if ntype in ("arena", "arena_out", "video_in"):
            continue

        if ntype == "smooth":
            ctx = _exec_smooth(state, data, temporal_state=temporal_state)
            smooth_ctx[nid] = ctx
            outputs[nid] = ctx["preview"]
            thumb_names[nid] = _write_node_thumb(out_dir, nid, ctx["preview"])
            continue

        if ntype == "freq_fuse":
            ctx = _exec_freq_fuse(state, data)
            smooth_ctx[nid] = ctx
            outputs[nid] = ctx["preview"]
            thumb_names[nid] = _write_node_thumb(out_dir, nid, ctx["preview"])
            continue

        # Resolve optional upstream Smooth / Freq Fuse context for look/scene depth.
        sm_id = _upstream_smooth_id(edges, nid, nodes)
        sm = smooth_ctx.get(sm_id) if sm_id else None
        use_depth = sm["depth"] if sm else depth
        use_inv = sm["inv_smooth"] if sm else inv_smooth
        use_cliff = float(sm["cliff"]) if sm else cliff
        near_ov = sm.get("near_override") if sm else None
        far_ov = sm.get("far_override") if sm else None

        if ntype == "look":
            rendered = _exec_look(
                use_depth,
                use_cliff,
                use_inv,
                data,
                looks_cache,
                photo=photo,
                near_override=near_ov,
                far_override=far_ov,
            )
            base, edge = _incoming_plate(edges, nid, outputs, nodes)
            # Legacy node-level blend if edge has no data but node still does.
            if base is not None and not (edge or {}).get("data") and (
                data.get("blend_mode") or data.get("opacity") is not None
            ):
                outputs[nid] = _exec_blend(base, rendered, data)
            else:
                outputs[nid] = _composite_from_edge(rendered, base, edge)
            preset = data.get("preset")
            cmap_override = data.get("cmap")
            preset_cmap = (LOOK_PRESETS.get(preset) or {}).get("cmap")
            effective_cmap = (
                cmap_override
                if cmap_override not in (None, "", "inherit")
                else preset_cmap
            )
            # Don't poison the shared preset cache with seasoning overrides
            # (especially photo / wash, which depend on source RGB).
            if (
                preset
                and preset in LOOK_PRESETS
                and preset != "scenery"
                and cmap_override in (None, "", "inherit")
                and near_ov is None
                and effective_cmap not in ("photo", "wash")
                and data.get("blur_amount") is None
            ):
                looks_cache[preset] = rendered
            thumb_names[nid] = _write_node_thumb(out_dir, nid, outputs[nid])
            continue

        if ntype == "scene":
            rendered = _exec_scene(use_depth, use_cliff, data)
            base, edge = _incoming_plate(edges, nid, outputs, nodes)
            if base is not None and not (edge or {}).get("data") and (
                data.get("blend_mode") or data.get("opacity") is not None
            ):
                outputs[nid] = _exec_blend(base, rendered, data)
            else:
                outputs[nid] = _composite_from_edge(rendered, base, edge)
            looks_cache["scenery"] = rendered
            thumb_names[nid] = _write_node_thumb(out_dir, nid, outputs[nid])
            continue

        if ntype == "relief":
            base, _edge = _incoming_plate(edges, nid, outputs, nodes)
            if base is None:
                raise ValueError(
                    f"Relief node {nid} needs an incoming look/scene/blend plate"
                )
            outputs[nid] = _exec_relief(base, data)
            thumb_names[nid] = _write_node_thumb(out_dir, nid, outputs[nid])
            continue

        if ntype == "blend":
            incoming = _incoming(edges, nid)
            base_img = overlay_img = None
            for e in incoming:
                src = outputs.get(e["source"])
                if src is None:
                    continue
                handle = e.get("targetHandle") or "base"
                if handle == "overlay":
                    overlay_img = src
                else:
                    base_img = src
            if base_img is None and overlay_img is None:
                raise ValueError(f"Blend node {nid} has no inputs")
            if base_img is None:
                base_img = overlay_img
            if overlay_img is None:
                overlay_img = base_img
            outputs[nid] = _exec_blend(base_img, overlay_img, data)
            thumb_names[nid] = _write_node_thumb(out_dir, nid, outputs[nid])
            continue

        if ntype in ("output", "video_out"):
            incoming = _incoming(edges, nid)
            if not incoming:
                raise ValueError(f"{ntype} node has no input")
            src_id = incoming[0]["source"]
            src = outputs.get(src_id)
            if src is None:
                raise ValueError(f"{ntype} input not ready")
            outputs[nid] = src
            continue

        raise ValueError(f"Unknown node type: {ntype}")

    final = None
    for n in nodes.values():
        if n.get("type") in _SINK_TYPES and n["id"] in outputs:
            final = outputs[n["id"]]
            break
    if final is None:
        for nid in reversed(order):
            if nodes[nid].get("type") in (
                "blend",
                "look",
                "scene",
                "relief",
                "smooth",
                "freq_fuse",
            ) and nid in outputs:
                final = outputs[nid]
                break
    if final is None:
        raise ValueError("Graph produced no output image")

    blend_path = out_dir / "blend_live.png"
    imageio.imwrite(blend_path, final)

    # Keep only the presets this graph used — drop full-res plates from RAM later
    # via the run LRU. Don't persist the output array (it's on disk as blend_live).
    used_looks = {
        k: v for k, v in looks_cache.items() if k in LOOK_PRESETS
    }
    update_run(
        run_id,
        looks=used_looks,
        blend=None,
        blend_path=str(blend_path),
        graph=graph,
    )

    from depth_core.pipeline import load_recipe, save_recipe

    recipe = {
        "version": 2,
        "model_key": state.get("model_key"),
        "max_size": state.get("max_size"),
        "infer_backend": state.get("infer_backend"),
        "infer_device": state.get("infer_device"),
        "graph": graph,
    }
    existing = load_recipe(out_dir) or {}
    arena = state.get("arena") or existing.get("arena")
    if arena:
        recipe["arena"] = arena
    save_recipe(out_dir, recipe)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "run_id": run_id,
        "output_url": f"/api/runs/{run_id}/files/blend_live.png",
        "elapsed_ms": round(elapsed_ms, 1),
        "node_previews": {
            nid: f"/api/runs/{run_id}/files/{name}"
            for nid, name in thumb_names.items()
        },
    }


def render_tune_turbo_preview(
    run_id: str,
    *,
    node_id: Optional[str] = None,
) -> dict[str, Any]:
    """Write a native-res grayscale picking plate for the Tune windows modal."""
    state = get_run(run_id)
    if state is None:
        raise KeyError(f"Unknown run_id: {run_id}")
    depth = state.get("depth")
    if depth is None:
        raise ValueError("Run has no depth yet")

    frame_index = int(state.get("frame_index") or 0)
    out_dir = Path(state["out_dir"])
    path = out_dir / "tune_turbo.png"
    # Prefix bumps past older colored-turbo caches.
    cache_key = f"gray:{frame_index}:{node_id or ''}"
    if state.get("tune_turbo_key") == cache_key and path.is_file():
        return {
            "url": f"/api/runs/{run_id}/files/tune_turbo.png",
            "frame_index": frame_index,
        }

    work = np.asarray(depth, dtype=np.float32)
    if node_id:
        graph = state.get("graph")
        if not graph:
            from depth_core.pipeline import load_recipe

            recipe = load_recipe(out_dir) or {}
            graph = recipe.get("graph")
        if graph:
            nodes = {n["id"]: n for n in graph.get("nodes", [])}
            edges = graph.get("edges", [])
            sm_id = _upstream_smooth_id(edges, node_id, nodes)
            if sm_id:
                proc = nodes.get(sm_id) or {}
                data = dict(proc.get("data") or {})
                if proc.get("type") == "freq_fuse":
                    ctx = _exec_freq_fuse(state, data)
                else:
                    ctx = _exec_smooth(state, data)
                work = ctx["depth"]

    rgb = render_linear_inverse_turbo(work)
    imageio.imwrite(path, rgb)
    update_run(run_id, tune_turbo_key=cache_key)
    return {
        "url": f"/api/runs/{run_id}/files/tune_turbo.png",
        "frame_index": frame_index,
    }


def write_node_previews(run_id: str, outputs: dict[str, np.ndarray]) -> None:
    state = get_run(run_id)
    if not state:
        return
    out_dir = Path(state["out_dir"])
    for nid, img in outputs.items():
        _write_node_thumb(out_dir, nid, img)
