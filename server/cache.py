"""Bounded in-memory run cache. Heavy arrays (depth / rgb / looks) stay hot
for a few recent runs and reload from the slim on-disk working set.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

from depth_core.pipeline import OUTPUT_DIR, _subject_hi_pct
from depth_core.store import (
    find_depth_path,
    find_rgb_path,
    load_run_arrays,
    ram_run_limit,
)

# Metadata always; arrays only for ids in _HOT.
_RUNS: dict[str, dict[str, Any]] = {}
_HOT: OrderedDict[str, None] = OrderedDict()
_PINNED: set[str] = set()
_HEAVY_KEYS = ("depth", "rgb", "looks", "blend", "vis_prompt")


def _pinned_ids() -> set[str]:
    pinned: set[str] = set(_PINNED)
    try:
        from server.status import pinned_run_ids

        pinned |= pinned_run_ids()
    except Exception:
        pass
    for run_id, state in _RUNS.items():
        status = str(state.get("infer_status") or "")
        if status in ("pending", "running"):
            pinned.add(run_id)
    return pinned


def pin_run(run_id: str) -> None:
    _PINNED.add(run_id)
    if run_id in _RUNS:
        _mark_hot(run_id)


def unpin_run(run_id: str) -> None:
    _PINNED.discard(run_id)


def _drop_heavy_in_place(state: dict[str, Any]) -> None:
    state["depth"] = None
    state["rgb"] = None
    state["looks"] = {}
    state["blend"] = None
    state["vis_prompt"] = None


def drop_heavy(run_id: str) -> None:
    state = _RUNS.get(run_id)
    if state is None:
        return
    _drop_heavy_in_place(state)
    _HOT.pop(run_id, None)


def forget_run(run_id: str) -> None:
    _RUNS.pop(run_id, None)
    _HOT.pop(run_id, None)


def _mark_hot(run_id: str) -> None:
    _HOT.pop(run_id, None)
    _HOT[run_id] = None
    limit = ram_run_limit()
    pinned = _pinned_ids()
    while len(_HOT) > limit:
        # Prefer evicting unpinned, oldest first.
        victim = None
        for rid in _HOT:
            if rid not in pinned:
                victim = rid
                break
        if victim is None:
            break
        _HOT.pop(victim, None)
        if victim in _RUNS:
            _drop_heavy_in_place(_RUNS[victim])


def _hydrate_arrays(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("depth") is not None:
        return state
    out_dir = Path(state["out_dir"])
    kind = state.get("kind") or "still"
    frame_index = int(state.get("frame_index") or 0)
    from depth_core.video import load_frame_arrays

    if kind == "video" or int(state.get("frame_count") or 1) > 1:
        try:
            rgb, depth = load_frame_arrays(out_dir, frame_index)
        except FileNotFoundError:
            rgb_path = out_dir / "frames" / f"{frame_index:06d}" / "rgb.jpg"
            if not rgb_path.is_file():
                rgb, depth = load_run_arrays(out_dir, state.get("run_id"))
            else:
                import imageio

                rgb = imageio.imread(rgb_path)
                depth = None
        state["rgb"] = rgb
        state["depth"] = depth
    else:
        rgb, depth = load_run_arrays(out_dir, state.get("run_id"))
        state["rgb"] = rgb
        state["depth"] = depth
    if depth is not None and state.get("cliff") is None:
        state["cliff"] = float(_subject_hi_pct(depth))
    if not state.get("looks"):
        state["looks"] = {}
    return state


def put_run(state: dict[str, Any]) -> str:
    out_dir = Path(state["out_dir"])
    run_id = out_dir.name
    state = dict(state)
    state["run_id"] = run_id
    if "kind" not in state:
        state["kind"] = "video" if state.get("frame_count", 1) > 1 else "still"
    if "looks" not in state or state["looks"] is None:
        state["looks"] = {}
    _RUNS[run_id] = state
    if state.get("depth") is not None or state.get("rgb") is not None:
        _mark_hot(run_id)
    return run_id


def get_run(run_id: str) -> Optional[dict[str, Any]]:
    if run_id in _RUNS:
        state = _RUNS[run_id]
        if state.get("depth") is None and str(state.get("infer_status") or "") not in (
            "pending",
        ):
            _hydrate_arrays(state)
        if state.get("depth") is not None:
            _mark_hot(run_id)
        return state
    return load_run_from_disk(run_id)


def load_run_from_disk(run_id: str) -> Optional[dict[str, Any]]:
    out_dir = OUTPUT_DIR / run_id
    if not out_dir.is_dir():
        return None

    from depth_core.video import load_frame_arrays, load_frames_meta

    meta = load_frames_meta(out_dir)
    kind = "video" if meta and int(meta.get("frame_count") or 0) > 1 else "still"
    frame_count = int(meta.get("frame_count") or 1) if meta else 1
    ready_count = int(meta.get("ready_count") or frame_count) if meta else frame_count
    infer_status = str(meta.get("infer_status") or "done") if meta else "done"
    frame_index = 0

    rgb = None
    depth = None
    if kind == "video":
        try:
            rgb, depth = load_frame_arrays(out_dir, frame_index)
        except FileNotFoundError:
            rgb_path = out_dir / "frames" / f"{frame_index:06d}" / "rgb.jpg"
            if not rgb_path.is_file():
                # Prepared (pending) run: no RGB yet.
                if not find_rgb_path(out_dir, run_id):
                    return None
            else:
                import imageio

                rgb = imageio.imread(rgb_path)
                depth = None
    else:
        if not find_depth_path(out_dir, run_id):
            return None
        rgb, depth = load_run_arrays(out_dir, run_id)

    cliff = _subject_hi_pct(depth) if depth is not None else None
    state: dict[str, Any] = {
        "run_id": run_id,
        "kind": kind,
        "frame_index": frame_index,
        "frame_count": frame_count,
        "ready_count": ready_count,
        "infer_status": infer_status,
        "sample_fps": float(meta.get("sample_fps") or 10) if meta else 10.0,
        "out_dir": str(out_dir.resolve()),
        "depth": depth,
        "cliff": cliff,
        "rgb": rgb,
        "looks": {},
        "look_paths": {},
        "metric_path": str(out_dir / f"{run_id}_depth_mm.png")
        if (out_dir / f"{run_id}_depth_mm.png").exists()
        else None,
        "blend_path": str(
            (out_dir / "blend_live.png")
            if (out_dir / "blend_live.png").exists()
            else (out_dir / f"{run_id}_blend.png")
        )
        if (out_dir / "blend_live.png").exists()
        or (out_dir / f"{run_id}_blend.png").exists()
        else None,
        "ply_path": str(out_dir / f"{run_id}_pointcloud.ply")
        if (out_dir / f"{run_id}_pointcloud.ply").exists()
        else None,
        "preview_gif": (meta or {}).get("preview_gif"),
        "source_file": (meta or {}).get("source_file"),
    }
    if meta:
        state["model_key"] = meta.get("model_key")
        state["max_size"] = meta.get("max_size")
    from depth_core.pipeline import load_recipe

    recipe = load_recipe(out_dir)
    if recipe:
        if recipe.get("arena"):
            state["arena"] = recipe["arena"]
        if recipe.get("model_key") and not state.get("model_key"):
            state["model_key"] = recipe.get("model_key")
        if recipe.get("max_size") and not state.get("max_size"):
            state["max_size"] = recipe.get("max_size")
        if recipe.get("infer_backend"):
            state["infer_backend"] = recipe.get("infer_backend")
    _RUNS[run_id] = state
    if depth is not None:
        _mark_hot(run_id)
    return state


def set_frame(run_id: str, index: int) -> dict[str, Any]:
    """Swap active depth/rgb to frame index (video runs)."""
    state = get_run(run_id)
    if state is None:
        raise KeyError(run_id)
    out_dir = Path(state["out_dir"])
    count = int(state.get("frame_count") or 1)
    if count <= 1 and state.get("kind") != "video":
        if index != 0:
            raise ValueError("Still run has only one frame")
        return state

    from depth_core.video import load_frame_arrays, load_frames_meta

    meta = load_frames_meta(out_dir)
    ready = int(
        (meta or {}).get("ready_count") or state.get("ready_count") or count
    )
    ready = max(1, min(ready, count))
    if index < 0 or index >= ready:
        raise ValueError(
            f"Frame index {index} not ready (0..{ready - 1} of {count})"
        )

    rgb, depth = load_frame_arrays(out_dir, index)
    cliff = float(_subject_hi_pct(depth)) if depth is not None else None
    return update_run(
        run_id,
        frame_index=index,
        rgb=rgb,
        depth=depth,
        cliff=cliff,
        looks={},
        ready_count=ready,
        infer_status=str(
            (meta or {}).get("infer_status") or state.get("infer_status") or "done"
        ),
    )


def update_run(run_id: str, **kwargs: Any) -> dict[str, Any]:
    state = get_run(run_id)
    if state is None:
        raise KeyError(run_id)
    state = dict(state)
    state.update(kwargs)
    _RUNS[run_id] = state
    if any(k in kwargs for k in _HEAVY_KEYS) and (
        state.get("depth") is not None or state.get("rgb") is not None
    ):
        _mark_hot(run_id)
    return state


def ram_stats() -> dict[str, Any]:
    hot = list(_HOT.keys())
    nbytes = 0
    for rid in hot:
        st = _RUNS.get(rid) or {}
        for key in _HEAVY_KEYS:
            val = st.get(key)
            if hasattr(val, "nbytes"):
                nbytes += int(val.nbytes)
            elif isinstance(val, dict):
                for v in val.values():
                    if hasattr(v, "nbytes"):
                        nbytes += int(v.nbytes)
    return {
        "runs_cached": len(_RUNS),
        "hot_runs": hot,
        "hot_arrays_mb": round(nbytes / (1024 * 1024), 1),
        "ram_run_limit": ram_run_limit(),
    }
