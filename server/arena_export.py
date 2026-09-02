"""Build Are.na block titles + recreate metadata from a Depth Ledger run."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from depth_core.pipeline import LOOK_PRESETS, SCENERY_MATTE_MODES, load_recipe
from server.arena import (
    ArenaError,
    fetch_block,
    fetch_channel,
    parse_arena_ref,
    source_attribution_from_block,
    _iter_channel_contents,
)

# Are.na metadata: scalars only; string values have a fairly low maxLength
# (oneOf string|number|boolean — long JSON blobs 400 with "maxLength").
_META_STR_MAX = 200

_RECIPE_COMMENT_RE = re.compile(
    r"<!--\s*depth-ledger-recipe\s*(\{.*?\})\s*-->",
    re.DOTALL,
)
_RECIPE_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{\s*\"app\"\s*:\s*\"depth-ledger\".*?\})\s*```",
    re.DOTALL,
)

# UI-only / ephemeral node fields — not needed to recreate the cook.
_EPHEMERAL_NODE_KEYS = {
    "status",
    "error",
    "preview",
    "collapsed",
    "batch_i",
    "batch_n",
    "batch_progress",
    "batch_phase",
    "batch_message",
    "batch_downloaded",
    "batch_inferred",
    "batch_cached",
    "batch_cooked",
    "batch_published",
    "batch_items",
    "last_block_url",
    "last_block_id",
    "block",  # large preview payload; source lives under recipe.arena
}

# Drop from export — not needed to re-cook the look.
_SKIP_NODE_TYPES = {"arena_out", "output"}


def run_label_from_state(state: dict[str, Any]) -> str:
    """Same identity as the History dropdown label (output folder name)."""
    out = state.get("out_dir")
    if out:
        return Path(out).name
    return str(state.get("run_id") or "run")


def _clip(s: str, n: int = _META_STR_MAX) -> str:
    s = str(s or "")
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)] + "…"


def block_description_text(block: dict[str, Any]) -> str:
    """Normalize Are.na description (string or markdown object) to plain text."""
    d = block.get("description")
    if d is None:
        return ""
    if isinstance(d, str):
        return d
    if isinstance(d, dict):
        for key in ("markdown", "plain", "html", "content"):
            v = d.get(key)
            if isinstance(v, str) and v.strip():
                return v
    return str(d)


def parse_recipe_json(text: str) -> Optional[dict[str, Any]]:
    """Pull a depth-ledger recipe object out of block description text."""
    if not text:
        return None
    for rx in (_RECIPE_COMMENT_RE, _RECIPE_FENCE_RE):
        m = rx.search(text)
        if not m:
            continue
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and (
            data.get("app") == "depth-ledger" or data.get("graph")
        ):
            return data
    return None


def extract_recipe_from_block(block: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Return recreate payload from an Are.na block, or None."""
    if not isinstance(block, dict):
        return None
    meta = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
    # Fast reject when metadata is present and clearly not ours.
    if meta and meta.get("depth_ledger_app") not in (None, "depth-ledger"):
        return None
    recipe = parse_recipe_json(block_description_text(block))
    if recipe:
        return recipe
    return None


def recipe_list_entry(block: dict[str, Any], recipe: dict[str, Any]) -> dict[str, Any]:
    bid = block.get("id")
    title = block.get("title") or block.get("generated_title") or ""
    summary = (
        recipe.get("summary")
        or (block.get("metadata") or {}).get("depth_ledger_summary")
        or summarize_kitchen(recipe.get("graph"))
    )
    return {
        "block_id": bid,
        "title": title,
        "arena_url": f"https://www.are.na/block/{bid}" if bid is not None else None,
        "run_name": recipe.get("run_name")
        or (block.get("metadata") or {}).get("depth_ledger_run"),
        "summary": summary,
        "model_key": recipe.get("model_key"),
        "max_size": recipe.get("max_size"),
        "recipe": recipe,
    }


def scan_arena_kitchens(
    raw: str,
    *,
    limit: int = 40,
    max_pages: int = 8,
) -> dict[str, Any]:
    """Find Depth Ledger recipes on an Are.na block or channel."""
    ref = parse_arena_ref(raw)
    found: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    def _consider(block: dict[str, Any]) -> None:
        nonlocal found
        if len(found) >= limit:
            return
        bid = block.get("id")
        try:
            iid = int(bid) if bid is not None else None
        except (TypeError, ValueError):
            iid = None
        if iid is not None and iid in seen_ids:
            return
        recipe = extract_recipe_from_block(block)
        if recipe is None and iid is not None:
            meta = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
            # List payloads often omit description — refetch when metadata marks ours.
            if meta.get("depth_ledger_app") == "depth-ledger":
                try:
                    full = fetch_block(iid)
                    recipe = extract_recipe_from_block(full)
                    block = full
                except ArenaError:
                    recipe = None
        if recipe is None:
            return
        if iid is not None:
            seen_ids.add(iid)
        found.append(recipe_list_entry(block, recipe))

    channel_meta = None
    if ref["kind"] == "block":
        block = fetch_block(ref["id"])
        _consider(block)
        kind = "block"
    else:
        slug = str(ref["id"]).strip()
        try:
            channel_meta = fetch_channel(slug)
        except ArenaError:
            channel_meta = {"slug": slug}
        kind = "channel"
        for item in _iter_channel_contents(slug, max_pages=max_pages, per=100):
            if len(found) >= limit:
                break
            btype = str(
                item.get("type") or item.get("base_type") or item.get("class") or ""
            )
            if btype.lower() in {"channel", "channels"}:
                continue
            _consider(item)

    return {
        "kind": kind,
        "input": raw.strip(),
        "channel": {
            "id": (channel_meta or {}).get("id"),
            "slug": (channel_meta or {}).get("slug")
            or (ref["id"] if kind == "channel" else None),
            "title": (channel_meta or {}).get("title"),
        }
        if kind == "channel"
        else None,
        "count": len(found),
        "kitchens": found,
    }


def strip_graph_for_export(graph: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Keep kitchen topology + cook params; drop UI chrome."""
    if not isinstance(graph, dict):
        return None
    keep_ids: set[str] = set()
    nodes = []
    for n in graph.get("nodes") or []:
        if not isinstance(n, dict):
            continue
        if n.get("type") in _SKIP_NODE_TYPES:
            continue
        data = {
            k: v
            for k, v in dict(n.get("data") or {}).items()
            if k not in _EPHEMERAL_NODE_KEYS
        }
        if n.get("type") == "depth":
            data.pop("run_id", None)
            data.pop("modal_cost", None)
            data.pop("infer_device", None)
        nid = n.get("id")
        if nid:
            keep_ids.add(str(nid))
        nodes.append(
            {
                "id": nid,
                "type": n.get("type"),
                "position": n.get("position") or {"x": 0, "y": 0},
                "data": data,
            }
        )
    edges = []
    for e in graph.get("edges") or []:
        if not isinstance(e, dict):
            continue
        src, tgt = e.get("source"), e.get("target")
        if src not in keep_ids or tgt not in keep_ids:
            continue
        edge: dict[str, Any] = {
            "id": e.get("id"),
            "source": src,
            "target": tgt,
        }
        if e.get("sourceHandle") is not None:
            edge["sourceHandle"] = e["sourceHandle"]
        if e.get("targetHandle") is not None:
            edge["targetHandle"] = e["targetHandle"]
        if e.get("data"):
            ed = dict(e["data"])
            slim = {
                k: ed[k]
                for k in ("blend_mode", "opacity", "blendable")
                if k in ed
            }
            if slim:
                edge["data"] = slim
        edges.append(edge)
    return {
        "version": int(graph.get("version") or 2),
        "nodes": nodes,
        "edges": edges,
    }


def summarize_kitchen(graph: Optional[dict[str, Any]]) -> str:
    """Short human line of look / blend / scene / relief combo."""
    if not isinstance(graph, dict):
        return "Depth Ledger kitchen"
    bits: list[str] = []
    for n in graph.get("nodes") or []:
        if not isinstance(n, dict):
            continue
        ntype = n.get("type")
        d = dict(n.get("data") or {})
        if ntype == "look":
            preset = str(d.get("preset") or "")
            label = (LOOK_PRESETS.get(preset) or {}).get("label") or preset or "Look"
            lo = d.get("lo_pct")
            hi = d.get("hi_pct")
            gamma = d.get("gamma")
            detail = label
            if lo is not None and hi is not None:
                detail += f" {float(lo):.0f}–{float(hi):.0f}"
            if gamma is not None:
                detail += f" γ{float(gamma):.2f}"
            bits.append(detail)
        elif ntype == "scene":
            mode = str(d.get("mode") or "")
            label = SCENERY_MATTE_MODES.get(mode, mode) or "Scene"
            bits.append(f"Scene/{label}")
        elif ntype == "relief":
            alpha = d.get("gc_alpha", 10)
            side = d.get("max_side", 512)
            inv = " inv" if d.get("invert") else ""
            bits.append(f"Relief α{float(alpha):.0f}@{side}{inv}")
        elif ntype == "blend":
            mode = d.get("mode") or d.get("blend_mode") or "normal"
            opacity = d.get("opacity", 1)
            bits.append(f"Blend {mode}@{float(opacity):.2f}")
    for e in graph.get("edges") or []:
        ed = dict((e or {}).get("data") or {})
        if not ed.get("blend_mode") and ed.get("opacity") is None:
            continue
        mode = ed.get("blend_mode") or "normal"
        opacity = ed.get("opacity", 1)
        bits.append(f"{mode}@{float(opacity):.2f}")
    seen: set[str] = set()
    uniq: list[str] = []
    for b in bits:
        if b not in seen:
            seen.add(b)
            uniq.append(b)
    return " · ".join(uniq) if uniq else "Depth Ledger kitchen"


def build_recreate_payload(
    state: dict[str, Any],
    *,
    graph: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """JSON-serializable recipe sufficient to recreate the cook (not the depth map)."""
    out_dir = state.get("out_dir")
    recipe = load_recipe(out_dir) if out_dir else None
    recipe = recipe or {}
    g = graph or state.get("graph") or recipe.get("graph")
    stripped = strip_graph_for_export(g)
    arena = state.get("arena") or recipe.get("arena")
    arena_slim = None
    if isinstance(arena, dict):
        block = arena.get("block") if isinstance(arena.get("block"), dict) else {}
        arena_slim = {
            "kind": arena.get("kind"),
            "mode": arena.get("mode"),
            "input": arena.get("input"),
            "block": {
                k: block.get(k)
                for k in ("id", "title", "arena_url")
                if block.get(k) is not None
            }
            if block
            else None,
        }
    return {
        "app": "depth-ledger",
        "version": 2,
        "run_name": run_label_from_state(state),
        "model_key": state.get("model_key") or recipe.get("model_key"),
        "max_size": state.get("max_size") or recipe.get("max_size"),
        "infer_backend": state.get("infer_backend") or recipe.get("infer_backend"),
        "graph": stripped,
        "arena": arena_slim,
        "summary": summarize_kitchen(stripped or g),
    }


def build_arena_publish_fields(
    state: dict[str, Any],
    *,
    graph: Optional[dict[str, Any]] = None,
    user_title: Optional[str] = None,
    roll: Optional[tuple[int, int]] = None,
) -> dict[str, Any]:
    """Title, description, and metadata for ``publish_image_to_channel``.

    Full recipe JSON goes in the description (Are.na metadata strings are
    too short for a kitchen graph). Metadata keeps short searchable scalars.
    """
    run_label = run_label_from_state(state)
    payload = build_recreate_payload(state, graph=graph)
    summary = str(payload.get("summary") or "Depth Ledger kitchen")
    recipe_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    title_parts: list[str] = []
    ut = (user_title or "").strip()
    if ut:
        title_parts.append(ut)
    title_parts.append(run_label)
    if roll is not None:
        title_parts.append(f"roll {roll[0]}/{roll[1]}")
    title = " · ".join(title_parts)

    arena_block = None
    arena = state.get("arena")
    if isinstance(arena, dict):
        arena_block = arena.get("block")
    attribution = source_attribution_from_block(
        arena_block if isinstance(arena_block, dict) else None
    )

    desc_lines = [
        f"**Depth Ledger** `{run_label}`",
        "",
        summary,
    ]
    src = attribution.get("original_source_url")
    if src:
        desc_lines.extend(["", f"Source: {src}"])
    desc_lines.extend(
        [
            "",
            "<!-- depth-ledger-recipe",
            recipe_json,
            "-->",
            "",
            "_Recipe embedded above (HTML comment) — load via Are.na kitchen in the top bar._",
        ]
    )
    description = "\n".join(desc_lines)

    metadata: dict[str, Any] = dict(attribution.get("metadata") or {})
    model = str(payload.get("model_key") or "")
    max_size = payload.get("max_size")
    try:
        max_size_num: Any = int(max_size) if max_size is not None else ""
    except (TypeError, ValueError):
        max_size_num = _clip(str(max_size or ""))

    for k, v in list(metadata.items()):
        if isinstance(v, str):
            metadata[k] = _clip(v)

    metadata.update(
        {
            "depth_ledger_app": "depth-ledger",
            "depth_ledger_version": 2,
            "depth_ledger_run": _clip(run_label),
            "depth_ledger_model": _clip(model),
            "depth_ledger_max_size": max_size_num,
            "depth_ledger_summary": _clip(summary),
        }
    )

    return {
        "title": title,
        "description": description,
        "original_source_url": attribution.get("original_source_url"),
        "original_source_title": attribution.get("original_source_title"),
        "metadata": metadata,
        "run_label": run_label,
        "summary": summary,
        "recipe": payload,
    }
