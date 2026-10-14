"""Folder of stills → one locked Depth Pro look → named image folder.

The first image is the look reference (focal length + meter window). Later
images reuse that exact scale and grade. Output plates land in a subfolder
the caller names, using the source filenames.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import imageio
import numpy as np

from depth_core.film import (
    infer_indices,
    load_params,
    normalize_params,
    render_vis,
    save_params,
    suggest_vis_bounds,
)
from depth_core.paths import make_run_dir
from depth_core.pipeline import is_promptda, normalize_model_key
from depth_core.video import load_frame_arrays, load_frames_meta, save_frames_meta

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
BATCH_MAX_IMAGES = 200
BATCH_MAX_FOLDERS = 64
BATCH_FOLDER_WORKERS = 8
MAX_SIDE_CAP = 1536
RESERVED_FOLDERS = {"frames", "sources", "_export"}
_LOCAL_FOLDER_LOCK = threading.Lock()

ProgressCb = Optional[Any]


def _progress(cb: ProgressCb, p: float, msg: str) -> None:
    if cb:
        cb(p, msg)


def _natural_key(name: str) -> list[tuple[int, Any]]:
    key: list[tuple[int, Any]] = []
    for part in re.split(r"(\d+)", name.lower()):
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return key


def sanitize_folder_name(name: str) -> str:
    raw = (name or "").strip().replace("\\", "/").split("/")[-1]
    raw = re.sub(r"[^\w.\- ]+", "_", raw, flags=re.UNICODE).strip(" ._")
    raw = re.sub(r"\s+", " ", raw)
    if not raw:
        return "depth"
    if raw.lower() in RESERVED_FOLDERS:
        raise ValueError(f"Folder name “{raw}” is reserved. Pick another name.")
    return raw[:80]


def _sanitize_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w.\- ]+", "_", stem, flags=re.UNICODE).strip(" ._")
    stem = re.sub(r"\s+", " ", stem)
    return (stem[:80] or "image")


def _unique_png_name(name: str, used: set[str]) -> str:
    stem = _sanitize_stem(name)
    candidate = f"{stem}.png"
    n = 2
    while candidate.lower() in used:
        candidate = f"{stem}_{n}.png"
        n += 1
    used.add(candidate.lower())
    return candidate


def _load_rgb(path: Path) -> np.ndarray:
    arr = np.asarray(imageio.imread(path))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    else:
        raise ValueError(f"Unsupported image shape {arr.shape} in {path.name}")
    if arr.dtype != np.uint8:
        work = arr.astype(np.float32)
        peak = float(np.nanmax(work)) if work.size else 1.0
        if not np.isfinite(peak) or peak <= 0:
            peak = 1.0
        if peak <= 1.0:
            work = work * 255.0
        elif peak > 255.0:
            work = work * (255.0 / peak)
        arr = np.clip(work, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _resize_max_side(img: np.ndarray, max_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    cap = int(max_size)
    if m <= cap:
        return img
    scale = cap / float(m)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _frame_dir(out_dir: Path, index: int) -> Path:
    return out_dir / "frames" / f"{index:06d}"


def _source_path(out_dir: Path, index: int, suffix: str) -> Path:
    return out_dir / "sources" / f"{index:06d}{suffix}"


def _clamp_max_size(max_size: int) -> int:
    return int(np.clip(int(max_size), 64, MAX_SIDE_CAP))


def _mark_batch(out_dir: Path) -> None:
    meta = load_frames_meta(out_dir) or {}
    meta["kind"] = "batch"
    meta["pipeline"] = "batch"
    save_frames_meta(out_dir, meta)


def select_images(
    pairs: list[tuple[str, Path]],
) -> tuple[list[tuple[str, Path]], int]:
    """Sort by filename and keep image files, capped at BATCH_MAX_IMAGES."""
    kept: list[tuple[str, Path]] = []
    for name, path in pairs:
        suffix = Path(name).suffix.lower() or Path(path).suffix.lower()
        if suffix not in IMAGE_EXTS:
            continue
        if not Path(path).is_file():
            continue
        kept.append((name, Path(path)))
    kept.sort(key=lambda item: _natural_key(item[0]))
    omitted = max(0, len(kept) - BATCH_MAX_IMAGES)
    return kept[:BATCH_MAX_IMAGES], omitted


def images_in_folder(folder: Path | str) -> tuple[list[tuple[str, Path]], int]:
    """Top-level images only, so a previous save folder is not ingested again."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Not a folder: {root}")
    pairs = [
        (p.name, p)
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return select_images(pairs)


def prepare_batch(
    pairs: list[tuple[str, Path]],
    model_key: str = "depth-pro",
    max_size: int = 756,
    *,
    infer_backend: Optional[str] = None,
    folder_name: str = "depth",
    source_dir: Optional[str] = None,
    on_progress: ProgressCb = None,
) -> dict[str, Any]:
    """Copy a sorted image series into a run. No depth yet."""
    from depth_core.infer_backend import resolve_backend

    key = normalize_model_key(model_key)
    if is_promptda(key):
        raise ValueError("Batch stills use Depth Pro. Switch the model off PromptDA.")
    chosen, omitted = select_images(pairs)
    if not chosen:
        raise ValueError("No images found. Use jpg, png, webp, tif, or bmp.")

    max_sz = _clamp_max_size(max_size)
    folder = sanitize_folder_name(folder_name)
    _progress(on_progress, 0.02, f"Importing {len(chosen)} images…")

    label = f"{Path(chosen[0][0]).stem}_{uuid.uuid4().hex[:4]}"
    out_dir = make_run_dir(Path(label), key)
    (out_dir / "sources").mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    sources: list[dict[str, str]] = []
    width = height = 0
    for i, (name, src) in enumerate(chosen):
        _progress(
            on_progress,
            0.05 + 0.9 * (i / max(len(chosen), 1)),
            f"Import {i + 1}/{len(chosen)}…",
        )
        suffix = Path(name).suffix.lower() or src.suffix.lower() or ".jpg"
        if suffix not in IMAGE_EXTS:
            suffix = ".jpg"
        stored = _source_path(out_dir, i, suffix)
        shutil.copyfile(src, stored)
        rgb = _resize_max_side(_load_rgb(stored), max_sz)
        fd = _frame_dir(out_dir, i)
        fd.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(fd / "rgb.jpg", rgb)
        if i == 0:
            height, width = rgb.shape[:2]
            imageio.imwrite(out_dir / f"{out_dir.name}_rgb.jpg", rgb)
        display = Path(name).name
        names.append(display)
        sources.append({"file": str(stored.relative_to(out_dir)), "name": display})

    params = normalize_params(
        {
            "anchor_focal": True,
            "ema_enabled": False,
            "bounds_locked": False,
            "phase": "pending",
            "preview_frames": 2,
        }
    )
    save_params(out_dir, params)
    meta = {
        "kind": "batch",
        "pipeline": "batch",
        "source": Path(chosen[0][0]).name,
        "source_file": sources[0]["file"],
        "names": names,
        "sources": sources,
        "frame_count": len(names),
        "ready_count": 0,
        "ready_indices": [],
        "infer_status": "pending",
        "sample_fps": 1.0,
        "max_size": max_sz,
        "model_key": key,
        "infer_backend": resolve_backend(infer_backend),
        "frame_width": int(width),
        "frame_height": int(height),
        "export_folder": folder,
        "export_dir": None,
        "source_dir": str(Path(source_dir).expanduser().resolve()) if source_dir else None,
        "saved_count": 0,
        "omitted": omitted,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    save_frames_meta(out_dir, meta)
    return {"out_dir": str(out_dir.resolve()), "omitted": omitted, "frame_count": len(names)}


def _resize_all(
    out_dir: Path,
    max_size: int,
    on_progress: ProgressCb = None,
) -> tuple[int, int, int]:
    meta = load_frames_meta(out_dir) or {}
    sources = list(meta.get("sources") or [])
    n = int(meta.get("frame_count") or len(sources) or 0)
    if n < 1 or len(sources) < n:
        raise ValueError("Batch sources are missing; import the folder again.")
    width = height = 0
    for i in range(n):
        _progress(on_progress, 0.02 + 0.2 * (i / max(n, 1)), f"Resizing image {i + 1}/{n}…")
        rel = str(sources[i].get("file") or "")
        src = out_dir / rel
        if not src.is_file():
            raise FileNotFoundError(f"Missing source for image {i + 1}: {rel}")
        rgb = _resize_max_side(_load_rgb(src), int(max_size))
        fd = _frame_dir(out_dir, i)
        fd.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(fd / "rgb.jpg", rgb)
        (fd / "depth.npz").unlink(missing_ok=True)
        (fd / "vis.png").unlink(missing_ok=True)
        if i == 0:
            height, width = rgb.shape[:2]
            imageio.imwrite(out_dir / f"{out_dir.name}_rgb.jpg", rgb)
    meta["max_size"] = int(max_size)
    meta["frame_width"] = int(width)
    meta["frame_height"] = int(height)
    meta["ready_count"] = 0
    meta["ready_indices"] = []
    meta["infer_status"] = "pending"
    meta["kind"] = "batch"
    meta["pipeline"] = "batch"
    save_frames_meta(out_dir, meta)
    return n, width, height


def _apply_max_size(
    out_dir: Path,
    params: dict[str, Any],
    max_sz: int,
    on_progress: ProgressCb = None,
) -> tuple[dict[str, Any], bool]:
    meta = load_frames_meta(out_dir) or {}
    prepared = int(meta.get("max_size") or max_sz)
    max_sz = _clamp_max_size(max_sz)
    if prepared == max_sz:
        return params, False
    old_w = meta.get("frame_width")
    _progress(on_progress, 0.02, f"Max side {prepared}→{max_sz}; resizing…")
    _n, fw, _fh = _resize_all(out_dir, max_sz, on_progress)
    if params.get("focallength_override") and old_w and fw:
        scale = float(fw) / float(old_w)
        if np.isfinite(scale) and scale > 0:
            params["focallength_override"] = float(params["focallength_override"]) * scale
    else:
        params["focallength_px"] = None
        params["inferred_with_f_px"] = None
    return params, True


def _focal_force(params: dict[str, Any], size_changed: bool) -> bool:
    inferred = params.get("inferred_with_f_px")
    want_f = params.get("focallength_override") or (
        params.get("focallength_px") if params.get("anchor_focal") else None
    )
    if size_changed:
        return True
    if inferred is not None and want_f is not None and abs(float(inferred) - float(want_f)) > 0.5:
        return True
    return False


def run_preview(
    out_dir: Path | str,
    params: Optional[dict[str, Any]] = None,
    *,
    model_key: Optional[str] = None,
    max_size: Optional[int] = None,
    infer_backend: Optional[str] = None,
    on_progress: ProgressCb = None,
    on_partial: Optional[Any] = None,
) -> dict[str, Any]:
    """Depth-map image 0 only, then grade it with the current meter window."""
    from depth_core.film import _update_ready_meta

    out_dir = Path(out_dir)
    meta = load_frames_meta(out_dir)
    if not meta or meta.get("kind") != "batch":
        raise ValueError(f"No batch under {out_dir}")
    params = normalize_params({**load_params(out_dir), **(params or {}), "ema_enabled": False})
    key = str(model_key or meta.get("model_key") or "depth-pro")
    if is_promptda(normalize_model_key(key)):
        raise ValueError("Batch stills use Depth Pro.")
    max_sz = _clamp_max_size(int(max_size or meta.get("max_size") or 756))
    params, size_changed = _apply_max_size(out_dir, params, max_sz, on_progress)
    force = _focal_force(params, size_changed)
    params["phase"] = "running"
    params["ema_enabled"] = False
    save_params(out_dir, params)
    _update_ready_meta(out_dir, [], "running", infer_backend=infer_backend, max_size=max_sz)
    _mark_batch(out_dir)

    locked, ready, billing = infer_indices(
        out_dir,
        [0],
        params=params,
        model_key=key,
        max_size=max_sz,
        infer_backend=infer_backend,
        force=force,
        on_progress=on_progress,
        on_partial=on_partial,
    )
    if ready:
        depths = [load_frame_arrays(out_dir, i)[1] for i in ready]
        smin, smax = suggest_vis_bounds(depths)
        params["suggested_vis_min_m"] = smin
        params["suggested_vis_max_m"] = smax
        if not params.get("bounds_locked"):
            params["vis_min_m"] = smin
            params["vis_max_m"] = smax
    if locked is not None:
        params["focallength_px"] = float(locked)
        params["inferred_with_f_px"] = float(locked)
    params["preview_ready"] = len(ready)
    params["phase"] = "preview"
    params["ema_enabled"] = False
    save_params(out_dir, params)
    _update_ready_meta(out_dir, ready, "preview", infer_backend=infer_backend, max_size=max_sz)
    _mark_batch(out_dir)
    _progress(on_progress, 0.92, "Rendering first image…")
    if ready:
        render_vis(out_dir, params, ready)
    _progress(on_progress, 1.0, "Preview ready")
    if billing:
        params["modal_billing"] = billing
    return params


def run_batch(
    out_dir: Path | str,
    params: Optional[dict[str, Any]] = None,
    *,
    model_key: Optional[str] = None,
    max_size: Optional[int] = None,
    infer_backend: Optional[str] = None,
    folder_name: Optional[str] = None,
    on_progress: ProgressCb = None,
    on_partial: Optional[Any] = None,
) -> dict[str, Any]:
    """Copy the locked look across every image and write the named folder."""
    from depth_core.film import _update_ready_meta

    out_dir = Path(out_dir)
    meta = load_frames_meta(out_dir)
    if not meta or meta.get("kind") != "batch":
        raise ValueError(f"No batch under {out_dir}")
    params = normalize_params({**load_params(out_dir), **(params or {}), "ema_enabled": False})
    params["bounds_locked"] = True
    params["ema_enabled"] = False
    n = int(meta.get("frame_count") or 0)
    if n < 1:
        raise ValueError("Batch is empty")
    key = str(model_key or meta.get("model_key") or "depth-pro")
    if is_promptda(normalize_model_key(key)):
        raise ValueError("Batch stills use Depth Pro.")
    max_sz = _clamp_max_size(int(max_size or meta.get("max_size") or 756))
    params, size_changed = _apply_max_size(out_dir, params, max_sz, on_progress)
    meta = load_frames_meta(out_dir) or meta
    n = int(meta.get("frame_count") or n)
    force = _focal_force(params, size_changed)
    if folder_name:
        meta["export_folder"] = sanitize_folder_name(folder_name)
        save_frames_meta(out_dir, meta)

    params["phase"] = "running"
    save_params(out_dir, params)
    existing = [i for i in range(n) if (_frame_dir(out_dir, i) / "depth.npz").is_file()]
    _update_ready_meta(
        out_dir, existing, "running", infer_backend=infer_backend, max_size=max_sz
    )
    _mark_batch(out_dir)

    locked, ready, billing = infer_indices(
        out_dir,
        list(range(n)),
        params=params,
        model_key=key,
        max_size=max_sz,
        infer_backend=infer_backend,
        force=force,
        on_progress=on_progress,
        on_partial=on_partial,
    )
    if locked is not None:
        params["focallength_px"] = float(locked)
        params["inferred_with_f_px"] = float(locked)
    if ready:
        depths = [load_frame_arrays(out_dir, i)[1] for i in ready]
        smin, smax = suggest_vis_bounds(depths)
        params["suggested_vis_min_m"] = smin
        params["suggested_vis_max_m"] = smax
    params["preview_ready"] = len(ready)
    params["phase"] = "done"
    params["ema_enabled"] = False
    save_params(out_dir, params)
    _update_ready_meta(out_dir, ready, "done", infer_backend=infer_backend, max_size=max_sz)
    _mark_batch(out_dir)
    _progress(on_progress, 0.9, "Rendering locked look…")
    render_vis(out_dir, params, ready)
    _progress(on_progress, 0.96, "Saving folder…")
    saved = export_folder(out_dir, folder_name)
    _progress(on_progress, 1.0, f"Saved {saved['saved_count']} images")
    if billing:
        params["modal_billing"] = billing
    params["export_dir"] = saved["export_dir"]
    params["saved_count"] = saved["saved_count"]
    return params


def _export_dir_for(out_dir: Path, meta: dict[str, Any], folder: str) -> Path:
    """New folder inside the source directory, or inside the run if it was uploaded."""
    source = meta.get("source_dir")
    if source:
        root = Path(str(source)).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Source folder is missing: {root}")
        export_dir = (root / folder).resolve()
        if export_dir.parent != root:
            raise ValueError("Invalid folder name")
        return export_dir
    export_dir = (out_dir / folder).resolve()
    if not str(export_dir).startswith(str(out_dir)):
        raise ValueError("Invalid folder name")
    return export_dir


def export_folder(out_dir: Path | str, folder_name: Optional[str] = None) -> dict[str, Any]:
    """Write vis plates into <source>/<folder>, named after the source files."""
    out_dir = Path(out_dir).resolve()
    meta = load_frames_meta(out_dir) or {}
    n = int(meta.get("frame_count") or 0)
    names = list(meta.get("names") or [])
    if n < 1:
        raise ValueError("Nothing to save")
    folder = sanitize_folder_name(folder_name or str(meta.get("export_folder") or "depth"))
    export_dir = _export_dir_for(out_dir, meta, folder)
    export_dir.mkdir(parents=True, exist_ok=True)

    used: set[str] = set()
    saved: list[str] = []
    params = load_params(out_dir)
    for i in range(n):
        src = _frame_dir(out_dir, i) / "vis.png"
        if not src.is_file():
            raise ValueError(
                f"Image {i + 1} has no depth plate yet. Preview, then run the batch."
            )
        label = names[i] if i < len(names) else f"{i:06d}.png"
        dest_name = _unique_png_name(label, used)
        shutil.copyfile(src, export_dir / dest_name)
        saved.append(dest_name)

    settings = {
        "model_key": meta.get("model_key"),
        "max_size": meta.get("max_size"),
        "vis_min_m": params.get("vis_min_m"),
        "vis_max_m": params.get("vis_max_m"),
        "gamma": params.get("gamma"),
        "cmap": params.get("cmap"),
        "focallength_px": params.get("focallength_px"),
        "anchor_focal": params.get("anchor_focal"),
        "count": len(saved),
        "files": saved,
    }
    (export_dir / "settings.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    meta["export_folder"] = folder
    meta["export_dir"] = str(export_dir)
    meta["saved_count"] = len(saved)
    meta["saved_names"] = saved
    meta["kind"] = "batch"
    meta["pipeline"] = "batch"
    save_frames_meta(out_dir, meta)
    return {
        "export_dir": str(export_dir),
        "export_folder": folder,
        "saved_count": len(saved),
        "saved_names": saved,
    }


def batch_info(out_dir: Path | str) -> dict[str, Any]:
    out_dir = Path(out_dir)
    meta = load_frames_meta(out_dir) or {}
    params = load_params(out_dir)
    n = int(meta.get("frame_count") or 0)
    ready = [i for i in range(n) if (_frame_dir(out_dir, i) / "depth.npz").is_file()]
    vis_ready = [i for i in range(n) if (_frame_dir(out_dir, i) / "vis.png").is_file()]
    return {
        "pipeline": "batch",
        "kind": "batch",
        "run_id": out_dir.name,
        "frame_count": n,
        "ready_count": len(ready),
        "ready_indices": ready,
        "vis_ready": vis_ready,
        "infer_status": meta.get("infer_status") or params.get("phase") or "pending",
        "names": list(meta.get("names") or []),
        "max_size": meta.get("max_size"),
        "frame_width": meta.get("frame_width"),
        "frame_height": meta.get("frame_height"),
        "model_key": meta.get("model_key"),
        "infer_mode": meta.get("infer_mode"),
        "infer_backend": meta.get("infer_backend"),
        "export_folder": meta.get("export_folder") or "depth",
        "export_dir": meta.get("export_dir"),
        "source_dir": meta.get("source_dir"),
        "saved_count": int(meta.get("saved_count") or 0),
        "saved_names": list(meta.get("saved_names") or []),
        "omitted": int(meta.get("omitted") or 0),
        "params": params,
    }


def folder_summary(folder: Path | str) -> dict[str, Any]:
    root = Path(folder).expanduser().resolve()
    chosen, omitted = images_in_folder(root)
    if not chosen:
        raise ValueError(
            f"No images in {root.name} or its subfolders. Use jpg, png, webp, tif, or bmp."
        )
    return {
        "path": str(root),
        "name": root.name,
        "count": len(chosen),
        "omitted": omitted,
        "names": [name for name, _path in chosen[:8]],
    }


def discover_batches(folder: Path | str, *, max_depth: int = 4) -> tuple[list[dict[str, Any]], int]:
    """One batch per folder that actually holds images.

    A parent like Collated Clips, whose pictures live in child folders, becomes
    one batch per child. A folder that already has images is a single batch, so
    a previous save folder inside it is not queued again.
    """
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Not a folder: {root}")
    found: list[Path] = []

    def walk(path: Path, depth: int) -> None:
        chosen, _omitted = images_in_folder(path)
        if chosen:
            found.append(path)
            return
        if depth >= max_depth:
            return
        children = [
            p
            for p in path.iterdir()
            if p.is_dir() and not p.name.startswith(".") and p.name.lower() not in RESERVED_FOLDERS
        ]
        children.sort(key=lambda p: _natural_key(p.name))
        for child in children:
            walk(child, depth + 1)

    walk(root, 0)
    if not found:
        raise ValueError(
            f"No images in {root.name} or its subfolders. Use jpg, png, webp, tif, or bmp."
        )
    omitted_folders = max(0, len(found) - BATCH_MAX_FOLDERS)
    summaries = [folder_summary(path) for path in found[:BATCH_MAX_FOLDERS]]
    return summaries, omitted_folders


def run_folders(
    folders: list[str],
    params: Optional[dict[str, Any]] = None,
    *,
    model_key: str = "depth-pro",
    max_size: int = 756,
    infer_backend: Optional[str] = None,
    folder_name: str = "depth",
    on_update: Optional[Any] = None,
    should_cancel: Optional[Any] = None,
) -> dict[str, Any]:
    """Depth-map every folder with one look. Modal runs the folders together."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from depth_core.infer_backend import resolve_backend

    if not folders:
        raise ValueError("Add at least one folder")

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for raw in folders:
        summaries, _omitted = discover_batches(raw)
        for info in summaries:
            if info["path"] in seen:
                continue
            seen.add(info["path"])
            unique.append(info)
            if len(unique) >= BATCH_MAX_FOLDERS:
                break
        if len(unique) >= BATCH_MAX_FOLDERS:
            break

    backend = resolve_backend(infer_backend)
    look = dict(params or {})
    look["ema_enabled"] = False
    look["bounds_locked"] = True
    if not look.get("focallength_override"):
        look["focallength_px"] = None
        look["inferred_with_f_px"] = None

    items: list[dict[str, Any]] = [
        {
            "path": info["path"],
            "name": info["name"],
            "count": info["count"],
            "omitted": info["omitted"],
            "status": "queued",
            "progress": 0.0,
            "saved_count": 0,
            "export_dir": None,
            "error": None,
        }
        for info in unique
    ]
    lock = threading.Lock()

    def publish(msg: str) -> None:
        with lock:
            snap = [dict(it) for it in items]
            frac = sum(float(it["progress"]) for it in items) / max(len(items), 1)
        if on_update:
            on_update(frac, msg, snap)

    def one(index: int) -> None:
        item = items[index]
        if should_cancel and should_cancel():
            item["status"] = "cancelled"
            item["progress"] = 1.0
            publish("Cancelled")
            return
        item["status"] = "running"
        item["progress"] = 0.02
        publish(f"{item['name']} · importing…")
        try:
            chosen, _omitted = images_in_folder(item["path"])

            def on_p(p: float, msg: str) -> None:
                if should_cancel and should_cancel():
                    raise RuntimeError("cancelled")
                item["progress"] = 0.05 + 0.9 * float(p)
                publish(f"{item['name']} · {msg}")

            state = prepare_batch(
                chosen,
                model_key,
                max_size,
                infer_backend=infer_backend,
                folder_name=folder_name,
                source_dir=item["path"],
                on_progress=on_p,
            )
            gate = _LOCAL_FOLDER_LOCK if backend != "modal" else nullcontext()
            with gate:
                if should_cancel and should_cancel():
                    raise RuntimeError("cancelled")
                result = run_batch(
                    state["out_dir"],
                    dict(look),
                    model_key=model_key,
                    max_size=max_size,
                    infer_backend=infer_backend,
                    folder_name=folder_name,
                    on_progress=on_p,
                )
            item["status"] = "done"
            item["progress"] = 1.0
            item["saved_count"] = int(result.get("saved_count") or 0)
            item["export_dir"] = result.get("export_dir")
            publish(f"Saved {item['name']}")
        except Exception as exc:
            item["status"] = "cancelled" if str(exc) == "cancelled" else "error"
            item["progress"] = 1.0
            item["error"] = str(exc)
            publish(f"{item['name']} · {exc}")

    workers = len(unique) if backend == "modal" else 1
    workers = max(1, min(workers, BATCH_FOLDER_WORKERS))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, i) for i in range(len(unique))]
        for fut in as_completed(futures):
            fut.result()

    saved = sum(int(it["saved_count"] or 0) for it in items)
    return {
        "folders": [dict(it) for it in items],
        "saved_count": saved,
        "folder_count": len(unique),
        "error_count": sum(1 for it in items if it["status"] == "error"),
        "cancelled": any(it["status"] == "cancelled" for it in items),
        "infer_backend": backend,
    }
