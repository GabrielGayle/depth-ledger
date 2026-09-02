"""Working cache: content-addressed infer results + disk budget for outputs/.

Are.na is the long-term store for published plates + recipes. Local disk only
keeps a bounded working set so rolls can skip GPU on repeat blocks and the
kitchen can re-cook without re-downloading the whole archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from depth_core.paths import OUTPUT_DIR


def _model_slug(model_key: str) -> str:
    key = (model_key or "depth-pro").strip().lower().replace("_", "-")
    return key or "depth-pro"

INFER_DIR = OUTPUT_DIR / "_infer"
INDEX_NAME = "index.json"
RGB_NAME = "rgb.jpg"
DEPTH_NAME = "depth.npz"
JPEG_QUALITY = 88

# Derived files that are cheap to recreate from depth + recipe.
_COMPACT_DELETE_GLOBS = (
    "*_look_*.png",
    "look_*.png",
    "*_blend.png",
    "*_depth_mm.png",
    "*pointcloud*",
    "operator_*",
    "tune_turbo.png",
    "*_near_far.json",
    "*_relief.png",
    "*_matte.exr",
    "*_depth.exr",
    "*_inv_smooth.exr",
)

_PROTECTED_DIR_PREFIX = "_"
_PROTECTED_FILE_PREFIXES = ("_default",)

_LOCK = threading.Lock()
_STATS = {
    "hits": 0,
    "misses": 0,
    "bytes_saved": 0,
}
_DISK_STATS: dict[str, Any] = {"at": 0.0, "data": None}
_DISK_STATS_TTL_S = 20.0


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def ram_run_limit() -> int:
    """How many runs keep depth/rgb arrays in process memory."""
    return max(1, _env_int("DEPTH_LEDGER_RAM_RUNS", 2))


def disk_budget_bytes() -> int:
    """Cap for outputs/ (run dirs + infer cache), in bytes."""
    gb = max(0.25, _env_float("DEPTH_LEDGER_DISK_GB", 3.0))
    return int(gb * (1024**3))


def infer_cache_budget_bytes() -> int:
    gb = max(0.1, _env_float("DEPTH_LEDGER_INFER_CACHE_GB", 1.5))
    return int(gb * (1024**3))


def sha256_file(path: Path, limit: int = 16) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:limit]


def infer_cache_key(
    model_key: str,
    max_size: int,
    *,
    arena_block_id: Optional[int | str] = None,
    image_path: Optional[Path | str] = None,
    image_sha: Optional[str] = None,
) -> str:
    model = _model_slug(model_key)
    size = int(max_size)
    if arena_block_id is not None and str(arena_block_id).strip():
        return f"a{str(arena_block_id).strip()}_{model}_{size}"
    digest = (image_sha or "").strip()
    if not digest and image_path:
        digest = sha256_file(Path(image_path))
    if not digest:
        raise ValueError("infer cache key needs arena_block_id or image bytes")
    return f"s{digest}_{model}_{size}"


def _index_path() -> Path:
    return INFER_DIR / INDEX_NAME


def _read_index() -> dict[str, Any]:
    path = _index_path()
    if not path.is_file():
        return {"entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"entries": {}}
    if not isinstance(data, dict):
        return {"entries": {}}
    entries = data.get("entries")
    if not isinstance(entries, dict):
        data["entries"] = {}
    return data


def _write_index(data: dict[str, Any]) -> None:
    INFER_DIR.mkdir(parents=True, exist_ok=True)
    path = _index_path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _entry_dir(key: str) -> Path:
    return INFER_DIR / key


def _dir_nbytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                total += fp.stat().st_size
            except OSError:
                continue
    return total


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        try:
            dest.unlink()
        except OSError:
            pass
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _write_jpeg(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        im = Image.fromarray(arr, mode="L")
    else:
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        im = Image.fromarray(arr, mode="RGB")
    im.save(path, format="JPEG", quality=JPEG_QUALITY, optimize=True)


def _write_depth(path: Path, depth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, depth=np.asarray(depth, dtype=np.float32))


def load_infer_cache(key: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return (rgb, depth) if cached, else None."""
    d = _entry_dir(key)
    rgb_path = d / RGB_NAME
    depth_path = d / DEPTH_NAME
    if not rgb_path.is_file() or not depth_path.is_file():
        with _LOCK:
            _STATS["misses"] += 1
        return None
    try:
        from PIL import Image

        with Image.open(rgb_path) as im:
            rgb = np.asarray(im.convert("RGB"))
        depth = np.load(depth_path)["depth"].astype(np.float32)
    except Exception:
        with _LOCK:
            _STATS["misses"] += 1
        return None
    with _LOCK:
        _STATS["hits"] += 1
        idx = _read_index()
        ent = (idx.get("entries") or {}).get(key)
        if isinstance(ent, dict):
            ent["hits"] = int(ent.get("hits") or 0) + 1
            ent["atime"] = datetime.now(timezone.utc).isoformat()
            _write_index(idx)
    return rgb, depth


def save_infer_cache(
    key: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    meta: Optional[dict[str, Any]] = None,
) -> Path:
    """Write (or refresh) an infer-cache entry. Returns the entry directory."""
    d = _entry_dir(key)
    d.mkdir(parents=True, exist_ok=True)
    rgb_path = d / RGB_NAME
    depth_path = d / DEPTH_NAME
    if not rgb_path.is_file():
        _write_jpeg(rgb_path, rgb)
    if not depth_path.is_file():
        _write_depth(depth_path, depth)
    nbytes = _dir_nbytes(d)
    payload = {
        "key": key,
        "bytes": nbytes,
        "mtime": datetime.now(timezone.utc).isoformat(),
        "atime": datetime.now(timezone.utc).isoformat(),
        "hits": 0,
    }
    if meta:
        payload.update({k: v for k, v in meta.items() if v is not None})
    with _LOCK:
        idx = _read_index()
        entries = idx.setdefault("entries", {})
        prev = entries.get(key) if isinstance(entries.get(key), dict) else {}
        if prev.get("hits"):
            payload["hits"] = int(prev["hits"])
        entries[key] = payload
        _write_index(idx)
    return d


def materialize_run_arrays(
    out_dir: Path,
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    cache_key: Optional[str] = None,
    cache_meta: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    """Write slim rgb.jpg + depth.npz into a run dir (hardlink from cache when possible)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = out_dir / RGB_NAME
    depth_path = out_dir / DEPTH_NAME
    if cache_key:
        cdir = save_infer_cache(cache_key, rgb, depth, meta=cache_meta)
        _link_or_copy(cdir / RGB_NAME, rgb_path)
        _link_or_copy(cdir / DEPTH_NAME, depth_path)
    else:
        _write_jpeg(rgb_path, rgb)
        _write_depth(depth_path, depth)

    run_id = out_dir.name
    alias_rgb = out_dir / f"{run_id}_rgb.jpg"
    alias_depth = out_dir / f"{run_id}_depth.npz"
    if not alias_rgb.exists():
        _link_or_copy(rgb_path, alias_rgb)
    if not alias_depth.exists():
        _link_or_copy(depth_path, alias_depth)
    return {
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "metric_path": "",
    }


def find_rgb_path(out_dir: Path | str, run_id: Optional[str] = None) -> Optional[Path]:
    out = Path(out_dir)
    rid = run_id or out.name
    for cand in (
        out / RGB_NAME,
        out / f"{rid}_rgb.jpg",
    ):
        if cand.is_file():
            return cand
    jpgs = sorted(out.glob("*_rgb.jpg"))
    return jpgs[0] if jpgs else None


def find_depth_path(out_dir: Path | str, run_id: Optional[str] = None) -> Optional[Path]:
    out = Path(out_dir)
    rid = run_id or out.name
    for cand in (
        out / DEPTH_NAME,
        out / f"{rid}_depth.npz",
    ):
        if cand.is_file():
            return cand
    npzs = sorted(out.glob("*_depth.npz"))
    return npzs[0] if npzs else None


def load_run_arrays(
    out_dir: Path | str, run_id: Optional[str] = None
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    out = Path(out_dir)
    rgb = depth = None
    rgb_path = find_rgb_path(out, run_id)
    depth_path = find_depth_path(out, run_id)
    if rgb_path and rgb_path.is_file():
        try:
            from PIL import Image

            with Image.open(rgb_path) as im:
                rgb = np.asarray(im.convert("RGB"))
        except Exception:
            rgb = None
    if depth_path and depth_path.is_file():
        try:
            depth = np.load(depth_path)["depth"].astype(np.float32)
        except Exception:
            depth = None
    return rgb, depth


def compact_run(out_dir: Path | str) -> int:
    """Delete derived look/blend/PLY files. Returns bytes freed (best-effort)."""
    out = Path(out_dir)
    if not out.is_dir():
        return 0
    freed = 0
    seen: set[Path] = set()
    for pattern in _COMPACT_DELETE_GLOBS:
        for path in out.glob(pattern):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            try:
                sz = path.stat().st_size
                path.unlink()
                freed += sz
            except OSError:
                continue
    return freed


def is_video_run(out_dir: Path) -> bool:
    return (out_dir / "frames_meta.json").is_file() or (out_dir / "frames").is_dir()


def list_run_dirs() -> list[Path]:
    if not OUTPUT_DIR.is_dir():
        return []
    return [
        p
        for p in OUTPUT_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(_PROTECTED_DIR_PREFIX)
    ]


def outputs_nbytes() -> int:
    return _dir_nbytes(OUTPUT_DIR)


def infer_cache_nbytes() -> int:
    return _dir_nbytes(INFER_DIR)


def rss_mb() -> Optional[float]:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        pass
    try:
        import resource

        v = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return round(v / (1024 * 1024), 1)
        return round(v / 1024.0, 1)
    except Exception:
        return None


def cache_stats(*, refresh: bool = False) -> dict[str, Any]:
    now = time.time()
    with _LOCK:
        hits = int(_STATS["hits"])
        misses = int(_STATS["misses"])
        cached = _DISK_STATS["data"]
        cached_at = float(_DISK_STATS["at"] or 0)
    if (
        not refresh
        and cached is not None
        and (now - cached_at) < _DISK_STATS_TTL_S
    ):
        out = dict(cached)
        out["hits"] = hits
        out["misses"] = misses
        out["rss_mb"] = rss_mb()
        return out

    idx = _read_index()
    entries = idx.get("entries") or {}
    disk = {
        "infer_entries": len(entries),
        "infer_cache_mb": round(infer_cache_nbytes() / (1024 * 1024), 1),
        "outputs_mb": round(outputs_nbytes() / (1024 * 1024), 1),
        "disk_budget_mb": round(disk_budget_bytes() / (1024 * 1024), 1),
        "ram_run_limit": ram_run_limit(),
        "run_dirs": len(list_run_dirs()),
    }
    with _LOCK:
        _DISK_STATS["at"] = now
        _DISK_STATS["data"] = disk
    return {
        "hits": hits,
        "misses": misses,
        "rss_mb": rss_mb(),
        **disk,
    }


def _mtime(path: Path) -> float:
    recipe = path / "recipe.json"
    try:
        if recipe.is_file():
            return recipe.stat().st_mtime
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _prune_infer_cache(budget: int, protect_keys: set[str]) -> int:
    """Drop oldest infer-cache entries until under budget. Returns bytes freed."""
    with _LOCK:
        idx = _read_index()
        entries = dict(idx.get("entries") or {})
    items: list[tuple[float, str, Path, int]] = []
    for key, meta in entries.items():
        if key in protect_keys:
            continue
        d = _entry_dir(key)
        nbytes = int((meta or {}).get("bytes") or _dir_nbytes(d))
        at = 0.0
        stamp = (meta or {}).get("atime") or (meta or {}).get("mtime")
        if stamp:
            try:
                at = datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).timestamp()
            except ValueError:
                at = _mtime(d)
        else:
            at = _mtime(d)
        items.append((at, key, d, nbytes))
    items.sort(key=lambda t: t[0])
    used = infer_cache_nbytes()
    freed = 0
    dropped: list[str] = []
    for _at, key, d, nbytes in items:
        if used - freed <= budget:
            break
        if d.is_dir():
            try:
                shutil.rmtree(d)
            except OSError:
                continue
        freed += nbytes
        dropped.append(key)
    if dropped:
        with _LOCK:
            idx = _read_index()
            ents = idx.setdefault("entries", {})
            for key in dropped:
                ents.pop(key, None)
            _write_index(idx)
    return freed


def enforce_disk_budget(protect_run_ids: Optional[set[str]] = None) -> dict[str, Any]:
    """Compact leftovers, then delete oldest still-run dirs, then infer cache.

    Video runs are evicted last. Protected run ids (UI current / in-flight) stay.
    """
    protect = {str(x) for x in (protect_run_ids or set()) if x}
    compact_freed = 0
    for run_dir in list_run_dirs():
        compact_freed += compact_run(run_dir)

    budget = disk_budget_bytes()
    deleted_runs: list[str] = []
    used = outputs_nbytes()

    def _candidates(*, videos: bool) -> list[Path]:
        dirs = [
            p
            for p in list_run_dirs()
            if p.name not in protect and is_video_run(p) == videos
        ]
        dirs.sort(key=_mtime)
        return dirs

    while used > budget:
        pool = _candidates(videos=False) or _candidates(videos=True)
        if not pool:
            break
        victim = pool[0]
        sz = _dir_nbytes(victim)
        try:
            shutil.rmtree(victim)
        except OSError:
            break
        deleted_runs.append(victim.name)
        used = max(0, used - sz)

    infer_budget = min(infer_cache_budget_bytes(), max(64 * 1024 * 1024, budget // 2))
    infer_freed = _prune_infer_cache(infer_budget, protect_keys=set())
    return {
        "compacted_bytes": compact_freed,
        "deleted_runs": deleted_runs,
        "infer_freed_bytes": infer_freed,
        "outputs_mb": round(outputs_nbytes() / (1024 * 1024), 1),
        "budget_mb": round(budget / (1024 * 1024), 1),
    }


def release_torch_memory() -> None:
    """Drop unused CUDA/MPS caches after a forward pass."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
