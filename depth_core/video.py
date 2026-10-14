"""RGB video → sampled frames → Depth Pro → multi-frame run layout."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import imageio
import numpy as np

from depth_core.pipeline import (
    DEFAULT_LOOK_A,
    DEFAULT_LOOK_B,
    DEFAULT_LOOK_C,
    DEFAULT_BLEND_MODE,
    DEFAULT_SCENERY_CLEAN,
    DEFAULT_SCENERY_FEATHER,
    DEFAULT_SCENERY_MATTE,
    DEFAULT_SCENERY_PROTECT,
    _infer_depth_pro,
    _make_run_dir,
    _subject_hi_pct,
    get_backend,
    is_promptda,
    normalize_model_key,
)

VIDEO_TARGET_FPS = 10.0
VIDEO_MAX_FRAMES = 60
VIDEO_MAX_DURATION_S = 30.0
VIDEO_PREVIEW_FRAMES = 6
VIDEO_PREVIEW_MAX_EDGE = 320

ProgressCb = Optional[Callable[[float, str], None]]
# ready_count, total_frames, partial_state|None, ready_indices|None
PartialCb = Optional[
    Callable[[int, int, Optional[dict[str, Any]], Optional[list[int]]], None]
]


def pick_preview_indices(n: int, count: int = VIDEO_PREVIEW_FRAMES) -> list[int]:
    """Evenly spaced indices into a clip of length n (always includes 0 and last when n>1)."""
    if n <= 0:
        return []
    k = max(1, min(int(count), n))
    if k == 1:
        return [0]
    if k >= n:
        return list(range(n))
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def write_preview_gif(
    out_dir: Path,
    frame_indices: list[int],
    *,
    max_edge: int = VIDEO_PREVIEW_MAX_EDGE,
    duration_ms: int = 120,
) -> Path:
    """Build a small looping GIF from existing frames/*/rgb.jpg."""
    from PIL import Image

    out_dir = Path(out_dir)
    frames: list[Image.Image] = []
    for i in frame_indices:
        path = _frame_dir(out_dir, i) / "rgb.jpg"
        if not path.is_file():
            continue
        img = Image.open(path).convert("RGB")
        w, h = img.size
        m = max(w, h)
        if m > max_edge:
            scale = max_edge / float(m)
            img = img.resize(
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                Image.Resampling.LANCZOS,
            )
        frames.append(img)
    if not frames:
        raise FileNotFoundError("No RGB frames available for preview GIF")
    gif_path = out_dir / "preview.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return gif_path


def _aggregate_modal_billing(parts: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not parts:
        return None
    gpu_type = str(parts[0].get("gpu_type") or "T4")
    gpu_seconds = sum(float(p.get("gpu_seconds") or 0) for p in parts)
    cost = sum(float(p.get("cost_usd") or 0) for p in parts)
    cold = any(bool(p.get("cold_start")) for p in parts)
    return {
        "gpu_type": gpu_type,
        "gpu_seconds": round(gpu_seconds, 4),
        "cost_usd": round(cost, 6),
        "cold_start": cold,
        "calls": len(parts),
    }


def _resize_max_side(img: np.ndarray, max_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_size:
        return img
    scale = max_size / float(m)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _frame_dir(out_dir: Path, index: int) -> Path:
    return out_dir / "frames" / f"{index:06d}"


def frames_meta_path(out_dir: Path | str) -> Path:
    return Path(out_dir) / "frames_meta.json"


def load_frames_meta(out_dir: Path | str) -> Optional[dict[str, Any]]:
    path = frames_meta_path(out_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_frames_meta(out_dir: Path | str, meta: dict[str, Any]) -> Path:
    path = frames_meta_path(out_dir)
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path


def load_frame_arrays(out_dir: Path | str, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb uint8, depth float32) for frame index."""
    fd = _frame_dir(Path(out_dir), index)
    rgb_path = fd / "rgb.jpg"
    depth_path = fd / "depth.npz"
    if not rgb_path.is_file() or not depth_path.is_file():
        raise FileNotFoundError(f"Frame {index} missing under {fd}")
    rgb = np.asarray(imageio.imread(rgb_path))
    depth = np.load(depth_path)["depth"].astype(np.float32)
    return rgb, depth


def compute_stride(source_fps: float, target_fps: float = VIDEO_TARGET_FPS) -> int:
    if source_fps <= 0:
        return 1
    return max(1, int(round(source_fps / target_fps)))


def extract_video_frame_indices(
    video_path: Path,
    *,
    target_fps: float = VIDEO_TARGET_FPS,
    max_frames: int = VIDEO_MAX_FRAMES,
    max_duration_s: float = VIDEO_MAX_DURATION_S,
) -> tuple[list[int], float, int, float]:
    """Pick source frame indices to sample.

    Returns (indices, source_fps, stride, duration_s).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 1e-3:
            source_fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_s = frame_count / source_fps if source_fps > 0 else 0.0
        if duration_s > max_duration_s + 0.5:
            # Truncate to first max_duration_s of source
            frame_count = min(frame_count, int(max_duration_s * source_fps))
            duration_s = frame_count / source_fps
        stride = compute_stride(source_fps, target_fps)
        indices = list(range(0, max(frame_count, 1), stride))
        if not indices and frame_count > 0:
            indices = [0]
        if len(indices) > max_frames:
            indices = indices[:max_frames]
        if not indices:
            raise ValueError("Video has no readable frames")
        return indices, source_fps, stride, duration_s
    finally:
        cap.release()


def _read_frame_bgr(cap: cv2.VideoCapture, index: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def _write_frames_meta(
    out_dir: Path,
    *,
    video_path: Path,
    source_fps: float,
    stride: int,
    sample_fps: float,
    n: int,
    ready_count: int,
    indices: list[int],
    duration_s: float,
    max_size: int,
    key: str,
    backend_used: str,
    infer_status: str,
    preview_indices: Optional[list[int]] = None,
    preview_gif: Optional[str] = None,
    source_file: Optional[str] = None,
    ready_indices: Optional[list[int]] = None,
    infer_mode: Optional[str] = None,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
) -> dict[str, Any]:
    meta = {
        "kind": "video",
        "source": video_path.name,
        "source_file": source_file or "source" + video_path.suffix.lower(),
        "source_fps": source_fps,
        "stride": stride,
        "sample_fps": sample_fps,
        "frame_count": n,
        "ready_count": int(ready_count),
        "ready_indices": list(ready_indices or []),
        "infer_status": infer_status,
        "source_indices": indices,
        "duration_s": duration_s,
        "max_size": int(max_size),
        "model_key": key,
        "infer_backend": backend_used,
        "infer_mode": infer_mode,
        "preview_indices": preview_indices or [],
        "preview_gif": preview_gif,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    save_frames_meta(out_dir, meta)
    return meta


def reextract_rgb_frames(
    out_dir: Path,
    *,
    max_size: int,
    on_progress: ProgressCb = None,
) -> tuple[int, int, int]:
    """Re-read source video into frames/*/rgb.jpg at a new max_size.

    Returns (frame_count, width, height) of frame 0 after extract.
    """
    out_dir = Path(out_dir)
    meta = load_frames_meta(out_dir) or {}
    source_file = str(meta.get("source_file") or "source.mp4")
    video_path = out_dir / source_file
    if not video_path.is_file():
        raise FileNotFoundError(f"Source video missing for re-extract: {video_path}")
    indices = list(meta.get("source_indices") or [])
    n = int(meta.get("frame_count") or len(indices) or 0)
    if n < 1 or not indices:
        raise ValueError("No frames to re-extract")

    def progress(p: float, msg: str) -> None:
        if on_progress:
            on_progress(p, msg)

    progress(0.02, f"Re-extracting RGB at max {max_size}…")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    width = height = 0
    try:
        for i, src_idx in enumerate(indices[:n]):
            progress(0.02 + 0.9 * (i / max(n, 1)), f"Re-extract frame {i + 1}/{n}…")
            bgr = _read_frame_bgr(cap, int(src_idx))
            if bgr is None:
                raise RuntimeError(f"Failed to read frame {src_idx}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = _resize_max_side(rgb, int(max_size))
            fd = _frame_dir(out_dir, i)
            fd.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(fd / "rgb.jpg", rgb)
            # Drop stale depth/vis so infer rewrites them at the new size
            (fd / "depth.npz").unlink(missing_ok=True)
            (fd / "depth_video.npz").unlink(missing_ok=True)
            (fd / "vis.png").unlink(missing_ok=True)
            if i == 0:
                height, width = rgb.shape[:2]
    finally:
        cap.release()

    meta["max_size"] = int(max_size)
    meta["frame_width"] = int(width)
    meta["frame_height"] = int(height)
    meta["ready_count"] = 0
    meta["ready_indices"] = []
    meta["infer_status"] = "pending"
    meta.pop("video_depth", None)
    save_frames_meta(out_dir, meta)
    progress(1.0, "Re-extract done")
    return n, width, height


def _build_video_state(
    *,
    out_dir: Path,
    run_prefix: str,
    first_color: np.ndarray,
    first_depth: np.ndarray,
    n: int,
    ready_count: int,
    sample_fps: float,
    source_fps: float,
    key: str,
    max_size: int,
    backend_used: str,
    infer_device: str,
    modal_billing: Any = None,
) -> dict[str, Any]:
    from depth_core.store import materialize_run_arrays

    cliff = _subject_hi_pct(first_depth)
    materialize_run_arrays(out_dir, first_color, first_depth)
    frame0 = _frame_dir(out_dir, 0) / "rgb.jpg"
    if frame0.is_file():
        dest = out_dir / f"{run_prefix}_rgb.jpg"
        if dest.resolve() != frame0.resolve():
            try:
                shutil.copyfile(frame0, dest)
            except OSError:
                pass

    state: dict[str, Any] = {
        "kind": "video",
        "frame_index": 0,
        "frame_count": n,
        "ready_count": int(ready_count),
        "infer_status": "running" if ready_count < n else "done",
        "sample_fps": sample_fps,
        "source_fps": source_fps,
        "rgb": first_color,
        "depth": first_depth,
        "cliff": cliff,
        "looks": {},
        "vis_prompt": None,
        "blend": None,
        "blend_path": None,
        "metric_path": None,
        "look_paths": {},
        "ply_path": None,
        "out_dir": str(out_dir.resolve()),
        "model_key": key,
        "max_size": int(max_size),
        "infer_backend": backend_used,
        "infer_device": infer_device,
        "look_a": DEFAULT_LOOK_A,
        "look_b": DEFAULT_LOOK_B,
        "look_c": DEFAULT_LOOK_C,
        "fade_b": 0.0,
        "fade_c": 0.0,
        "blend_mode_b": DEFAULT_BLEND_MODE,
        "blend_mode_c": "screen",
        "scenery_protect": DEFAULT_SCENERY_PROTECT,
        "scenery_feather": DEFAULT_SCENERY_FEATHER,
        "scenery_mode": DEFAULT_SCENERY_MATTE,
        "scenery_clean": DEFAULT_SCENERY_CLEAN,
    }
    if modal_billing:
        state["modal_billing"] = modal_billing
    return state


def run_video(
    video_file: str | Path,
    model_key: str = "depth-pro",
    max_size: int = 756,
    *,
    target_fps: float = VIDEO_TARGET_FPS,
    max_frames: int = VIDEO_MAX_FRAMES,
    max_duration_s: float = VIDEO_MAX_DURATION_S,
    on_progress: ProgressCb = None,
    on_partial: PartialCb = None,
    infer_backend: Optional[str] = None,
) -> dict[str, Any]:
    """Infer Depth Pro on sampled frames; write multi-frame run directory.

    Local backend publishes a scrubbable partial run after frame 0 via on_partial
    so the kitchen can open while remaining frames finish.
    """
    from depth_core.infer_backend import resolve_backend

    key = normalize_model_key(model_key)
    if is_promptda(key):
        raise ValueError(
            "RGB video uses Depth Pro only (PromptDA needs LiDAR prompts per frame). "
            "Switch the model to Depth Pro."
        )
    video_path = Path(video_file)
    if not video_path.is_file():
        raise ValueError(f"Video not found: {video_path}")

    def progress(p: float, msg: str) -> None:
        if on_progress:
            on_progress(p, msg)

    def publish_partial(
        ready: int,
        state: Optional[dict[str, Any]] = None,
        ready_indices: Optional[list[int]] = None,
    ) -> None:
        if on_partial:
            on_partial(ready, n, state, ready_indices)

    backend_used = resolve_backend(infer_backend)
    progress(0.02, "Reading video…")
    indices, source_fps, stride, duration_s = extract_video_frame_indices(
        video_path,
        target_fps=target_fps,
        max_frames=max_frames,
        max_duration_s=max_duration_s,
    )
    n = len(indices)
    sample_fps = source_fps / stride if stride else target_fps

    out_dir = _make_run_dir(video_path, key)
    run_prefix = out_dir.name
    frames_root = out_dir / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    _write_frames_meta(
        out_dir,
        video_path=video_path,
        source_fps=source_fps,
        stride=stride,
        sample_fps=sample_fps,
        n=n,
        ready_count=0,
        indices=indices,
        duration_s=duration_s,
        max_size=int(max_size),
        key=key,
        backend_used=backend_used,
        infer_status="running",
    )

    first_color: Optional[np.ndarray] = None
    first_depth: Optional[np.ndarray] = None
    infer_device = "cpu"
    modal_billing = None
    state: Optional[dict[str, Any]] = None

    if backend_used == "modal":
        progress(0.05, f"Extracting {n} frames for Modal…")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        frame_bytes: list[bytes] = []
        try:
            for i, src_idx in enumerate(indices):
                progress(0.05 + 0.15 * (i / max(n, 1)), f"Extract frame {i + 1}/{n}…")
                bgr = _read_frame_bgr(cap, src_idx)
                if bgr is None:
                    raise RuntimeError(f"Failed to read frame {src_idx}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = _resize_max_side(rgb, int(max_size))
                fd = _frame_dir(out_dir, i)
                fd.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(fd / "rgb.jpg", rgb)
                tmp_rgb = fd / "_input.jpg"
                imageio.imwrite(tmp_rgb, rgb)
                frame_bytes.append(tmp_rgb.read_bytes())
                tmp_rgb.unlink(missing_ok=True)
        finally:
            cap.release()

        progress(0.25, f"Modal Depth Pro · {n} frames (parallel)…")
        from depth_core.infer_modal import forward_rgb_batch_as_completed

        infer_device = "cuda"
        billing_parts: list[dict[str, Any]] = []
        ready_set: set[int] = set()
        written: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for i, res in forward_rgb_batch_as_completed(
            frame_bytes, key, int(max_size)
        ):
            color = np.asarray(res.color)
            depth = res.depth.astype(np.float32)
            fd = _frame_dir(out_dir, i)
            fd.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(fd / "rgb.jpg", color)
            np.savez_compressed(fd / "depth.npz", depth=depth)
            written[i] = (color, depth)
            if res.modal_billing:
                billing_parts.append(res.modal_billing)
            ready_set.add(i)
            ready = len(ready_set)
            progress(
                0.25 + 0.7 * (ready / max(n, 1)),
                f"Depth frame {ready}/{n}…",
            )

            if 0 in written and state is None:
                first_color, first_depth = written[0]
                state = _build_video_state(
                    out_dir=out_dir,
                    run_prefix=run_prefix,
                    first_color=first_color,
                    first_depth=first_depth,
                    n=n,
                    ready_count=ready,
                    sample_fps=sample_fps,
                    source_fps=source_fps,
                    key=key,
                    max_size=int(max_size),
                    backend_used=backend_used,
                    infer_device=infer_device,
                )
                _write_frames_meta(
                    out_dir,
                    video_path=video_path,
                    source_fps=source_fps,
                    stride=stride,
                    sample_fps=sample_fps,
                    n=n,
                    ready_count=ready,
                    indices=indices,
                    duration_s=duration_s,
                    max_size=int(max_size),
                    key=key,
                    backend_used=backend_used,
                    infer_status="running" if ready < n else "done",
                )
                publish_partial(ready, state)
            else:
                _write_frames_meta(
                    out_dir,
                    video_path=video_path,
                    source_fps=source_fps,
                    stride=stride,
                    sample_fps=sample_fps,
                    n=n,
                    ready_count=ready,
                    indices=indices,
                    duration_s=duration_s,
                    max_size=int(max_size),
                    key=key,
                    backend_used=backend_used,
                    infer_status="running" if ready < n else "done",
                )
                if state is not None:
                    state["ready_count"] = ready
                    state["infer_status"] = "running" if ready < n else "done"
                    publish_partial(ready, None)

        modal_billing = _aggregate_modal_billing(billing_parts)
        if 0 not in written:
            raise RuntimeError("Modal video infer produced no frame 0")
        first_color, first_depth = written[0]
        ready = n
        _write_frames_meta(
            out_dir,
            video_path=video_path,
            source_fps=source_fps,
            stride=stride,
            sample_fps=sample_fps,
            n=n,
            ready_count=ready,
            indices=indices,
            duration_s=duration_s,
            max_size=int(max_size),
            key=key,
            backend_used=backend_used,
            infer_status="done",
        )
    else:
        progress(0.05, f"Loading Depth Pro · {n} frames…")
        backend = get_backend(key)
        infer_device = backend.device
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            for i, src_idx in enumerate(indices):
                progress(
                    0.05 + 0.9 * (i / max(n, 1)),
                    f"Depth frame {i + 1}/{n}…",
                )
                bgr = _read_frame_bgr(cap, src_idx)
                if bgr is None:
                    raise RuntimeError(f"Failed to read frame {src_idx}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = _resize_max_side(rgb, int(max_size))

                fd = _frame_dir(out_dir, i)
                fd.mkdir(parents=True, exist_ok=True)
                tmp_rgb = fd / "_input.jpg"
                imageio.imwrite(tmp_rgb, rgb)

                color, depth = _infer_depth_pro(
                    backend, tmp_rgb, max_size=int(max_size)
                )
                tmp_rgb.unlink(missing_ok=True)
                from depth_core.store import release_torch_memory

                release_torch_memory()

                imageio.imwrite(fd / "rgb.jpg", color)
                np.savez_compressed(fd / "depth.npz", depth=depth.astype(np.float32))

                ready = i + 1
                if i == 0:
                    first_color = color
                    first_depth = depth.astype(np.float32)
                    assert first_color is not None and first_depth is not None
                    state = _build_video_state(
                        out_dir=out_dir,
                        run_prefix=run_prefix,
                        first_color=first_color,
                        first_depth=first_depth,
                        n=n,
                        ready_count=ready,
                        sample_fps=sample_fps,
                        source_fps=source_fps,
                        key=key,
                        max_size=int(max_size),
                        backend_used=backend_used,
                        infer_device=infer_device,
                    )
                    _write_frames_meta(
                        out_dir,
                        video_path=video_path,
                        source_fps=source_fps,
                        stride=stride,
                        sample_fps=sample_fps,
                        n=n,
                        ready_count=ready,
                        indices=indices,
                        duration_s=duration_s,
                        max_size=int(max_size),
                        key=key,
                        backend_used=backend_used,
                        infer_status="running" if ready < n else "done",
                    )
                    publish_partial(ready, state)
                else:
                    _write_frames_meta(
                        out_dir,
                        video_path=video_path,
                        source_fps=source_fps,
                        stride=stride,
                        sample_fps=sample_fps,
                        n=n,
                        ready_count=ready,
                        indices=indices,
                        duration_s=duration_s,
                        max_size=int(max_size),
                        key=key,
                        backend_used=backend_used,
                        infer_status="running" if ready < n else "done",
                    )
                    if state is not None:
                        state["ready_count"] = ready
                        state["infer_status"] = (
                            "running" if ready < n else "done"
                        )
                    publish_partial(ready, None)
        finally:
            cap.release()

    assert first_color is not None and first_depth is not None

    if state is None:
        # Single-frame or edge path: publish once at the end
        progress(0.96, "Writing run metadata…")
        state = _build_video_state(
            out_dir=out_dir,
            run_prefix=run_prefix,
            first_color=first_color,
            first_depth=first_depth,
            n=n,
            ready_count=n,
            sample_fps=sample_fps,
            source_fps=source_fps,
            key=key,
            max_size=int(max_size),
            backend_used=backend_used,
            infer_device=infer_device,
            modal_billing=modal_billing,
        )
        publish_partial(n, state)
    else:
        state["ready_count"] = n
        state["infer_status"] = "done"
        if modal_billing:
            state["modal_billing"] = modal_billing
        _write_frames_meta(
            out_dir,
            video_path=video_path,
            source_fps=source_fps,
            stride=stride,
            sample_fps=sample_fps,
            n=n,
            ready_count=n,
            indices=indices,
            duration_s=duration_s,
            max_size=int(max_size),
            key=key,
            backend_used=backend_used,
            infer_status="done",
        )
        publish_partial(n, None)

    progress(1.0, "Done")
    return state


def prepare_video(
    video_file: str | Path,
    model_key: str = "depth-pro",
    max_size: int = 756,
    *,
    target_fps: float = VIDEO_TARGET_FPS,
    max_frames: int = VIDEO_MAX_FRAMES,
    max_duration_s: float = VIDEO_MAX_DURATION_S,
    infer_backend: Optional[str] = None,
    on_progress: ProgressCb = None,
) -> dict[str, Any]:
    """Extract RGB frames + cheap 6-frame GIF; no depth yet.

    Returns a prepared run state (infer_status=pending) suitable for put_run.
    """
    from depth_core.infer_backend import resolve_backend

    key = normalize_model_key(model_key)
    if is_promptda(key):
        raise ValueError(
            "RGB video uses Depth Pro only (PromptDA needs LiDAR prompts per frame). "
            "Switch the model to Depth Pro."
        )
    video_path = Path(video_file)
    if not video_path.is_file():
        raise ValueError(f"Video not found: {video_path}")

    def progress(p: float, msg: str) -> None:
        if on_progress:
            on_progress(p, msg)

    backend_used = resolve_backend(infer_backend)
    progress(0.02, "Reading video…")
    indices, source_fps, stride, duration_s = extract_video_frame_indices(
        video_path,
        target_fps=target_fps,
        max_frames=max_frames,
        max_duration_s=max_duration_s,
    )
    n = len(indices)
    sample_fps = source_fps / stride if stride else target_fps

    out_dir = _make_run_dir(video_path, key)
    run_prefix = out_dir.name
    frames_root = out_dir / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    suffix = video_path.suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v"}:
        suffix = ".mp4"
    source_name = f"source{suffix}"
    source_dest = out_dir / source_name
    shutil.copyfile(video_path, source_dest)

    progress(0.1, f"Extracting {n} RGB frames…")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    first_color: Optional[np.ndarray] = None
    try:
        for i, src_idx in enumerate(indices):
            progress(0.1 + 0.7 * (i / max(n, 1)), f"Extract frame {i + 1}/{n}…")
            bgr = _read_frame_bgr(cap, src_idx)
            if bgr is None:
                raise RuntimeError(f"Failed to read frame {src_idx}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = _resize_max_side(rgb, int(max_size))
            fd = _frame_dir(out_dir, i)
            fd.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(fd / "rgb.jpg", rgb)
            if i == 0:
                first_color = rgb
    finally:
        cap.release()

    assert first_color is not None
    fh, fw = first_color.shape[:2]
    preview_idxs = pick_preview_indices(n, VIDEO_PREVIEW_FRAMES)
    progress(0.85, "Building preview GIF…")
    gif_path = write_preview_gif(out_dir, preview_idxs)
    imageio.imwrite(out_dir / f"{run_prefix}_rgb.jpg", first_color)

    _write_frames_meta(
        out_dir,
        video_path=video_path,
        source_fps=source_fps,
        stride=stride,
        sample_fps=sample_fps,
        n=n,
        ready_count=0,
        indices=indices,
        duration_s=duration_s,
        max_size=int(max_size),
        key=key,
        backend_used=backend_used,
        infer_status="pending",
        preview_indices=preview_idxs,
        preview_gif="preview.gif",
        source_file=source_name,
        ready_indices=[],
        frame_width=fw,
        frame_height=fh,
    )

    progress(1.0, "Prepared")
    return {
        "kind": "video",
        "frame_index": 0,
        "frame_count": n,
        "ready_count": 0,
        "infer_status": "pending",
        "sample_fps": sample_fps,
        "source_fps": source_fps,
        "rgb": first_color,
        "depth": None,
        "cliff": None,
        "looks": {},
        "vis_prompt": None,
        "blend": None,
        "blend_path": None,
        "metric_path": None,
        "look_paths": {},
        "ply_path": None,
        "out_dir": str(out_dir.resolve()),
        "model_key": key,
        "max_size": int(max_size),
        "frame_width": fw,
        "frame_height": fh,
        "infer_backend": backend_used,
        "infer_device": None,
        "preview_gif": "preview.gif",
        "preview_indices": preview_idxs,
        "source_file": source_name,
        "source_indices": indices,
        "duration_s": duration_s,
        "stride": stride,
    }


def infer_prepared_run(
    out_dir: str | Path,
    *,
    model_key: Optional[str] = None,
    max_size: Optional[int] = None,
    infer_backend: Optional[str] = None,
    on_progress: ProgressCb = None,
    on_partial: PartialCb = None,
) -> dict[str, Any]:
    """Run Depth Pro on an already-prepared video run (RGB frames on disk)."""
    from depth_core.infer_backend import resolve_backend

    out_dir = Path(out_dir)
    meta = load_frames_meta(out_dir)
    if not meta:
        raise ValueError(f"No frames_meta.json under {out_dir}")

    key = normalize_model_key(str(model_key or meta.get("model_key") or "depth-pro"))
    if is_promptda(key):
        raise ValueError(
            "RGB video uses Depth Pro only (PromptDA needs LiDAR prompts per frame)."
        )
    n = int(meta.get("frame_count") or 0)
    if n < 1:
        raise ValueError("Prepared run has no frames")
    indices = list(meta.get("source_indices") or list(range(n)))
    sample_fps = float(meta.get("sample_fps") or VIDEO_TARGET_FPS)
    source_fps = float(meta.get("source_fps") or sample_fps)
    stride = int(meta.get("stride") or 1)
    duration_s = float(meta.get("duration_s") or 0)
    max_sz = int(max_size or meta.get("max_size") or 756)
    backend_used = resolve_backend(infer_backend or meta.get("infer_backend"))
    video_name = str(meta.get("source") or "video.mp4")
    video_path = out_dir / str(meta.get("source_file") or "source.mp4")
    if not video_path.is_file():
        video_path = Path(video_name)
    preview_indices = list(meta.get("preview_indices") or [])
    preview_gif = meta.get("preview_gif")
    source_file = meta.get("source_file")
    run_prefix = out_dir.name
    prepared_max = int(meta.get("max_size") or max_sz)
    infer_mode = (
        "modal_parallel" if backend_used == "modal" else "local_sequential"
    )

    def progress(p: float, msg: str) -> None:
        if on_progress:
            on_progress(p, msg)

    # Re-extract RGB if the user bumped max side since prepare.
    if max_sz != prepared_max and video_path.is_file():
        progress(0.01, f"Max side {prepared_max}→{max_sz}; re-extracting…")
        _n, fw0, fh0 = reextract_rgb_frames(
            out_dir, max_size=max_sz, on_progress=on_progress
        )
        meta = load_frames_meta(out_dir) or meta
        n = int(meta.get("frame_count") or _n)
        frame_width, frame_height = fw0, fh0
    else:
        frame_width = meta.get("frame_width")
        frame_height = meta.get("frame_height")

    def publish_partial(
        ready: int,
        state: Optional[dict[str, Any]] = None,
        ready_indices: Optional[list[int]] = None,
    ) -> None:
        if on_partial:
            on_partial(ready, n, state, ready_indices)

    def _meta_kwargs(
        ready: int,
        status: str,
        *,
        ready_indices: Optional[list[int]] = None,
        fw: Any = None,
        fh: Any = None,
    ) -> dict[str, Any]:
        return dict(
            video_path=video_path if video_path.is_file() else Path(video_name),
            source_fps=source_fps,
            stride=stride,
            sample_fps=sample_fps,
            n=n,
            ready_count=ready,
            indices=indices,
            duration_s=duration_s,
            max_size=max_sz,
            key=key,
            backend_used=backend_used,
            infer_status=status,
            preview_indices=preview_indices,
            preview_gif=preview_gif,
            source_file=source_file,
            ready_indices=ready_indices or [],
            infer_mode=infer_mode,
            frame_width=fw if fw is not None else frame_width,
            frame_height=fh if fh is not None else frame_height,
        )

    _write_frames_meta(out_dir, **_meta_kwargs(0, "running", ready_indices=[]))

    first_color: Optional[np.ndarray] = None
    first_depth: Optional[np.ndarray] = None
    infer_device = "cpu"
    modal_billing = None
    state: Optional[dict[str, Any]] = None

    frame_bytes: list[bytes] = []
    for i in range(n):
        rgb_path = _frame_dir(out_dir, i) / "rgb.jpg"
        if not rgb_path.is_file():
            raise FileNotFoundError(f"Missing RGB frame {i}")
        frame_bytes.append(rgb_path.read_bytes())

    if backend_used == "modal":
        progress(0.1, f"Modal Depth Pro · {n} frames (parallel)…")
        from depth_core.infer_modal import forward_rgb_batch_as_completed

        infer_device = "cuda"
        billing_parts: list[dict[str, Any]] = []
        ready_set: set[int] = set()
        written: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for i, res in forward_rgb_batch_as_completed(frame_bytes, key, max_sz):
            color = np.asarray(res.color)
            depth = res.depth.astype(np.float32)
            fd = _frame_dir(out_dir, i)
            imageio.imwrite(fd / "rgb.jpg", color)
            np.savez_compressed(fd / "depth.npz", depth=depth)
            written[i] = (color, depth)
            if res.modal_billing:
                billing_parts.append(res.modal_billing)
            ready_set.add(i)
            ready = len(ready_set)
            ready_idx = sorted(ready_set)
            if frame_width is None and color is not None:
                frame_height, frame_width = color.shape[:2]
            progress(
                0.1 + 0.85 * (ready / max(n, 1)),
                f"Modal parallel · depth {ready}/{n} (frame {i})…",
            )

            status = "running" if ready < n else "done"
            if 0 in written and state is None:
                first_color, first_depth = written[0]
                state = _build_video_state(
                    out_dir=out_dir,
                    run_prefix=run_prefix,
                    first_color=first_color,
                    first_depth=first_depth,
                    n=n,
                    ready_count=ready,
                    sample_fps=sample_fps,
                    source_fps=source_fps,
                    key=key,
                    max_size=max_sz,
                    backend_used=backend_used,
                    infer_device=infer_device,
                )
                _write_frames_meta(
                    out_dir, **_meta_kwargs(ready, status, ready_indices=ready_idx)
                )
                publish_partial(ready, state, ready_idx)
            else:
                _write_frames_meta(
                    out_dir, **_meta_kwargs(ready, status, ready_indices=ready_idx)
                )
                if state is not None:
                    state["ready_count"] = ready
                    state["infer_status"] = status
                    publish_partial(ready, None, ready_idx)

        modal_billing = _aggregate_modal_billing(billing_parts)
        if 0 not in written:
            raise RuntimeError("Modal video infer produced no frame 0")
        first_color, first_depth = written[0]
        final_indices = sorted(ready_set)
    else:
        progress(0.05, f"Local Depth Pro · {n} frames (sequential)…")
        backend = get_backend(key)
        infer_device = backend.device
        final_indices: list[int] = []
        for i in range(n):
            progress(
                0.05 + 0.9 * (i / max(n, 1)),
                f"Local sequential · depth {i + 1}/{n}…",
            )
            rgb_path = _frame_dir(out_dir, i) / "rgb.jpg"
            color, depth = _infer_depth_pro(
                backend, rgb_path, max_size=max_sz
            )
            from depth_core.store import release_torch_memory

            release_torch_memory()
            fd = _frame_dir(out_dir, i)
            imageio.imwrite(fd / "rgb.jpg", color)
            np.savez_compressed(fd / "depth.npz", depth=depth.astype(np.float32))
            ready = i + 1
            final_indices = list(range(ready))
            if frame_width is None:
                frame_height, frame_width = color.shape[:2]
            status = "running" if ready < n else "done"
            if i == 0:
                first_color = color
                first_depth = depth.astype(np.float32)
                state = _build_video_state(
                    out_dir=out_dir,
                    run_prefix=run_prefix,
                    first_color=first_color,
                    first_depth=first_depth,
                    n=n,
                    ready_count=ready,
                    sample_fps=sample_fps,
                    source_fps=source_fps,
                    key=key,
                    max_size=max_sz,
                    backend_used=backend_used,
                    infer_device=infer_device,
                )
                _write_frames_meta(
                    out_dir,
                    **_meta_kwargs(ready, status, ready_indices=final_indices),
                )
                publish_partial(ready, state, final_indices)
            else:
                _write_frames_meta(
                    out_dir,
                    **_meta_kwargs(ready, status, ready_indices=final_indices),
                )
                if state is not None:
                    state["ready_count"] = ready
                    state["infer_status"] = status
                publish_partial(ready, None, final_indices)

    assert first_color is not None and first_depth is not None
    if frame_width is None:
        frame_height, frame_width = first_color.shape[:2]
    if state is None:
        state = _build_video_state(
            out_dir=out_dir,
            run_prefix=run_prefix,
            first_color=first_color,
            first_depth=first_depth,
            n=n,
            ready_count=n,
            sample_fps=sample_fps,
            source_fps=source_fps,
            key=key,
            max_size=max_sz,
            backend_used=backend_used,
            infer_device=infer_device,
            modal_billing=modal_billing,
        )
        publish_partial(n, state, list(range(n)))
    else:
        state["ready_count"] = n
        state["infer_status"] = "done"
        if modal_billing:
            state["modal_billing"] = modal_billing
        _write_frames_meta(
            out_dir,
            **_meta_kwargs(n, "done", ready_indices=list(range(n))),
        )
        publish_partial(n, None, list(range(n)))

    # Preserve prepare artifacts on state
    state["preview_gif"] = preview_gif
    state["preview_indices"] = preview_indices
    state["source_file"] = source_file
    state["frame_width"] = frame_width
    state["frame_height"] = frame_height
    state["infer_mode"] = infer_mode
    state["ready_indices"] = list(range(n))
    progress(1.0, "Done")
    return state


def export_frames_to_mp4(
    frame_pngs: list[Path],
    out_mp4: Path,
    fps: float,
) -> Path:
    """Encode ordered PNG frames to MP4 via ffmpeg (PATH or imageio-ffmpeg)."""
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found. Install system ffmpeg or `pip install imageio-ffmpeg`."
        )
    if not frame_pngs:
        raise ValueError("No frames to export")
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # Expect %06d.png naming in the same directory
    pattern = frame_pngs[0].parent / "%06d.png"
    # libx264 + yuv420p require even width/height (e.g. 756x425 fails).
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(max(fps, 1.0)),
        "-i",
        str(pattern),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_mp4.is_file():
        raise RuntimeError(proc.stderr.strip() or "ffmpeg export failed")
    return out_mp4
