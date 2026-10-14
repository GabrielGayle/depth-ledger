"""Modal remote inference client."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterator, Optional, Tuple

import imageio
import numpy as np

from depth_core.infer_backend import modal_app_name
from depth_core.modal_pricing import billing_dict

_WORKER = None
_WORKER_ERROR: Optional[str] = None


@dataclass
class ForwardResult:
    color: np.ndarray
    depth: np.ndarray
    prompt_np: Optional[np.ndarray]
    model_key: str
    device: str
    modal_billing: Optional[dict[str, Any]]
    focallength_px: Optional[float] = None


def modal_credentials_present() -> bool:
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return True
    try:
        return (Path.home() / ".modal.toml").is_file()
    except Exception:
        return False


def modal_available() -> bool:
    if not modal_credentials_present():
        return False
    try:
        import modal  # noqa: F401
    except ImportError:
        return False
    return True


def _get_worker():
    global _WORKER, _WORKER_ERROR
    if _WORKER is not None:
        return _WORKER
    if _WORKER_ERROR is not None:
        raise RuntimeError(_WORKER_ERROR)
    try:
        import modal

        app_name = modal_app_name()
        cls = modal.Cls.from_name(app_name, "DepthWorker")
        _WORKER = cls()
        return _WORKER
    except Exception as exc:
        _WORKER_ERROR = (
            f"Modal worker unavailable ({exc}). "
            "Run: modal deploy modal_app/worker.py"
        )
        raise RuntimeError(_WORKER_ERROR) from exc


def _record(billing: dict[str, Any]) -> None:
    try:
        from server.modal_usage import record_modal_usage

        record_modal_usage(billing)
    except Exception:
        pass


def _decode_result(payload: dict[str, Any]) -> ForwardResult:
    depth_buf = io.BytesIO(payload["depth"])
    depth = np.load(depth_buf)["depth"].astype(np.float32)
    color = imageio.imread(io.BytesIO(payload["rgb"]))
    prompt_np = None
    if payload.get("prompt_depth"):
        prompt_np = np.load(io.BytesIO(payload["prompt_depth"]))["depth"].astype(
            np.float32
        )
    gpu_type = str(payload.get("gpu_type") or "T4")
    gpu_seconds = float(payload.get("gpu_seconds") or 0.0)
    cold_start = bool(payload.get("cold_start"))
    billing = billing_dict(gpu_type, gpu_seconds, cold_start=cold_start)
    _record(billing)
    fl = payload.get("focallength_px")
    try:
        focallength_px = float(fl) if fl is not None else None
    except (TypeError, ValueError):
        focallength_px = None
    return ForwardResult(
        color=np.asarray(color),
        depth=depth,
        prompt_np=prompt_np,
        model_key=str(payload.get("model_key") or "depth-pro"),
        device=str(payload.get("device") or "cuda"),
        modal_billing=billing,
        focallength_px=focallength_px,
    )


def forward_rgb(
    rgb_path: Path,
    model_key: str,
    max_size: int,
    f_px: Optional[float] = None,
    estimate_focal: bool = False,
) -> ForwardResult:
    worker = _get_worker()
    rgb_bytes = Path(rgb_path).read_bytes()
    payload = worker.infer_rgb.remote(
        rgb_bytes, model_key, int(max_size), f_px, bool(estimate_focal)
    )
    return _decode_result(payload)


def forward_pair(
    rgb_path: Path,
    prompt_depth_path: Path,
    model_key: str,
    max_size: int,
) -> ForwardResult:
    worker = _get_worker()
    payload = worker.infer_pair.remote(
        Path(rgb_path).read_bytes(),
        Path(prompt_depth_path).read_bytes(),
        model_key,
        int(max_size),
    )
    return _decode_result(payload)


def forward_frames(
    frame_bytes_list: list[bytes],
    model_key: str,
    max_size: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], Optional[dict[str, Any]]]:
    worker = _get_worker()
    payload = worker.infer_frames.remote(frame_bytes_list, model_key, int(max_size))
    frames_out: list[tuple[np.ndarray, np.ndarray]] = []
    for item in payload.get("frames") or []:
        depth = np.load(io.BytesIO(item["depth"]))["depth"].astype(np.float32)
        color = imageio.imread(io.BytesIO(item["rgb"]))
        frames_out.append((np.asarray(color), depth))
    billing = None
    gpu_type = str(payload.get("gpu_type") or "T4")
    gpu_seconds = float(payload.get("gpu_seconds") or 0.0)
    cold_start = bool(payload.get("cold_start"))
    if gpu_seconds > 0:
        billing = billing_dict(gpu_type, gpu_seconds, cold_start=cold_start)
        _record(billing)
    return frames_out, billing


def forward_rgb_batch_as_completed(
    rgb_bytes_list: list[bytes],
    model_key: str,
    max_size: int,
    f_px: Optional[float] = None,
    estimate_focal: bool = False,
) -> Iterator[Tuple[int, ForwardResult]]:
    """Yield (index, result) as each Modal infer finishes (completion order).

    Spawns all calls up front so cook/publish can overlap remaining GPU work.
    """
    if not rgb_bytes_list:
        return
    worker = _get_worker()
    # Warm once so the batch doesn't pay cold start N times.
    try:
        worker.ping.remote()
    except Exception:
        pass

    calls: list[Tuple[int, Any]] = [
        (
            i,
            worker.infer_rgb.spawn(
                b, model_key, int(max_size), f_px, bool(estimate_focal)
            ),
        )
        for i, b in enumerate(rgb_bytes_list)
    ]

    def _cancel_all() -> None:
        for _, call in calls:
            try:
                call.cancel()
            except Exception:
                pass

    workers = min(16, max(1, len(calls)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fut_to_i = {pool.submit(call.get): i for i, call in calls}
        try:
            for fut in as_completed(fut_to_i):
                i = fut_to_i[fut]
                yield i, _decode_result(fut.result())
        except GeneratorExit:
            _cancel_all()
            raise
        except Exception:
            _cancel_all()
            raise


def forward_rgb_batch(
    rgb_bytes_list: list[bytes],
    model_key: str,
    max_size: int,
    f_px: Optional[float] = None,
    estimate_focal: bool = False,
) -> list[ForwardResult]:
    """Parallel Modal infer for batch jobs (ordered by input index)."""
    by_i: dict[int, ForwardResult] = {}
    for i, res in forward_rgb_batch_as_completed(
        rgb_bytes_list,
        model_key,
        max_size,
        f_px=f_px,
        estimate_focal=estimate_focal,
    ):
        by_i[i] = res
    return [by_i[i] for i in range(len(rgb_bytes_list))]


def modal_warmup() -> dict[str, Any]:
    worker = _get_worker()
    payload = worker.ping.remote()
    gpu_type = str(payload.get("gpu_type") or "T4")
    gpu_seconds = float(payload.get("gpu_seconds") or 0.0)
    billing = billing_dict(
        gpu_type, gpu_seconds, cold_start=bool(payload.get("cold_start"))
    )
    if gpu_seconds > 0:
        _record(billing)
    return {"ok": True, "modal_billing": billing, **payload}
