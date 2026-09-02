"""Live multi-system status aggregator for Depth Ledger console."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from depth_core.infer_backend import DEFAULT_INFER_BACKEND, modal_ready
from depth_core.modal_pricing import pricing_meta
from server.modal_usage import get_modal_usage

UI_HEARTBEAT_TTL_S = 8.0
BLENDER_TCP_TIMEOUT_S = 0.2
BLENDER_DEEP_TIMEOUT_S = 1.5
BLENDER_TCP_CACHE_S = 5.0
BLENDER_DEEP_CACHE_S = 30.0
DEFAULT_POLL_HINT_MS = 2000

_LOCK = threading.Lock()

_WARMUP: dict[str, Any] = {
    "state": "idle",  # idle|warming|ready|skipped|error
    "error": None,
    "at": None,
}

_LOCAL_INFER: dict[str, Any] = {
    "state": "unknown",  # idle|loading|running|failed|unknown
    "device": None,
    "last_run_id": None,
    "last_error": None,
    "at": None,
}

_UI: dict[str, Any] = {
    "mode": "unknown",
    "last_evaluate_at": None,
    "last_run_id": None,
    "heartbeat_at": None,
}

_MODAL_META: dict[str, Any] = {
    "state": "unknown",  # ready|undeployed|error|unknown
    "last_job": None,
    "last_error": None,
}

_BLENDER_CACHE: dict[str, Any] = {
    "tcp": None,
    "tcp_at": 0.0,
    "deep": None,
    "deep_at": 0.0,
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_warmup(state: str, error: Optional[str] = None) -> None:
    with _LOCK:
        _WARMUP["state"] = state
        _WARMUP["error"] = error
        _WARMUP["at"] = _iso_now()


def set_local_infer(
    state: str,
    *,
    device: Optional[str] = None,
    last_run_id: Optional[str] = None,
    last_error: Optional[str] = None,
) -> None:
    with _LOCK:
        _LOCAL_INFER["state"] = state
        if device is not None:
            _LOCAL_INFER["device"] = device
        if last_run_id is not None:
            _LOCAL_INFER["last_run_id"] = last_run_id
        if last_error is not None:
            _LOCAL_INFER["last_error"] = last_error
        elif state in ("idle", "running", "loading"):
            _LOCAL_INFER["last_error"] = None
        _LOCAL_INFER["at"] = _iso_now()


def set_modal_state(
    state: str,
    *,
    last_job: Optional[str] = None,
    last_error: Optional[str] = None,
) -> None:
    with _LOCK:
        _MODAL_META["state"] = state
        if last_job is not None:
            _MODAL_META["last_job"] = last_job
        if last_error is not None:
            _MODAL_META["last_error"] = last_error
        elif state in ("ready", "unknown"):
            _MODAL_META["last_error"] = None


def record_ui_heartbeat(
    *,
    mode: Optional[str] = None,
    last_evaluate_at: Optional[str] = None,
    last_run_id: Optional[str] = None,
) -> dict[str, Any]:
    with _LOCK:
        if mode:
            _UI["mode"] = mode
        if last_evaluate_at is not None:
            _UI["last_evaluate_at"] = last_evaluate_at
        if last_run_id is not None:
            _UI["last_run_id"] = last_run_id
        _UI["heartbeat_at"] = _iso_now()
        return dict(_UI)


def pinned_run_ids() -> set[str]:
    """Run ids the UI / local infer currently care about (keep arrays hot)."""
    with _LOCK:
        ids = {
            _UI.get("last_run_id"),
            _LOCAL_INFER.get("last_run_id"),
        }
    return {str(x) for x in ids if x}


def blender_env() -> tuple[str, int, Path]:
    host = os.environ.get("BLENDER_MCP_HOST", "localhost").strip() or "localhost"
    port_raw = os.environ.get("BLENDER_MCP_PORT", "9876").strip() or "9876"
    try:
        port = int(port_raw)
    except ValueError:
        port = 9876
    path = Path(
        os.environ.get(
            "BLENDER_PATH",
            "/Applications/Blender.app/Contents/MacOS/Blender",
        )
    )
    return host, port, path


def _probe_blender_tcp(host: str, port: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=BLENDER_TCP_TIMEOUT_S):
            pass
        return {
            "addon": "listening",
            "last_probe_ms": int((time.perf_counter() - t0) * 1000),
            "last_error": None,
        }
    except OSError as exc:
        return {
            "addon": "not_listening",
            "last_probe_ms": int((time.perf_counter() - t0) * 1000),
            "last_error": str(exc),
        }


def _probe_blender_deep(host: str, port: int) -> dict[str, Any]:
    """Null-framed execute on the Lab MCP add-on socket."""
    t0 = time.perf_counter()
    code = (
        "import bpy; "
        "result={'ok':True,'version':'.'.join(map(str, bpy.app.version))}"
    )
    payload = (
        json.dumps(
            {"type": "execute", "code": code, "strict_json": False},
            separators=(",", ":"),
        )
        + "\0"
    ).encode("utf-8")
    try:
        with socket.create_connection((host, port), timeout=BLENDER_DEEP_TIMEOUT_S) as sock:
            sock.settimeout(BLENDER_DEEP_TIMEOUT_S)
            sock.sendall(payload)
            chunks: list[bytes] = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\0" in chunk:
                    break
        raw = b"".join(chunks).split(b"\0", 1)[0]
        data = json.loads(raw.decode("utf-8"))
        ms = int((time.perf_counter() - t0) * 1000)
        if data.get("status") == "ok":
            result = data.get("result") or {}
            version = None
            if isinstance(result, dict):
                version = result.get("version")
            return {
                "bpy": "ok",
                "version": version,
                "last_probe_ms": ms,
                "last_error": None,
            }
        return {
            "bpy": "error",
            "version": None,
            "last_probe_ms": ms,
            "last_error": str(data.get("message") or data),
        }
    except Exception as exc:
        return {
            "bpy": "error",
            "version": None,
            "last_probe_ms": int((time.perf_counter() - t0) * 1000),
            "last_error": str(exc),
        }


def _cached_blender(deep: bool) -> dict[str, Any]:
    host, port, path = blender_env()
    now = time.monotonic()
    with _LOCK:
        tcp_cached = _BLENDER_CACHE["tcp"]
        tcp_at = float(_BLENDER_CACHE["tcp_at"] or 0)
        deep_cached = _BLENDER_CACHE["deep"]
        deep_at = float(_BLENDER_CACHE["deep_at"] or 0)

    need_tcp = tcp_cached is None or (now - tcp_at) > BLENDER_TCP_CACHE_S
    need_deep = deep and (
        deep_cached is None or (now - deep_at) > BLENDER_DEEP_CACHE_S
    )

    tcp = tcp_cached
    if need_tcp:
        tcp = _probe_blender_tcp(host, port)
        with _LOCK:
            _BLENDER_CACHE["tcp"] = tcp
            _BLENDER_CACHE["tcp_at"] = time.monotonic()

    deep_info = deep_cached or {
        "bpy": "unknown",
        "version": None,
        "last_probe_ms": 0,
        "last_error": None,
    }
    if need_deep and tcp and tcp.get("addon") == "listening":
        deep_info = _probe_blender_deep(host, port)
        with _LOCK:
            _BLENDER_CACHE["deep"] = deep_info
            _BLENDER_CACHE["deep_at"] = time.monotonic()
    elif tcp and tcp.get("addon") != "listening":
        deep_info = {
            "bpy": "unknown",
            "version": (deep_cached or {}).get("version"),
            "last_probe_ms": tcp.get("last_probe_ms") or 0,
            "last_error": tcp.get("last_error"),
        }

    return {
        "blender_path_ok": path.exists(),
        "addon": (tcp or {}).get("addon", "not_listening"),
        "bpy": deep_info.get("bpy", "unknown"),
        "version": deep_info.get("version"),
        "last_probe_ms": int(
            deep_info.get("last_probe_ms")
            or (tcp or {}).get("last_probe_ms")
            or 0
        ),
        "last_error": deep_info.get("last_error") or (tcp or {}).get("last_error"),
        "host": host,
        "port": port,
    }


def _active_jobs_and_runs() -> tuple[int, list[dict[str, str]]]:
    active_jobs = 0
    try:
        from server.routes import _JOBS, _JOBS_LOCK

        with _JOBS_LOCK:
            for job in _JOBS.values():
                if str(job.get("status") or "") == "running":
                    active_jobs += 1
    except Exception:
        pass

    active_runs: list[dict[str, str]] = []
    try:
        from server.cache import _RUNS

        for run_id, state in list(_RUNS.items()):
            status = str(state.get("infer_status") or "")
            if status in ("pending", "running"):
                active_runs.append({"run_id": run_id, "infer_status": status})
    except Exception:
        pass
    return active_jobs, active_runs


def _modal_block() -> dict[str, Any]:
    available = bool(modal_ready())
    with _LOCK:
        meta = dict(_MODAL_META)
    state = meta.get("state") or "unknown"
    if not available:
        state = "undeployed" if state in ("unknown", "ready") else state
    elif state == "unknown":
        state = "ready"
    return {
        "available": available,
        "state": state,
        "last_job": meta.get("last_job"),
        "usage": get_modal_usage(),
        "pricing": pricing_meta(),
        "last_error": meta.get("last_error"),
    }


def _ui_block() -> dict[str, Any]:
    with _LOCK:
        ui = dict(_UI)
    beat = ui.get("heartbeat_at")
    vite_up = "unknown"
    if beat:
        try:
            ts = datetime.fromisoformat(str(beat).replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            vite_up = "up" if age <= UI_HEARTBEAT_TTL_S else "down"
        except ValueError:
            vite_up = "unknown"
    return {
        "mode": ui.get("mode") or "unknown",
        "vite_up": vite_up,
        "last_evaluate_at": ui.get("last_evaluate_at"),
        "last_run_id": ui.get("last_run_id"),
    }


def build_status(*, deep: bool = False) -> dict[str, Any]:
    with _LOCK:
        warmup = dict(_WARMUP)
        local = dict(_LOCAL_INFER)

    active_jobs, active_runs = _active_jobs_and_runs()
    any_busy = active_jobs > 0 or bool(active_runs) or local.get("state") in (
        "loading",
        "running",
    )
    poll_hint = 2000 if any_busy else 5000

    cache_block: dict[str, Any] = {}
    try:
        from depth_core.store import cache_stats
        from server.cache import ram_stats

        cache_block = {**cache_stats(), **ram_stats()}
    except Exception:
        cache_block = {}

    return {
        "ts": _iso_now(),
        "poll_hint_ms": poll_hint,
        "api": {
            "ok": True,
            "warmup": warmup.get("state") or "idle",
            "infer_backend_default": DEFAULT_INFER_BACKEND,
            "active_jobs": active_jobs,
            "active_runs": active_runs,
            "warmup_error": warmup.get("error"),
        },
        "local_infer": {
            "state": local.get("state") or "unknown",
            "device": local.get("device"),
            "last_run_id": local.get("last_run_id"),
            "last_error": local.get("last_error"),
        },
        "modal": _modal_block(),
        "ui": _ui_block(),
        "blender_mcp": _cached_blender(deep=deep),
        "cache": cache_block,
    }


async def build_status_async(*, deep: bool = False) -> dict[str, Any]:
    """Run socket probes off the event loop."""
    return await asyncio.to_thread(build_status, deep=deep)
