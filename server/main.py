"""Depth Ledger FastAPI application."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from repo .env without overriding real env."""
    path = ROOT / ".env"
    if not path.is_file():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except OSError:
        pass


_load_dotenv()

# Enable OpenCV float EXR export for Blender bundles (disabled by default).
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from depth_core.pipeline import DEFAULT_MODEL, get_backend  # noqa: E402
from server.routes import router  # noqa: E402

WEB_DIST = ROOT / "web" / "dist"

app = FastAPI(title="Depth Ledger", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


def _prune_on_boot() -> None:
    try:
        from depth_core.infer_backend import configure_thread_pools

        configure_thread_pools()
    except Exception:
        pass
    try:
        from depth_core.store import cache_stats

        stats = cache_stats()
        print(
            "Working cache: "
            f"{stats.get('outputs_mb')} MB / {stats.get('disk_budget_mb')} MB budget"
            f" · {stats.get('run_dirs')} run dirs · POST /api/cache/prune to compact"
        )
    except Exception as exc:
        print(f"Working cache stats skipped: {exc}")


@app.on_event("startup")
def _startup():
    from server.status import set_local_infer, set_warmup

    _prune_on_boot()
    if os.environ.get("DEPTH_LEDGER_SKIP_WARMUP", "").strip() in {"1", "true", "yes"}:
        set_warmup("skipped")
        set_local_infer("idle")
        return
    from depth_core.infer_backend import DEFAULT_INFER_BACKEND

    if DEFAULT_INFER_BACKEND == "modal":
        print("INFER_BACKEND=modal — skipping local model warmup")
        set_warmup("skipped")
        set_local_infer("idle")
        return
    set_warmup("warming")
    set_local_infer("loading")
    try:
        loaded = get_backend(DEFAULT_MODEL)
        set_warmup("ready")
        set_local_infer("idle", device=getattr(loaded, "device", None))
    except Exception as exc:
        print(f"Warmup skipped / failed: {exc}")
        set_warmup("error", error=str(exc))
        set_local_infer("failed", last_error=str(exc))


if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
