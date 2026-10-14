"""Launch official Blender with a Depth Ledger run loaded."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BLEND_ASSET = ROOT / "blender" / "assets" / "DepthLedger_v1.blend"
LOAD_SCRIPT = ROOT / "blender" / "load_run.py"


def blender_binary() -> Path:
    return Path(
        os.environ.get(
            "BLENDER_PATH",
            "/Applications/Blender.app/Contents/MacOS/Blender",
        )
    )


def blender_app_bundle() -> Path:
    binary = blender_binary()
    # …/Blender.app/Contents/MacOS/Blender → …/Blender.app
    for parent in binary.parents:
        if parent.suffix == ".app":
            return parent
    return Path("/Applications/Blender.app")


def launch_blender_for_run(
    run_dir: Path | str,
    *,
    preset: str = "ledger",
) -> dict[str, Any]:
    """Spawn GUI Blender with DepthLedger_v1.blend + load_run.py for this run."""
    run_dir = Path(run_dir).resolve()
    blender = blender_binary()
    if not blender.is_file():
        raise FileNotFoundError(
            f"Blender not found at {blender}. Set BLENDER_PATH to "
            "/Applications/Blender.app/Contents/MacOS/Blender"
        )
    if not BLEND_ASSET.is_file():
        raise FileNotFoundError(
            f"Missing asset {BLEND_ASSET}. Run: make blender-assets"
        )
    if not LOAD_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing {LOAD_SCRIPT}")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    log_path = run_dir / "_blender_launch.log"
    args = [
        str(BLEND_ASSET),
        "--python",
        str(LOAD_SCRIPT),
        "--",
        str(run_dir),
        preset,
    ]

    # On macOS prefer `open -n -a … --args` so the window attaches to the
    # interactive session (raw binary from uvicorn can look like “nothing”).
    if sys.platform == "darwin":
        app = blender_app_bundle()
        cmd = ["open", "-n", "-a", str(app), "--args", *args]
    else:
        cmd = [str(blender), *args]

    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write("cmd: " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    return {
        "ok": True,
        "pid": proc.pid,
        "cmd": cmd,
        "blend": str(BLEND_ASSET),
        "run_dir": str(run_dir),
        "preset": preset,
        "log": str(log_path),
        "blender": str(blender),
    }
