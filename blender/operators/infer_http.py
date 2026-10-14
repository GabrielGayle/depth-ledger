"""HTTP infer bridge: Blender / Cursor → Depth Ledger FastAPI (no torch in bpy).

Env:
  DEPTH_LEDGER_URL  default http://127.0.0.1:8000
  DEPTH_LEDGER_INFER_BACKEND  local|modal  (optional form field)

CLI:
  blender --background --python blender/operators/infer_http.py -- \\
    /path/to/rgb.jpg [model_key] [max_size]

Or call from bpy:

  from pathlib import Path
  import importlib.util
  ...
  result = infer_rgb(Path('photo.jpg'))
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional


def _ledger_url() -> str:
    return os.environ.get("DEPTH_LEDGER_URL", "http://127.0.0.1:8000").rstrip("/")


def _argv_after_double_dash() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return []


def _multipart(
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = "----DepthLedgerBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(f"{value}\r\n".encode())
    for name, (filename, data, content_type) in files.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\n'
            ).encode()
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.extend(data)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _http_json(
    method: str,
    url: str,
    *,
    data: Optional[bytes] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 600.0,
) -> Any:
    req = urllib.request.Request(url, data=data, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {url}: {detail}") from e


def warmup(
    *,
    model_key: str = "depth-pro",
    infer_backend: Optional[str] = None,
) -> dict[str, Any]:
    backend = infer_backend or os.environ.get("DEPTH_LEDGER_INFER_BACKEND")
    q = f"model_key={model_key}"
    if backend:
        q += f"&infer_backend={backend}"
    return _http_json("POST", f"{_ledger_url()}/api/warmup?{q}", timeout=120.0)


def infer_rgb(
    image_path: Path,
    *,
    model_key: str = "depth-pro",
    max_size: int = 756,
    infer_backend: Optional[str] = None,
) -> dict[str, Any]:
    image_path = Path(image_path)
    data = image_path.read_bytes()
    fields = {
        "model_key": model_key,
        "max_size": str(int(max_size)),
    }
    backend = infer_backend or os.environ.get("DEPTH_LEDGER_INFER_BACKEND")
    if backend:
        fields["infer_backend"] = backend
    body, content_type = _multipart(
        fields,
        {
            "file": (
                image_path.name,
                data,
                "application/octet-stream",
            )
        },
    )
    return _http_json(
        "POST",
        f"{_ledger_url()}/api/infer/rgb",
        data=body,
        headers={"Content-Type": content_type},
        timeout=900.0,
    )


def export_blender(run_id: str, **kwargs: Any) -> dict[str, Any]:
    return _http_json(
        "POST",
        f"{_ledger_url()}/api/runs/{run_id}/export/blender",
        data=json.dumps(kwargs or {}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        timeout=120.0,
    )


def download_file(url_path: str, dest: Path) -> Path:
    url = url_path if url_path.startswith("http") else f"{_ledger_url()}{url_path}"
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120.0) as resp:
        dest.write_bytes(resp.read())
    return dest


def infer_and_export(
    image_path: Path,
    out_dir: Path,
    *,
    model_key: str = "depth-pro",
    max_size: int = 756,
    infer_backend: Optional[str] = None,
) -> dict[str, Any]:
    """Infer → export blender bundle → download EXR/JSON into out_dir."""
    print(f"Warmup {_ledger_url()} …")
    try:
        warmup(model_key=model_key, infer_backend=infer_backend)
    except Exception as exc:
        print(f"Warmup warning: {exc}")

    print(f"Infer {image_path} …")
    result = infer_rgb(
        image_path,
        model_key=model_key,
        max_size=max_size,
        infer_backend=infer_backend,
    )
    run_id = result["run_id"]
    print(f"run_id={run_id} — exporting blender bundle…")
    bundle = export_blender(run_id, write_inv=True, write_matte=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {"run_id": run_id}
    for key in (
        "depth_exr_url",
        "inv_smooth_exr_url",
        "near_far_url",
        "matte_exr_url",
        "rgb_url",
    ):
        url = bundle.get(key)
        if not url:
            continue
        name = Path(str(url)).name
        # urls look like /api/runs/ID/files/NAME
        parts = str(url).rstrip("/").split("/")
        name = parts[-1]
        path = download_file(str(url), out_dir / name)
        saved[key] = str(path)
    (out_dir / "infer_result.json").write_text(
        json.dumps({"infer": result, "export": bundle, "saved": saved}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved to {out_dir}")
    return {"infer": result, "export": bundle, "saved": saved}


def main() -> None:
    args = _argv_after_double_dash()
    if not args:
        print(
            "Usage: blender --python infer_http.py -- /path/to/rgb.jpg "
            "[model_key] [max_size] [out_dir]"
        )
        return
    image = Path(args[0])
    model = args[1] if len(args) > 1 else "depth-pro"
    max_size = int(args[2]) if len(args) > 2 else 756
    out_dir = Path(args[3]) if len(args) > 3 else image.parent / f"{image.stem}_dl"
    infer_and_export(image, out_dir, model_key=model, max_size=max_size)


if __name__ == "__main__":
    main()
