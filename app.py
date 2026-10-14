#!/usr/bin/env python3
"""Deprecated Gradio entrypoint — use the FastAPI + Vite app instead.

  Terminal 1:  PYTHONPATH=. uvicorn server.main:app --reload --port 8000
  Terminal 2:  cd web && npm run dev

Then open http://127.0.0.1:5173
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "Depth Ledger no longer uses Gradio.\n\n"
        "  # API (keeps the model warm; reloads only on Python changes)\n"
        "  PYTHONPATH=. uvicorn server.main:app --reload --port 8000\n\n"
        "  # UI (hot-reloads CSS / React — no model reboot)\n"
        "  cd web && npm install && npm run dev\n\n"
        "Open http://127.0.0.1:5173\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
