.PHONY: api web install ffmpeg modal-setup modal-deploy blender-assets boot up down restart status boot-fast

# Prefer project venv when present (avoids "command not found: pip/modal")
VENV := $(wildcard .venv/bin)
PY := $(if $(VENV),$(VENV)/python,python3)
PIP := $(if $(VENV),$(VENV)/pip,pip3)
MODAL := $(if $(VENV),$(VENV)/modal,modal)
UVICORN := $(if $(VENV),$(VENV)/uvicorn,uvicorn)
BLENDER ?= /Applications/Blender.app/Contents/MacOS/Blender
DEV := ./scripts/dev.sh

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e ./DepthPro
	$(PIP) install -e ./PromptDA
	cd web && npm install

# Prefer system ffmpeg; imageio-ffmpeg (in requirements) is the fallback for MP4 export.
ffmpeg:
	@command -v ffmpeg >/dev/null 2>&1 && echo "ffmpeg already on PATH" || ( \
		command -v brew >/dev/null 2>&1 && brew install ffmpeg || \
		echo "No system ffmpeg — MP4 export will use imageio-ffmpeg from the venv" \
	)

# One-command local stack (frees stale ports, starts API + Vite, waits until healthy).
boot up:
	@$(DEV) boot

# Skip Depth Pro warmup — faster after API/UI edits; model loads on first generate.
boot-fast:
	@DEPTH_LEDGER_SKIP_WARMUP=1 $(DEV) boot

restart:
	@$(DEV) restart

down:
	@$(DEV) down

status:
	@$(DEV) status

api:
	PYTHONPATH=. $(UVICORN) server.main:app --reload --reload-dir server --reload-dir depth_core --reload-dir modal_app --host 127.0.0.1 --port 8000

web:
	cd web && npm run dev -- --host 127.0.0.1 --port 5173

# One-time Modal auth (opens browser). Must run in your terminal — not headless.
modal-setup:
	$(MODAL) setup

# Deploy DepthWorker to Modal (run `make modal-setup` first). First build is slow.
modal-deploy:
	$(MODAL) deploy modal_app/worker.py

# Rebuild Blender 5.2 Depth Ledger asset library (official Blender.app only).
blender-assets:
	$(BLENDER) --background --python blender/build_assets.py
