# Depth Ledger

Local metric depth kitchen — [Apple Depth Pro](https://github.com/apple/ml-depth-pro)
or [PromptDA](https://promptda.github.io/) (RGB + LiDAR) — with a **node-based**
looks mixer (React Flow), an Ableton-inspired **Operator** synthesizer mode, and
a FastAPI backend.

Infer once, then tweak Looks / Scene / Blend nodes **or** Operator oscillators
without re-running the model. The Vite UI hot-reloads independently so CSS and
React changes do **not** reboot the model server.

## Architecture

| Process | Command | Port |
|---------|---------|------|
| API + models | `PYTHONPATH=. uvicorn server.main:app --reload --port 8000` | 8000 |
| UI (HMR) | `cd web && npm run dev` | 5173 (proxies `/api` → 8000) |

## Models

| Model | Input | Device default |
|-------|--------|----------------|
| `depth-pro` | Single RGB image **or RGB video** | MPS → CPU |
| `promptda-small` / `promptda-large` | RGB + uint16 mm prompt depth | CPU |

### Video

**Generate Video** uploads mp4/mov/webm, samples ~10 fps (cap **60 frames** /
~**30s**), runs **Depth Pro** per frame. On **local**, the kitchen opens after
the first frame so you can scrub/tweak while the rest fills in. **Modal** still
batches the full clip before opening. Video runs load:

`Video In → Depth → Smooth → Look(Ledger) → Look(Sockets) → Scene → Video Out`

Load alternate stacks from the rail **Video setup** menu (EMA / Flow /
Freq Fuse stub). Frames are resized so the longest edge ≤ **Max** (header /
Video Out; up to **1536**). **Smooth** stabilizes flicker before looks:
locked/spline windows, EMA, or flow-guided fusion (Farneback + occlusion mask).
**Freq Fuse** blends a stub temporal video-depth base with Depth Pro
high-frequency edges. **Video Out** owns Infer, scrub / filmstrip / Export MP4,
and per-frame status. **Video In** shows a short RGB preview GIF.
**Local** infer is sequential; **Modal** runs frames in parallel.
PromptDA is not supported for plain RGB video.

## Node kitchen

Default graph after generate:

`Depth → Look(Ledger) → Look(Sockets) → Scene → Output`

Blend mode / opacity live **on the wire** between image nodes. Click a
connection pill to open the blend modal. Look / Scene take a single input
(depth or another plate). Select a node (blue) or wire, then Backspace to
remove it.

| Node | Role |
|------|------|
| **Depth** | Cached metric depth for the run |
| **Video In** | Load / prepare clip + short RGB GIF preview; Infer |
| **Video Out** | Video plate sink — scrub, filmstrip, Export MP4 |
| **Smooth** | Video temporal / window lock (before looks) |
| **Freq Fuse** | Stub video low-freq + Depth Pro edges (testable #3) |
| **Look** | Preset grade + window/gamma (one input) |
| **Scene** | Background matte plate (one input) |
| **Blend** | Optional standalone mixer (`base` / `overlay`) |
| **Output** | Still final preview |

Node thumbs are ~128px for snappy live updates; the output plate stays full-res.

## Operator mode

Topbar **Mode → Operator** opens a separate synthesizer UI (not the node graph).
Same infer / `run_id` substrate as the kitchen.

| Section | Role |
|---------|------|
| **Oscillators A–D** | Each is a **Look** (preset, window, gamma, cmap, level) on the shared depth |
| **Algorithms 1–11** | Ableton-style FM / additive routing between oscillators |
| **Filter / Shaper** | Spatial lowpass, highpass/formant, drive, curve remaps |
| **Envelope / LFO** | Modulate levels, cutoff, windows, spread over time `t` (video frames) |
| **Global** | Time scalar, Z **spread**, multi-voice outs |

**Voices:** plate (`operator_live.png`), optional PLY point cloud, vector-field
PNG/NPZ, JPEG/PNG stream at `/api/runs/{id}/operator/stream`. Set `OPERATOR_NDI=1`
for a best-effort NDI send when the `ndi` voice is enabled (Spout/Syphon not
included).

Operator defaults persist in `outputs/_default_operator.json` (Save default in
the Operator rail). Video **Export MP4** bakes the patch with per-frame `t`.

## Requirements

- Python 3.9–3.11
- Node.js 18+ (for the web UI)
- ~2 GB free for the Depth Pro checkpoint
- Optional: `ffmpeg` for Stray Scanner `rgb.mp4` frames and **video MP4 export**

## Setup

```bash
cd ~/Projects/promptda-depth-generator   # or your clone path
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -e ./DepthPro
pip install -e ./PromptDA

# Depth Pro weights (~1.8 GB)
mkdir -p checkpoints
# see DepthPro/get_pretrained_models.sh or download depth_pro.pt into checkpoints/

cd web && npm install && cd ..
```

## Run

One command (frees stale `:8000` / `:5173`, starts API + Vite, waits until healthy):

```bash
make boot          # or: make restart / make down / make status
make boot-fast     # skip Depth Pro warmup (loads on first generate)
```

Open **http://127.0.0.1:5173** — logs live in `.logs/`.

Or two terminals (same as before):

```bash
make api    # uvicorn --reload :8000
make web    # Vite HMR :5173
```

**Save as default** stores the current node layout (positions, wires, blend
settings) in `outputs/_default_graph.json` and uses it for every new generate.
**Reset default** clears it back to the built-in stack.

## Modal (remote GPU infer)

Depth Ledger can run the **neural forward pass** on Modal (T4 GPU) while looks,
scenery, graph eval, and Are.na stay local.

1. Create a [Modal](https://modal.com) account and authenticate **from the project venv**:

```bash
source .venv/bin/activate   # if you haven't already
make modal-setup            # opens browser — or: .venv/bin/modal setup
```

If you see `command not found: pip` or `modal`, you're not in the venv. Either
activate it, or call tools directly:

```bash
.venv/bin/pip install modal
.venv/bin/modal setup
```

2. Deploy the worker (first build downloads torch + checkpoints — expect several minutes):

```bash
make modal-deploy
# or: .venv/bin/modal deploy modal_app/worker.py
```

3. In the UI topbar, set **Infer → Modal**. A rough **~$** ticker shows estimated
GPU cost for the last call and the session total (list rates × GPU seconds —
not an exact invoice).

Or set the server default:

```bash
# .env
INFER_BACKEND=modal
```

**When to use Modal:** PromptDA (slow on Mac CPU), Are.na batch rolls, video
frames, or any time you want GPU without a local NVIDIA card.

**When to stay Local:** quick Depth Pro stills on Apple Silicon MPS, offline
dev, or before the Modal app is deployed.

Warm the remote worker:

```bash
curl -X POST 'http://127.0.0.1:8000/api/warmup?infer_backend=modal'
```

## Status console

Topbar strip polls `GET /api/status` (Modal · Local · UI · API · Blender MCP). UI heartbeats via `POST /api/status/ui`. Blender add-on is probed on `localhost:9876` (TCP + optional deep `bpy`).

## Blender (Geometry Nodes / Shader / Compositor)

See [`blender/README.md`](blender/README.md). Rebuild assets with `make blender-assets` (official Blender 5.2 only). Export a run for Blender with `POST /api/runs/{run_id}/export/blender` (float meters EXR + `near_far.json`). Infer from Blender/Cursor via HTTP (`blender/operators/infer_http.py`) — never torch-in-bpy.

## Working cache

Published plates and kitchen recipes live on **Are.na**. `outputs/` is a bounded
scratch pad, not an archive:

| Knob | Default | Role |
|------|---------|------|
| `DEPTH_LEDGER_DISK_GB` | `3` | Cap for `outputs/` (run dirs + infer cache) |
| `DEPTH_LEDGER_INFER_CACHE_GB` | `1.5` | Content-addressed depth maps (Are.na block + model + size) |
| `DEPTH_LEDGER_RAM_RUNS` | `2` | Runs that keep depth/RGB arrays in process memory |
| `DEPTH_LEDGER_DOWNLOAD_WORKERS` | `8` | Parallel Are.na still downloads |
| `DEPTH_LEDGER_COOK_WORKERS` | CPU count (max 8) | Parallel kitchen eval + Are.na publish |
| `DEPTH_LEDGER_INFER_WORKERS` | `1` | Local neural forwards (MPS/GPU stay serialized) |

Repeat rolls of the same Are.na block skip the GPU. Batch download, infer, and
cook overlap. Status drawer shows disk / RAM / cache hits. `POST /api/cache/prune`
enforces the budget immediately.

## Project layout

```
depth_core/     # models, looks, blend, scenery, infer, video, recipes, blender_export
modal_app/      # Modal DepthWorker (deployable GPU image)
server/         # FastAPI routes + graph evaluator + batch jobs + status
web/            # Vite + React + React Flow UI
blender/        # Blender 5.2 asset library + HTTP infer bridge
DepthPro/       # upstream Apple ml-depth-pro
PromptDA/       # upstream PromptDA
checkpoints/    # depth_pro.pt (gitignored)
outputs/        # working cache (runs + `_infer/`); Are.na is the archive
```

## License

App code is **MIT** (see `LICENSE`). Upstream:

- `DepthPro/` — Apple license
- `PromptDA/` — Apache 2.0

Model weights are not included.
