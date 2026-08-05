# Depth Ledger

Local Gradio app for metric depth maps on Apple Silicon / CPU. Switch between:

1. **[Apple Depth Pro](https://github.com/apple/ml-depth-pro)** — RGB only (single photo). Preferred default; runs on **MPS** when available.
2. **[PromptDA](https://promptda.github.io/)** — RGB + LiDAR prompt depth (Stray Scanner / ARKit). Defaults to **CPU** (DINOv2 is MPS-fragile on many Macs).

UI is styled to the Wondrous Isles brand language (Courier, hard borders, ink primary).

## Models at a glance

| Model | Input | Device default |
|-------|--------|----------------|
| `depth-pro` | Single RGB image | MPS → CPU |
| `promptda-small` / `promptda-large` | RGB + uint16 mm prompt depth | CPU |

Depth Pro internally processes at **1536×1536**. Optional EXIF focal length improves metric scale; the model can estimate FOV if missing.

## Looks builder

After each generate, the app builds **seven graded previews** from the same metric depth (no re-infer):

| Look | Style |
|------|--------|
| Ledger gray | Default high-contrast gray |
| Magma | Warm colormap |
| Punchy near | Tighter near window, more contrast |
| Soft far | Wider window, softer midtones |
| Portrait | Near-field window + expand-near grade (faces) |
| Sockets | Tight near window + local contrast for eyes/nose |
| Scenery | Far-field only — subject punched out, buildings graded |

**Scenery matte** (live, no re-infer): choose **Smart** (depth-edge grow), **Soft** (depth gate), or **Protect** (classic dilate); dial **Protect** down to recover buildings next to the subject, **Feather** for edge softness, and optionally clean edge speckles.

Pick **Look A / B / C**, then stack B and C over A with separate Photoshop-style **blend modes** and **opacities**. Result = `blend(blend(A, B), C)`. **Download blend** exports the result. Metric millimetre PNG remains available for 3D tooling.

## Requirements

- Python 3.9–3.11 (3.10+ enables the optional depth image slider)
- ~2 GB free for the Depth Pro checkpoint
- For PromptDA captures: iPhone 12 Pro / iPad Pro (or later Pro) + [Stray Scanner](https://apps.apple.com/us/app/stray-scanner/id1557051662)
- Optional: `ffmpeg` for extracting frames from Stray Scanner `rgb.mp4` (OpenCV fallback)

## Setup

```bash
cd ~/Projects/promptda-depth-generator
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch torchvision
pip install -r requirements.txt
pip install -e ./PromptDA
pip install -e ./DepthPro
```

Download the Depth Pro weights (~1.8 GB) into `checkpoints/`:

```bash
mkdir -p checkpoints
curl -L -o checkpoints/depth_pro.pt \
  https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt
# or: bash DepthPro/get_pretrained_models.sh
```

Upstream PromptDA pins CUDA/`xformers`; this project’s `requirements.txt` skips those and uses a current PyTorch build that works on macOS.

## Run

```bash
source .venv/bin/activate
python app.py
```

Open http://127.0.0.1:7860

### Tabs

- **RGB image** — Depth Pro (single photo). Use the bundled example under PromptDA assets.
- **Stray Scanner zip** — first frame RGB; PromptDA also uses LiDAR depth; Depth Pro ignores depth.
- **RGB + prompt depth** — PromptDA only (uint16 mm PNG).

### Environment overrides

| Variable | Default | Meaning |
|----------|---------|---------|
| `PROMPTDA_DEVICE` | auto | Force `cpu`, `mps`, or `cuda` |
| `PROMPTDA_MODEL` | `depth-pro` | `depth-pro`, `promptda-small`, `promptda-large` (legacy: `small` / `large`) |
| `PROMPTDA_MAX_SIZE` | `756` | Max image side for PromptDA only |
| `PORT` | `7860` | Gradio port |

**Note:** PromptDA on MPS often hits a Metal NDArray assertion; the app keeps PromptDA on **CPU** unless you force a device. Depth Pro prefers **MPS** on Apple Silicon.

## Capture your own data (Stray Scanner)

1. Capture a scene in Stray Scanner (charging port facing down or right works best).
2. In the iOS Files app, compress the capture into a zip and AirDrop / copy it to this Mac.
3. In the UI, open **Stray Scanner zip**, upload the zip, and click **Generate depth**.

With PromptDA + `camera_matrix.csv`, the app also exports a `.ply` point cloud. Outputs land under `outputs/` as uint16 millimeter depth PNGs.

## Project layout

```
promptda-depth-generator/
  app.py                 # Gradio UI + model registry
  requirements.txt       # Mac-friendly deps
  PromptDA/              # Upstream PromptDA (editable install)
  DepthPro/              # Upstream Apple ml-depth-pro (editable install)
  checkpoints/           # depth_pro.pt (gitignored)
  outputs/               # Generated depth maps / PLYs
```

## License

This project's app code is **MIT** (see `LICENSE`). Bundled upstream projects keep their own terms:

- `DepthPro/` — Apple license
- `PromptDA/` — Apache 2.0

Model weights (`checkpoints/*.pt`) are **not** included; download them as described in Setup.
