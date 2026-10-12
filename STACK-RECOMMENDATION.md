# Stack recommendation: depth-ledger ↔ gabrielgaylesplat

This document records the integration contract between **depth-ledger** (this repo) and **gabrielgaylesplat** (private viewer). Follow it when wiring capture → depth → splat → viewer.

## Recommended pipeline

1. **Metric depth** — `depth-ledger` CLI or Python API (`depth_ledger.run_metric_depth`).
2. **DepthPaper bundle** — folder with `rgb.jpg`, `depth_mm.png`, `camera.json` ([schema](docs/depthpaper-bundle.md)).
3. **Lift (optional)** — `depth-ledger lift` → `splat.ply` + `splat.spz` for the splat viewer path.
4. **Viewer** — gabrielgaylesplat loads **`.ply`** and **`.spz`** via its splat loader stack (Niantic SPZ / standard 3DGS PLY).

## Viewer loader alignment

gabrielgaylesplat should consume:

| Asset | Format | Notes |
|-------|--------|--------|
| DepthPaper bundle | `docs/depthpaper-bundle.md` | For depth-backed UI, reprojection, or future single-image tools. |
| `splat.ply` | 3DGS binary PLY | Written by `spz.save_splat_to_ply`, RDF coordinates. |
| `splat.spz` | Niantic SPZ v4 | Same splat; preferred for size. Load with `SpzLoader` / `@playcanvas/splat-transform` / `spz.load_spz`. |

Do **not** depend on Gradio `outputs/` naming (`*_depth_mm.png`, recipe JSON, etc.) for the viewer — use the **DepthPaper** filenames only.

## Python module boundaries

| Module | Responsibility |
|--------|----------------|
| `depth_ledger.models` | Depth Pro / PromptDA inference, device selection. |
| `depth_ledger.bundle` | DepthPaper read/write. |
| `depth_ledger.lift` | RGB-D → GaussianCloud → `.ply` / `.spz`. |
| `app.py` | Gradio UI only; imports `depth_ledger` for inference. |

## Dependencies

- **Inference**: `torch`, editable `DepthPro` + `PromptDA`, `checkpoints/depth_pro.pt`.
- **Lift**: optional `spz` (`pip install 'git+https://github.com/nianticlabs/spz.git'` or `pip install -e ".[lift]"`).

## Example end-to-end

```bash
source .venv/bin/activate
pip install -e . -e ./PromptDA -e ./DepthPro
pip install 'git+https://github.com/nianticlabs/spz.git'

depth-ledger bundle PromptDA/assets/example_images/image.jpg \
  -o ./viewer_drop/example \
  --model depth-pro \
  --lift
```

Copy `./viewer_drop/example/` (bundle + `splat.ply` + `splat.spz`) into gabrielgaylesplat’s asset intake or point the viewer at those paths.
