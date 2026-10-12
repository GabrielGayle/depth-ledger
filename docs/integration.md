# Integration with gabrielgaylesplat

This repo produces assets consumed by the **gabrielgaylesplat** test ground. The viewer stack uses **Spark 2.3.1** (`@sparkjsdev/spark`) on **three.js ≥ 0.180**, not PlayCanvas.

## Pipeline

1. **depth-ledger** infers metric depth (Depth Pro or PromptDA + Stray intrinsics).
2. A **DepthPaper bundle** (`rgb.jpg`, `depth_mm.png`, `camera.json`) feeds the displaced paper layer.
3. **`depth-ledger lift`** writes `splat.ply` and `splat.spz` for Spark’s splat loader (PLY/SPZ in, RDF coordinates).

## Contracts

| Asset | Spec |
|-------|------|
| DepthPaper bundle | [depthpaper-bundle.md](depthpaper-bundle.md) |
| Web splat delivery | `.spz` (preferred size); `.ply` for interchange |
| Spark | Load SPZ/PLY via Spark 2.x `SplatMesh` APIs |

## Typical commands

```bash
depth-ledger bundle photo.jpg -o public/assets/my_scene --model depth-pro --lift
```

Copy the output folder into the viewer’s asset path or reference URLs from look JSON.

## PromptDA / Stray

PromptDA bundles **require** `camera_matrix.csv` (pass `--camera-matrix`). Intrinsics are scaled to the inference resolution using the same 1920-max-dimension rule as the Gradio Stray zip path.
