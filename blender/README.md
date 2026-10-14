# Depth Ledger ↔ Blender 5.2

Official Blender only: `/Applications/Blender.app` (5.2 LTS). Do not use broken custom 4.3 builds.

## Asset library

| File | Contents |
|------|----------|
| [`assets/DepthLedger_v1.blend`](assets/DepthLedger_v1.blend) | `DL_Prep`, `DL_Look`, `DL_Matte` (shader), `DL_Displace` (GN), `DL_Look_Material`, compositor `DL_Comp_Default` |
| [`build_assets.py`](build_assets.py) | Rebuild the `.blend` |
| [`recipe_mix_map.json`](recipe_mix_map.json) | Kitchen edge → Compositor Mix mapping |
| [`operators/infer_http.py`](operators/infer_http.py) | HTTP infer bridge (no torch in bpy) |
| [`load_run.py`](load_run.py) | Load API-exported EXR + `near_far.json` into the look material |

Rebuild:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background \
  --python blender/build_assets.py
```

## Data contract (from Depth Ledger API)

`POST /api/runs/{run_id}/export/blender` writes:

| Artifact | Meaning |
|----------|---------|
| `{run}_depth.exr` | float32 meters (RGB identical channels) |
| `{run}_inv_smooth.exr` | bilateral inverse-depth (preferred for `DL_Look`) |
| `{run}_near_far.json` | `near_m`, `far_m`, per-preset windows, cliff |
| `{run}_matte.exr` | scenery keep-matte (smart baked in Python) |
| `{run}_relief.png` | optional Weyrich bas-relief bake |

Prefer **inv EXR + JSON windows** — do not convert mm PNG inside Blender.

## Default kitchen in Compositor

`DL_Comp_Default` mirrors:

`Ledger → Soft Light@0.55 → Sockets → Screen@0.45 → Scene → File Output`

Load look plates (or render `DL_Look_Material` twice with ledger/sockets gamma) into `DL_Ledger` / `DL_Sockets` / `DL_Scene` image nodes.

## Infer from Blender (Option A — HTTP)

```bash
export DEPTH_LEDGER_URL=http://127.0.0.1:8000
# In Blender Scripting, run operators/infer_http.py
# or: blender --python blender/operators/infer_http.py -- /path/to/rgb.jpg
```

Requires the FastAPI server (`make api`). Never import `depth_core` / torch into `bpy`.

**Kitchen one-click:** with a run loaded, click **Open in Blender** in the topbar. That calls `POST /api/runs/{id}/open/blender` (export EXR/JSON + launch `DepthLedger_v1.blend` with `load_run.py`).

## MCP health

Add-on listens on `localhost:9876`. Status console probes TCP (+ optional deep `bpy` execute). Pin blender-mcp `mcp[cli]==1.9.4`.

## Non-goals (v1)

- Full Operator in Blender
- Are.na inside Blender
- torch-in-bpy
- Exact Poisson / smart CC matte in GN (API bake instead)
