# DepthPaper bundle

A **DepthPaper bundle** is a single-directory interchange format between **depth-ledger** and **gabrielgaylesplat** (Spark viewer). One bundle is one RGB frame with aligned metric depth and pinhole intrinsics.

## Layout

```
my_scene/
  rgb.jpg          # 8-bit RGB, same resolution as depth
  depth_mm.png     # 16-bit PNG, metric depth in millimetres
  camera.json      # Pinhole intrinsics + metadata
```

Optional outputs from `depth-ledger lift`:

```
my_scene/
  splat.ply        # Standard 3DGS binary PLY (RDF)
  splat.spz        # Niantic SPZ for Spark web delivery
```

## `depth_mm.png`

- Single channel, unsigned 16-bit.
- Value `d` encodes depth **`d / 1000` metres** along camera +Z.
- Invalid pixels: `0`.
- Clipped to **65535 mm** when written.

## `camera.json`

| Field | Type | Description |
|-------|------|-------------|
| `format_version` | string | `"1.0"` |
| `width`, `height` | int | Pixel size (matches rgb + depth) |
| `fx`, `fy`, `cx`, `cy` | float | Pinhole intrinsics (pixels) |
| `f_px` | float | Focal length in pixels (equals `fx` for square pixels) |
| `near_m`, `far_m` | float | Scene depth window (metres) from valid depth percentiles |
| `world_units` | string | `"meters"` for 3D/unprojection |
| `depth_units` | string | `"millimeters"` for `depth_mm.png` |
| `focal_estimate_source` | string | `exif` \| `depth_pro_fov` \| `camera_matrix` \| `default_pinhole` |

### Example

```json
{
  "format_version": "1.0",
  "width": 1536,
  "height": 1536,
  "fx": 1420.5,
  "fy": 1420.5,
  "cx": 767.5,
  "cy": 767.5,
  "f_px": 1420.5,
  "near_m": 0.42,
  "far_m": 12.8,
  "world_units": "meters",
  "depth_units": "millimeters",
  "focal_estimate_source": "depth_pro_fov"
}
```

## Coordinates

- **Camera**: OpenCV pinhole — X right, Y down, Z forward (metres when unprojecting).
- **Lift output**: RDF via Niantic `spz`, compatible with **Spark 2.3.1** SPZ/PLY loading.

## CLI

```bash
depth-ledger bundle path/to/photo.jpg -o ./out/my_scene --model depth-pro
depth-ledger bundle path/to/photo.jpg -o ./out/my_scene --model depth-pro --lift
depth-ledger lift ./out/my_scene
```

PromptDA:

```bash
depth-ledger bundle rgb.jpg -o ./out/scene --model promptda-small \
  --prompt-depth depth.png --camera-matrix camera_matrix.csv
```
