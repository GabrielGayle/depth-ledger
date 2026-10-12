# DepthPaper bundle

A **DepthPaper bundle** is a single-directory interchange format between **depth-ledger** (metric depth inference) and **gabrielgaylesplat** (viewer / splat stack). One bundle corresponds to one RGB frame with aligned metric depth and pinhole intrinsics.

## Layout

```
my_scene/
  rgb.jpg          # 8-bit RGB, same resolution as depth
  depth_mm.png     # 16-bit PNG, metric depth in millimetres
  camera.json      # Pinhole intrinsics + metadata
```

Optional outputs from `depth-ledger lift` (not part of the core bundle contract):

```
my_scene/
  splat.ply        # Standard 3D Gaussian Splatting PLY (binary little-endian)
  splat.spz        # Niantic SPZ (same splat, RDF coordinates)
```

## `depth_mm.png`

- Single channel, unsigned 16-bit.
- Value `d` encodes depth **`d / 1000` metres** along the camera +Z axis.
- Invalid or unknown pixels should be `0` (treated as no depth).
- Values are clipped to **65535 mm** (~65.5 m) when written by depth-ledger.

## `camera.json`

| Field | Type | Description |
|-------|------|-------------|
| `format_version` | string | Schema version (currently `"1.0"`). |
| `width` | int | Image width in pixels (matches `rgb.jpg` / `depth_mm.png`). |
| `height` | int | Image height in pixels. |
| `fx` | float | Focal length in pixels (x). |
| `fy` | float | Focal length in pixels (y). |
| `cx` | float | Principal point x (pixels). |
| `cy` | float | Principal point y (pixels). |
| `units` | string | Scene / depth scale hint for consumers (`"millimeters"`). |
| `depth_units` | string | Always `"millimeters"` for `depth_mm.png`. |
| `focal_estimate_source` | string | How `fx`/`fy` were obtained (see below). |

### `focal_estimate_source`

| Value | Meaning |
|-------|---------|
| `exif` | Focal length from image EXIF (Depth Pro path). |
| `depth_pro_fov` | Depth Pro FOV head estimate. |
| `promptda_intrinsics` | PromptDA pipeline (placeholder: max side as focal). |
| `default_pinhole` | Fallback when no EXIF / FOV estimate exists. |

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
  "units": "millimeters",
  "depth_units": "millimeters",
  "focal_estimate_source": "depth_pro_fov"
}
```

## Coordinate convention

- **Camera**: OpenCV-style pinhole — **X right, Y down, Z forward** (metres in unprojection; millimetres in PNG).
- **3DGS PLY / SPZ from lift**: Written in **RDF** via [Niantic `spz`](https://github.com/nianticlabs/spz), matching common PLY splat loaders (`3dgs-loader` / `SpzLoader`, PlayCanvas `splat-transform`, etc.).

## CLI

```bash
# Write bundle only
depth-ledger bundle path/to/photo.jpg -o ./out/my_scene --model depth-pro

# Bundle + splat lift
depth-ledger bundle path/to/photo.jpg -o ./out/my_scene --model depth-pro --lift

# Lift an existing bundle
depth-ledger lift ./out/my_scene
```
