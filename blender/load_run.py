"""Load a Depth Ledger Blender export bundle into the current .blend.

Usage (Blender Scripting or CLI):

  blender assets/DepthLedger_v1.blend --python blender/load_run.py -- \\
    /path/to/outputs/RUN_ID

Expects:
  {run}_inv_smooth.exr (preferred) or {run}_depth.exr
  {run}_near_far.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy


def _argv_after_double_dash() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return []


def load_near_far(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_look_defaults(look_node, nf: dict, preset: str = "ledger") -> None:
    presets = nf.get("presets") or {}
    cfg = presets.get(preset) or nf
    if "Near_m" in look_node.inputs:
        look_node.inputs["Near_m"].default_value = float(cfg.get("near_m", 1.0))
    if "Far_m" in look_node.inputs:
        look_node.inputs["Far_m"].default_value = float(cfg.get("far_m", 10.0))
    if "Gamma" in look_node.inputs:
        look_node.inputs["Gamma"].default_value = float(cfg.get("gamma", 0.85))


def ensure_viewer_plate(mat: bpy.types.Material, img: bpy.types.Image) -> bpy.types.Object:
    """Guarantee a visible DL_Plate with the look material + camera view."""
    plate = bpy.data.objects.get("DL_Plate")
    if plate is None:
        bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
        plate = bpy.context.active_object
        plate.name = "DL_Plate"

    # Aspect-fit plane to image
    w = max(int(img.size[0]), 1)
    h = max(int(img.size[1]), 1)
    aspect = w / h
    if aspect >= 1.0:
        plate.scale = (aspect, 1.0, 1.0)
    else:
        plate.scale = (1.0, 1.0 / aspect, 1.0)

    if plate.data.materials:
        plate.data.materials[0] = mat
    else:
        plate.data.materials.append(mat)

    cam = bpy.data.objects.get("DL_Camera")
    if cam is None or cam.type != "CAMERA":
        cam_data = bpy.data.cameras.new("DL_Camera")
        cam_data.type = "ORTHO"
        cam = bpy.data.objects.new("DL_Camera", cam_data)
        bpy.context.scene.collection.objects.link(cam)
        cam.location = (0.0, 0.0, 3.0)
        cam.rotation_euler = (0.0, 0.0, 0.0)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 2.2 * max(plate.scale.x, plate.scale.y)
    bpy.context.scene.camera = cam

    # Select + activate plate
    bpy.ops.object.select_all(action="DESELECT")
    plate.select_set(True)
    bpy.context.view_layer.objects.active = plate

    # Material Preview + camera view in every 3D viewport
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type != "VIEW_3D":
                    continue
                space.shading.type = "MATERIAL"
                space.region_3d.view_perspective = "CAMERA"
            area.tag_redraw()

    # Also pin the EXR in any Image Editor so something is obviously loaded
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "IMAGE_EDITOR":
                for space in area.spaces:
                    if space.type == "IMAGE_EDITOR":
                        space.image = img

    return plate


def load_run(run_dir: Path, preset: str = "ledger") -> None:
    run_dir = Path(run_dir)
    run_id = run_dir.name
    nf_path = run_dir / f"{run_id}_near_far.json"
    inv_path = run_dir / f"{run_id}_inv_smooth.exr"
    depth_path = run_dir / f"{run_id}_depth.exr"
    exr = inv_path if inv_path.is_file() else depth_path
    if not exr.is_file():
        raise FileNotFoundError(f"No EXR in {run_dir} — run export/blender first")
    if not nf_path.is_file():
        raise FileNotFoundError(nf_path)

    nf = load_near_far(nf_path)
    img = bpy.data.images.load(str(exr), check_existing=True)
    try:
        img.colorspace_settings.name = "Non-Color"
    except Exception:
        pass
    img.reload()

    mat = bpy.data.materials.get("DL_Look_Material")
    if mat is None or mat.node_tree is None:
        raise RuntimeError("Open DepthLedger_v1.blend (missing DL_Look_Material)")

    tex = None
    look = None
    for n in mat.node_tree.nodes:
        if n.type == "TEX_IMAGE":
            tex = n
        if n.type == "GROUP" and n.node_tree and n.node_tree.name == "DL_Look":
            look = n
    if tex is None or look is None:
        raise RuntimeError("DL_Look_Material missing Image Texture or DL_Look group")

    tex.image = img
    apply_look_defaults(look, nf, preset=preset)
    ensure_viewer_plate(mat, img)

    # Kitchen look PNG as a second visual check (Image Editor / compositor)
    look_png = run_dir / f"{run_id}_look_{preset}.png"
    if not look_png.is_file():
        look_png = run_dir / f"look_{preset}_live.png"
    if look_png.is_file():
        ref = bpy.data.images.load(str(look_png), check_existing=True)
        ref.name = f"DL_Kitchen_{preset}"
        print(f"Also loaded Kitchen plate {look_png.name}")

    print(
        f"Loaded {exr.name} · preset={preset} "
        f"near={look.inputs['Near_m'].default_value:.3f} "
        f"far={look.inputs['Far_m'].default_value:.3f} "
        f"→ see DL_Plate in Material Preview (camera view)"
    )


def main() -> None:
    args = _argv_after_double_dash()
    if not args:
        print(
            "Usage: blender DepthLedger_v1.blend --python load_run.py -- "
            "/path/to/run_dir [preset]"
        )
        return
    run_dir = Path(args[0])
    preset = args[1] if len(args) > 1 else "ledger"
    load_run(run_dir, preset=preset)


if __name__ == "__main__":
    main()
