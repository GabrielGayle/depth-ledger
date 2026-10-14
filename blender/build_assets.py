"""Build Depth Ledger Blender 5.2 asset library (shader + compositor + GN).

Run:
  /Applications/Blender.app/Contents/MacOS/Blender --background \\
    --python blender/build_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets" / "DepthLedger_v1.blend"

# Tent math (matches depth_core.pipeline._visualize_inverse_depth):
#   peak = 1/near; lo = 1/far; span = peak - lo
#   far_side (inv <= peak): (inv - lo) / span
#   near_side: 1 - (inv - peak) / span
#   clip 0..1 → power(gamma)


def _clear_orphans() -> None:
    for block in list(bpy.data.node_groups):
        if block.name.startswith("DL_"):
            bpy.data.node_groups.remove(block)


def _new_shader_group(name: str) -> bpy.types.ShaderNodeTree:
    ng = bpy.data.node_groups.new(name, "ShaderNodeTree")
    return ng


def _sock(tree, in_out: str, sock_type: str, name: str, **kwargs):
    """Blender 4+/5 interface sockets."""
    iface = tree.interface
    item = iface.new_socket(name=name, in_out=in_out, socket_type=sock_type)
    for k, v in kwargs.items():
        if hasattr(item, k):
            setattr(item, k, v)
    return item


def build_dl_depth_source() -> bpy.types.ShaderNodeTree:
    """DL_DepthSource: pass-through Image → Float (R channel as meters/inv)."""
    ng = _new_shader_group("DL_DepthSource")
    _sock(ng, "INPUT", "NodeSocketColor", "Image")
    _sock(ng, "OUTPUT", "NodeSocketFloat", "Value")
    _sock(ng, "OUTPUT", "NodeSocketColor", "Color")
    nodes = ng.nodes
    links = ng.links
    nin = nodes.new("NodeGroupInput")
    nout = nodes.new("NodeGroupOutput")
    nin.location = (-300, 0)
    nout.location = (200, 0)
    sep = nodes.new("ShaderNodeSeparateColor")
    sep.location = (-40, 0)
    links.new(nin.outputs["Image"], sep.inputs["Color"])
    links.new(sep.outputs["Red"], nout.inputs["Value"])
    links.new(nin.outputs["Image"], nout.inputs["Color"])
    return ng


def build_dl_prep() -> bpy.types.ShaderNodeTree:
    """DL_Prep: Near_m / Far_m → peak / lo / span for tent."""
    ng = _new_shader_group("DL_Prep")
    _sock(ng, "INPUT", "NodeSocketFloat", "Near_m", default_value=1.0, min_value=0.01)
    _sock(ng, "INPUT", "NodeSocketFloat", "Far_m", default_value=10.0, min_value=0.02)
    _sock(ng, "OUTPUT", "NodeSocketFloat", "Peak")
    _sock(ng, "OUTPUT", "NodeSocketFloat", "Lo")
    _sock(ng, "OUTPUT", "NodeSocketFloat", "Span")

    nodes = ng.nodes
    links = ng.links
    nin = nodes.new("NodeGroupInput")
    nout = nodes.new("NodeGroupOutput")
    nin.location = (-400, 0)
    nout.location = (400, 0)

    # peak = 1 / Near
    inv_near = nodes.new("ShaderNodeMath")
    inv_near.operation = "DIVIDE"
    inv_near.inputs[0].default_value = 1.0
    inv_near.location = (-100, 120)

    inv_far = nodes.new("ShaderNodeMath")
    inv_far.operation = "DIVIDE"
    inv_far.inputs[0].default_value = 1.0
    inv_far.location = (-100, -40)

    span = nodes.new("ShaderNodeMath")
    span.operation = "SUBTRACT"
    span.location = (120, 40)

    # max(span, 1e-12)
    span_max = nodes.new("ShaderNodeMath")
    span_max.operation = "MAXIMUM"
    span_max.inputs[1].default_value = 1e-12
    span_max.location = (280, 40)

    links.new(nin.outputs["Near_m"], inv_near.inputs[1])
    links.new(nin.outputs["Far_m"], inv_far.inputs[1])
    links.new(inv_near.outputs[0], span.inputs[0])
    links.new(inv_far.outputs[0], span.inputs[1])
    links.new(span.outputs[0], span_max.inputs[0])
    links.new(inv_near.outputs[0], nout.inputs["Peak"])
    links.new(inv_far.outputs[0], nout.inputs["Lo"])
    links.new(span_max.outputs[0], nout.inputs["Span"])
    return ng


def build_dl_look() -> bpy.types.ShaderNodeTree:
    """DL_Look: Inv + Near/Far + Gamma → gray tent plate (cmap gray)."""
    ng = _new_shader_group("DL_Look")
    _sock(ng, "INPUT", "NodeSocketFloat", "Inv", default_value=0.5)
    _sock(ng, "INPUT", "NodeSocketFloat", "Near_m", default_value=1.0, min_value=0.01)
    _sock(ng, "INPUT", "NodeSocketFloat", "Far_m", default_value=10.0, min_value=0.02)
    _sock(ng, "INPUT", "NodeSocketFloat", "Gamma", default_value=0.85, min_value=0.05)
    _sock(ng, "OUTPUT", "NodeSocketColor", "Look")
    _sock(ng, "OUTPUT", "NodeSocketFloat", "Norm")

    nodes = ng.nodes
    links = ng.links
    nin = nodes.new("NodeGroupInput")
    nout = nodes.new("NodeGroupOutput")
    nin.location = (-900, 0)
    nout.location = (900, 0)

    # Embed Prep
    prep = nodes.new("ShaderNodeGroup")
    prep.node_tree = bpy.data.node_groups["DL_Prep"]
    prep.location = (-650, -80)
    links.new(nin.outputs["Near_m"], prep.inputs["Near_m"])
    links.new(nin.outputs["Far_m"], prep.inputs["Far_m"])

    inv = nin.outputs["Inv"]
    peak = prep.outputs["Peak"]
    lo = prep.outputs["Lo"]
    span = prep.outputs["Span"]

    # far_side value: (inv - lo) / span
    inv_minus_lo = nodes.new("ShaderNodeMath")
    inv_minus_lo.operation = "SUBTRACT"
    inv_minus_lo.location = (-400, 160)
    links.new(inv, inv_minus_lo.inputs[0])
    links.new(lo, inv_minus_lo.inputs[1])

    far_side = nodes.new("ShaderNodeMath")
    far_side.operation = "DIVIDE"
    far_side.location = (-200, 160)
    links.new(inv_minus_lo.outputs[0], far_side.inputs[0])
    links.new(span, far_side.inputs[1])

    # near_side: 1 - (inv - peak) / span
    inv_minus_peak = nodes.new("ShaderNodeMath")
    inv_minus_peak.operation = "SUBTRACT"
    inv_minus_peak.location = (-400, -40)
    links.new(inv, inv_minus_peak.inputs[0])
    links.new(peak, inv_minus_peak.inputs[1])

    near_div = nodes.new("ShaderNodeMath")
    near_div.operation = "DIVIDE"
    near_div.location = (-200, -40)
    links.new(inv_minus_peak.outputs[0], near_div.inputs[0])
    links.new(span, near_div.inputs[1])

    near_side = nodes.new("ShaderNodeMath")
    near_side.operation = "SUBTRACT"
    near_side.inputs[0].default_value = 1.0
    near_side.location = (0, -40)
    links.new(near_div.outputs[0], near_side.inputs[1])

    # mix: if inv <= peak use far_side else near_side
    # compare: inv <= peak → 1
    cmp = nodes.new("ShaderNodeMath")
    cmp.operation = "LESS_THAN"  # returns 1 if inv < peak; close enough with EQUAL band
    cmp.location = (-200, 320)
    # Use: 1 if inv <= peak ≈ NOT (inv > peak)
    # Blender LESS_EQUAL not always available; use: compare inv - peak
    # Actually use Map Range or Mix with factor from Greater Than
    gt = nodes.new("ShaderNodeMath")
    gt.operation = "GREATER_THAN"
    gt.location = (-200, 280)
    links.new(inv, gt.inputs[0])
    links.new(peak, gt.inputs[1])

    # fac = gt → 1 means near_side
    mix = nodes.new("ShaderNodeMath")
    mix.operation = "MULTIPLY"  # placeholder; use MixRGB/Mix float
    # ShaderNodeMix (Blender 3.4+) float mode
    mixf = nodes.new("ShaderNodeMix")
    mixf.data_type = "FLOAT"
    mixf.location = (120, 80)
    links.new(gt.outputs[0], mixf.inputs["Factor"])
    links.new(far_side.outputs[0], mixf.inputs["A"])
    links.new(near_side.outputs[0], mixf.inputs["B"])

    # clip 0..1
    clip_min = nodes.new("ShaderNodeMath")
    clip_min.operation = "MAXIMUM"
    clip_min.inputs[1].default_value = 0.0
    clip_min.location = (320, 80)
    links.new(mixf.outputs["Result"], clip_min.inputs[0])

    clip_max = nodes.new("ShaderNodeMath")
    clip_max.operation = "MINIMUM"
    clip_max.inputs[1].default_value = 1.0
    clip_max.location = (480, 80)
    links.new(clip_min.outputs[0], clip_max.inputs[0])

    # gamma: power
    pow_n = nodes.new("ShaderNodeMath")
    pow_n.operation = "POWER"
    pow_n.location = (640, 80)
    links.new(clip_max.outputs[0], pow_n.inputs[0])
    links.new(nin.outputs["Gamma"], pow_n.inputs[1])

    # to color
    comb = nodes.new("ShaderNodeCombineColor")
    comb.mode = "RGB"
    comb.location = (780, 40)
    links.new(pow_n.outputs[0], comb.inputs["Red"])
    links.new(pow_n.outputs[0], comb.inputs["Green"])
    links.new(pow_n.outputs[0], comb.inputs["Blue"])

    links.new(comb.outputs["Color"], nout.inputs["Look"])
    links.new(pow_n.outputs[0], nout.inputs["Norm"])

    # Remove unused placeholder
    nodes.remove(mix)
    nodes.remove(cmp)
    return ng


def build_dl_matte() -> bpy.types.ShaderNodeTree:
    """DL_Matte: soft depth gate — Depth vs Cut/Near with protect/feather approx."""
    ng = _new_shader_group("DL_Matte")
    _sock(ng, "INPUT", "NodeSocketFloat", "Depth", default_value=2.0)
    _sock(ng, "INPUT", "NodeSocketFloat", "Cut_m", default_value=1.5)
    _sock(ng, "INPUT", "NodeSocketFloat", "Near_m", default_value=3.0)
    _sock(ng, "INPUT", "NodeSocketFloat", "Protect", default_value=0.22)
    _sock(ng, "INPUT", "NodeSocketFloat", "BakedKeep", default_value=1.0)
    _sock(ng, "INPUT", "NodeSocketFloat", "UseBaked", default_value=0.0)
    _sock(ng, "OUTPUT", "NodeSocketFloat", "Keep")

    nodes = ng.nodes
    links = ng.links
    nin = nodes.new("NodeGroupInput")
    nout = nodes.new("NodeGroupOutput")
    nin.location = (-600, 0)
    nout.location = (500, 0)

    # t = (depth - cut) / max(near - cut, 1e-3)
    d_cut = nodes.new("ShaderNodeMath")
    d_cut.operation = "SUBTRACT"
    d_cut.location = (-350, 120)
    links.new(nin.outputs["Depth"], d_cut.inputs[0])
    links.new(nin.outputs["Cut_m"], d_cut.inputs[1])

    n_cut = nodes.new("ShaderNodeMath")
    n_cut.operation = "SUBTRACT"
    n_cut.location = (-350, -40)
    links.new(nin.outputs["Near_m"], n_cut.inputs[0])
    links.new(nin.outputs["Cut_m"], n_cut.inputs[1])

    span = nodes.new("ShaderNodeMath")
    span.operation = "MAXIMUM"
    span.inputs[1].default_value = 1e-3
    span.location = (-180, -40)
    links.new(n_cut.outputs[0], span.inputs[0])

    t = nodes.new("ShaderNodeMath")
    t.operation = "DIVIDE"
    t.location = (-20, 80)
    links.new(d_cut.outputs[0], t.inputs[0])
    links.new(span.outputs[0], t.inputs[1])

    tclip = nodes.new("ShaderNodeClamp")
    tclip.location = (140, 80)
    links.new(t.outputs[0], tclip.inputs["Value"])

    # gate_kill ≈ (1-t)^(0.45+1.6*protect) — simplify power≈1 for soft
    one_minus = nodes.new("ShaderNodeMath")
    one_minus.operation = "SUBTRACT"
    one_minus.inputs[0].default_value = 1.0
    one_minus.location = (300, 120)
    links.new(tclip.outputs["Result"], one_minus.inputs[1])

    # keep = 1 - gate_kill ≈ tclip for soft approx, then mix with baked
    keep_soft = tclip  # soft keep ≈ t (farther → keep more)

    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "FLOAT"
    mix.location = (360, -40)
    links.new(nin.outputs["UseBaked"], mix.inputs["Factor"])
    links.new(keep_soft.outputs["Result"], mix.inputs["A"])
    links.new(nin.outputs["BakedKeep"], mix.inputs["B"])
    links.new(mix.outputs["Result"], nout.inputs["Keep"])
    return ng


def build_dl_displace() -> bpy.types.GeometryNodeTree:
    """DL_Displace: Set Position from gray height (creative 3D)."""
    ng = bpy.data.node_groups.new("DL_Displace", "GeometryNodeTree")
    _sock(ng, "INPUT", "NodeSocketGeometry", "Geometry")
    _sock(ng, "INPUT", "NodeSocketFloat", "Height", default_value=0.0)
    _sock(ng, "INPUT", "NodeSocketFloat", "Scale", default_value=0.25)
    _sock(ng, "OUTPUT", "NodeSocketGeometry", "Geometry")

    nodes = ng.nodes
    links = ng.links
    nin = nodes.new("NodeGroupInput")
    nout = nodes.new("NodeGroupOutput")
    nin.location = (-400, 0)
    nout.location = (400, 0)

    # Normal * height * scale
    normal = nodes.new("GeometryNodeInputNormal")
    normal.location = (-200, -120)

    mul_h = nodes.new("ShaderNodeMath")
    mul_h.operation = "MULTIPLY"
    mul_h.location = (-40, -40)
    links.new(nin.outputs["Height"], mul_h.inputs[0])
    links.new(nin.outputs["Scale"], mul_h.inputs[1])

    scale_v = nodes.new("ShaderNodeVectorMath")
    scale_v.operation = "SCALE"
    scale_v.location = (120, -80)
    links.new(normal.outputs["Normal"], scale_v.inputs[0])
    links.new(mul_h.outputs[0], scale_v.inputs["Scale"])

    setpos = nodes.new("GeometryNodeSetPosition")
    setpos.location = (280, 40)
    links.new(nin.outputs["Geometry"], setpos.inputs["Geometry"])
    links.new(scale_v.outputs[0], setpos.inputs["Offset"])
    links.new(setpos.outputs["Geometry"], nout.inputs["Geometry"])
    return ng


def build_compositor_default_tree() -> None:
    """Create compositor group: Ledger Soft-Light Sockets Screen Scene → Out."""
    scene = bpy.context.scene
    tree = bpy.data.node_groups.new("DL_Comp_Default", "CompositorNodeTree")
    scene.compositing_node_group = tree
    # Avoid deprecated Scene.use_nodes when possible
    try:
        scene.use_nodes = True
    except Exception:
        pass

    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    def img(name, loc):
        n = nodes.new("CompositorNodeImage")
        n.label = name
        n.name = name
        n.location = loc
        return n

    ledger = img("DL_Ledger", (-800, 200))
    sockets = img("DL_Sockets", (-800, 0))
    scene_n = img("DL_Scene", (-800, -200))

    mix1 = nodes.new("ShaderNodeMix")
    mix1.data_type = "RGBA"
    mix1.blend_type = "SOFT_LIGHT"
    mix1.inputs["Factor"].default_value = 0.55
    mix1.label = "Soft Light @0.55"
    mix1.location = (-400, 100)

    mix2 = nodes.new("ShaderNodeMix")
    mix2.data_type = "RGBA"
    mix2.blend_type = "SCREEN"
    mix2.inputs["Factor"].default_value = 0.45
    mix2.label = "Screen @0.45"
    mix2.location = (-100, 40)

    viewer = nodes.new("CompositorNodeViewer")
    viewer.location = (200, 40)
    out = nodes.new("CompositorNodeOutputFile")
    out.label = "DL_FileOutput"
    out.location = (200, -200)
    try:
        out.file_output_items.new("RGBA", "Image")
    except Exception:
        pass

    # Group output for node-group based compositor
    gout = nodes.new("NodeGroupOutput")
    gout.location = (200, -40)
    try:
        tree.interface.new_socket(
            name="Image", in_out="OUTPUT", socket_type="NodeSocketColor"
        )
    except Exception:
        pass

    links.new(ledger.outputs["Image"], mix1.inputs["A"])
    links.new(sockets.outputs["Image"], mix1.inputs["B"])
    links.new(mix1.outputs["Result"], mix2.inputs["A"])
    links.new(scene_n.outputs["Image"], mix2.inputs["B"])
    links.new(mix2.outputs["Result"], viewer.inputs["Image"])
    if "Image" in out.inputs:
        links.new(mix2.outputs["Result"], out.inputs["Image"])
    if "Image" in gout.inputs:
        links.new(mix2.outputs["Result"], gout.inputs["Image"])


def build_material_demo() -> None:
    """Demo material using Image Texture (inv EXR) → DL_Look."""
    mat = bpy.data.materials.new("DL_Look_Material")
    nt = mat.node_tree
    if nt is None:
        # Blender 5+: ensure shader tree exists
        try:
            mat.use_nodes = True
            nt = mat.node_tree
        except Exception:
            raise RuntimeError("Material has no node_tree")
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.location = (400, 0)
    look = nt.nodes.new("ShaderNodeGroup")
    look.node_tree = bpy.data.node_groups["DL_Look"]
    look.location = (100, 0)
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.label = "inv_smooth.exr (R=inv)"
    tex.location = (-300, 0)
    tex.extension = "CLIP"
    texcoord = nt.nodes.new("ShaderNodeTexCoord")
    texcoord.location = (-520, 0)
    nt.links.new(texcoord.outputs["UV"], tex.inputs["Vector"])
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    sep.location = (-80, 0)
    nt.links.new(tex.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs["Red"], look.inputs["Inv"])
    nt.links.new(look.outputs["Look"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])

    look.inputs["Near_m"].default_value = 1.0
    look.inputs["Far_m"].default_value = 10.0
    look.inputs["Gamma"].default_value = 0.85


def build_viewer_scene() -> None:
    """Mesh plate + ortho camera so the look material is visible on open."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    mat = bpy.data.materials.get("DL_Look_Material")

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
    plate = bpy.context.active_object
    plate.name = "DL_Plate"
    if mat is not None:
        if plate.data.materials:
            plate.data.materials[0] = mat
        else:
            plate.data.materials.append(mat)

    cam_data = bpy.data.cameras.new("DL_Camera")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 2.2
    cam = bpy.data.objects.new("DL_Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = (0.0, 0.0, 3.0)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.camera = cam

    world = bpy.data.worlds.new("DL_World")
    bpy.context.scene.world = world
    if world.node_tree:
        bg = world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs[0].default_value = (0.02, 0.02, 0.02, 1.0)
            bg.inputs[1].default_value = 1.0

    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024

    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type != "VIEW_3D":
                    continue
                space.shading.type = "MATERIAL"
                space.region_3d.view_perspective = "CAMERA"


def mark_assets() -> None:
    for name in ("DL_DepthSource", "DL_Prep", "DL_Look", "DL_Matte", "DL_Displace"):
        ng = bpy.data.node_groups.get(name)
        if ng:
            ng.asset_mark()
            ng.asset_data.catalog_id = "00000000-0000-0000-0000-000000000001"
    mat = bpy.data.materials.get("DL_Look_Material")
    if mat:
        mat.asset_mark()


def main() -> None:
    _clear_orphans()
    # Fresh file
    bpy.ops.wm.read_factory_settings(use_empty=True)
    build_dl_depth_source()
    build_dl_prep()
    build_dl_look()
    build_dl_matte()
    build_dl_displace()
    build_material_demo()
    build_compositor_default_tree()
    build_viewer_scene()
    mark_assets()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        sys.exit(1)
