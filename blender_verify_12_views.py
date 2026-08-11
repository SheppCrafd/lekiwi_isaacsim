"""
Run this INSIDE Blender: Scripting tab -> Open -> select this file -> Run Script
(or paste its contents into a new text block and press the Play/Run button).

Imports both final USD deliverables fresh (clears the scene first, so nothing
stale from a previous manual import lingers), then renders 12 angles of each
to PNG so they can be inspected without needing to drive Blender interactively.

Output goes to:
  C:\\Users\\mwall\\lekiwi_isaacsim\\blender_verification\\camera\\view_XX.png
  C:\\Users\\mwall\\lekiwi_isaacsim\\blender_verification\\lidar\\view_XX.png
"""
import bpy
import math
import os
from contextlib import suppress

REPO = r"C:\Users\mwall\lekiwi_isaacsim"
TARGETS = [
    ("camera", os.path.join(REPO, "usd", "lekiwi_camera.usd")),
    ("lidar", os.path.join(REPO, "usd", "lekiwi_lidar.usd")),
]
N_VIEWS = 12
ELEVATION_DEG = 28  # camera height angle above the horizontal, degrees


def clear_scene():
    # Direct data-API removal instead of select+delete operators -- avoids
    # any dependency on there being an active selection/context, which
    # varies depending on what's already in the scene when this is run.
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for block_type in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights, bpy.data.materials):
        for block in list(block_type):
            if block.users == 0:
                block_type.remove(block)


def import_usd(path):
    bpy.ops.wm.usd_import(filepath=path)


def scene_bounds():
    mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not mesh_objs:
        return (0, 0, 0), 1.0
    min_co = [math.inf, math.inf, math.inf]
    max_co = [-math.inf, -math.inf, -math.inf]
    for obj in mesh_objs:
        for corner in obj.bound_box:
            world_co = obj.matrix_world @ bpy_vector(corner)
            for i in range(3):
                min_co[i] = min(min_co[i], world_co[i])
                max_co[i] = max(max_co[i], world_co[i])
    center = tuple((min_co[i] + max_co[i]) / 2 for i in range(3))
    radius = max(max_co[i] - min_co[i] for i in range(3)) / 2
    return center, max(radius, 0.05)


def bpy_vector(co):
    from mathutils import Vector
    return Vector(co)


def setup_light():
    """
    Root cause of the "flat/unlit" quirk flagged in plan.md Phase 0 (not fixed at the
    time): a single fixed-direction SUN light can't evenly illuminate a full 360-degree
    camera orbit -- roughly half the 12 views end up facing away from it, rendering as
    near-silhouettes with no visible surface detail (good enough for layout/shape
    checks, per the original note, but not for anything finer). Fixed by adding a
    second SUN light that's re-aimed every frame to always point from the camera toward
    the subject (a "headlamp") -- see render_views(), which updates its rotation
    alongside the camera's each iteration. This one (VerifyFillSun) stays fixed as a
    subtle secondary light so shading still reads as 3D, not flat headlamp-on-a-camera.
    """
    fill_light_data = bpy.data.lights.new(name="VerifyFillSun", type="SUN")
    fill_light_data.energy = 1.5  # secondary -- half the headlamp's strength, just enough to keep shadows non-black
    fill_light_obj = bpy.data.objects.new(name="VerifyFillSun", object_data=fill_light_data)
    bpy.context.collection.objects.link(fill_light_obj)
    fill_light_obj.rotation_euler = (math.radians(55), 0, math.radians(35))

    headlamp_data = bpy.data.lights.new(name="VerifyHeadlampSun", type="SUN")
    headlamp_data.energy = 3.0
    headlamp_obj = bpy.data.objects.new(name="VerifyHeadlampSun", object_data=headlamp_data)
    bpy.context.collection.objects.link(headlamp_obj)

    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("VerifyWorld")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.65, 0.65, 0.68, 1.0)
        bg.inputs[1].default_value = 1.0

    return headlamp_obj


def setup_camera(center, radius):
    cam_data = bpy.data.cameras.new("VerifyCam")
    cam_obj = bpy.data.objects.new("VerifyCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj


def point_camera_at(cam_obj, target):
    from mathutils import Vector
    direction = Vector(target) - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def render_views(cam_obj, headlamp_obj, center, radius, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    distance = radius * 3.2 + 0.3
    elev_rad = math.radians(ELEVATION_DEG)
    scene = bpy.context.scene
    scene.render.resolution_x = 960
    scene.render.resolution_y = 720
    # "BLENDER_EEVEE_NEXT" only exists on Blender 4.2+; older versions use
    # "BLENDER_EEVEE" instead. Previously a bare try/except/pass here silently
    # swallowed the mismatch on older Blender, leaving whichever engine the scene
    # already defaulted to with no indication anything was skipped (found by static
    # analysis flagging the bare except, not by reading it) -- now falls back
    # explicitly and prints which engine actually ended up active, so a wrong-engine
    # render (e.g. flat Workbench shading, ignoring the lighting setup below) is
    # visible in the script's own output instead of a silent surprise.
    if hasattr(scene.render, "engine"):
        with suppress(TypeError):
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        if scene.render.engine != "BLENDER_EEVEE_NEXT":
            with suppress(TypeError):
                scene.render.engine = "BLENDER_EEVEE"
    print(f"render engine: {scene.render.engine}")

    for i in range(N_VIEWS):
        az = math.radians(i * (360.0 / N_VIEWS))
        x = center[0] + distance * math.cos(elev_rad) * math.cos(az)
        y = center[1] + distance * math.cos(elev_rad) * math.sin(az)
        z = center[2] + distance * math.sin(elev_rad)
        cam_obj.location = (x, y, z)
        point_camera_at(cam_obj, center)

        # SUN lights only have a direction (their local -Z axis), no position -- copying
        # the camera's exact rotation makes the headlamp shine in the same direction the
        # camera looks (camera position -> center), so every one of the 12 views gets
        # front-lit regardless of azimuth, not just the ones that happen to face
        # VerifyFillSun's fixed direction.
        headlamp_obj.rotation_euler = cam_obj.rotation_euler.copy()

        scene.render.filepath = os.path.join(out_dir, f"view_{i:02d}_az{int(i*360/N_VIEWS):03d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"rendered {scene.render.filepath}")


for label, usd_path in TARGETS:
    print(f"=== {label}: {usd_path} ===")
    clear_scene()
    import_usd(usd_path)
    headlamp_obj = setup_light()
    center, radius = scene_bounds()
    print(f"  scene center={center} radius={radius:.3f}")
    cam_obj = setup_camera(center, radius)
    out_dir = os.path.join(REPO, "blender_verification", label)
    render_views(cam_obj, headlamp_obj, center, radius, out_dir)

print("ALL DONE. Check C:\\Users\\mwall\\lekiwi_isaacsim\\blender_verification\\")
