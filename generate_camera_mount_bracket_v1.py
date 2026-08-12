"""
Generates urdf/meshes/camera_mount_bracket_v1.stl (and prints/camera_mount_bracket_v1.stl,
a duplicate copy for whoever's doing the actual printing).

Run with: python generate_camera_mount_bracket_v1.py
Requires: trimesh, manifold3d (both already used elsewhere in this project's toolchain).

This script is a deliberate departure from this project's earlier practice of not
persisting a generator script for either 3D-printable mount (lidar_mount_block_v1.stl,
the original camera_mount_bracket_v1.stl) -- only the output STL was ever committed,
which meant the base-plate hole positions, collision-clearance reasoning, and Arducam
board mounting-hole pattern all had to be reverse-engineered from the STL itself the
next time a bug needed fixing (2026-08-11, see plan_log.md). Persisting this script
avoids repeating that archaeology.

Geometry, in the base-plate's own frame (mm), matches what's baked directly into
usd/lekiwi_camera.usd's Camera_Mount_bracket_v1_visual/_collision Mesh prims (in
meters) -- see that USD's own docstring precedent (points baked directly, no xformOps
on the parent Xform).

-- Flange (bolts to the base plate) --
Two side rails (x:[-46,-34] and [34,46], y:[74,101], z:[0,2]) plus a front crossbar
(x:[-46,46], y:[93,101], z:[0,2]) that connects them -- together routing AROUND
servo_controller_mount_v3 (real bbox x:[-22.4,22.8] y:[30.1,90.3] z:[0,11]mm) rather
than covering it. All three pieces are unioned into ONE watertight body: an earlier
version of this bracket authored the rails and crossbar as three separate solids that
only *touched* at a shared z=0 plane without ever overlapping in X (crossbar stopped
at x=+-30, rails started at x=+-34) -- geometrically disconnected, a real 4mm gap per
side, caught visually in a Blender render (2026-08-11) despite passing every
watertight/collision check that was only ever run on each piece in isolation. Fixed
here by widening the crossbar to the rails' own full x:[-46,46] width so the pieces
directly overlap before the boolean union, instead of only asserting they share a
touching plane.

Base-plate mount holes (5 total, not 6): real hole positions extracted from
urdf/meshes/camera_base_base_plate_layer1_v5.stl via trimesh.mesh.section() in the
original design session -- two on the rails at (x=+-40, y=79), three on the crossbar
at (x=-19.94/0/20.06, y=98.94). Only 5 of the 6 real base-plate holes originally
extracted are used: the frame's reshape to route around the servo mount removed the
solid material a 6th, centered rear-row hole (x~0, y~79) would have needed, so it's
correctly unused rather than missing.

-- Riser (holds the Arducam board) --
Stands on the crossbar's front edge (x:[-20,20], y:[99,101], z:[2,48]).

Board hole pattern (2026-08-11, corrected): SQUARE, 34x34mm pitch (the larger of two
real pitch options), sourced directly from Arducam's own B0200 Quick Start Guide
(uctronics.com/download/Amazon/B0200.pdf -- "Board Size 38x38mm (Hole pitch 28x28mm,
34x34mm)"), i.e. this SKU's *own* published mechanical spec. This replaces the
original design's 16x28mm RECTANGLE, which had been borrowed from the B0201 sibling
SKU's datasheet because B0200's own datasheet PDF couldn't be located at the time --
a well-founded but wrong guess (same-family board, different mount pattern). The
34mm pitch (vs. the smaller 28mm option) was picked per explicit instruction to use
the larger/outer holes. Hole diameter still isn't published by Arducam for either
pitch option, so it stays at the same picked-default M2.5-clearance 3.0mm (r=1.5mm)
used previously -- flagged, not re-verified.

Center cutout: Ø20mm, clears both the M12 lens body (Ø14mm) and the USB cable --
assumed to route straight back through this same central opening (this board's real
cable-exit location isn't independently confirmed by any Arducam photo/drawing for
this SKU, same category of assumption as the hole diameter above). At the real 34mm
hole pitch the nearest board hole is ~24mm from center, well clear of the 10mm-radius
cutout (more margin than the old 16x28 pattern's ~16mm had).

Collision clearance against every other component on /LeKiwi/base (19 total, checked
directly against usd/lekiwi_camera.usd, not just the one part that clips closest) is
re-verified by a separate script, not this one -- see plan_log.md's "Real collision
found and fixed by The Temper" note for the protocol (bbox-overlap pass, then dense
surface-point + vertex containment sampling for anything that overlaps).
"""
import numpy as np
import trimesh

OUT_URDF = "urdf/meshes/camera_mount_bracket_v1.stl"
OUT_PRINT = "prints/camera_mount_bracket_v1.stl"


def cyl_z(radius, x, y, z0=-5, z1=7):
    """Vertical hole through a thin flange/rail (axis along Z)."""
    c = trimesh.creation.cylinder(radius=radius, height=z1 - z0, sections=48)
    c.apply_translation([x, y, (z0 + z1) / 2])
    return c


def cyl_y(radius, x, z, y0=97, y1=103):
    """Horizontal hole through the thin riser wall (axis along Y)."""
    c = trimesh.creation.cylinder(radius=radius, height=y1 - y0, sections=48)
    c.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0]))
    c.apply_translation([x, (y0 + y1) / 2, z])
    return c


def build():
    left_rail = trimesh.creation.box(extents=[12, 27, 2])
    left_rail.apply_translation([-40, 87.5, 1])  # x:[-46,-34] y:[74,101] z:[0,2]

    right_rail = trimesh.creation.box(extents=[12, 27, 2])
    right_rail.apply_translation([40, 87.5, 1])  # x:[34,46] y:[74,101] z:[0,2]

    # Full x:[-46,46] width -- directly overlaps both rails (the fix: previously
    # x:[-30,30], leaving a disconnected 4mm gap on each side).
    crossbar_flange = trimesh.creation.box(extents=[92, 8, 2])
    crossbar_flange.apply_translation([0, 97, 1])  # x:[-46,46] y:[93,101] z:[0,2]

    riser_wall = trimesh.creation.box(extents=[40, 2, 46])
    riser_wall.apply_translation([0, 100, 25])  # x:[-20,20] y:[99,101] z:[2,48]

    solid = trimesh.boolean.union(
        [left_rail, right_rail, crossbar_flange, riser_wall], engine="manifold"
    )

    holes = [
        # base-plate mount holes -- real positions, unchanged from the original design
        cyl_z(1.75, -40, 79),
        cyl_z(1.75, 40, 79),
        cyl_z(1.75, -19.94, 98.94),
        cyl_z(1.75, 0.00, 98.94),
        cyl_z(1.75, 20.06, 98.94),
        # Arducam B0200 board mount holes -- real 34x34mm square pitch
        cyl_y(1.5, 17, 42),
        cyl_y(1.5, 17, 8),
        cyl_y(1.5, -17, 42),
        cyl_y(1.5, -17, 8),
        # M12 lens body + USB cable center cutout
        cyl_y(10.0, 0, 25),
    ]
    cut = trimesh.boolean.union(holes, engine="manifold")
    final = trimesh.boolean.difference([solid, cut], engine="manifold")
    final.remove_unreferenced_vertices()
    final.fix_normals()
    return final


if __name__ == "__main__":
    mesh = build()
    assert mesh.is_watertight, "generated bracket is not watertight"
    assert mesh.body_count == 1, f"generated bracket has {mesh.body_count} disconnected bodies, expected 1"
    print(f"watertight: {mesh.is_watertight}, bodies: {mesh.body_count}, bounds(mm): {mesh.bounds.tolist()}")
    mesh.export(OUT_URDF)
    mesh.export(OUT_PRINT)
    print(f"wrote {OUT_URDF} and {OUT_PRINT}")
