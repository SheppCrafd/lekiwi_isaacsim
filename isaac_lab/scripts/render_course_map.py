"""
Render a generated cone course to a dimensioned top-down PNG floor plan, and print a
tape-measure build spec (feet-inches + meters) for laying it out in real life.

No Isaac Sim / omni dependency (matplotlib + the pure-numpy course_generator only), so
this runs on a normal machine -- useful both for offline course previews and for
producing the physical "hero course" build reference.

Usage:
    python render_course_map.py --seed 0 --out hero_course.png
    python render_course_map.py --hero --out hero_course.png   # fixed real-world-sized layout, see build_hero_course()
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lekiwi_tasks", "cone_nav"))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from course_generator import (
    CourseGeneratorCfg,
    CourseLayout,
    generate_course,
    validate_layout,
)

M_TO_IN = 39.3701

_SHAPE_COLOR = {"cone": "#e2622a", "pylon": "#d9a441", "barrel": "#2f6f9f"}

# Distinct per-cone color for map legibility only -- real cones are one physical color
# (the hero course is deliberately all-"cone"-shape, see build_hero_course()); this has
# nothing to do with the sim's own material/color domain randomization (Phase 6).
_CONE_PALETTE = [
    "#e2622a", "#2f6f9f", "#3fa34d", "#a04fc9", "#d9a441", "#c9455b", "#3fb6b6", "#7a7a7a",
]


def feet_inches(meters: float) -> str:
    total_in = meters * M_TO_IN
    feet = int(total_in // 12)
    inches = total_in - feet * 12
    return f'{feet}\'{inches:.1f}"'


def build_hero_course(seed: int = 0) -> tuple[CourseLayout, CourseGeneratorCfg]:
    """
    A fixed, real-world-buildable course: 12ft x 20ft (3.6576m x 6.096m, well above the
    100 sqft / 9.29 sqm minimum), 8 cones, all identical -- real 18in traffic cones
    with a ~10in-diameter base (a common, purchasable size), one shape/size keeps the
    shopping list simple for a physical build. Generously-sized 3ft-diameter end zone
    (bigger than the sim default 0.7m/27in -- easier to mark and see when you're the
    one walking the course). Spawn is not fixed to one point -- see spawn handling
    below and print_build_spec()'s note about it.

    The end zone position is fixed at the far end, centered laterally, matching the
    generator's own canonical placement (goal_x = length - margin/2, goal_y = width/2)
    -- "end zone always stays the same" falls out naturally: only the cones (and where
    you start the robot) vary if you want a different run, the end zone and course
    footprint don't move.

    Cone/goal positions are rounded to the nearest inch after generation (nicer for a
    tape measure) and re-validated -- if rounding ever breaks a safety check, the next
    seed is tried instead of silently shipping an invalid course. The example spawn
    marker is rounded too, but it's illustrative only -- course_generator.py now
    randomizes spawn across the whole open area (matching plan.md's own task spec), so
    for the physical build any point in the shaded safe zone on the map works, not just
    the one marked.
    """
    fixed_cone_radius = 0.127  # 5in radius = 10in diameter base
    fixed_cone_height = 0.4572  # 18in tall
    cfg = CourseGeneratorCfg(
        width_range_m=(3.6576, 3.6576),  # 12 ft, fixed (min==max => deterministic footprint)
        length_range_m=(6.096, 6.096),  # 20 ft, fixed
        num_cones_range=(14, 14),  # bumped from 8 -- the 12x20ft footprint had visibly empty space at 8
        goal_radius_m=0.4572,  # 3 ft diameter
    )
    # Restrict to real traffic cones only, all one fixed size, for the physical build --
    # NOT the sim's wide 0.25-2ft/0.125-1ft randomization range (course_generator.py's
    # CONE_SIZE_RANGE_M): the hero course is one fixed real build, still "all identical"
    # per the earlier explicit request, unaffected by that sim-side widening.
    import course_generator as _cg

    original_shapes = _cg.CONE_SHAPES
    original_range = _cg.CONE_SIZE_RANGE_M
    _cg.CONE_SHAPES = ("cone",)
    _cg.CONE_SIZE_RANGE_M = {"radius": (fixed_cone_radius, fixed_cone_radius), "height": (fixed_cone_height, fixed_cone_height)}
    try:
        s = seed
        while True:
            layout = generate_course(s, cfg)
            rounded = _round_layout_to_inch(layout)
            ok, reason = validate_layout(rounded, cfg)
            if ok:
                return rounded, cfg
            s += 1  # rounding broke a safety check -- try the next seed rather than ship it
    finally:
        _cg.CONE_SHAPES = original_shapes
        _cg.CONE_SIZE_RANGE_M = original_range


def _round_layout_to_inch(layout: CourseLayout) -> CourseLayout:
    inch = 1.0 / M_TO_IN
    r = lambda v: round(v / inch) * inch
    rounded_cones = [
        type(c)(x=r(c.x), y=r(c.y), radius=r(c.radius), height=r(c.height), shape=c.shape)
        for c in layout.cones
    ]
    return CourseLayout(
        seed=layout.seed,
        width_m=r(layout.width_m),
        length_m=r(layout.length_m),
        cones=rounded_cones,
        goal_x=r(layout.goal_x),
        goal_y=r(layout.goal_y),
        goal_radius_m=r(layout.goal_radius_m),
        spawn_x=r(layout.spawn_x),
        spawn_y=r(layout.spawn_y),
        spawn_heading_rad=layout.spawn_heading_rad,
    )


def print_build_spec(layout: CourseLayout) -> None:
    print(f"=== Cone course build spec (seed {layout.seed}) ===")
    print(
        f"Course footprint: {feet_inches(layout.length_m)} long (nav axis) x "
        f"{feet_inches(layout.width_m)} wide  ({layout.length_m:.2f}m x {layout.width_m:.2f}m)"
    )
    print("Origin (0,0) = the corner at the SPAWN end, lateral-min side. Measure X down the")
    print("course length, Y across its width, from that corner.\n")

    print(
        f"SPAWN -- NOT fixed. Start the robot anywhere in the shaded safe zone on the map "
        f"(anywhere at least ~0.7ft / 0.2m clear of a cone or the end zone). "
        f"Example point used for this printout: X={feet_inches(layout.spawn_x)} "
        f"({layout.spawn_x:.2f}m)  Y={feet_inches(layout.spawn_y)} ({layout.spawn_y:.2f}m). "
        f"Starting heading can point any direction -- the sim trains against a full "
        f"random heading too, not just facing down the course."
    )
    print(
        f"END ZONE (fixed -- always the same spot, never moves between runs):  "
        f"center X={feet_inches(layout.goal_x)} ({layout.goal_x:.2f}m)   "
        f"Y={feet_inches(layout.goal_y)} ({layout.goal_y:.2f}m)   "
        f"diameter={feet_inches(layout.goal_radius_m * 2)} ({layout.goal_radius_m * 2:.2f}m)"
    )
    print()
    cone_dia_in = layout.cones[0].radius * 2 * M_TO_IN if layout.cones else 0.0
    cone_h_in = layout.cones[0].height * M_TO_IN if layout.cones else 0.0
    print(
        f"Cones (shopping list: {len(layout.cones)}x standard traffic cone, "
        f"{cone_dia_in:.0f}in base dia x {cone_h_in:.0f}in tall, all identical):"
    )
    for i, c in enumerate(layout.cones, start=1):
        print(f"  #{i}: X={feet_inches(c.x):<8} ({c.x:.2f}m)   Y={feet_inches(c.y):<8} ({c.y:.2f}m)")


def render_course_png(layout: CourseLayout, out_path: str, title: str, clearance_m: float = 0.2) -> None:
    fig, ax = plt.subplots(figsize=(12, 9 * layout.width_m / layout.length_m + 1.5))

    # Whole floor washed pale green = "safe to start the robot anywhere in here" (spawn
    # is no longer one fixed point, see course_generator.py) -- cone/end-zone clearance
    # discs drawn on top in the "not safe" color punch holes out of that reading.
    ax.add_patch(
        mpatches.Rectangle((0, 0), layout.length_m, layout.width_m, facecolor="#eaf7ec", edgecolor="none", zorder=0)
    )
    for c in layout.cones:
        ax.add_patch(
            mpatches.Circle((c.x, c.y), c.radius + clearance_m, facecolor="#fbeaea", edgecolor="none", zorder=1)
        )
    ax.add_patch(
        mpatches.Circle(
            (layout.goal_x, layout.goal_y), layout.goal_radius_m + clearance_m,
            facecolor="#fbeaea", edgecolor="none", zorder=1,
        )
    )

    ax.add_patch(
        mpatches.Rectangle((0, 0), layout.length_m, layout.width_m, fill=False, edgecolor="black", linewidth=2, zorder=5)
    )

    # End zone -- drawn on top, distinctly colored/hatched so it reads as "fixed" vs. cones
    ax.add_patch(
        mpatches.Circle(
            (layout.goal_x, layout.goal_y),
            layout.goal_radius_m,
            facecolor="#2fa84f",
            alpha=0.35,
            edgecolor="#1c6b31",
            linewidth=2,
            hatch="//",
            zorder=2,
        )
    )
    ax.annotate(
        f"END ZONE (fixed)\n{layout.goal_radius_m*2:.2f}m / {feet_inches(layout.goal_radius_m*2)} dia",
        (layout.goal_x, layout.goal_y),
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#1c6b31",
        zorder=6,
    )

    for i, c in enumerate(layout.cones, start=1):
        color = _CONE_PALETTE[(i - 1) % len(_CONE_PALETTE)]
        ax.add_patch(mpatches.Circle((c.x, c.y), c.radius, facecolor=color, edgecolor="black", zorder=3))
        ax.annotate(f"#{i}", (c.x, c.y), ha="center", va="center", fontsize=8, color="white", fontweight="bold", zorder=4)
        ax.annotate(
            f"X={feet_inches(c.x)}\nY={feet_inches(c.y)}",
            (c.x, c.y - c.radius - 0.12),
            ha="center",
            va="top",
            fontsize=7,
        )

    # Spawn is NOT a fixed point -- the whole pale-green wash is the legal spawn area
    # (anywhere at least `clearance_m` clear of a cone or the end zone, i.e. anywhere
    # you could physically stand the robot without it touching something). This marker
    # is one illustrative example, drawn faded/dashed on purpose so it doesn't read as
    # "the" spawn point.
    ax.plot(layout.spawn_x, layout.spawn_y, marker="s", markersize=9, color="#333333",
            alpha=0.55, markeredgewidth=1.5, fillstyle="none", zorder=3)
    ax.annotate(
        "e.g. spawn here\n(anywhere in green\nzone also works)",
        (layout.spawn_x, layout.spawn_y),
        xytext=(layout.spawn_x, layout.spawn_y - 0.55),
        ha="center",
        va="top",
        fontsize=7,
        color="#555555",
        style="italic",
    )
    dx, dy = 0.4 * math.cos(layout.spawn_heading_rad), 0.4 * math.sin(layout.spawn_heading_rad)
    ax.annotate("", xy=(layout.spawn_x + dx, layout.spawn_y + dy), xytext=(layout.spawn_x, layout.spawn_y),
                arrowprops=dict(arrowstyle="->", color="#333333", linewidth=1.5, alpha=0.55))

    # Overall dimension callouts
    ax.annotate(
        "", xy=(layout.length_m, -0.5), xytext=(0, -0.5),
        arrowprops=dict(arrowstyle="<->", color="black"),
    )
    ax.text(layout.length_m / 2, -0.7, f"{feet_inches(layout.length_m)}  ({layout.length_m:.2f}m)",
            ha="center", fontsize=10, fontweight="bold")
    ax.annotate(
        "", xy=(-0.5, layout.width_m), xytext=(-0.5, 0),
        arrowprops=dict(arrowstyle="<->", color="black"),
    )
    ax.text(-0.7, layout.width_m / 2, f"{feet_inches(layout.width_m)}  ({layout.width_m:.2f}m)",
            ha="center", fontsize=10, fontweight="bold", rotation=90)

    ax.set_xlim(-1.4, layout.length_m + 0.6)
    ax.set_ylim(-1.4, layout.width_m + 0.6)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Nav axis / course length (m)")
    ax.set_ylabel("Lateral axis / course width (m)")
    ax.grid(True, linestyle=":", alpha=0.4)

    legend_handles = [
        mpatches.Patch(facecolor="#eaf7ec", edgecolor="#888888", label="Spawn anywhere here"),
        mpatches.Patch(facecolor="#fbeaea", edgecolor="#888888", label="Keep-out (too close to cone/goal)"),
    ] + [
        mpatches.Patch(color=_CONE_PALETTE[i % len(_CONE_PALETTE)], label=f"Cone #{i+1}")
        for i in range(len(layout.cones))
    ]
    ax.legend(
        handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7,
        title="Cone colors = ID only\n(all cones physically identical)", title_fontsize=7,
        borderaxespad=0,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hero", action="store_true", help="Use the fixed real-world-buildable hero course instead of a raw generator seed")
    parser.add_argument("--out", type=str, default="course_map.png")
    args = parser.parse_args()

    if args.hero:
        layout, _cfg = build_hero_course(seed=args.seed)
        title = f"LeKiwi Cone Course -- Hero Build (12ft x 20ft, {len(layout.cones)} cones)"
    else:
        layout = generate_course(args.seed, CourseGeneratorCfg())
        title = f"LeKiwi Cone Course -- seed {args.seed}"

    print_build_spec(layout)
    render_course_png(layout, args.out, title)


if __name__ == "__main__":
    main()
