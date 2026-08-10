"""
Shared scene/action/reward/termination/event wiring for both cone-nav variants.
env_cfg_camera.py and env_cfg_lidar.py each subclass LekiwiConeNavEnvCfgBase, adding
only their sensor and the matching observation group -- everything else (rewards,
terminations, actuation/course-regen events, action space) is sensor-agnostic and
lives here once.

plan.md Phase 3's settled physics/control numbers (copied from the real
MuammerBay/isaac_so_arm101 Isaac Lab reference, see that phase's writeup for the
verification trail): sim.dt = 1/60s, decimation = 2 -> 30Hz control rate.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as base_mdp

from . import mdp
from .course_generator import CourseGeneratorCfg

MAX_CONES = 14  # matches course_generator's num_cones_range upper bound (course_generator.py)
MAX_CLUTTER_PROPS = 16  # matches mdp.events.randomize_surroundings_clutter's num_props_range upper bound

# Course footprints run up to length_range_m=7.0 / width_range_m=6.0 (course_generator
# defaults) plus a clutter band beyond that -- 9m spacing keeps neighboring envs' clutter
# from ever bleeding into each other even at the largest generated course + full clutter
# band, satisfying Phase 4's per-env isolation requirement.
ENV_SPACING = 9.0


@configclass
class CourseSceneCfg:
    """Non-physics, task-level constants read by mdp/events.py and cone_nav_env.py."""

    max_cones: int = MAX_CONES
    max_clutter_props: int = MAX_CLUTTER_PROPS
    generator: CourseGeneratorCfg = CourseGeneratorCfg()


def _cone_prop_cfg(name: str) -> RigidObjectCfg:
    """
    One physical cone prim per slot -- see mdp/events.py:regenerate_course's docstring
    and mdp/events.py module docstring for why size/shape variety is approximated as
    continuous scale on a single real cone mesh (sim_utils.ConeCfg) rather than
    swapping between literal cone/pylon/barrel primitive types every reset: shape *type*
    switching would need a MultiAssetSpawnerCfg (fixed per-env at scene build, not
    reset), which doesn't fit "regenerate every episode." Documented simplification,
    not a silent one.

    Kinematic, not dynamic (plan.md Phase 5's still-open static-vs-dynamic-cones
    question) -- default choice for now since the task's own collision check
    (mdp/rewards.py, mdp/terminations.py) is contact-based and doesn't need cones to
    physically topple. Flip kinematic_enabled=False here to try the dynamic-rigid-body
    variant Phase 5 leaves as a documented option, not implemented default.
    """
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.ConeCfg(
            radius=0.15,
            height=0.4,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.35, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0)),
    )


def _clutter_prop_cfg(name: str) -> RigidObjectCfg:
    """One background-clutter prim per slot -- see mdp.events.randomize_surroundings_clutter."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0)),
    )


@configclass
class LekiwiSceneCfgBase(InteractiveSceneCfg):
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(200.0, 200.0)),
    )

    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0),
    )

    robot_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base", history_length=1, track_air_time=False
    )

    def __post_init__(self):
        for i in range(MAX_CONES):
            setattr(self, f"cone_{i}", _cone_prop_cfg(f"cone_{i}"))
        for i in range(MAX_CLUTTER_PROPS):
            setattr(self, f"clutter_{i}", _clutter_prop_cfg(f"clutter_{i}"))


@configclass
class ActionsCfg:
    base_velocity: mdp.BodyVelocityActionCfg = mdp.BodyVelocityActionCfg(asset_name="robot")


@configclass
class RewardsCfg:
    approach_goal: RewTerm = RewTerm(func=mdp.rewards.approach_goal_potential, weight=1.0)
    success: RewTerm = RewTerm(func=mdp.rewards.success_bonus, weight=100.0)
    cone_collision: RewTerm = RewTerm(func=mdp.rewards.cone_collision_penalty, weight=5.0)
    action_smoothness: RewTerm = RewTerm(func=mdp.rewards.action_smoothness_penalty, weight=0.01)


@configclass
class TerminationsCfg:
    # Episode timeout/truncation (plan.md Phase 5: "without one, an episode that never
    # hits a cone or reaches the goal never resets, which breaks batched on-policy
    # training") -- Isaac Lab's own built-in term, driven by
    # ManagerBasedRLEnvCfg.episode_length_s below, not a custom one.
    time_out: DoneTerm = DoneTerm(func=base_mdp.time_out, time_out=True)
    cone_collision: DoneTerm = DoneTerm(func=mdp.terminations.cone_collision)
    success: DoneTerm = DoneTerm(func=mdp.terminations.goal_reached_and_held)
    out_of_bounds: DoneTerm = DoneTerm(func=mdp.terminations.out_of_bounds)


@configclass
class EventsCfg:
    regenerate_course: EventTerm = EventTerm(func=mdp.events.regenerate_course, mode="reset")
    randomize_actuation: EventTerm = EventTerm(func=mdp.events.randomize_actuation, mode="reset")
    randomize_clutter: EventTerm = EventTerm(func=mdp.events.randomize_surroundings_clutter, mode="reset")


@configclass
class CurriculumCfg:
    # Phase 6's last unchecked item: anneal obstacle density rather than training
    # against full difficulty from step 0. See mdp/curriculum.py's docstring.
    # UNCONFIRMED ORDERING (Phase 1/2 to check): whether CurriculumManager runs before
    # or after EventsCfg's "reset"-mode terms within a single env.reset() call. If this
    # runs after regenerate_course on the same reset, the difficulty update lags by
    # exactly one reset for the env(s) being reset that step -- a harmless one-step
    # staleness for a signal this slow-moving (steps, not episodes), not a correctness
    # bug either way.
    anneal_course_difficulty: CurrTerm = CurrTerm(func=mdp.curriculum.anneal_course_difficulty)


@configclass
class LekiwiConeNavEnvCfgBase(ManagerBasedRLEnvCfg):
    scene: LekiwiSceneCfgBase = LekiwiSceneCfgBase(num_envs=2500, env_spacing=ENV_SPACING)
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    course: CourseSceneCfg = CourseSceneCfg()

    def __post_init__(self):
        self.sim.dt = 1.0 / 60.0
        self.decimation = 2
        self.episode_length_s = 30.0  # ~900 control steps at 30Hz -- generous relative to course size, tune once Phase 8 eval shows typical solve time
        self.sim.render_interval = self.decimation
