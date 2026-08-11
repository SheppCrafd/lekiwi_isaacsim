"""
Evaluate a trained policy on the HELD-OUT eval seed range (plan.md Phase 8: "roll out
each trained policy on a held-out set of generated environments/seeds, separate from
training -- currently missing entirely. Without this there's no way to measure real
generalization vs. training-curve reward"). This is that missing piece.

Sets events.regenerate_course's eval_mode=True, which switches
course_generator's seed draw from the training range [0, TRAIN_SEED_UPPER) to the
held-out range [TRAIN_SEED_UPPER, TOTAL_SEEDS) (course_generator.py's seed-convention
docstring) -- seeds a training run should never have sampled.

    python play.py --task camera --checkpoint logs/rsl_rl/lekiwi_conenav_camera/model_2500.pt --num_eval_envs 256
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["camera", "lidar"], required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_eval_envs", type=int, default=256)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import lekiwi_tasks.cone_nav  # noqa: F401


def main() -> None:
    task_id = f"Lekiwi-ConeNav-{args.task.capitalize()}-v0"
    # Real Isaac Lab entry-point resolution helpers (isaaclab_tasks.utils), confirmed
    # via source research -- see train.py's docstring for the same note.
    env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=args.num_eval_envs)
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    env_cfg.events.regenerate_course.params = dict(env_cfg.events.regenerate_course.params or {})
    env_cfg.events.regenerate_course.params["eval_mode"] = True

    env = gym.make(task_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    num_episodes_target = 500  # Phase 8: real generalization measurement, not a handful of anecdotal rollouts
    successes = 0
    collisions = 0
    out_of_bounds = 0
    timeouts = 0
    episodes_done = 0
    episode_lengths: list[int] = []
    steps_since_reset = torch.zeros(env.num_envs, dtype=torch.long)

    # BUGFIX (2026-08-10): this used to read extras["log"].get("Episode_Termination/success")
    # inside the per-env_id loop below and treat it as if it were THAT env's own outcome.
    # extras["log"] is a batch-level aggregate logged once per env.step() call, not
    # indexable per env_id -- so every env in done_ids on a step where ANY env succeeded
    # got credited, wildly overcounting whenever multiple of the num_eval_envs finish in
    # the same step with mixed outcomes (the common case). It also had a dead
    # `and env_id in done_ids` check (always true -- we're iterating done_ids itself) and
    # declared collisions/out_of_bounds counters that were never incremented or printed.
    #
    # Fixed by reading each termination TERM's own per-env boolean straight from the
    # unwrapped env's TerminationManager (env.unwrapped.termination_manager.get_term(name),
    # name = the TerminationsCfg field name in env_cfg_base.py: "success"/"cone_collision"/
    # "out_of_bounds"/"time_out") right after env.step() -- a real, per-env signal, not a
    # batch-level log line. TerminationManager.get_term() follows the same pattern this
    # project already confirmed for ActionManager.get_term() (mdp/events.py:randomize_actuation)
    # -- both are isaaclab.managers.ManagerBase subclasses sharing that interface -- but
    # this specific call is still unverified against a real install (Phase 1/2), same
    # caveat as everything else in this file that touches Isaac Lab's manager internals.
    obs, _ = env.get_observations()
    while episodes_done < num_episodes_target:
        with torch.inference_mode():
            actions = policy(obs)
        obs, _rew, dones, extras = env.step(actions)
        steps_since_reset += 1

        term_mgr = env.unwrapped.termination_manager
        term_success = term_mgr.get_term("success")
        term_collision = term_mgr.get_term("cone_collision")
        term_oob = term_mgr.get_term("out_of_bounds")
        term_timeout = term_mgr.get_term("time_out")

        done_ids = dones.nonzero(as_tuple=False).flatten().tolist()
        for env_id in done_ids:
            episodes_done += 1
            episode_lengths.append(int(steps_since_reset[env_id].item()))
            steps_since_reset[env_id] = 0
            if term_success[env_id]:
                successes += 1
            if term_collision[env_id]:
                collisions += 1
            if term_oob[env_id]:
                out_of_bounds += 1
            if term_timeout[env_id]:
                timeouts += 1

    print(f"[play.py] {episodes_done} held-out episodes:")
    print(f"  success rate:      {successes / episodes_done:.3f}")
    print(f"  collision rate:    {collisions / episodes_done:.3f}")
    print(f"  out-of-bounds rate:{out_of_bounds / episodes_done:.3f}")
    print(f"  timeout rate:      {timeouts / episodes_done:.3f}")
    print(f"  mean ep length:    {sum(episode_lengths) / len(episode_lengths):.1f} steps")
    print(
        "  NOTE: per-episode attribution above reads env.unwrapped.termination_manager."
        "get_term(name) directly (a real per-env boolean per registered TerminationsCfg "
        "term), not extras['log'] parsing -- more solid than the old approach, but still "
        "unverified against a real Isaac Lab install (Phase 1/2). If get_term() doesn't "
        "exist or behaves differently in your installed version, that's the first thing "
        "to fix here."
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
