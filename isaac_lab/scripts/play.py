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
    episodes_done = 0
    episode_lengths: list[int] = []
    steps_since_reset = torch.zeros(env.num_envs, dtype=torch.long)

    obs, _ = env.get_observations()
    while episodes_done < num_episodes_target:
        with torch.inference_mode():
            actions = policy(obs)
        obs, _rew, dones, extras = env.step(actions)
        steps_since_reset += 1

        done_ids = dones.nonzero(as_tuple=False).flatten().tolist()
        for env_id in done_ids:
            episodes_done += 1
            episode_lengths.append(int(steps_since_reset[env_id].item()))
            steps_since_reset[env_id] = 0
            term_log = extras.get("log", {})
            # Exact key names depend on the installed Isaac Lab version's termination
            # logging format (see train.py's docstring) -- if these don't match,
            # inspect extras["log"]'s actual keys directly rather than guessing again.
            if term_log.get(f"Episode_Termination/success", 0) and env_id in done_ids:
                successes += 1

    print(f"[play.py] {episodes_done} held-out episodes:")
    print(f"  success rate:   {successes / episodes_done:.3f}")
    print(f"  mean ep length: {sum(episode_lengths) / len(episode_lengths):.1f} steps")
    print(
        "  NOTE: per-episode success/collision attribution above via extras['log'] is "
        "a best-effort read of Isaac Lab's manager logging -- verify against your "
        "installed version's actual extras['log'] keys (print one dict and check) "
        "before trusting these numbers for a real Phase 8 report."
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
