"""
Train one of the two cone-nav policies (plan.md Phase 7). Run inside Isaac Sim's own
Python environment on the Google Cloud instance (Phase 1) -- this cannot run anywhere
in this repo's own dev environment, there's no Isaac Sim installed here.

    python train.py --task camera --num_envs 2500 --headless
    python train.py --task lidar  --num_envs 2500 --headless

Checkpointing + auto-resume (plan.md Phase 1's explicit requirement -- spot instances
get preempted with short notice, and this project is paying for two separate 50-hour
runs, so losing progress mid-run is real money, not just annoyance): logs go to a
STABLE directory per (task, run_name), not a fresh timestamp every launch. On startup,
this script looks for the newest checkpoint already in that directory and resumes from
it automatically unless --fresh_start is passed -- so if a spot instance gets reclaimed
and you just re-run the same command on a new instance (with the log dir on persistent
storage, e.g. a mounted GCS bucket -- set that up yourself, this script doesn't), it
picks up where it left off with no extra flags to remember.

Metrics beyond the reward curve (plan.md Phase 7: "track real metrics during each run,
not just the reward curve: episode success rate, collision rate, average episode
length"): these come for free from Isaac Lab's own manager framework, not custom code
here -- the RewardManager/TerminationManager automatically log each term's per-episode
rate into `extras["log"]`, which rsl_rl's tensorboard writer picks up. Once training is
running, look for these tags in tensorboard rather than only the top-level reward curve:
    Episode_Termination/success        -- success rate
    Episode_Termination/cone_collision -- collision rate
    Episode_Termination/out_of_bounds
    Episode_Termination/time_out
    Episode/episode_length             -- (rsl_rl's own standard key)
If your installed Isaac Lab version logs these under slightly different tag names,
that's a version thing to confirm in Phase 1/2, not a sign this script is missing
something.
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["camera", "lidar"], required=True)
parser.add_argument("--num_envs", type=int, default=None, help="Override the env cfg's default (2500 for both variants)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log_root", type=str, default="logs/rsl_rl")
parser.add_argument("--fresh_start", action="store_true", help="Ignore any existing checkpoint and start over")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac Sim/omni imports must come after AppLauncher starts the Kit process -- importing
# them earlier fails, this ordering is not optional.
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import lekiwi_tasks.cone_nav  # noqa: F401 -- registers Lekiwi-ConeNav-{Camera,Lidar}-v0 with gym


def _find_latest_checkpoint(run_dir: str) -> str | None:
    if not os.path.isdir(run_dir):
        return None
    checkpoints = [f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda f: int(f.removeprefix("model_").removesuffix(".pt")))
    return os.path.join(run_dir, checkpoints[-1])


def main() -> None:
    task_id = f"Lekiwi-ConeNav-{args.task.capitalize()}-v0"
    # parse_env_cfg / load_cfg_from_registry are Isaac Lab's own real entry-point
    # resolution helpers (isaaclab_tasks.utils) -- confirmed via source research, not
    # guessed. They handle string/class/instance entry-point kwargs internally, so
    # there's no need to hand-introspect gym.spec(...).kwargs (an earlier, less robust
    # version of this script did that directly).
    env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")

    env_cfg.seed = args.seed

    run_dir = os.path.join(args.log_root, agent_cfg.experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    resume_path = None if args.fresh_start else _find_latest_checkpoint(run_dir)
    if resume_path:
        print(f"[train.py] Resuming from {resume_path}")
    else:
        print(f"[train.py] No checkpoint found in {run_dir}, starting fresh")

    env = gym.make(task_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=run_dir, device=agent_cfg.device)
    if resume_path:
        runner.load(resume_path)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
