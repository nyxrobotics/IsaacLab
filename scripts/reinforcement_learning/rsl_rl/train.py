# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)

# =====================================================================
# EMERGENCY CHECKPOINT + HANG WATCHDOG (minimal add-on)
#   - Saves emergency checkpoint on:
#       * suspected hang (no progress heartbeat)
#       * SIGTERM/SIGINT
#       * unhandled exception
#   - Exits non-zero so an external bash loop can restart.
# =====================================================================
parser.add_argument(
    "--run_dir",
    type=str,
    default=None,
    help="Fixed directory for logs/checkpoints (reused across restarts). If set, logs go here.",
)
parser.add_argument(
    "--auto_resume",
    action="store_true",
    default=False,
    help="If set, automatically resume from the latest checkpoint found in run_dir.",
)
parser.add_argument(
    "--hang_timeout_s",
    type=int,
    default=0,
    help="If >0 and no progress heartbeat for this many seconds (after iter>=1), save emergency checkpoint and exit.",
)
parser.add_argument(
    "--startup_timeout_s",
    type=int,
    default=600,
    help="Startup hang timeout in seconds before Learning iteration 1 appears (iter<1).",
)

# =====================================================================
# NCCL / ProcessGroup timeout knobs (env-based; set as early as possible)
#   - This does NOT solve the hang root-cause.
#   - It just makes ALLREDUCE time out earlier instead of waiting 10 minutes.
# =====================================================================
parser.add_argument(
    "--nccl_timeout_s",
    type=int,
    default=60,
    help="Timeout (seconds) for NCCL collectives / process group watchdog (best-effort via env vars).",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# ---------------------------------------------------------------------
# Apply NCCL timeout env vars BEFORE launching the app / importing torch.
# ---------------------------------------------------------------------
import os

# Best-effort: enable blocking wait + async error handling so timeouts surface.
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

# Many setups still honor the process group timeout via env; set both.
# NOTE: some stacks ignore TORCH_NCCL_TIMEOUT; keeping both costs nothing.
if args_cli.nccl_timeout_s is not None and int(args_cli.nccl_timeout_s) > 0:
    os.environ["TORCH_NCCL_TIMEOUT"] = str(int(args_cli.nccl_timeout_s))
    os.environ["TORCH_DISTRIBUTED_DEFAULT_TIMEOUT"] = str(int(args_cli.nccl_timeout_s))

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime
from pathlib import Path
import time
import threading
import faulthandler
import traceback
import signal

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# -----------------------------
# Heartbeat + watchdog helpers
# -----------------------------
faulthandler.enable(all_threads=True)

_LAST_HEARTBEAT_TIME = time.time()
_LAST_HEARTBEAT_TAG = "startup"


def _touch_heartbeat(tag: str) -> None:
    global _LAST_HEARTBEAT_TIME, _LAST_HEARTBEAT_TAG
    _LAST_HEARTBEAT_TIME = time.time()
    _LAST_HEARTBEAT_TAG = tag


def _get_heartbeat_age_s() -> float:
    return time.time() - _LAST_HEARTBEAT_TIME


def _is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _find_latest_checkpoint(run_dir: str) -> str | None:
    root = Path(run_dir)
    if not root.exists():
        return None
    exts = (".pt", ".pth", ".ckpt")
    candidates: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in exts:
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def _emergency_save(runner, out_dir: str | None, tag: str) -> None:
    if out_dir is None or runner is None:
        return
    if not _is_rank0():
        return
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = str(Path(out_dir) / f"emergency_{tag}_{ts}.pt")

        if hasattr(runner, "save"):
            runner.save(out_path)
        elif hasattr(runner, "save_checkpoint"):
            runner.save_checkpoint(out_path)
        else:
            print("[WARN] Runner has no save/save_checkpoint. Skipping emergency checkpoint.", flush=True)
            return

        print(f"[INFO] Emergency checkpoint saved: {out_path}", flush=True)
    except Exception:
        print("[ERROR] Emergency checkpoint failed.", flush=True)
        traceback.print_exc()


def _start_hang_watchdog(get_runner_fn, out_dir: str | None, startup_timeout_s: int, hang_timeout_s: int) -> None:
    # Rank0 only
    if not _is_rank0():
        return
    if (startup_timeout_s is None or startup_timeout_s <= 0) and (hang_timeout_s is None or hang_timeout_s <= 0):
        return

    def _get_iter(runner) -> int:
        if runner is None:
            return 0
        it = getattr(runner, "current_learning_iteration", None)
        if it is None:
            return 0
        try:
            return int(it)
        except Exception:
            return 0

    def _worker():
        while True:
            time.sleep(5)
            dt = _get_heartbeat_age_s()

            runner = None
            try:
                runner = get_runner_fn()
            except Exception:
                runner = None

            it = _get_iter(runner)
            timeout_s = startup_timeout_s if it < 1 else hang_timeout_s

            if timeout_s is None or timeout_s <= 0:
                continue

            if dt > timeout_s:
                print(
                    f"[ERROR] Hang suspected: no heartbeat for {dt:.1f}s (last_tag={_LAST_HEARTBEAT_TAG}, iter={it}).",
                    flush=True,
                )
                try:
                    faulthandler.dump_traceback(all_threads=True)
                except Exception:
                    pass

                _emergency_save(runner, out_dir, tag="hang")

                # Exit non-zero so an external supervisor (bash loop) can restart.
                try:
                    os.killpg(os.getpgid(0), signal.SIGTERM)
                except Exception:
                    pass
                time.sleep(2)
                os._exit(1)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    print(
        f"[INFO] Hang watchdog enabled on rank0 (startup_timeout_s={startup_timeout_s}, hang_timeout_s={hang_timeout_s}).",
        flush=True,
    )


# -----------------------------
# Patch PPO.update only to mark progress (heartbeat)
#   - No behavior changes to the algorithm.
# -----------------------------
try:
    from rsl_rl.algorithms.ppo import PPO as _PPO

    _orig_ppo_update = _PPO.update

    def _hb_ppo_update(self, *args, **kwargs):
        _touch_heartbeat("ppo_update_enter")
        out = _orig_ppo_update(self, *args, **kwargs)
        _touch_heartbeat("ppo_update_exit")
        return out

    _PPO.update = _hb_ppo_update
    if _is_rank0():
        print("[INFO] Patched rsl_rl PPO.update with heartbeat only.", flush=True)
except Exception:
    if _is_rank0():
        print("[WARN] Failed to patch PPO.update for heartbeat. Continuing without heartbeat patch.", flush=True)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    if _is_rank0():
        # show effective NCCL timeout config (best-effort)
        print(
            "[INFO] NCCL timeout env: "
            f"TORCH_NCCL_TIMEOUT={os.environ.get('TORCH_NCCL_TIMEOUT')} "
            f"TORCH_DISTRIBUTED_DEFAULT_TIMEOUT={os.environ.get('TORCH_DISTRIBUTED_DEFAULT_TIMEOUT')} "
            f"NCCL_BLOCKING_WAIT={os.environ.get('NCCL_BLOCKING_WAIT')} "
            f"NCCL_ASYNC_ERROR_HANDLING={os.environ.get('NCCL_ASYNC_ERROR_HANDLING')}",
            flush=True,
        )

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # override to fixed run_dir if provided (for restarts)
    if args_cli.run_dir is not None:
        log_dir = os.path.abspath(args_cli.run_dir)
        log_root_path = log_dir
        os.makedirs(log_dir, exist_ok=True)
        if _is_rank0():
            print(f"[INFO] Using fixed run_dir for logs/checkpoints: {log_dir}", flush=True)

    effective_run_dir = os.path.abspath(log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner_holder = {"runner": runner}

    def _handle_signal(sig: int, _frame) -> None:
        if _is_rank0():
            print(f"[WARN] Received signal {sig}. Attempting emergency checkpoint then exiting.", flush=True)
        _emergency_save(runner_holder.get("runner"), effective_run_dir, tag=f"signal{sig}")
        os._exit(128 + int(sig))

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    _start_hang_watchdog(
        lambda: runner_holder.get("runner"),
        effective_run_dir,
        args_cli.startup_timeout_s,
        args_cli.hang_timeout_s,
    )

    runner.add_git_repo_to_log(__file__)

    if args_cli.auto_resume and args_cli.run_dir is not None:
        ckpt = _find_latest_checkpoint(effective_run_dir)
        if ckpt is not None:
            if _is_rank0():
                print(f"[INFO] Auto-resume: loading latest checkpoint: {ckpt}", flush=True)
            runner.load(ckpt)
        else:
            if _is_rank0():
                print(f"[INFO] Auto-resume: no checkpoint found in run_dir={effective_run_dir}", flush=True)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # Keep TOTAL iterations fixed across restarts
    def _get_current_iter() -> int:
        it = getattr(runner, "current_learning_iteration", 0)
        try:
            return int(it)
        except Exception:
            return 0

    total_target = int(agent_cfg.max_iterations)
    already_done = _get_current_iter()
    remaining = max(0, total_target - already_done)

    if _is_rank0():
        print(
            f"[INFO] Iteration plan: total_target={total_target}, already_done={already_done}, remaining={remaining}",
            flush=True,
        )

    _touch_heartbeat("before_learn")
    try:
        if remaining > 0:
            runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)
        else:
            if _is_rank0():
                print("[INFO] Nothing to do: already reached max_iterations. Exiting.", flush=True)
    except BaseException:
        if _is_rank0():
            print("[ERROR] Unhandled exception during training. Saving emergency checkpoint.", flush=True)
        _emergency_save(runner, effective_run_dir, tag="exception")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
