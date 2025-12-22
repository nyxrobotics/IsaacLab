# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import datetime as _datetime
import faulthandler
import importlib.metadata as metadata
import os
from pathlib import Path
import platform
import signal
import sys
import threading
import time
import traceback

import torch
import torch.nn.functional as Functional
from packaging import version
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.actor_critic import ActorCritic
from torch.distributions import Normal

from isaaclab.app import AppLauncher

# =====================================================================
# Heartbeat (in-memory only)
# =====================================================================
_LAST_HEARTBEAT_TIME = time.time()
_LAST_HEARTBEAT_TAG = "startup"
_FIRST_ITERATION_DONE = False


def _touch_heartbeat(tag: str) -> None:
    global _LAST_HEARTBEAT_TIME, _LAST_HEARTBEAT_TAG
    _LAST_HEARTBEAT_TIME = time.time()
    _LAST_HEARTBEAT_TAG = tag


def _get_heartbeat_age_s() -> float:
    return time.time() - _LAST_HEARTBEAT_TIME


def _is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


# =====================================================================
# SAFETY PATCH 1: Clamp policy.log_std after every PPO.update
#   + warn when invalid values are detected
# =====================================================================
_original_ppo_update = PPO.update


def _safe_ppo_update(self, *args, **kwargs):
    global _FIRST_ITERATION_DONE

    # Progress heartbeat: entering PPO.update means the training loop is alive.
    _touch_heartbeat("ppo_update_enter")

    # Run original update
    out = _original_ppo_update(self, *args, **kwargs)

    # Progress heartbeat: PPO.update finished successfully
    _touch_heartbeat("ppo_update")
    _FIRST_ITERATION_DONE = True

    # Safety guard for log_std
    with torch.no_grad():
        policy = self.policy
        if hasattr(policy, "log_std"):
            log_std = policy.log_std.data

            # Detect invalid values
            invalid_mask = ~torch.isfinite(log_std)
            if invalid_mask.any():
                print(
                    "[WARNING] Detected invalid policy.log_std values (NaN or Inf). "
                    "They have been reset to 0.0.",
                    flush=True,
                )
                log_std[invalid_mask] = 0.0

            # Detect values outside safe range
            too_low = (log_std < -100.0)
            too_high = (log_std > 100.0)
            if too_low.any() or too_high.any():
                print(
                    "[WARNING] Detected policy.log_std outside safe range (-100, 100). "
                    "Values have been clamped.",
                    flush=True,
                )

            # Clamp range so std = exp(log_std) stays valid
            log_std.clamp_(min=-100.0, max=100.0)

            # Write back
            policy.log_std.data.copy_(log_std)

    return out


# Patch PPO.update
PPO.update = _safe_ppo_update
print("[INFO] Patched rsl_rl PPO.update with log_std clamp + in-memory heartbeat", flush=True)

# =====================================================================
# SAFETY PATCH 2:
#   Patch ActorCritic._update_distribution so Normal(mean, std) never
#   receives negative/non-finite std.
# =====================================================================


def _safe_update_distribution(self, obs):
    """Replacement for ActorCritic._update_distribution.

    Handles both:
      - state_dependent_std=True:
          * noise_std_type == "scalar": std may be negative -> softplus + eps
          * noise_std_type == "log": std = exp(log_std) (optionally clamp log_std)
      - state_dependent_std=False:
          * "scalar": parameter std might become invalid -> clamp
          * "log": clamp log_std -> exp
    """
    if getattr(self, "_std_guard_printed", False) is False:
        print(
            "[INFO] Patched ActorCritic._update_distribution: std is guarded (finite, >= 1e-6).",
            flush=True,
        )
        self._std_guard_printed = True

    if self.state_dependent_std:
        mean_and_std = self.actor(obs)

        if self.noise_std_type == "scalar":
            mean, std = torch.unbind(mean_and_std, dim=-2)
            std = Functional.softplus(std) + 1e-6
            std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
            std = std.clamp(min=1e-6, max=1e6)

        elif self.noise_std_type == "log":
            mean, log_std = torch.unbind(mean_and_std, dim=-2)
            log_std = torch.nan_to_num(log_std, nan=0.0, posinf=0.0, neginf=0.0)
            log_std = log_std.clamp(-100.0, 100.0)
            std = torch.exp(log_std)

        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )

    else:
        mean = self.actor(obs)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
            std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
            std = std.clamp(min=1e-6, max=1e6)

        elif self.noise_std_type == "log":
            log_std = torch.nan_to_num(self.log_std, nan=0.0, posinf=0.0, neginf=0.0)
            log_std = log_std.clamp(-100.0, 100.0)
            std = torch.exp(log_std).expand_as(mean)

        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )

    # Create distribution (this must never throw)
    self.distribution = Normal(mean, std)


# Patch ActorCritic._update_distribution
ActorCritic._update_distribution = _safe_update_distribution

# Local imports
import cli_args  # isort: skip

# =====================================================================
# CLI args
# =====================================================================
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")

parser.add_argument(
    "--run_dir",
    type=str,
    default=None,
    help="Fixed directory for logs/checkpoints (reused across restarts).",
)
parser.add_argument(
    "--auto_resume",
    action="store_true",
    default=False,
    help="Automatically resume from the latest checkpoint found in run_dir.",
)
parser.add_argument(
    "--hang_timeout_s",
    type=int,
    default=None,
    help="If no heartbeat update for this many seconds, attempt emergency checkpoint and exit. "
    "Default: 600 seconds (10 minutes).",
)
parser.add_argument(
    "--startup_timeout_s",
    type=int,
    default=1800,
    help="If no heartbeat update before iteration 1 for this many seconds, attempt emergency checkpoint and exit.",
)

# Append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# ---------------------------------------------------------------------
# NCCL timeout / watchdog timeout synchronization
#   - Use longer default timeout (600s = 10 minutes) for stability
#   - If user specifies --hang_timeout_s, we also set TORCH_NCCL_TIMEOUT
#     to the same value so collective timeouts happen within ~that window.
# ---------------------------------------------------------------------
_env_torch_nccl_timeout = os.environ.get("TORCH_NCCL_TIMEOUT", "")
try:
    _env_torch_nccl_timeout_s = int(_env_torch_nccl_timeout) if _env_torch_nccl_timeout else 600
except Exception:
    _env_torch_nccl_timeout_s = 600

if args_cli.hang_timeout_s is None:
    # Default to 600 seconds (10 minutes) instead of 180
    args_cli.hang_timeout_s = max(_env_torch_nccl_timeout_s, 600)
else:
    os.environ["TORCH_NCCL_TIMEOUT"] = str(int(args_cli.hang_timeout_s))

# Ensure startup_timeout_s is always at least hang_timeout_s unless explicitly set smaller.
try:
    if args_cli.startup_timeout_s is None:
        args_cli.startup_timeout_s = max(int(args_cli.hang_timeout_s), 1800)
    else:
        args_cli.startup_timeout_s = int(args_cli.startup_timeout_s)
except Exception:
    args_cli.startup_timeout_s = 1800

# ---------------------------------------------------------------------
# Patch torch.distributed.init_process_group to inject timeout derived from
# TORCH_NCCL_TIMEOUT when caller doesn't specify one.
# ---------------------------------------------------------------------
try:
    import torch.distributed as _dist

    _orig_init_pg = _dist.init_process_group
    _patched_pg_printed = False

    def _init_process_group_with_timeout(*pg_args, **pg_kwargs):
        global _patched_pg_printed
        try:
            _sec = int(os.environ.get("TORCH_NCCL_TIMEOUT", "600"))
        except Exception:
            _sec = 600

        if pg_kwargs.get("timeout", None) is None:
            pg_kwargs["timeout"] = _datetime.timedelta(seconds=_sec)
            if not _patched_pg_printed:
                print(
                    f"[INFO] Patched torch.distributed.init_process_group timeout to {_sec}s "
                    "(affects ProcessGroupNCCL collectives).",
                    flush=True,
                )
                _patched_pg_printed = True

        return _orig_init_pg(*pg_args, **pg_kwargs)

    _dist.init_process_group = _init_process_group_with_timeout

    # These envs help NCCL fail fast / surface errors earlier.
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("NCCL_TIMEOUT", os.environ.get("TORCH_NCCL_TIMEOUT", "600"))
except Exception:
    pass

# =====================================================================
# WATCHDOG / AUTO-RESUME utilities
# =====================================================================
faulthandler.enable(all_threads=True)


def _list_checkpoints_sorted(run_dir: str) -> list[str]:
    """Return checkpoint-like files in run_dir sorted by mtime (newest first)."""
    exts = (".pt", ".pth", ".ckpt")
    root = Path(run_dir)
    if not root.exists():
        return []
    candidates: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in exts:
            candidates.append(p)
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return [str(p) for p in candidates]


def _find_latest_checkpoint(run_dir: str) -> str | None:
    ckpts = _list_checkpoints_sorted(run_dir)
    return ckpts[0] if ckpts else None


def _try_load_checkpoints(runner, checkpoint_paths: list[str]) -> str | None:
    """Try loading checkpoints in order until one succeeds."""
    if runner is None:
        return None
    for p in checkpoint_paths:
        try:
            runner.load(p)
            return p
        except Exception as exc:
            print(f"[WARN] Failed to load checkpoint: {p} ({type(exc).__name__}: {exc})", flush=True)
    return None


def _find_latest_checkpoint_in_tree(root_dir: str) -> tuple[str | None, str | None]:
    root = Path(root_dir)
    if not root.exists():
        return None, None
    patterns = ["**/*.pt", "**/*.pth", "**/*.ckpt"]
    files: list[Path] = []
    for pat in patterns:
        files.extend(root.glob(pat))
    files = [p for p in files if p.is_file()]
    if not files:
        return None, None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    ckpt = files[0]
    return str(ckpt), str(ckpt.parent)


def _maybe_reuse_existing_run_dir(default_log_dir: str) -> tuple[str | None, str | None]:
    parent = str(Path(default_log_dir).parent)
    return _find_latest_checkpoint_in_tree(parent)


def _emergency_save(runner, run_dir: str | None, tag: str) -> None:
    if run_dir is None or runner is None:
        return
    if not _is_rank0():
        return
    try:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = str(Path(run_dir) / f"emergency_{tag}_{ts}.pt")
        if hasattr(runner, "save"):
            runner.save(out)
        elif hasattr(runner, "save_checkpoint"):
            runner.save_checkpoint(out)
        else:
            print("[WARN] Runner has no save/save_checkpoint. Skipping emergency checkpoint.", flush=True)
            return
        print(f"[INFO] Emergency checkpoint saved: {out}", flush=True)
    except Exception:
        print("[ERROR] Emergency checkpoint failed.", flush=True)
        traceback.print_exc()


def _start_hang_watchdog(
    get_runner_fn,
    run_dir: str | None,
    hang_timeout_s: int,
    startup_timeout_s: int,
    distributed: bool = False,
) -> None:
    if hang_timeout_s <= 0 and startup_timeout_s <= 0:
        return

    if distributed and (not _is_rank0()):
        return

    def _worker():
        while True:
            time.sleep(5)
            dt = _get_heartbeat_age_s()

            if not _FIRST_ITERATION_DONE:
                limit = startup_timeout_s
                phase = "startup"
            else:
                limit = hang_timeout_s
                phase = "training"

            if limit is not None and limit > 0 and dt > limit:
                print(
                    f"[ERROR] Hang suspected during {phase}: no heartbeat for {dt:.1f}s "
                    f"(limit={limit}s, last_tag={_LAST_HEARTBEAT_TAG}).",
                    flush=True,
                )
                try:
                    faulthandler.dump_traceback(all_threads=True)
                except Exception:
                    pass

                try:
                    runner = get_runner_fn()
                except Exception:
                    runner = None

                _emergency_save(runner, run_dir, tag=f"hang_{phase}")

                try:
                    os.killpg(os.getpgid(0), signal.SIGTERM)
                except Exception:
                    pass
                time.sleep(2)
                os._exit(1)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    print(
        f"[INFO] Hang watchdog enabled (startup_timeout_s={startup_timeout_s}, hang_timeout_s={hang_timeout_s}).",
        flush=True,
    )


def _get_completed_learning_iterations(runner) -> int:
    """Try to infer how many learning iterations were already completed."""
    if runner is None:
        return 0

    for attr in (
        "current_learning_iteration",
        "current_iteration",
        "learning_iteration",
        "iteration",
        "it",
    ):
        v = getattr(runner, attr, None)
        if isinstance(v, int) and v >= 0:
            return v

    state = getattr(runner, "state_dict", None)
    if callable(state):
        try:
            sd = state()
            if isinstance(sd, dict):
                for k in ("current_learning_iteration", "current_iteration", "iteration", "it"):
                    v = sd.get(k, None)
                    if isinstance(v, int) and v >= 0:
                        return v
        except Exception:
            pass

    return 0


# Always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =====================================================================
# Check for minimum supported RSL-RL version.
# =====================================================================
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        "Please install the correct version of RSL-RL.\n"
        f"Existing version is: '{installed_version}' and required version is: '{RSL_RL_VERSION}'.\n"
        "To install the correct version, run:\n\n\t"
        + " ".join(cmd)
        + "\n",
        flush=True,
    )
    exit(1)

# =====================================================================
# The rest follows.
# =====================================================================
import gymnasium as gym
from datetime import datetime

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


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""

    if args_cli.run_dir is not None:
        os.environ["ISAACLAB_RUN_DIR"] = os.path.abspath(args_cli.run_dir)

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}", flush=True)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}", flush=True)
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if args_cli.run_dir is not None:
        log_dir = os.path.abspath(args_cli.run_dir)
        log_root_path = log_dir
        os.makedirs(log_dir, exist_ok=True)
        print(f"[INFO] Using fixed run_dir for logs/checkpoints: {log_dir}", flush=True)
    elif args_cli.auto_resume:
        ckpt, ckpt_parent = _maybe_reuse_existing_run_dir(log_dir)
        if ckpt_parent is not None:
            if _is_rank0():
                print(f"[INFO] Auto-resume (no --run_dir): reusing existing run_dir={ckpt_parent}", flush=True)
            log_dir = ckpt_parent
            log_root_path = ckpt_parent

    effective_run_dir = os.path.abspath(log_dir)

    _touch_heartbeat("before_env_make")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    resume_path = None
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.", flush=True)
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    _touch_heartbeat("before_runner_init")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner_holder = {"runner": runner}

    def _handle_signal(sig: int, _frame) -> None:
        print(f"[WARN] Received signal {sig}. Attempting emergency checkpoint then exiting.", flush=True)
        _emergency_save(runner_holder.get("runner"), effective_run_dir, tag=f"signal{sig}")
        os._exit(128 + int(sig))

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    _start_hang_watchdog(
        lambda: runner_holder.get("runner"),
        effective_run_dir,
        args_cli.hang_timeout_s,
        args_cli.startup_timeout_s,
        distributed=args_cli.distributed,
    )

    # Auto-resume logic - try to load latest checkpoint from run_dir
    if args_cli.auto_resume:
        ckpt_candidates = _list_checkpoints_sorted(effective_run_dir)
        if ckpt_candidates:
            if _is_rank0():
                print(
                    f"[INFO] Auto-resume: probing {len(ckpt_candidates)} checkpoint(s) in run_dir={effective_run_dir}",
                    flush=True,
                )
            loaded = _try_load_checkpoints(runner, ckpt_candidates)
            if loaded is not None:
                if _is_rank0():
                    print(f"[INFO] Auto-resume: loaded checkpoint: {loaded}", flush=True)
            else:
                if _is_rank0():
                    print(
                        f"[WARN] Auto-resume: no usable checkpoint found in run_dir={effective_run_dir}. Starting from scratch.",
                        flush=True,
                    )
        else:
            if _is_rank0():
                print(f"[INFO] Auto-resume: no checkpoint found in run_dir={effective_run_dir}", flush=True)

    # Add git repo to log (moved outside of auto_resume block - this was the bug!)
    runner.add_git_repo_to_log(__file__)

    # Load specific resume_path if provided (separate from auto-resume)
    if resume_path is not None:
        if _is_rank0():
            print(f"[INFO] Loading model checkpoint from: {resume_path}", flush=True)
        try:
            runner.load(resume_path)
            if _is_rank0():
                print(f"[INFO] Loaded checkpoint: {resume_path}", flush=True)
        except Exception as exc:
            if _is_rank0():
                print(
                    f"[WARN] Failed to load requested checkpoint: {resume_path} "
                    f"({type(exc).__name__}: {exc}).",
                    flush=True,
                )
            # Optional fallback: walk back through run_dir checkpoints if enabled
            if args_cli.auto_resume:
                ckpt_candidates = [p for p in _list_checkpoints_sorted(effective_run_dir) if p != resume_path]
                if ckpt_candidates:
                    if _is_rank0():
                        print(
                            f"[INFO] Fallback auto-resume: probing {len(ckpt_candidates)} checkpoint(s) in run_dir={effective_run_dir}",
                            flush=True,
                        )
                    loaded = _try_load_checkpoints(runner, ckpt_candidates)
                    if loaded is not None:
                        if _is_rank0():
                            print(f"[INFO] Fallback auto-resume: loaded checkpoint: {loaded}", flush=True)
                    else:
                        if _is_rank0():
                            print("[WARN] Fallback auto-resume: no usable checkpoint found. Starting from scratch.", flush=True)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    _touch_heartbeat("before_learn")

    requested = int(agent_cfg.max_iterations)
    completed = _get_completed_learning_iterations(runner)
    remaining = max(requested - completed, 0)

    if completed > 0 and _is_rank0():
        print(
            f"[INFO] Resume adjustment: completed={completed}, requested={requested}, remaining={remaining}",
            flush=True,
        )

    if remaining <= 0:
        if _is_rank0():
            print("[INFO] Nothing to do: remaining iterations is 0. Exiting.", flush=True)
        env.close()
        return

    try:
        runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)
    except BaseException:
        _emergency_save(runner, effective_run_dir, tag="exception")
        raise

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
