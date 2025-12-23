# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL.

This file contains small safety/robustness patches to improve stability in long-running,
distributed training (auto-resume, watchdog, and numerical guards).
"""

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
import types as _types

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
# SAFETY PATCH 0: make Normal(scale) always valid (finite, >= eps)
#   Prevents hard-crashes like:
#     RuntimeError: normal expects all elements of std >= 0.0
# =====================================================================
import torch.distributions.normal as _dist_normal  # noqa: E402

_ORIG_NORMAL_INIT = _dist_normal.Normal.__init__
_WARNED_NEG_STD = False


def _safe_normal_init(self, loc, scale, validate_args=None, *args, **kwargs):
    global _WARNED_NEG_STD

    eps = 1e-3

    if torch.is_tensor(scale):
        scale_clean = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)

        if (not _WARNED_NEG_STD) and torch.any(scale_clean < 0):
            try:
                mn = float(scale_clean.min().detach().cpu())
                mx = float(scale_clean.max().detach().cpu())
                print(
                    f"[WARN] Normal(scale) had negative values. min={mn}, max={mx}. Applying abs()+clamp_min().",
                    flush=True,
                )
            except Exception:
                print("[WARN] Normal(scale) had negative values. Applying abs()+clamp_min().", flush=True)
            _WARNED_NEG_STD = True

        scale_safe = scale_clean.abs().clamp_min(eps)
    else:
        try:
            scale_val = float(scale)
        except Exception:
            scale_val = 0.0
        scale_safe = max(abs(scale_val), eps)

    return _ORIG_NORMAL_INIT(self, loc, scale_safe, validate_args=validate_args)


_dist_normal.Normal.__init__ = _safe_normal_init
print("[INFO] Patched torch.distributions.Normal to enforce scale >= 1e-3.", flush=True)


# =====================================================================
# SAFETY PATCH 1: PPO.update guard
#   - heartbeat updates
#   - detect NaN/Inf in PPO.update outputs and skip contaminating the run
#   - clamp policy.log_std to a safe range
# =====================================================================
_original_ppo_update = PPO.update


def _tensor_has_bad(x) -> bool:
    if x is None:
        return False
    if torch.is_tensor(x):
        return (not torch.isfinite(x).all()).item()
    return False


def _safe_ppo_update(self, *args, **kwargs):
    global _FIRST_ITERATION_DONE

    _touch_heartbeat("ppo_update_enter")

    out = _original_ppo_update(self, *args, **kwargs)

    # Detect invalid update outputs early.
    bad = False
    if isinstance(out, (tuple, list)):
        for v in out:
            bad = bad or _tensor_has_bad(v)
    elif isinstance(out, dict):
        for v in out.values():
            bad = bad or _tensor_has_bad(v)
    elif torch.is_tensor(out):
        bad = _tensor_has_bad(out)

    if bad:
        print("[ERROR] Detected NaN/Inf in PPO.update outputs. Skipping this update.", flush=True)

        # Best-effort: clear gradients so the next step starts cleanly.
        try:
            for opt_name in ("optimizer", "actor_optimizer", "critic_optimizer"):
                opt = getattr(self, opt_name, None)
                if opt is not None and hasattr(opt, "zero_grad"):
                    opt.zero_grad(set_to_none=True)
        except Exception:
            pass

        # Best-effort: reduce LR to recover from a blow-up.
        try:
            for opt_name in ("optimizer", "actor_optimizer", "critic_optimizer"):
                opt = getattr(self, opt_name, None)
                if opt is None:
                    continue
                for pg in opt.param_groups:
                    pg["lr"] = float(pg.get("lr", 0.0)) * 0.5
            print("[WARN] Halved optimizer learning rate(s) due to NaN/Inf.", flush=True)
        except Exception:
            pass

        _touch_heartbeat("ppo_update_nan_skipped")
        _FIRST_ITERATION_DONE = True
        return out

    _touch_heartbeat("ppo_update")
    _FIRST_ITERATION_DONE = True

    # Clamp log_std so std = exp(log_std) stays sane.
    with torch.no_grad():
        policy = getattr(self, "policy", None)
        if policy is not None and hasattr(policy, "log_std"):
            log_std = policy.log_std.data

            invalid_mask = ~torch.isfinite(log_std)
            if invalid_mask.any():
                print("[WARNING] Detected invalid policy.log_std (NaN/Inf). Reset to 0.0.", flush=True)
                log_std[invalid_mask] = 0.0

            log_std.clamp_(min=-100.0, max=100.0)
            policy.log_std.data.copy_(log_std)

    return out


PPO.update = _safe_ppo_update
print("[INFO] Patched rsl_rl PPO.update with NaN/Inf guard + log_std clamp + heartbeat.", flush=True)


# =====================================================================
# SAFETY PATCH 2:
#   Patch ActorCritic._update_distribution so Normal(mean, std) never
#   receives negative/non-finite std, and std keeps a minimum exploration.
# =====================================================================
def _safe_update_distribution(self, obs):
    if getattr(self, "_std_guard_printed", False) is False:
        print(
            "[INFO] Patched ActorCritic._update_distribution: std is guarded (finite, >= 1e-3).",
            flush=True,
        )
        self._std_guard_printed = True

    std_min = 1e-3
    std_max = 1e3

    if self.state_dependent_std:
        mean_and_std = self.actor(obs)

        if self.noise_std_type == "scalar":
            mean, std = torch.unbind(mean_and_std, dim=-2)
            std = Functional.softplus(std) + std_min
            std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
            std = std.clamp(min=std_min, max=std_max)

        elif self.noise_std_type == "log":
            mean, log_std = torch.unbind(mean_and_std, dim=-2)
            log_std = torch.nan_to_num(log_std, nan=0.0, posinf=0.0, neginf=0.0)
            log_std = log_std.clamp(-100.0, 100.0)
            std = torch.exp(log_std).clamp(min=std_min, max=std_max)

        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )

    else:
        mean = self.actor(obs)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
            std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
            std = std.clamp(min=std_min, max=std_max)

        elif self.noise_std_type == "log":
            log_std = torch.nan_to_num(self.log_std, nan=0.0, posinf=0.0, neginf=0.0)
            log_std = log_std.clamp(-100.0, 100.0)
            std = torch.exp(log_std).expand_as(mean).clamp(min=std_min, max=std_max)

        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )

    self.distribution = Normal(mean, std)


ActorCritic._update_distribution = _safe_update_distribution


# =====================================================================
# SAFETY PATCH 3:
#   Sanitize actions before env.step() to avoid simulator freeze.
#   - Replace NaN/Inf with 0
#   - Clamp to action_space bounds (or fallback to [-1, 1])
# =====================================================================
def _sanitize_actions_for_env(env, actions: torch.Tensor) -> torch.Tensor:
    a = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)

    low = None
    high = None
    try:
        if hasattr(env, "action_space") and env.action_space is not None:
            if hasattr(env.action_space, "low") and hasattr(env.action_space, "high"):
                low = env.action_space.low
                high = env.action_space.high
    except Exception:
        low = None
        high = None

    if low is not None and high is not None:
        low_t = torch.as_tensor(low, device=a.device, dtype=a.dtype)
        high_t = torch.as_tensor(high, device=a.device, dtype=a.dtype)
        a = torch.max(torch.min(a, high_t), low_t)
    else:
        a = a.clamp(-1.0, 1.0)

    return a


# Local imports
import cli_args  # isort: skip  # noqa: E402

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
    "Default: TORCH_NCCL_TIMEOUT (or 180 if unset).",
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
# ---------------------------------------------------------------------
_env_torch_nccl_timeout = os.environ.get("TORCH_NCCL_TIMEOUT", "")
try:
    _env_torch_nccl_timeout_s = int(_env_torch_nccl_timeout) if _env_torch_nccl_timeout else 180
except Exception:
    _env_torch_nccl_timeout_s = 180

if args_cli.hang_timeout_s is None:
    args_cli.hang_timeout_s = _env_torch_nccl_timeout_s
else:
    os.environ["TORCH_NCCL_TIMEOUT"] = str(int(args_cli.hang_timeout_s))

try:
    if args_cli.startup_timeout_s is None:
        args_cli.startup_timeout_s = max(int(args_cli.hang_timeout_s), 1800)
    else:
        args_cli.startup_timeout_s = int(args_cli.startup_timeout_s)
except Exception:
    args_cli.startup_timeout_s = 1800

# ---------------------------------------------------------------------
# Patch torch.distributed.init_process_group to inject timeout derived from TORCH_NCCL_TIMEOUT.
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


def _runner_has_nonfinite_params(runner) -> bool:
    """Return True if runner's policy parameters contain NaN/Inf."""
    try:
        alg = getattr(runner, "alg", None)
        if alg is None:
            return False

        policy = getattr(alg, "policy", None)
        if policy is None:
            policy = getattr(runner, "policy", None)

        if policy is None or not hasattr(policy, "parameters"):
            return False

        with torch.no_grad():
            for p in policy.parameters():
                if p is None:
                    continue
                if torch.is_tensor(p) and (not torch.isfinite(p).all()).item():
                    return True
    except Exception:
        # If we cannot validate, do not mark as bad here.
        return False

    return False


def _delete_bad_checkpoint(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
        print(f"[WARN] Deleted bad checkpoint: {path}", flush=True)
    except Exception as exc:
        print(f"[WARN] Failed to delete checkpoint {path}: {type(exc).__name__}: {exc}", flush=True)


def _try_load_checkpoints(runner, checkpoint_paths: list[str]) -> str | None:
    """Try loading checkpoints in order until one succeeds and looks numerically sane.

    Strategy:
      1) newest -> oldest
      2) if load fails OR loaded model has NaN/Inf parameters, treat it as bad
      3) delete bad checkpoint (rank0 only) and continue searching older ones
      4) if none works, return None
    """
    if runner is None:
        return None

    for p in checkpoint_paths:
        try:
            runner.load(p)
        except Exception as exc:
            print(f"[WARN] Failed to load checkpoint: {p} ({type(exc).__name__}: {exc})", flush=True)
            if _is_rank0():
                _delete_bad_checkpoint(p)
            continue

        # Validate the loaded checkpoint.
        if _runner_has_nonfinite_params(runner):
            print(f"[WARN] Loaded checkpoint has NaN/Inf parameters, skipping: {p}", flush=True)
            if _is_rank0():
                _delete_bad_checkpoint(p)
            continue

        return p

    return None


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
        "To install the correct version, run:\n\n\t" + " ".join(cmd) + "\n",
        flush=True,
    )
    exit(1)

# =====================================================================
# The rest follows.
# =====================================================================
import gymnasium as gym  # noqa: E402
from datetime import datetime  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_pickle, dump_yaml  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

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

    # Apply SAFETY PATCH 3: sanitize actions in env.step()
    try:
        _orig_step = env.step

        def _step_sanitized(self, actions):
            actions = _sanitize_actions_for_env(self, actions)
            return _orig_step(actions)

        env.step = _types.MethodType(_step_sanitized, env)
        if _is_rank0():
            print("[INFO] Patched env.step: sanitize actions (nan_to_num + clamp).", flush=True)
    except Exception as e:
        if _is_rank0():
            print(f"[WARN] Failed to patch env.step for action sanitization: {e}", flush=True)

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

    if args_cli.auto_resume:
        # Prefer newest, but walk back until a loadable + numerically sane checkpoint is found.
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

        runner.add_git_repo_to_log(__file__)

    if resume_path is not None:
        if _is_rank0():
            print(f"[INFO] Loading model checkpoint from: {resume_path}", flush=True)
        try:
            runner.load(resume_path)
            if _runner_has_nonfinite_params(runner):
                raise RuntimeError("Loaded requested checkpoint contains NaN/Inf parameters.")
            if _is_rank0():
                print(f"[INFO] Loaded checkpoint: {resume_path}", flush=True)
        except Exception as exc:
            if _is_rank0():
                print(
                    f"[WARN] Failed to load requested checkpoint: {resume_path} "
                    f"({type(exc).__name__}: {exc}).",
                    flush=True,
                )
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
                            print(
                                "[WARN] Fallback auto-resume: no usable checkpoint found. Starting from scratch.",
                                flush=True,
                            )

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
