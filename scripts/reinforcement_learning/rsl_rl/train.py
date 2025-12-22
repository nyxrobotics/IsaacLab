# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# =====================================================================
# SAFETY PATCH 1: Clamp policy.log_std after every PPO.update
#   + warn when invalid values are detected
# =====================================================================
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import torch
torch._dynamo.disable()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import os
import torch
from rsl_rl.algorithms.ppo import PPO

# =====================================================================
# HEARTBEAT (defined early so safety patches can use it)
# =====================================================================
import time as _time
_LAST_HEARTBEAT_TIME = _time.time()
_LAST_HEARTBEAT_TAG = "startup"

def _touch_heartbeat(tag: str) -> None:
    global _LAST_HEARTBEAT_TIME, _LAST_HEARTBEAT_TAG
    _LAST_HEARTBEAT_TIME = _time.time()
    _LAST_HEARTBEAT_TAG = tag

def _get_heartbeat_age_s() -> float:
    return _time.time() - _LAST_HEARTBEAT_TIME


_original_ppo_update = PPO.update

def _safe_ppo_update(self, *args, **kwargs):
    # Run original update
    out = _original_ppo_update(self, *args, **kwargs)

    # progress heartbeat: if PPO.update runs, training is still making progress
    _touch_heartbeat("ppo_update")
    hb_path = os.environ.get("ISAACLAB_HEARTBEAT_PATH")
    if hb_path and _is_rank0():
        _write_heartbeat_file(hb_path)

    # safety guard for log_std
    with torch.no_grad():
        policy = self.policy
        if hasattr(policy, "log_std"):
            log_std = policy.log_std.data

            # detect invalid values
            invalid_mask = ~torch.isfinite(log_std)
            if invalid_mask.any():
                print(
                    "[WARNING] Detected invalid policy.log_std values (NaN or Inf). "
                    "They have been reset to 0.0."
                )
                log_std[invalid_mask] = 0.0

            # detect values outside safe range
            too_low  = (log_std < -20.0)
            too_high = (log_std > 2.0)

            if too_low.any() or too_high.any():
                print(
                    "[WARNING] Detected policy.log_std outside safe range "
                    "(-20, 2). Values have been clamped."
                )

            # clamp range so std = exp(log_std) stays valid
            log_std.clamp_(min=-20.0, max=2.0)

            # write back
            policy.log_std.data.copy_(log_std)

    return out

# Patch PPO.update
PPO.update = _safe_ppo_update
print("[INFO] Patched rsl_rl PPO.update with log_std clamp + heartbeat", flush=True)

# =====================================================================
# SAFETY PATCH 2:
#   Patch ActorCritic._update_distribution so Normal(mean, std) never
#   receives negative/non-finite std. This prevents the crash *before*
#   it happens (better than catching in act()).
# =====================================================================
import torch.nn.functional as Functional
from torch.distributions import Normal
from rsl_rl.modules.actor_critic import ActorCritic

def _safe_update_distribution(self, obs):
    """
    Replacement for ActorCritic._update_distribution.

    Handles both:
      - state_dependent_std=True:
          * noise_std_type == "scalar": std may be negative -> softplus + eps
          * noise_std_type == "log": std = exp(log_std) (optionally clamp log_std)
      - state_dependent_std=False:
          * "scalar": parameter std might become invalid -> clamp
          * "log": clamp log_std -> exp
    """
    if getattr(self, "_std_guard_printed", False) is False:
        # Print once per process
        print("[INFO] Patched ActorCritic._update_distribution: std is guarded (finite, >= 1e-6).", flush=True)
        self._std_guard_printed = True

    if self.state_dependent_std:
        mean_and_std = self.actor(obs)

        if self.noise_std_type == "scalar":
            mean, std = torch.unbind(mean_and_std, dim=-2)

            # Convert potentially-negative std into strictly-positive std.
            std = Functional.softplus(std) + 1e-6
            std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
            std = std.clamp(min=1e-6, max=10.0)

        elif self.noise_std_type == "log":
            mean, log_std = torch.unbind(mean_and_std, dim=-2)

            # Keep log_std in a sane range to avoid Inf/NaN.
            log_std = torch.nan_to_num(log_std, nan=0.0, posinf=0.0, neginf=0.0)
            log_std = log_std.clamp(-20.0, 2.0)

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
            std = std.clamp(min=1e-6, max=10.0)

        elif self.noise_std_type == "log":
            log_std = torch.nan_to_num(self.log_std, nan=0.0, posinf=0.0, neginf=0.0)
            log_std = log_std.clamp(-20.0, 2.0)
            std = torch.exp(log_std).expand_as(mean)

        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )

    # Create distribution (this must never throw)
    self.distribution = Normal(mean, std)

# Patch ActorCritic._update_distribution
ActorCritic._update_distribution = _safe_update_distribution

# =====================================================================
# SAFETY PATCH 3:
#   - Keep minibatch iteration count identical across all ranks.
#   - Decisions (yield / reuse / skip / pad) are made only from
#     all_reduce results so every rank follows the same control flow.
#   - No forced termination: when consistency cannot be guaranteed,
#     the step is skipped on all ranks.
# =====================================================================

import torch.distributed as dist

def _dist_rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

def _ddp_ready() -> bool:
    return dist.is_available() and dist.is_initialized()

def _ddp_device() -> torch.device:
    # Prefer current CUDA device if available; otherwise CPU.
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")

def _ddp_sum_i32(x: int) -> int:
    if not _ddp_ready():
        return x
    t = torch.tensor([x], device=_ddp_device(), dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())

def _ddp_max_i32(x: int) -> int:
    if not _ddp_ready():
        return x
    t = torch.tensor([x], device=_ddp_device(), dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item())

# Try patching RolloutStorage minibatch generator (names differ by rsl_rl version)
_rollout_storage_cls = None
for _mod_path, _cls_name in [
    ("rsl_rl.storage.rollout_storage", "RolloutStorage"),
    ("rsl_rl.storage", "RolloutStorage"),
]:
    try:
        _m = __import__(_mod_path, fromlist=[_cls_name])
        _rollout_storage_cls = getattr(_m, _cls_name)
        break
    except Exception:
        pass

if _rollout_storage_cls is not None and hasattr(_rollout_storage_cls, "mini_batch_generator"):
    _orig_mbg = _rollout_storage_cls.mini_batch_generator

    def _patched_mini_batch_generator(self, *args, **kwargs):
        last_good = None
        local_yield_count = 0

        # Print once per rank
        if not getattr(self, "_mbg_patch_printed", False):
            print(
                f"[INFO][rank{_dist_rank()}] Patched mini_batch_generator (DDP-safe, rank0-free): "
                "if any rank yields None -> all ranks reuse last_good; "
                "if last_good missing on any rank -> all ranks skip that step; "
                "pad to global max yields (only if last_good exists on all ranks).",
                flush=True,
            )
            self._mbg_patch_printed = True

        for batch in _orig_mbg(self, *args, **kwargs):
            local_is_none = 1 if (batch is None) else 0

            # If not distributed, keep the old local behavior.
            if not _ddp_ready():
                if batch is None:
                    if last_good is not None:
                        print(
                            f"[WARN][rank{_dist_rank()}] minibatch_generator yielded None -> reusing last_good",
                            flush=True,
                        )
                        yield last_good
                        local_yield_count += 1
                    else:
                        # No forced stop: just skip this step.
                        print(
                            f"[WARN][rank{_dist_rank()}] minibatch_generator yielded None but last_good is None -> skip",
                            flush=True,
                        )
                    continue
                last_good = batch
                yield batch
                local_yield_count += 1
                continue

            # DDP path: decide action using only all_reduce (no rank0).
            world_size = dist.get_world_size()
            n_none = _ddp_sum_i32(local_is_none)

            if n_none > 0:
                # At least one rank got None. To keep all ranks aligned,
                # all ranks take the same branch.
                local_has_last = 1 if (last_good is not None) else 0
                n_has_last = _ddp_sum_i32(local_has_last)

                if n_has_last == world_size:
                    # Everyone has last_good -> everyone yields last_good
                    if local_is_none:
                        print(
                            f"[WARN][rank{_dist_rank()}] batch=None detected globally (n_none={n_none}/{world_size}) "
                            "-> yielding last_good (aligned)",
                            flush=True,
                        )
                    yield last_good
                    local_yield_count += 1
                else:
                    # Someone lacks last_good -> no forced termination; skip on all ranks.
                    if local_is_none:
                        print(
                            f"[WARN][rank{_dist_rank()}] batch=None detected globally (n_none={n_none}/{world_size}) "
                            f"but last_good missing on some rank (n_has_last={n_has_last}/{world_size}) "
                            "-> skipping this step on all ranks (aligned)",
                            flush=True,
                        )
                    # Everyone skips this step, so yield count stays aligned.
                continue

            # n_none == 0: all ranks have a valid batch -> safe to yield
            last_good = batch
            yield batch
            local_yield_count += 1

        # Optional padding: ensure all ranks yield the same number of batches.
        if _ddp_ready():
            world_size = dist.get_world_size()
            global_max = _ddp_max_i32(local_yield_count)

            if global_max > local_yield_count:
                local_has_last = 1 if (last_good is not None) else 0
                n_has_last = _ddp_sum_i32(local_has_last)

                if n_has_last == world_size:
                    pad = global_max - local_yield_count
                    print(
                        f"[INFO][rank{_dist_rank()}] Padding minibatches to global_max={global_max} "
                        f"(local={local_yield_count}, pad={pad}) using last_good (aligned)",
                        flush=True,
                    )
                    for _ in range(pad):
                        yield last_good
                else:
                    # If not everyone has last_good, don't pad (still aligned: those ranks also can't pad).
                    print(
                        f"[WARN][rank{_dist_rank()}] global_max={global_max} > local={local_yield_count} "
                        f"but last_good missing on some rank (n_has_last={n_has_last}/{world_size}) "
                        "-> padding disabled to avoid desync",
                        flush=True,
                    )

    _rollout_storage_cls.mini_batch_generator = _patched_mini_batch_generator
else:
    print(f"[WARN][rank{_dist_rank()}] Could not patch RolloutStorage.mini_batch_generator (class not found).", flush=True)

# =====================================================================


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

parser.add_argument("--run_dir", type=str, default=None, help="Fixed directory for logs/checkpoints (reused across restarts).")
parser.add_argument("--auto_resume", action="store_true", default=False, help="Automatically resume from the latest checkpoint found in run_dir.")
parser.add_argument("--heartbeat_path", type=str, default=None, help="Path where rank0 writes a heartbeat timestamp during training.")
parser.add_argument("--hang_timeout_s", type=int, default=180, help="If no heartbeat update for this many seconds, attempt emergency checkpoint and exit.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# =====================================================================
# WATCHDOG / AUTO-RESUME utilities
# =====================================================================
import time
import threading
import faulthandler
import traceback
from pathlib import Path

faulthandler.enable(all_threads=True)

def _is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def _write_heartbeat_file(path: str) -> None:
    # Keep this as small/robust as possible.
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(time.time()))
    except Exception:
        # Never crash training due to heartbeat IO.
        pass

def _find_latest_checkpoint(run_dir: str) -> str | None:
    # Pick the most recently modified checkpoint-like file.
    exts = (".pt", ".pth", ".ckpt")
    candidates: list[Path] = []
    root = Path(run_dir)
    if not root.exists():
        return None
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in exts:
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(candidates[0])

def _find_latest_checkpoint_in_tree(root_dir: str) -> tuple[str | None, str | None]:
    """Search for the most recently modified checkpoint under root_dir.
    Returns (ckpt_path, ckpt_parent_dir)."""
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
    """Try to reuse an existing run dir when --auto_resume is set without --run_dir.
    We search under the parent directory of default_log_dir and pick the newest checkpoint."""
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

def _start_hang_watchdog(get_runner_fn, run_dir: str | None, heartbeat_path: str | None, timeout_s: int) -> None:
    # Best-effort watchdog: if Python threads can still run, try to save and exit.
    # If the process is fully wedged (e.g., stuck in native code while holding the GIL),
    # an external watchdog must still kill/restart the process.
    if timeout_s <= 0:
        return

    def _worker():
        while True:
            time.sleep(5)
            dt = _get_heartbeat_age_s()
            if dt > timeout_s:
                print(
                    f"[ERROR] Hang suspected: no heartbeat for {dt:.1f}s (last_tag={_LAST_HEARTBEAT_TAG}).",
                    flush=True,
                )
                try:
                    faulthandler.dump_traceback(all_threads=True)
                except Exception:
                    pass
                # Try an emergency checkpoint (best effort).
                try:
                    runner = get_runner_fn()
                except Exception:
                    runner = None
                _emergency_save(runner, run_dir, tag="hang")
                # Always exit so an external supervisor can restart.
                os._exit(1)
            # Update heartbeat file even if we only touched in-memory markers.
            if heartbeat_path and _is_rank0():
                _write_heartbeat_file(heartbeat_path)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    print(f"[INFO] Hang watchdog enabled (timeout_s={timeout_s}).", flush=True)


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

    # Export paths for any patched components that run outside main() scope.
    if args_cli.heartbeat_path is not None:
        os.environ["ISAACLAB_HEARTBEAT_PATH"] = args_cli.heartbeat_path
    if args_cli.run_dir is not None:
        os.environ["ISAACLAB_RUN_DIR"] = os.path.abspath(args_cli.run_dir)
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    # If run_dir is provided, reuse it across restarts (no timestamp subdir).
    # If run_dir is not provided but --auto_resume is set, try to reuse the most recent run dir
    # under the same experiment parent directory so restarts keep writing to the same place.
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

    effective_run_dir = args_cli.run_dir if args_cli.run_dir is not None else log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
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

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # Start an in-process hang watchdog (best effort) and update an external heartbeat file.
    runner_holder = {"runner": runner}
    _start_hang_watchdog(lambda: runner_holder.get("runner"), effective_run_dir, args_cli.heartbeat_path, args_cli.hang_timeout_s)

    # Auto-resume from the latest checkpoint in run_dir, if requested.
    if args_cli.auto_resume and effective_run_dir is not None:
        ckpt = _find_latest_checkpoint(effective_run_dir)
        if ckpt is not None:
            if _is_rank0():
                print(f"[INFO] Auto-resume: loading latest checkpoint: {ckpt}", flush=True)
            runner.load(ckpt)
        else:
            if _is_rank0():
                print(f"[INFO] Auto-resume: no checkpoint found in run_dir={effective_run_dir}", flush=True)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    # run training
    _touch_heartbeat("before_learn")
    hb_path = os.environ.get("ISAACLAB_HEARTBEAT_PATH")
    if hb_path and _is_rank0():
        _write_heartbeat_file(hb_path)
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except BaseException:
        # Best-effort emergency checkpoint on failures. (Hangs are handled by watchdogs.)
        _emergency_save(runner, effective_run_dir, tag="exception")
        raise

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
