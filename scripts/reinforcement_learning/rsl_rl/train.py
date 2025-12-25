# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL.

This version adds:
  - PPO/ActorCritic safety guards for invalid std/log_std
  - In-memory heartbeat + hang watchdog
  - Auto-resume from run_dir without changing the original checkpoint format
  - Sidecar extras file: "<checkpoint>.extras.pt" (optional)
      * RNG states (python / numpy / torch cpu / torch cuda)
      * Runner counters (best-effort)
      * IsaacLab scene snapshot (best-effort) for env state restore
  - IMPORTANT: After resume, this script forces a single env.reset() so the pending
    env snapshot is applied immediately (some runner paths do not reset at start).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import datetime as _datetime
import faulthandler
import importlib.metadata as metadata
import os
from pathlib import Path
import platform
import random
import signal
import sys
import threading
import time
import traceback
from typing import Any

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

_EXTRAS_SUFFIX = ".extras.pt"


def _is_checkpoint_file(p: Path) -> bool:
    exts = (".pt", ".pth", ".ckpt")
    if not (p.is_file() and p.suffix in exts):
        return False
    # Never treat sidecar extras as checkpoints.
    if str(p).endswith(_EXTRAS_SUFFIX):
        return False
    return True


def _list_checkpoints_sorted(run_dir: str) -> list[str]:
    """Return checkpoint-like files in run_dir sorted by mtime (newest first)."""
    root = Path(run_dir)
    if not root.exists():
        return []
    candidates: list[Path] = []
    for p in root.rglob("*"):
        if _is_checkpoint_file(p):
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
    files = [p for p in files if _is_checkpoint_file(p)]
    if not files:
        return None, None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    ckpt = files[0]
    return str(ckpt), str(ckpt.parent)


def _maybe_reuse_existing_run_dir(default_log_dir: str) -> tuple[str | None, str | None]:
    parent = str(Path(default_log_dir).parent)
    return _find_latest_checkpoint_in_tree(parent)


def _torch_load_any(path: str, map_location: str = "cpu") -> Any:
    """torch.load that works across PyTorch versions (weights_only default changed in 2.6)."""
    try:
        # PyTorch 2.6+: weights_only defaults True -> we want full pickle for extras.
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older versions: no weights_only argument
        return torch.load(path, map_location=map_location)
    except Exception:
        # Fallback: try with weights_only=False even if it exists but failed due to allowlist.
        # If this fails too, bubble up.
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except Exception:
            return torch.load(path, map_location=map_location)


def _torch_save_any(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def _extras_path_for_checkpoint(ckpt_path: str) -> str:
    return f"{ckpt_path}{_EXTRAS_SUFFIX}"


def _capture_rng_state() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        out["python_random_state"] = random.getstate()
    except Exception:
        pass
    try:
        import numpy as np

        out["numpy_random_state"] = np.random.get_state()
    except Exception:
        pass
    try:
        out["torch_random_state_cpu"] = torch.random.get_rng_state()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            out["torch_random_state_cuda_all"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return out


def _restore_rng_state(state: dict[str, Any]) -> None:
    try:
        if "python_random_state" in state:
            random.setstate(state["python_random_state"])
    except Exception:
        pass
    try:
        import numpy as np

        if "numpy_random_state" in state:
            np.random.set_state(state["numpy_random_state"])
    except Exception:
        pass
    try:
        if "torch_random_state_cpu" in state:
            torch.random.set_rng_state(state["torch_random_state_cpu"])
    except Exception:
        pass
    try:
        if torch.cuda.is_available() and "torch_random_state_cuda_all" in state:
            torch.cuda.set_rng_state_all(state["torch_random_state_cuda_all"])
    except Exception:
        pass


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


def _capture_runner_counters(runner) -> dict[str, Any]:
    """Best-effort capture of runner counters that influence scheduling."""
    out: dict[str, Any] = {}
    if runner is None:
        return out

    for k in (
        "current_learning_iteration",
        "current_iteration",
        "learning_iteration",
        "iteration",
        "it",
        "total_timesteps",
        "tot_timesteps",
        "num_timesteps",
    ):
        v = getattr(runner, k, None)
        if isinstance(v, int):
            out[k] = int(v)

    # Sometimes stored inside runner.writer or runner.logger; skip (too version-specific).
    return out


def _restore_runner_counters(runner, counters: dict[str, Any]) -> int:
    """Restore any runner counter attributes that exist. Return how many were restored."""
    if runner is None or not isinstance(counters, dict):
        return 0
    restored = 0
    for k, v in counters.items():
        if not isinstance(v, int):
            continue
        if hasattr(runner, k):
            try:
                setattr(runner, k, int(v))
                restored += 1
            except Exception:
                pass
    return restored


def _capture_isaaclab_scene_snapshot(env) -> dict[str, Any] | None:
    """Best-effort IsaacLab scene snapshot.

    This intentionally avoids relying on task-specific env.state_dict() since many
    IsaacLab envs do not implement it. Instead, it tries to capture the scene's
    articulations/rigid bodies state tensors and later write them back to sim.

    Returns a CPU-only dict, or None if it could not capture anything meaningful.
    """
    try:
        unwrapped = env.unwrapped
    except Exception:
        return None

    scene = getattr(unwrapped, "scene", None)
    if scene is None:
        return None

    snap: dict[str, Any] = {"_kind": "isaaclab_scene_v1", "articulations": {}, "rigid_objects": {}}

    # Articulations
    arts = getattr(scene, "articulations", None)
    if isinstance(arts, dict):
        for name, art in arts.items():
            data = getattr(art, "data", None)
            if data is None:
                continue
            # Common IsaacLab tensors: root_state_w, joint_pos, joint_vel
            root_state = getattr(data, "root_state_w", None)
            joint_pos = getattr(data, "joint_pos", None)
            joint_vel = getattr(data, "joint_vel", None)

            # Some tasks use different names; try a couple more
            if root_state is None:
                root_state = getattr(data, "root_state", None)
            if joint_pos is None:
                joint_pos = getattr(data, "joint_pos_w", None)
            if joint_vel is None:
                joint_vel = getattr(data, "joint_vel_w", None)

            if root_state is None and joint_pos is None and joint_vel is None:
                continue

            entry: dict[str, Any] = {}
            try:
                if torch.is_tensor(root_state):
                    entry["root_state"] = root_state.detach().clone().cpu()
            except Exception:
                pass
            try:
                if torch.is_tensor(joint_pos):
                    entry["joint_pos"] = joint_pos.detach().clone().cpu()
            except Exception:
                pass
            try:
                if torch.is_tensor(joint_vel):
                    entry["joint_vel"] = joint_vel.detach().clone().cpu()
            except Exception:
                pass

            if entry:
                snap["articulations"][str(name)] = entry

    # Rigid objects (optional)
    rigs = getattr(scene, "rigid_objects", None)
    if isinstance(rigs, dict):
        for name, robj in rigs.items():
            data = getattr(robj, "data", None)
            if data is None:
                continue
            root_state = getattr(data, "root_state_w", None)
            if root_state is None:
                root_state = getattr(data, "root_state", None)
            if root_state is None:
                continue
            entry = {}
            try:
                if torch.is_tensor(root_state):
                    entry["root_state"] = root_state.detach().clone().cpu()
            except Exception:
                pass
            if entry:
                snap["rigid_objects"][str(name)] = entry

    n_art = len(snap["articulations"])
    n_rig = len(snap["rigid_objects"])
    if n_art == 0 and n_rig == 0:
        return None

    if _is_rank0():
        print(f"[INFO] Captured IsaacLab scene snapshot (articulations={n_art}, rigid_objects={n_rig}).", flush=True)
    return snap


def _apply_isaaclab_scene_snapshot(env, snap: dict[str, Any]) -> bool:
    """Best-effort apply IsaacLab scene snapshot to sim."""
    if not isinstance(snap, dict) or snap.get("_kind") != "isaaclab_scene_v1":
        return False

    try:
        unwrapped = env.unwrapped
    except Exception:
        return False

    scene = getattr(unwrapped, "scene", None)
    if scene is None:
        return False

    ok_any = False

    # Articulations
    arts = getattr(scene, "articulations", None)
    if isinstance(arts, dict):
        for name, payload in snap.get("articulations", {}).items():
            art = arts.get(name, None)
            if art is None or not isinstance(payload, dict):
                continue
            data = getattr(art, "data", None)
            if data is None:
                continue

            # Move tensors to env device
            device = getattr(unwrapped, "device", None)
            if device is None:
                # Many IsaacLab envs store tensors on cuda:0 but do not expose device; infer from existing buffers.
                cur = getattr(data, "joint_pos", None)
                if torch.is_tensor(cur):
                    device = cur.device
                else:
                    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

            root_state = payload.get("root_state", None)
            joint_pos = payload.get("joint_pos", None)
            joint_vel = payload.get("joint_vel", None)

            try:
                if torch.is_tensor(root_state):
                    rs = root_state.to(device=device)
                    # Write into buffers if present
                    if hasattr(data, "root_state_w") and torch.is_tensor(getattr(data, "root_state_w")):
                        getattr(data, "root_state_w").copy_(rs)
                    elif hasattr(data, "root_state") and torch.is_tensor(getattr(data, "root_state")):
                        getattr(data, "root_state").copy_(rs)

                    # Write to sim if method exists
                    if hasattr(art, "write_root_state_to_sim"):
                        art.write_root_state_to_sim(rs)
                    else:
                        # Some versions split pose/vel; try best-effort
                        if hasattr(art, "write_root_pose_to_sim") and rs.shape[-1] >= 7:
                            art.write_root_pose_to_sim(rs[..., 0:7])
                        if hasattr(art, "write_root_velocity_to_sim") and rs.shape[-1] >= 13:
                            art.write_root_velocity_to_sim(rs[..., 7:13])
                    ok_any = True
            except Exception:
                pass

            try:
                if torch.is_tensor(joint_pos) or torch.is_tensor(joint_vel):
                    jp = joint_pos.to(device=device) if torch.is_tensor(joint_pos) else None
                    jv = joint_vel.to(device=device) if torch.is_tensor(joint_vel) else None

                    if jp is not None and hasattr(data, "joint_pos") and torch.is_tensor(getattr(data, "joint_pos")):
                        getattr(data, "joint_pos").copy_(jp)
                    if jv is not None and hasattr(data, "joint_vel") and torch.is_tensor(getattr(data, "joint_vel")):
                        getattr(data, "joint_vel").copy_(jv)

                    if hasattr(art, "write_joint_state_to_sim"):
                        # IsaacLab typically expects (joint_pos, joint_vel)
                        art.write_joint_state_to_sim(jp, jv)
                    else:
                        if jp is not None and hasattr(art, "write_joint_pos_to_sim"):
                            art.write_joint_pos_to_sim(jp)
                        if jv is not None and hasattr(art, "write_joint_vel_to_sim"):
                            art.write_joint_vel_to_sim(jv)
                    ok_any = True
            except Exception:
                pass

    # Rigid objects
    rigs = getattr(scene, "rigid_objects", None)
    if isinstance(rigs, dict):
        for name, payload in snap.get("rigid_objects", {}).items():
            robj = rigs.get(name, None)
            if robj is None or not isinstance(payload, dict):
                continue
            data = getattr(robj, "data", None)
            if data is None:
                continue

            device = getattr(unwrapped, "device", None)
            if device is None:
                cur = getattr(data, "root_state_w", None)
                if torch.is_tensor(cur):
                    device = cur.device
                else:
                    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

            root_state = payload.get("root_state", None)
            if not torch.is_tensor(root_state):
                continue

            try:
                rs = root_state.to(device=device)
                if hasattr(data, "root_state_w") and torch.is_tensor(getattr(data, "root_state_w")):
                    getattr(data, "root_state_w").copy_(rs)
                elif hasattr(data, "root_state") and torch.is_tensor(getattr(data, "root_state")):
                    getattr(data, "root_state").copy_(rs)

                if hasattr(robj, "write_root_state_to_sim"):
                    robj.write_root_state_to_sim(rs)
                else:
                    if hasattr(robj, "write_root_pose_to_sim") and rs.shape[-1] >= 7:
                        robj.write_root_pose_to_sim(rs[..., 0:7])
                    if hasattr(robj, "write_root_velocity_to_sim") and rs.shape[-1] >= 13:
                        robj.write_root_velocity_to_sim(rs[..., 7:13])
                ok_any = True
            except Exception:
                pass

    if _is_rank0():
        print(f"[INFO] Restore IsaacLab scene snapshot: ok_any={ok_any}.", flush=True)
    return ok_any


def _save_extras_sidecar(env, runner, checkpoint_path: str) -> None:
    """Save sidecar extras file next to the original checkpoint."""
    if not _is_rank0():
        return

    extras: dict[str, Any] = {"_kind": "rslrl_extras_v1"}

    # RNG
    extras["rng"] = _capture_rng_state()

    # Runner counters (best-effort)
    extras["runner_counters"] = _capture_runner_counters(runner)

    # Env snapshot (best-effort)
    snap = _capture_isaaclab_scene_snapshot(env)
    if snap is not None:
        extras["env_scene_snapshot"] = snap
    else:
        extras["env_scene_snapshot"] = None
        print("[INFO] No stateful env found (no scene snapshot captured).", flush=True)

    out_path = _extras_path_for_checkpoint(checkpoint_path)
    try:
        _torch_save_any(extras, out_path)
        print(f"[INFO] Saved extras sidecar: {out_path}", flush=True)
    except Exception as exc:
        print(f"[WARN] Failed to save extras sidecar: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()


def _load_extras_sidecar(env, runner, checkpoint_path: str) -> dict[str, Any] | None:
    """Load sidecar extras file if present. Returns pending env snapshot dict or None."""
    path = _extras_path_for_checkpoint(checkpoint_path)
    if not Path(path).exists():
        return None

    try:
        extras = _torch_load_any(path, map_location="cpu")
    except Exception as exc:
        print(f"[WARN] Failed to load extras sidecar: {type(exc).__name__}: {exc}", flush=True)
        return None

    if not isinstance(extras, dict):
        return None

    # RNG
    rng = extras.get("rng", None)
    if isinstance(rng, dict):
        _restore_rng_state(rng)
        if _is_rank0():
            print("[INFO] Restored RNG state from sidecar.", flush=True)

    # Runner counters
    counters = extras.get("runner_counters", None)
    if isinstance(counters, dict):
        n = _restore_runner_counters(runner, counters)
        if _is_rank0() and n > 0:
            print(f"[INFO] Restored {n} runner counters from sidecar.", flush=True)

    # Env snapshot is applied after a reset (or forced reset).
    snap = extras.get("env_scene_snapshot", None)
    if isinstance(snap, dict) and snap.get("_kind") == "isaaclab_scene_v1":
        return snap

    return None


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

                # Emergency save handled by caller via signal or exception path in learn().
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


def _emergency_save(env, runner, run_dir: str | None, tag: str) -> None:
    if run_dir is None or runner is None:
        return
    if not _is_rank0():
        return
    try:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = str(Path(run_dir) / f"emergency_{tag}_{ts}.pt")

        # Save sidecar first (so even if main save crashes, we still keep env/rng hints).
        _save_extras_sidecar(env, runner, out)

        if hasattr(runner, "save_checkpoint"):
            runner.save_checkpoint(out)
            print("[INFO] runner.save_checkpoint", flush=True)
        elif hasattr(runner, "save"):
            runner.save(out)
            print("[INFO] runner.save", flush=True)
        else:
            print("[WARN] Runner has no save/save_checkpoint. Skipping emergency checkpoint.", flush=True)
            return

        print(f"[INFO] Emergency checkpoint saved: {out}", flush=True)
    except Exception:
        print("[ERROR] Emergency checkpoint failed.", flush=True)
        traceback.print_exc()


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

# Pending env snapshot holder (applied after reset)
_resume_env_state_holder: dict[str, Any] = {"pending": None, "applied_once": False}


def _maybe_apply_pending_env_state(env) -> None:
    """Apply pending env snapshot once, after env.reset()."""
    if _resume_env_state_holder.get("applied_once", False):
        return
    snap = _resume_env_state_holder.get("pending", None)
    if not isinstance(snap, dict):
        return

    ok = False
    try:
        ok = _apply_isaaclab_scene_snapshot(env, snap)
    except Exception as exc:
        if _is_rank0():
            print(f"[WARN] Applying env snapshot failed: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()

    _resume_env_state_holder["applied_once"] = True
    _resume_env_state_holder["pending"] = None
    if _is_rank0() and ok:
        print("[INFO] Applied env state from sidecar after env.reset().", flush=True)


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

    # Wrap env for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Patch env.reset to apply pending env snapshot right after reset returns.
    if not hasattr(env, "_reset_patched_for_resume"):
        _orig_reset = env.reset

        def _reset_with_resume_apply(*a, **kw):
            obs = _orig_reset(*a, **kw)
            _maybe_apply_pending_env_state(env)
            return obs

        env.reset = _reset_with_resume_apply  # type: ignore[assignment]
        env._reset_patched_for_resume = True  # type: ignore[attr-defined]

    _touch_heartbeat("before_runner_init")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner_holder = {"runner": runner}

    # Patch runner.save (and save_checkpoint if exists) to also write sidecar.
    if not hasattr(runner, "_save_patched_for_extras"):
        if hasattr(runner, "save") and callable(getattr(runner, "save")):
            _orig_save = runner.save

            def _save_wrapped(path: str, *a, **kw):
                _save_extras_sidecar(env, runner, path)
                return _orig_save(path, *a, **kw)

            runner.save = _save_wrapped  # type: ignore[assignment]

        if hasattr(runner, "save_checkpoint") and callable(getattr(runner, "save_checkpoint")):
            _orig_save_ckpt = runner.save_checkpoint

            def _save_ckpt_wrapped(path: str, *a, **kw):
                _save_extras_sidecar(env, runner, path)
                return _orig_save_ckpt(path, *a, **kw)

            runner.save_checkpoint = _save_ckpt_wrapped  # type: ignore[assignment]

        runner._save_patched_for_extras = True  # type: ignore[attr-defined]

    # Patch runner.load to also load sidecar and then FORCE one env.reset() so it is applied immediately.
    if not hasattr(runner, "_load_patched_for_extras"):
        if hasattr(runner, "load") and callable(getattr(runner, "load")):
            _orig_load = runner.load

            def _load_wrapped(path: str, *a, **kw):
                ret = _orig_load(path, *a, **kw)

                pending = _load_extras_sidecar(env, runner, path)
                if pending is not None:
                    _resume_env_state_holder["pending"] = pending
                    _resume_env_state_holder["applied_once"] = False
                    if _is_rank0():
                        print("[INFO] Resume: env state is pending and will be applied after env.reset().", flush=True)

                    # Force a reset now: some runner paths do not call reset on resume.
                    try:
                        env.reset()
                    except Exception as exc:
                        if _is_rank0():
                            print(f"[WARN] Forced env.reset() after resume failed: {type(exc).__name__}: {exc}", flush=True)

                return ret

            runner.load = _load_wrapped  # type: ignore[assignment]

        runner._load_patched_for_extras = True  # type: ignore[attr-defined]

    def _handle_signal(sig: int, _frame) -> None:
        print(f"[WARN] Received signal {sig}. Attempting emergency checkpoint then exiting.", flush=True)
        _emergency_save(env, runner_holder.get("runner"), effective_run_dir, tag=f"signal{sig}")
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

    # -----------------------------------------------------------------
    # Auto-resume: scan run_dir and load newest usable checkpoint.
    # -----------------------------------------------------------------
    loaded_ckpt: str | None = None
    if args_cli.auto_resume:
        ckpt_candidates = _list_checkpoints_sorted(effective_run_dir)
        if ckpt_candidates:
            if _is_rank0():
                print(
                    f"[INFO] Auto-resume: probing {len(ckpt_candidates)} checkpoint(s) in run_dir={effective_run_dir}",
                    flush=True,
                )
            loaded_ckpt = _try_load_checkpoints(runner, ckpt_candidates)
            if loaded_ckpt is not None:
                if _is_rank0():
                    print(f"[INFO] Auto-resume: loaded checkpoint: {loaded_ckpt}", flush=True)
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

    # -----------------------------------------------------------------
    # If Hydra resume path is requested, try it after auto-resume.
    # -----------------------------------------------------------------
    if resume_path is not None:
        if _is_rank0():
            print(f"[INFO] Loading model checkpoint from: {resume_path}", flush=True)
        try:
            runner.load(resume_path)
            loaded_ckpt = resume_path
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
                        loaded_ckpt = loaded
                        if _is_rank0():
                            print(f"[INFO] Fallback auto-resume: loaded checkpoint: {loaded}", flush=True)
                    else:
                        if _is_rank0():
                            print("[WARN] Fallback auto-resume: no usable checkpoint found. Starting from scratch.", flush=True)

    # Dump configs
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
        runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=(completed == 0))
    except BaseException:
        _emergency_save(env, runner, effective_run_dir, tag="exception")
        raise

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
