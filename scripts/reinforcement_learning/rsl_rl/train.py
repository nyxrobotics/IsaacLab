# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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


# =====================================================================
# DIST DEBUG PATCH:
# - Wrap torch.distributed.all_reduce to log every call (rank, shape, finite)
# - Wrap PPO.update to:
#   - add CUDA sync pre-check
#   - add pre-update barrier to detect desync earlier
#   - add a GLOBAL non-finite gate (any-rank -> all-ranks skip update)
#   - print traceback on the failing rank and exit fast so other ranks don't hang.
#
# Notes:
# - DO NOT use all_gather_object here. It can hang/crash due to object serialization.
# - If you want a per-rank "state" view, use fixed-size tensors.
# =====================================================================
import os
import time
import traceback
import faulthandler
import torch
from rsl_rl.algorithms.ppo import PPO

import threading
import torch.distributed as dist

faulthandler.enable(all_threads=True)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _rank() -> int:
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _world() -> int:
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return int(os.environ.get("WORLD_SIZE", "1"))


def _log(msg: str):
    print(f"[DISTDBG][{_now()}][rank{_rank()}/{_world()}] {msg}", flush=True)


def _fail_fast(reason: str):
    _log(f"FAILFAST: {reason}")

    # Try to tear down distributed cleanly (avoid other ranks hanging)
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        _log(f"FAILFAST: destroy_process_group failed: {repr(e)}")

    # Try to close simulation app if available
    try:
        if _rank() == 0 and "simulation_app" in globals():
            simulation_app.close()
    except Exception as e:
        _log(f"FAILFAST: simulation_app.close failed: {repr(e)}")

    os._exit(1)


def _tinfo(t: torch.Tensor) -> str:
    if not torch.is_tensor(t):
        return f"type={type(t)}"
    with torch.no_grad():
        dev = str(t.device)
        shp = tuple(t.shape)
        dt = str(t.dtype)
        finite = bool(torch.isfinite(t).all().item()) if t.numel() > 0 else True
        return f"shape={shp} dtype={dt} device={dev} finite={finite}"


# ---------------------------------------------------------------------
# 1) Monkeypatch torch.distributed.all_reduce
# ---------------------------------------------------------------------
if hasattr(dist, "all_reduce"):
    _orig_all_reduce = dist.all_reduce

    def _all_reduce_logged(tensor, *args, **kwargs):
        _log(f"ENTER all_reduce: {_tinfo(tensor)}")
        try:
            out = _orig_all_reduce(tensor, *args, **kwargs)
            _log(f"EXIT  all_reduce: {_tinfo(tensor)}")
            return out
        except Exception as e:
            _log(f"EXC   all_reduce: {_tinfo(tensor)} err={repr(e)}")
            _log("TRACEBACK:\n" + traceback.format_exc())
            _fail_fast("all_reduce exception")

    dist.all_reduce = _all_reduce_logged
    _log("Patched torch.distributed.all_reduce with logging+failfast.")
else:
    _log("WARN: torch.distributed.all_reduce not found; skip patch.")


# ---------------------------------------------------------------------
# Rank-local progress state (iteration/step/phase) for desync visibility
# ---------------------------------------------------------------------
_state_lock = threading.Lock()
_state = {"phase": "init", "iter": None, "step": None, "ep": None}


def _get_attr_any(obj, names):
    for n in names:
        if hasattr(obj, n):
            try:
                v = getattr(obj, n)
                if callable(v):
                    continue
                return v
            except Exception:
                pass
    return None


def _update_state_from_runner(runner_obj, phase: str):
    it = _get_attr_any(
        runner_obj,
        [
            "current_learning_iteration",
            "learning_iteration",
            "iteration",
            "it",
            "_it",
        ],
    )
    step = _get_attr_any(
        runner_obj,
        [
            "tot_timesteps",
            "total_timesteps",
            "num_timesteps",
            "global_step",
            "step",
            "_step",
        ],
    )
    ep = _get_attr_any(
        runner_obj,
        [
            "tot_episodes",
            "total_episodes",
            "episode",
            "ep",
            "_ep",
        ],
    )

    with _state_lock:
        _state["phase"] = phase
        _state["iter"] = it
        _state["step"] = step
        _state["ep"] = ep


def _state_str() -> str:
    with _state_lock:
        return f"phase={_state['phase']} iter={_state['iter']} step={_state['step']} ep={_state['ep']}"


def _set_phase(phase: str):
    with _state_lock:
        _state["phase"] = phase


# ---------------------------------------------------------------------
# WATCHDOG: periodic + stall traceback dumps
# ---------------------------------------------------------------------
_progress_lock = threading.Lock()
_last_progress = time.time()
_last_tag = "init"


def _touch(tag: str):
    global _last_progress, _last_tag
    with _progress_lock:
        _last_progress = time.time()
        _last_tag = tag


def _watchdog_thread():
    dump_every = 60
    stall_sec = 120
    last_dump = time.time()
    while True:
        time.sleep(5)
        now = time.time()
        with _progress_lock:
            dt = now - _last_progress
            tag = _last_tag

        if now - last_dump > dump_every:
            _log("PERIODIC: dumping tracebacks")
            faulthandler.dump_traceback(all_threads=True)
            last_dump = now

        if dt > stall_sec:
            _log(f"STALL: no progress for {dt:.1f}s (last_tag={tag}) -> dumping tracebacks")
            faulthandler.dump_traceback(all_threads=True)
            _touch(f"watchdog_dump_after_{tag}")


threading.Thread(target=_watchdog_thread, daemon=True).start()
_log("Watchdog thread started (periodic+stall traceback dumps enabled).")


# ---------------------------------------------------------------------
# Runner phase patching (best-effort across rsl_rl versions)
# ---------------------------------------------------------------------
try:
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner

    _log("Loaded OnPolicyRunner for phase patching.")

    _collect_names = [
        "collect_rollouts",
        "collect_rollouts_and_compute_returns",
        "collect",
        "_collect_rollouts",
    ]

    for _name in _collect_names:
        if hasattr(OnPolicyRunner, _name):
            _orig_collect = getattr(OnPolicyRunner, _name)

            def _collect_logged(self, *args, __orig=_orig_collect, __nm=_name, **kwargs):
                _update_state_from_runner(self, phase=f"before_{__nm}")
                _touch(f"before_{__nm}")
                _log(f"ENTER runner.{__nm} ({_state_str()})")
                try:
                    out = __orig(self, *args, **kwargs)
                    _log(f"EXIT  runner.{__nm} ({_state_str()})")
                    _update_state_from_runner(self, phase=f"after_{__nm}")
                    _touch(f"after_{__nm}")
                    return out
                except Exception as e:
                    _log(f"EXC   runner.{__nm} err={repr(e)} ({_state_str()})")
                    _log("TRACEBACK:\n" + traceback.format_exc())
                    _fail_fast("runner collect exception")

            setattr(OnPolicyRunner, _name, _collect_logged)
            _log(f"Patched OnPolicyRunner.{_name} with logging.")
            break
    else:
        _log("WARN: No known collect method found on OnPolicyRunner; collect phase won't be logged.")

    _update_names = [
        "update",
        "_update",
        "_update_policy",
        "_update_alg",
        "_update_algorithm",
    ]

    for _uname in _update_names:
        if hasattr(OnPolicyRunner, _uname):
            _orig_u = getattr(OnPolicyRunner, _uname)

            def _update_callsite_logged(self, *args, __orig=_orig_u, __nm=_uname, **kwargs):
                _update_state_from_runner(self, phase=f"before_{__nm}")
                _touch(f"before_{__nm}")
                _log(f"ENTER runner.{__nm} ({_state_str()})")
                try:
                    out = __orig(self, *args, **kwargs)
                    _log(f"EXIT  runner.{__nm} ({_state_str()})")
                    _update_state_from_runner(self, phase=f"after_{__nm}")
                    _touch(f"after_{__nm}")
                    return out
                except Exception:
                    _log(f"EXC   runner.{__nm} ({_state_str()})")
                    _log("TRACEBACK:\n" + traceback.format_exc())
                    _fail_fast("runner update callsite exception")

            setattr(OnPolicyRunner, _uname, _update_callsite_logged)
            _log(f"Patched OnPolicyRunner.{_uname} with logging.")
            break
    else:
        _log("WARN: No known update method found on OnPolicyRunner; update phase may not be logged.")

except Exception as e:
    _log(f"WARN: Failed to patch runner phases: {repr(e)}")


# ---------------------------------------------------------------------
# GLOBAL non-finite gate for update (any-rank -> all-ranks skip)
# ---------------------------------------------------------------------
def _any_nonfinite_in(obj) -> bool:
    """Return True if obj (tensor/list/dict/has __dict__) contains any non-finite tensor value."""
    try:
        if torch.is_tensor(obj):
            if obj.numel() == 0:
                return False
            return not torch.isfinite(obj).all().item()
        if isinstance(obj, dict):
            return any(_any_nonfinite_in(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(_any_nonfinite_in(v) for v in obj)
        # Best-effort: inspect attributes on objects (avoid deep recursion explosions)
        if hasattr(obj, "__dict__"):
            for v in obj.__dict__.values():
                if torch.is_tensor(v):
                    if _any_nonfinite_in(v):
                        return True
    except Exception:
        pass
    return False


def _get_rollout_candidates(ppo) -> list:
    """Try common holder names across rsl_rl versions."""
    cands = []
    for name in ["storage", "rollout_storage", "buffer", "experience_buffer"]:
        if hasattr(ppo, name):
            cands.append(getattr(ppo, name))
    return cands


def _global_should_skip_update(ppo) -> bool:
    """Compute a global skip decision: if any rank sees non-finite, all ranks skip."""
    local_bad = False

    for st in _get_rollout_candidates(ppo):
        if _any_nonfinite_in(st):
            local_bad = True
            break

    if not local_bad:
        for name in ["obs", "observations", "actions", "returns", "advantages", "values", "old_log_probs"]:
            if hasattr(ppo, name) and _any_nonfinite_in(getattr(ppo, name)):
                local_bad = True
                break

    if dist.is_available() and dist.is_initialized():
        dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        flag = torch.tensor([1 if local_bad else 0], device=dev, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item() == 1)

    return local_bad


def _try_clear_rollout_storage(ppo):
    """Best-effort: prevent 'poison' tensors from being reused after skip."""
    for st in _get_rollout_candidates(ppo):
        try:
            if hasattr(st, "clear") and callable(getattr(st, "clear")):
                st.clear()
            elif hasattr(st, "reset") and callable(getattr(st, "reset")):
                st.reset()
        except Exception:
            _log("WARN: failed to clear/reset rollout storage:\n" + traceback.format_exc())


# ---------------------------------------------------------------------
# PPO.update wrapper: pre-sync + pre-barrier + global skip gate + failfast
# ---------------------------------------------------------------------
_orig_update = PPO.update


def _update_with_barrier(self, *args, **kwargs):
    _touch("enter_update_wrapper")
    _set_phase("enter_ppo_update_wrapper")
    _log("ENTER PPO.update wrapper")

    if dist.is_available() and dist.is_initialized():
        # 1) CUDA sync: detect CUDA-side stall before comms
        if torch.cuda.is_available():
            _touch("before_cuda_sync_update")
            _set_phase("before_cuda_synchronize")
            _log("ENTER torch.cuda.synchronize() (pre-barrier)")
            try:
                _log(
                    f"CUDA current_device={torch.cuda.current_device()} "
                    f"name={torch.cuda.get_device_name(torch.cuda.current_device())}"
                )
                torch.cuda.synchronize()
                _set_phase("after_cuda_synchronize")
                _log("PASS  torch.cuda.synchronize() (pre-barrier)")
            except Exception:
                _log("TRACEBACK:\n" + traceback.format_exc())
                _fail_fast("cuda.synchronize exception")

        # 2) Barrier: detect rank desync / comm hang earlier
        _touch("before_dist_barrier_update")
        _set_phase("before_dist_barrier")
        _log(f"PRE-BARRIER STATE: {_state_str()}")
        _log("ENTER dist.barrier() (pre-update)")
        try:
            dist.barrier()
            _log("PASS  dist.barrier() (pre-update)")
        except Exception:
            _log("TRACEBACK:\n" + traceback.format_exc())
            _fail_fast("barrier exception")

        _set_phase("after_dist_barrier")
        _touch("after_dist_barrier_update")
    else:
        _log("SKIP dist.barrier(): dist not initialized")
        _touch("skip_dist_barrier_update")

    # 3) GLOBAL skip gate: any-rank non-finite -> all-ranks skip update consistently
    _touch("before_global_skip_gate")
    _set_phase("before_global_skip_gate")
    try:
        if _global_should_skip_update(self):
            _log("GLOBAL SKIP: non-finite detected on at least one rank -> skip PPO.update and continue")
            _touch("skipped_update_due_to_nonfinite")
            _try_clear_rollout_storage(self)
            return None
    except Exception:
        _log("TRACEBACK:\n" + traceback.format_exc())
        _fail_fast("global skip gate exception")

    # 4) Actual PPO.update
    _touch("enter_real_ppo_update")
    _set_phase("enter_real_ppo_update")
    _log("ENTER PPO.update")
    try:
        out = _orig_update(self, *args, **kwargs)
        _log("EXIT  PPO.update")
        _touch("exit_real_ppo_update")
        _set_phase("exit_real_ppo_update")
        return out
    except Exception:
        _log("TRACEBACK:\n" + traceback.format_exc())
        _fail_fast("PPO.update exception")


PPO.update = _update_with_barrier
_log("Patched PPO.update with pre-barrier + global non-finite skip gate + failfast.")
# =====================================================================


# ---------------------------------------------------------------------
# add argparse arguments
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

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

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
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
import logging
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

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
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning("IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported.")

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

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
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
