# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# =====================================================================
# SAFETY PATCH 1: Clamp policy.log_std after every PPO.update
#   + warn when invalid values are detected
# =====================================================================
from rsl_rl.algorithms.ppo import PPO

_original_ppo_update = PPO.update

def _safe_ppo_update(self, *args, **kwargs):
    # Run original update
    out = _original_ppo_update(self, *args, **kwargs)

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

# =====================================================================
# SAFETY PATCH 2: Guard ActorCritic.act so that invalid std doesn't crash
# =====================================================================
from rsl_rl.modules.actor_critic import ActorCritic
from torch.distributions import Normal

_original_act = ActorCritic.act

def _safe_act(self, *args, **kwargs):
    try:
        return _original_act(self, *args, **kwargs)
    except RuntimeError as e:
        msg = str(e)
        if "normal expects all elements of std" not in msg:
            raise

        print("[WARNING] Invalid std detected. Reconstructing distribution with clamped std...")

        with torch.no_grad():
            # ActorCritic が保持する mean / std を使用
            if not (hasattr(self, "action_std") and hasattr(self, "action_mean")):
                print("  - policy has no action_std or action_mean")
                raise

            std = self.action_std

            # NaN / Inf / 非正値を修正
            invalid = (~torch.isfinite(std)) | (std <= 0.0)
            std = torch.where(invalid, torch.full_like(std, 0.1), std)

            # 最終 clamping
            std = torch.clamp(std, min=1e-6, max=10.0)

            # ★ NOTE: self.action_std に書き込んではいけない！
            # distribution を作り直すだけでOK
            safe_dist = Normal(self.action_mean, std)
            self.distribution = safe_dist
            self.entropy = safe_dist.entropy()

        return self.distribution.sample()


# ActorCritic.act を差し替え
ActorCritic.act = _safe_act
# =====================================================================


# =====================================================================
# SAFETY PATCH 3: If rsl_rl minibatch generator yields None,
#        reuse the last valid minibatch (keeps the exact 12-tuple structure).
# =====================================================================

def _dist_rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

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
        none_count = 0
        total_count = 0

        for batch in _orig_mbg(self, *args, **kwargs):
            total_count += 1

            if batch is None:
                none_count += 1

                if last_good is None:
                    # If the first batch is None, we cannot safely fabricate the expected 12-tuple.
                    raise RuntimeError(
                        f"[rank{_dist_rank()}] mini_batch_generator yielded None before any valid batch "
                        f"(none_count={none_count}, total={total_count})."
                    )

                print(
                    f"[WARN][rank{_dist_rank()}] minibatch_generator yielded None -> reusing last_good "
                    f"(none_count={none_count}, total={total_count})",
                    flush=True,
                )
                yield last_good
                continue

            # Keep the last valid batch (this preserves the exact expected tuple length, devices, dtypes)
            last_good = batch
            yield batch

    _rollout_storage_cls.mini_batch_generator = _patched_mini_batch_generator
    print(
        f"[INFO][rank{_dist_rank()}] Patched {_rollout_storage_cls.__name__}.mini_batch_generator: "
        f"None -> reuse last_good",
        flush=True,
    )
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
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
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
import torch
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
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

        log_dir += f"_rank{app_launcher.local_rank}"

    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

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
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
