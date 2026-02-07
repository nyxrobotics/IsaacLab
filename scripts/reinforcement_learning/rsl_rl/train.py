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
import os
import torch
from datetime import datetime

from collections.abc import Mapping, Sequence


class TerminateOnNonFiniteWrapper(gym.Wrapper):
    """Force-terminate envs that output NaN/Inf in obs/reward/done.

    This prevents non-finite values from entering the rollout buffer and
    crashing PPO updates (e.g., invalid std in Normal distribution).
    """

    def __init__(self, env, obs_fill_value: float = 0.0, reward_fill_value: float = 0.0):
        super().__init__(env)
        self._obs_fill_value = float(obs_fill_value)
        self._reward_fill_value = float(reward_fill_value)

    @staticmethod
    def _isfinite_per_env(x: torch.Tensor) -> torch.Tensor:
        # Returns (num_envs,) mask indicating if all elements are finite for each env.
        if x.ndim == 0:
            return torch.isfinite(x).view(1)
        if x.ndim == 1:
            return torch.isfinite(x)
        reduce_dims = tuple(range(1, x.ndim))
        return torch.isfinite(x).all(dim=reduce_dims)

    def _obs_bad_mask(self, obs, num_envs: int, device=None) -> torch.Tensor:
        bad = torch.zeros(num_envs, dtype=torch.bool, device=device)

        def _acc(o):
            nonlocal bad
            if torch.is_tensor(o):
                fin = self._isfinite_per_env(o)
                if fin.numel() == num_envs:
                    bad |= ~fin
            elif isinstance(o, Mapping):
                for v in o.values():
                    _acc(v)
            elif isinstance(o, Sequence) and not isinstance(o, (str, bytes)):
                for v in o:
                    _acc(v)

        _acc(obs)
        return bad

    def _sanitize_obs(self, obs, bad_envs: torch.Tensor):
        if torch.is_tensor(obs):
            out = obs.clone()
            if out.ndim >= 1:
                out[bad_envs] = self._obs_fill_value
            return out
        if isinstance(obs, Mapping):
            return {k: self._sanitize_obs(v, bad_envs) for k, v in obs.items()}
        if isinstance(obs, Sequence) and not isinstance(obs, (str, bytes)):
            return type(obs)(self._sanitize_obs(v, bad_envs) for v in obs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Determine num_envs and device from reward/terminated or env.
        device = None
        num_envs = 1
        if torch.is_tensor(reward) and reward.ndim >= 1:
            num_envs = reward.shape[0]
            device = reward.device
        elif torch.is_tensor(terminated) and terminated.ndim >= 1:
            num_envs = terminated.shape[0]
            device = terminated.device
        else:
            if hasattr(self.unwrapped, "device"):
                device = self.unwrapped.device

        bad_obs = self._obs_bad_mask(obs, num_envs, device=device)

        bad_reward = torch.zeros(num_envs, dtype=torch.bool, device=device)
        if torch.is_tensor(reward):
            r = reward if reward.ndim >= 1 else reward.view(1)
            bad_reward = ~torch.isfinite(r)

        bad_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        for d in (terminated, truncated):
            if torch.is_tensor(d):
                dd = d if d.ndim >= 1 else d.view(1)
                bad_done |= ~torch.isfinite(dd.to(torch.float32))

        bad = bad_obs | bad_reward | bad_done

        if torch.any(bad):
            # Force terminate/truncate the bad envs so the runner resets them.
            if torch.is_tensor(terminated):
                terminated = terminated.clone()
                (terminated if terminated.ndim >= 1 else terminated.view(1))[bad] = True
            if torch.is_tensor(truncated):
                truncated = truncated.clone()
                (truncated if truncated.ndim >= 1 else truncated.view(1))[bad] = True

            # Sanitize outputs so rollouts won't contain NaN/Inf.
            obs = self._sanitize_obs(obs, bad)
            if torch.is_tensor(reward):
                reward = reward.clone()
                (reward if reward.ndim >= 1 else reward.view(1))[bad] = self._reward_fill_value

            if isinstance(info, dict):
                info = dict(info)
                info["nonfinite_terminated"] = bad

        return obs, reward, terminated, truncated, info


import omni
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
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # terminate envs that produce NaN/Inf to keep rollouts numerically stable
    env = TerminateOnNonFiniteWrapper(env)

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
