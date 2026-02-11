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

# -------------------------------------------------------------------------------------
# Safety guards against NaN/Inf explosions during long training runs.
# These wrappers/patches are intentionally lightweight and keep comments in English only.
# -------------------------------------------------------------------------------------

from collections.abc import Mapping, Sequence


class RewardClipWrapper(gym.Wrapper):
    """Clip rewards and replace non-finite rewards with 0.0 to keep PPO stable."""

    def __init__(self, env, min_reward: float = -1e4, max_reward: float = 1e4):
        super().__init__(env)
        self._min_reward = float(min_reward)
        self._max_reward = float(max_reward)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if torch.is_tensor(reward):
            reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
            reward = reward.clamp_(self._min_reward, self._max_reward)
        return obs, reward, terminated, truncated, info


class TerminateOnNonFiniteWrapper(gym.Wrapper):
    """Force-terminate envs that produce NaN/Inf in obs/reward/done to avoid poisoning PPO."""

    def __init__(self, env, obs_fill_value: float = 0.0, reward_fill_value: float = 0.0):
        super().__init__(env)
        self._obs_fill_value = float(obs_fill_value)
        self._reward_fill_value = float(reward_fill_value)

    @staticmethod
    def _reduce_isfinite(x: torch.Tensor) -> torch.Tensor:
        """Return per-env finiteness mask for a (N, ...) tensor."""
        if x.ndim <= 1:
            return torch.isfinite(x)
        return torch.isfinite(x).all(dim=tuple(range(1, x.ndim)))

    def _obs_bad_mask(self, obs, num_envs: int, device) -> torch.Tensor:
        bad = torch.zeros(num_envs, dtype=torch.bool, device=device)

        def _acc(o):
            nonlocal bad
            if torch.is_tensor(o):
                if o.ndim >= 1 and o.shape[0] == num_envs:
                    bad |= ~self._reduce_isfinite(o)
                else:
                    # Scalar or unexpected shape: treat any non-finite as global bad
                    if not torch.isfinite(o).all():
                        bad |= True
            elif isinstance(o, Mapping):
                for v in o.values():
                    _acc(v)
            elif isinstance(o, Sequence) and not isinstance(o, (str, bytes)):
                for v in o:
                    _acc(v)

        _acc(obs)
        return bad

    def _sanitize_obs(self, obs, bad_env_mask: torch.Tensor):
        if torch.is_tensor(obs):
            obs = obs.clone()
            if obs.ndim >= 1 and obs.shape[0] == bad_env_mask.shape[0]:
                obs[bad_env_mask] = self._obs_fill_value
            else:
                obs = torch.nan_to_num(obs, nan=self._obs_fill_value, posinf=self._obs_fill_value, neginf=self._obs_fill_value)
            return obs
        if isinstance(obs, Mapping):
            return {k: self._sanitize_obs(v, bad_env_mask) for k, v in obs.items()}
        if isinstance(obs, Sequence) and not isinstance(obs, (str, bytes)):
            return type(obs)(self._sanitize_obs(v, bad_env_mask) for v in obs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Infer num_envs/device from tensors
        device = None
        num_envs = 1
        for t in (reward, terminated, truncated):
            if torch.is_tensor(t) and t.ndim >= 1:
                num_envs = int(t.shape[0])
                device = t.device
                break
        if device is None:
            # Try to locate device from obs
            def _find_tensor(o):
                if torch.is_tensor(o):
                    return o
                if isinstance(o, Mapping):
                    for v in o.values():
                        r = _find_tensor(v)
                        if r is not None:
                            return r
                if isinstance(o, Sequence) and not isinstance(o, (str, bytes)):
                    for v in o:
                        r = _find_tensor(v)
                        if r is not None:
                            return r
                return None
            ot = _find_tensor(obs)
            if ot is not None:
                device = ot.device
                if ot.ndim >= 1:
                    num_envs = int(ot.shape[0])
            else:
                device = torch.device('cpu')

        bad_obs = self._obs_bad_mask(obs, num_envs=num_envs, device=device)
        bad_reward = torch.zeros(num_envs, dtype=torch.bool, device=device)
        if torch.is_tensor(reward) and reward.ndim >= 1 and reward.shape[0] == num_envs:
            bad_reward |= ~torch.isfinite(reward)
        bad_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        for d in (terminated, truncated):
            if torch.is_tensor(d) and d.ndim >= 1 and d.shape[0] == num_envs:
                bad_done |= ~torch.isfinite(d.to(torch.float32))

        bad = bad_obs | bad_reward | bad_done

        if torch.any(bad):
            # Force terminate and sanitize to keep rollout buffers clean.
            if torch.is_tensor(terminated) and terminated.ndim >= 1 and terminated.shape[0] == num_envs:
                terminated = terminated.clone()
                terminated[bad] = True
            if torch.is_tensor(truncated) and truncated.ndim >= 1 and truncated.shape[0] == num_envs:
                truncated = truncated.clone()
                truncated[bad] = True
            obs = self._sanitize_obs(obs, bad)
            if torch.is_tensor(reward) and reward.ndim >= 1 and reward.shape[0] == num_envs:
                reward = reward.clone()
                reward[bad] = self._reward_fill_value
            if isinstance(info, dict):
                info = dict(info)
                info['nonfinite_terminated'] = bad

        return obs, reward, terminated, truncated, info


def patch_rslrl_actor_critic_for_safe_std(min_std: float = 1e-6) -> None:
    """Monkey-patch rsl_rl ActorCritic.act to clamp std and avoid crashes from NaN/negative std."""
    try:
        from rsl_rl.modules.actor_critic import ActorCritic  # type: ignore
    except Exception as exc:  # pragma: no cover
        omni.log.warn(f'Failed to import rsl_rl ActorCritic for patching: {exc}')
        return

    if getattr(ActorCritic, '_isaaclab_safe_std_patched', False):
        return

    orig_act = ActorCritic.act

    def safe_act(self, observations, masks=None, hidden_states=None):
        # rsl_rl stores distribution on self; we sanitize parameters before sampling.
        action = None
        try:
            action = orig_act(self, observations, masks=masks, hidden_states=hidden_states)
            return action
        except RuntimeError as e:
            msg = str(e)
            if 'std' not in msg and 'Normal' not in msg and 'normal expects all elements of std' not in msg:
                raise
            # Attempt to salvage by clamping distribution parameters.
            if hasattr(self, 'distribution') and self.distribution is not None:
                try:
                    loc = torch.nan_to_num(self.distribution.loc, nan=0.0, posinf=0.0, neginf=0.0)
                    scale = torch.nan_to_num(self.distribution.scale, nan=min_std, posinf=1e3, neginf=min_std)
                    scale = scale.clamp_min(min_std)
                    self.distribution = torch.distributions.Normal(loc, scale)
                    return self.distribution.sample()
                except Exception:
                    pass
            # As a last resort, return zeros so training can continue; buffers still get a valid tensor.
            if torch.is_tensor(observations):
                batch = observations.shape[0] if observations.ndim >= 1 else 1
            elif isinstance(observations, Mapping):
                t = next((v for v in observations.values() if torch.is_tensor(v)), None)
                batch = t.shape[0] if t is not None and t.ndim >= 1 else 1
            else:
                batch = 1
            device = observations.device if torch.is_tensor(observations) else (t.device if 't' in locals() and t is not None else 'cpu')
            return torch.zeros((batch, self.num_actions), device=device, dtype=torch.float32)

    ActorCritic.act = safe_act  # type: ignore[assignment]
    ActorCritic._isaaclab_safe_std_patched = True



@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # Patch rsl_rl to avoid hard-crashes when std becomes NaN/negative in very long runs.
    patch_rslrl_actor_critic_for_safe_std()
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

    # Guard against NaN/Inf explosions: terminate bad envs and keep rewards bounded.
    env = TerminateOnNonFiniteWrapper(env)
    env = RewardClipWrapper(env)

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