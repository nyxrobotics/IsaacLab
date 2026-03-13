# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import fnmatch
import math
import os
import re
from dataclasses import MISSING
from typing import Any

import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..assets.canele_cfg import CANELE_MINIMAL_CFG
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg
from .io_descriptors import history_observation_descriptor
from .rewards import canele_rewards_env
from .rewards import canele_rewards_joint
from .rewards import canele_rewards_link
from .rewards import canele_rewards_walk
from .terminations import canele_terminations

from pxr import Usd


IMU_ORIENTATION_NOISE_STD = 0.015
IMU_GYRO_NOISE_STD = 0.01

ACTUATOR_NOISE_STD_DEG = 0.5
ACTUATOR_NOISE_STD_RAD = math.radians(ACTUATOR_NOISE_STD_DEG)

ACTUATOR_DELAY_MIN = 1
ACTUATOR_DELAY_MAX = 4

BACKLASH_DEFAULT_DEG = 1.6
BACKLASH_DEFAULT_RAD = math.radians(BACKLASH_DEFAULT_DEG)

BACKLASH_RANDOM_RANGE_DEG = (0.5, 2.0)
BACKLASH_RANDOM_RANGE_RAD = (
    math.radians(BACKLASH_RANDOM_RANGE_DEG[0]),
    math.radians(BACKLASH_RANDOM_RANGE_DEG[1]),
)

JOINT_FRICTION_RANGE = (0.1, 0.3)
JOINT_EFFORT_LIMIT_RATIO_RANGE = (0.8, 1.0)
JOINT_DAMPING_RANGE = (0.6, 0.7)


def find_prim_paths(usd_path: str, pattern: str) -> list[str]:
    """Find USD prim paths whose last path element matches a glob pattern."""
    stage = Usd.Stage.Open(usd_path)
    results = []
    for prim in stage.Traverse():
        name = prim.GetPath().name
        if fnmatch.fnmatch(name, pattern):
            results.append(str(prim.GetPath()))
    return results


def _apply_gaussian_noise(values: torch.Tensor, std: float) -> torch.Tensor:
    """Apply additive Gaussian noise if std is positive."""
    if std <= 0.0:
        return values
    return values + torch.randn_like(values) * std


def _get_history_buffer(env, attr_name: str, feature_dim: int) -> torch.Tensor:
    """Return or lazily create a [num_envs, 4, feature_dim] history buffer on the env."""
    hist = getattr(env, attr_name, None)
    if (
        hist is None
        or hist.shape[0] != env.num_envs
        or hist.shape[1] != 4
        or hist.shape[2] != feature_dim
        or hist.device != env.device
    ):
        hist = torch.zeros(env.num_envs, 4, feature_dim, device=env.device)
        setattr(env, attr_name, hist)
    return hist


def _update_history(
    env, attr_name: str, values: torch.Tensor, *, init_with_current: bool
) -> torch.Tensor:
    """Append current values to a 4-step per-env history buffer."""
    hist = _get_history_buffer(env, attr_name, int(values.shape[-1]))
    env_step_count = getattr(env, "episode_length_buf", None)
    if env_step_count is None:
        reset_mask = torch.zeros(
            values.shape[0], dtype=torch.bool, device=values.device
        )
    else:
        reset_mask = env_step_count == 0

    non_reset_mask = ~reset_mask
    if torch.any(non_reset_mask):
        hist[non_reset_mask, :-1] = hist[non_reset_mask, 1:].clone()
        hist[non_reset_mask, -1] = values[non_reset_mask]

    if torch.any(reset_mask):
        if init_with_current:
            hist[reset_mask] = values[reset_mask].unsqueeze(1).repeat(1, 4, 1)
        else:
            hist[reset_mask] = 0.0

    return hist.reshape(values.shape[0], -1)


def _ensure_env_buffer(
    env,
    attr_name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype | None = None,
    fill_value: float | int = 0.0,
) -> torch.Tensor:
    """Create a tensor buffer on env if missing or mismatched."""
    buf = getattr(env, attr_name, None)
    expected_shape = (env.num_envs, *shape)
    dtype = dtype or torch.float32
    if (
        buf is None
        or tuple(buf.shape) != expected_shape
        or buf.device != env.device
        or buf.dtype != dtype
    ):
        if dtype.is_floating_point:
            buf = torch.full(
                expected_shape, float(fill_value), device=env.device, dtype=dtype
            )
        else:
            buf = torch.full(
                expected_shape, int(fill_value), device=env.device, dtype=dtype
            )
        setattr(env, attr_name, buf)
    return buf


def _resolve_joint_ids(asset, joint_names: list[str] | None) -> list[int]:
    """Resolve joint ids from names while preserving order."""
    if not joint_names:
        return list(range(asset.num_joints))
    if hasattr(asset, "find_joints"):
        found = asset.find_joints(joint_names, preserve_order=True)
        if isinstance(found, tuple):
            ids = found[0]
        else:
            ids = found
        if isinstance(ids, slice):
            return list(range(asset.num_joints))[ids]
        if torch.is_tensor(ids):
            return ids.tolist()
        return list(ids)
    return list(range(asset.num_joints))


def _match_values(
    joint_names: list[str],
    value: float | dict[str, float] | None,
    *,
    default: float = 0.0,
) -> torch.Tensor:
    """Map scalar or regex-keyed dict values to a per-joint tensor."""
    out = torch.full((len(joint_names),), float(default), dtype=torch.float32)
    if value is None:
        return out
    if isinstance(value, (int, float)):
        out[:] = float(value)
        return out
    if isinstance(value, dict):
        for pattern, pattern_value in value.items():
            for i, joint_name in enumerate(joint_names):
                if re.fullmatch(pattern, joint_name) or re.match(pattern, joint_name):
                    out[i] = float(pattern_value)
        return out
    raise TypeError(f"Unsupported value type for joint mapping: {type(value)}")


def _get_joint_limits_for_ids(
    asset, joint_ids: list[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return lower and upper soft joint limits for the selected joints."""
    limits = asset.data.soft_joint_pos_limits
    if limits.dim() == 3:
        lower = limits[0, joint_ids, 0].to(device=device)
        upper = limits[0, joint_ids, 1].to(device=device)
    else:
        lower = limits[joint_ids, 0].to(device=device)
        upper = limits[joint_ids, 1].to(device=device)
    return lower, upper


def _get_current_joint_effort_limits(
    asset, joint_ids: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Return current per-joint effort limits from the runtime articulation interface."""
    if hasattr(asset, "root_physx_view") and asset.root_physx_view is not None:
        view = asset.root_physx_view
        if hasattr(view, "get_dof_max_forces"):
            effort = view.get_dof_max_forces()
            effort = torch.as_tensor(effort, device=device, dtype=torch.float32)
            if effort.dim() == 2:
                effort = effort[0, joint_ids]
            else:
                effort = effort[joint_ids]
            return effort

    if (
        hasattr(asset.data, "joint_effort_limits")
        and asset.data.joint_effort_limits is not None
    ):
        effort = asset.data.joint_effort_limits
        if effort.dim() == 2:
            return effort[0, joint_ids].to(device=device)
        return effort[joint_ids].to(device=device)

    raise AttributeError(
        "Could not resolve current joint effort limits from sim. "
        "Please inspect the articulation API available in this Isaac Lab build."
    )


def randomize_joint_drive_parameters(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_range: tuple[float, float] = JOINT_FRICTION_RANGE,
    effort_limit_ratio_range: tuple[float, float] = JOINT_EFFORT_LIMIT_RATIO_RANGE,
    damping_range: tuple[float, float] = JOINT_DAMPING_RANGE,
    backlash_range_rad: tuple[float, float] = BACKLASH_RANDOM_RANGE_RAD,
) -> None:
    """Randomize per-env, per-joint drive parameters and backlash."""
    asset = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.flatten().long()

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = torch.arange(asset.num_joints, device=env.device)
    elif not torch.is_tensor(joint_ids):
        joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
    else:
        joint_ids = joint_ids.to(device=env.device, dtype=torch.long)

    n_reset = int(env_ids.shape[0])
    n_joints = int(joint_ids.shape[0])

    friction = torch.empty((n_reset, n_joints), device=env.device).uniform_(
        *friction_range
    )
    damping = torch.empty((n_reset, n_joints), device=env.device).uniform_(
        *damping_range
    )
    backlash = torch.empty((n_reset, n_joints), device=env.device).uniform_(
        *backlash_range_rad
    )

    base_effort_limit = _get_current_joint_effort_limits(asset, joint_ids, env.device)
    effort_ratio = torch.empty((n_reset, n_joints), device=env.device).uniform_(
        *effort_limit_ratio_range
    )
    effort_limit = effort_ratio * base_effort_limit.unsqueeze(0)

    asset.write_joint_friction_coefficient_to_sim(
        friction, joint_ids=joint_ids, env_ids=env_ids
    )
    asset.write_joint_effort_limit_to_sim(
        effort_limit, joint_ids=joint_ids, env_ids=env_ids
    )
    asset.write_joint_damping_to_sim(damping, joint_ids=joint_ids, env_ids=env_ids)

    env_backlash = _ensure_env_buffer(env, "_canele_joint_backlash_rad", (n_joints,))
    env_backlash[env_ids] = backlash


class DelayedBacklashJointPositionAction(ActionTerm):
    """Joint position action term with delay, backlash, and actuator noise."""

    def __init__(self, cfg: "DelayedBacklashJointPositionActionCfg", env):
        super().__init__(cfg, env)

        self.cfg: DelayedBacklashJointPositionActionCfg = cfg
        self._joint_names = list(cfg.joint_names)
        self._joint_ids = _resolve_joint_ids(self._asset, self._joint_names)
        self._num_envs = env.num_envs
        self._device = env.device
        self._action_dim = len(self._joint_ids)

        self._raw_actions = torch.zeros(
            self._num_envs, self._action_dim, device=self._device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._pending_target = torch.zeros_like(self._raw_actions)
        self._last_applied_target = torch.zeros_like(self._raw_actions)
        self._gear_position = torch.zeros_like(self._raw_actions)
        self._last_direction = torch.zeros_like(self._raw_actions)
        self._delay_counter = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.long
        )
        self._delay_steps = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.long
        )

        self._lower_limits, self._upper_limits = _get_joint_limits_for_ids(
            self._asset, self._joint_ids, self._device
        )

        self._scale = _match_values(self._joint_names, cfg.scale, default=1.0).to(
            self._device
        )
        if cfg.offset is not None:
            self._offset = _match_values(self._joint_names, cfg.offset, default=0.0).to(
                self._device
            )
        elif cfg.use_default_offset:
            self._offset = (
                self._asset.data.default_joint_pos[0, self._joint_ids]
                .detach()
                .clone()
                .to(self._device)
            )
        else:
            self._offset = torch.zeros(self._action_dim, device=self._device)

        self._default_backlash = torch.full(
            (self._num_envs, self._action_dim),
            cfg.backlash_default_rad,
            device=self._device,
        )
        self.reset()

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids=None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)

        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._pending_target[env_ids] = joint_pos[env_ids]
        self._last_applied_target[env_ids] = joint_pos[env_ids]
        self._gear_position[env_ids] = joint_pos[env_ids]
        self._last_direction[env_ids] = 0.0
        self._delay_counter[env_ids] = 0
        self._delay_steps[env_ids] = 0
        return {}

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        clipped = actions.clamp(self.cfg.raw_action_min, self.cfg.raw_action_max)
        self._processed_actions[:] = clipped

        target = clipped * self._scale.unsqueeze(0) + self._offset.unsqueeze(0)
        target = torch.max(
            torch.min(target, self._upper_limits.unsqueeze(0)),
            self._lower_limits.unsqueeze(0),
        )
        self._pending_target[:] = target

        self._delay_counter.zero_()
        self._delay_steps = torch.randint(
            low=self.cfg.actuator_delay_min,
            high=self.cfg.actuator_delay_max + 1,
            size=(self._num_envs,),
            device=self._device,
            dtype=torch.long,
        )

    def apply_actions(self):
        ready_mask = self._delay_counter >= self._delay_steps
        self._delay_counter += 1

        active_target = torch.where(
            ready_mask.unsqueeze(-1),
            self._pending_target,
            self._last_applied_target,
        )

        backlash = getattr(self._env, "_canele_joint_backlash_rad", None)
        if backlash is None or backlash.shape != self._default_backlash.shape:
            backlash = self._default_backlash
        else:
            backlash = backlash.to(device=self._device)

        delta = active_target - self._gear_position
        direction = torch.sign(delta)
        direction_changed = (direction != self._last_direction) & (
            self._last_direction != 0
        )

        movement = torch.where(
            direction_changed,
            torch.clamp(torch.abs(delta) - backlash, min=0.0) * direction,
            delta,
        )

        self._gear_position += movement
        self._last_direction = torch.where(delta != 0, direction, self._last_direction)

        noisy_target = _apply_gaussian_noise(
            self._gear_position, self.cfg.actuator_noise_std_rad
        )
        noisy_target = torch.max(
            torch.min(noisy_target, self._upper_limits.unsqueeze(0)),
            self._lower_limits.unsqueeze(0),
        )

        self._last_applied_target[:] = active_target
        self._asset.set_joint_position_target(noisy_target, joint_ids=self._joint_ids)


@configclass
class DelayedBacklashJointPositionActionCfg(ActionTermCfg):
    """Configuration for a delayed joint position action with backlash and noise."""

    class_type: type[ActionTerm] = DelayedBacklashJointPositionAction

    asset_name: str = "robot"
    joint_names: list[str] = MISSING
    preserve_order: bool = True

    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] | None = None
    use_default_offset: bool = True

    raw_action_min: float = -1.0
    raw_action_max: float = 1.0

    actuator_delay_min: int = ACTUATOR_DELAY_MIN
    actuator_delay_max: int = ACTUATOR_DELAY_MAX
    actuator_noise_std_rad: float = ACTUATOR_NOISE_STD_RAD
    backlash_default_rad: float = BACKLASH_DEFAULT_RAD


@history_observation_descriptor(
    observation_type="CommandHistory",
    terms_per_step=3,
    history_length=4,
    units="normalized",
    source="base_velocity command [lin_vel_x, lin_vel_y, ang_vel_z]",
    normalization="minmax_to_minus1_plus1",
)
def canele_obs_cmd_vel_history(env) -> torch.Tensor:
    """Flattened 4-step history of normalized base velocity commands [vx, vy, wz]."""
    commands = env.command_manager.get_command("base_velocity")[:, :3]
    cmd_cfg = env.cfg.commands.base_velocity.ranges
    cmd_min = torch.tensor(
        [cmd_cfg.lin_vel_x[0], cmd_cfg.lin_vel_y[0], cmd_cfg.ang_vel_z[0]],
        device=env.device,
        dtype=commands.dtype,
    )
    cmd_max = torch.tensor(
        [cmd_cfg.lin_vel_x[1], cmd_cfg.lin_vel_y[1], cmd_cfg.ang_vel_z[1]],
        device=env.device,
        dtype=commands.dtype,
    )
    cmd_norm = 2.0 * (commands - cmd_min) / (cmd_max - cmd_min).clamp_min(1.0e-6) - 1.0
    return _update_history(
        env, "_canele_cmd_vel_hist", cmd_norm, init_with_current=True
    )


@history_observation_descriptor(
    observation_type="IMUHistory",
    terms_per_step=2,
    history_length=4,
    units="normalized",
    axes=["gravity_x", "gravity_y"],
    source="projected_gravity_b[:2] with additive IMU orientation noise",
    normalization="clamp_to_minus1_plus1",
)
def canele_obs_projected_gravity_history(
    env, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Flattened 4-step history of projected gravity x/y with IMU orientation noise."""
    asset = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b[:, :2]
    projected_gravity = _apply_gaussian_noise(
        projected_gravity, IMU_ORIENTATION_NOISE_STD
    )
    gravity_xy = projected_gravity.clamp(-1.0, 1.0)
    return _update_history(
        env, "_canele_projected_gravity_hist", gravity_xy, init_with_current=False
    )


@history_observation_descriptor(
    observation_type="IMUHistory",
    terms_per_step=3,
    history_length=4,
    units="normalized",
    axes=["ang_vel_x", "ang_vel_y", "ang_vel_z"],
    source="root_ang_vel_b[:3] with additive gyro noise",
    normalization="clamp_to_minus2_plus2_then_divide_by_2",
)
def canele_obs_ang_vel_history(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Flattened 4-step history of body-frame angular velocity with gyro noise."""
    asset = env.scene[asset_cfg.name]
    angular_vel = asset.data.root_ang_vel_b[:, :3]
    angular_vel = _apply_gaussian_noise(angular_vel, IMU_GYRO_NOISE_STD)
    angular_vel = angular_vel.clamp(-2.0, 2.0) / 2.0
    return _update_history(
        env, "_canele_ang_vel_hist", angular_vel, init_with_current=False
    )


@history_observation_descriptor(
    observation_type="ActionHistory",
    terms_per_step=13,
    history_length=4,
    units="normalized",
    source="env.action_manager.prev_action for the actuated joints",
    normalization="policy_action_clamped_to_minus1_plus1",
    include_joint_names=True,
)
def canele_obs_action_history(env) -> torch.Tensor:
    """Flattened 4-step history of previous policy actions."""
    action = env.action_manager.prev_action
    action = action.clamp(-1.0, 1.0)
    return _update_history(env, "_canele_action_hist", action, init_with_current=True)


@configclass
class CaneleRewards(RewardsCfg):
    """Reward terms for the MDP (Canele)."""

    termination_penalty = RewTerm(func=canele_rewards_env.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=canele_rewards_walk.track_lin_vel_xy_yaw_frame_exp_no_flight,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["right_toe_link", "left_toe_link"]
            ),
            "contact_time_eps": 1.0e-3,
            "force_eps": 1.0e-3,
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=canele_rewards_walk.track_ang_vel_z_world_exp_no_flight,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["right_toe_link", "left_toe_link"]
            ),
            "contact_time_eps": 1.0e-3,
            "force_eps": 1.0e-3,
        },
    )
    feet_air_time = RewTerm(
        func=canele_rewards_walk.feet_air_time_alternating_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["right_toe_link", "left_toe_link"]
            ),
            "linear_cmd_threshold": 0.0,
            "angular_cmd_threshold": 0.0,
            "body_tilt_threshold": 0.0,
            "air_min_time": 0.1,
            "air_max_time": 1.0,
            "min_contact_time": 0.1,
            "ema_alpha": 0.02,
            "air_balance_weight": 1.0,
            "contact_balance_weight": 0.0,
            "air_reward": 1.0,
            "contact_reward": 1.0,
        },
    )
    feet_slide = RewTerm(
        func=canele_rewards_walk.feet_slide_keep_flat,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["right_toe_link", "left_toe_link"]
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["right_toe_link", "left_toe_link"]
            ),
            "air_time_eps": 0.02,
        },
    )
    joint_deviation_hip = RewTerm(
        func=canele_rewards_joint.joint_action_deviation_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_hip_roll",
                    "left_hip_pitch",
                    "right_hip_roll",
                    "right_hip_pitch",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=canele_rewards_joint.joint_action_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["left_hip_yaw", "right_hip_yaw", "torso_yaw"]
            )
        },
    )
    flat_toe_penalty = RewTerm(
        func=canele_rewards_link.flat_orientation_links_l2,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["right_toe_link", "left_toe_link"]
            ),
            "margin": 0.0,
            "gain": 1.0,
        },
    )


@configclass
class CaneleRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: CaneleRewards = CaneleRewards()

    def __post_init__(self):
        super().__post_init__()

        usd_path = CANELE_MINIMAL_CFG.spawn.usd_path
        print("[DEBUG] Loading USD:", usd_path)

        base_paths = find_prim_paths(usd_path, "body_link")
        print("[DEBUG] Found base_link prims:", base_paths)
        if not base_paths:
            raise RuntimeError("body_link not found in USD!")
        base_link_full = base_paths[0]
        base_link_name = os.path.basename(base_link_full)

        print("[DEBUG] base_link_full:", base_link_full)
        print("[DEBUG] base_link_name:", base_link_name)

        ankle_paths = find_prim_paths(usd_path, "*_toe_link")
        print("[DEBUG] Found ankle yaw prims:", ankle_paths)
        if not ankle_paths:
            raise RuntimeError("*_toe_link not found in USD!")
        ankle_names = [os.path.basename(p) for p in ankle_paths]
        print("[DEBUG] ankle yaw link names:", ankle_names)

        self.scene.robot = CANELE_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.height_scanner = None
        if hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None

        actuated_joint_names: list[str] = []
        try:
            for actuator_cfg in self.scene.robot.actuators.values():
                actuated_joint_names.extend(list(actuator_cfg.joint_names_expr))
        except Exception as exc:
            print(
                "[DEBUG] Failed to collect actuated joints from robot.actuators:", exc
            )

        seen = set()
        actuated_joint_names = [
            joint_name
            for joint_name in actuated_joint_names
            if not (joint_name in seen or seen.add(joint_name))
        ]

        arm_joints = [
            "left_shoulder_yaw",
            "left_shoulder_pitch",
            "left_shoulder_roll",
            "left_elbow_yaw",
            "left_elbow_pitch",
            "left_wrist_yaw",
            "left_wrist_roll",
            "left_wrist_pitch",
            "right_shoulder_yaw",
            "right_shoulder_pitch",
            "right_shoulder_roll",
            "right_elbow_yaw",
            "right_elbow_pitch",
            "right_wrist_yaw",
            "right_wrist_roll",
            "right_wrist_pitch",
        ]
        actuated_joint_names = [
            joint_name
            for joint_name in actuated_joint_names
            if joint_name not in arm_joints
        ]

        def _make_actuated_asset_cfg() -> SceneEntityCfg:
            return SceneEntityCfg(
                "robot", joint_names=actuated_joint_names, preserve_order=True
            )

        if hasattr(self, "actions") and hasattr(self.actions, "joint_pos"):
            old_action_cfg = self.actions.joint_pos
            self.actions.joint_pos = DelayedBacklashJointPositionActionCfg(
                asset_name="robot",
                joint_names=actuated_joint_names,
                preserve_order=True,
                scale=getattr(old_action_cfg, "scale", 1.0),
                offset=getattr(old_action_cfg, "offset", None),
                use_default_offset=getattr(old_action_cfg, "use_default_offset", True),
                raw_action_min=-1.0,
                raw_action_max=1.0,
                actuator_delay_min=ACTUATOR_DELAY_MIN,
                actuator_delay_max=ACTUATOR_DELAY_MAX,
                actuator_noise_std_rad=ACTUATOR_NOISE_STD_RAD,
                backlash_default_rad=BACKLASH_DEFAULT_RAD,
            )
            print(
                "[DEBUG] Replaced joint_pos action term with delayed-backlash/noise model"
            )

        if hasattr(self.observations, "policy"):
            policy_obs = self.observations.policy
            policy_obs.enable_corruption = True
            policy_obs.concatenate_terms = True
            policy_obs.concatenate_dim = -1
            policy_obs.history_length = None
            policy_obs.flatten_history_dim = True

            policy_obs.base_lin_vel = None
            policy_obs.base_ang_vel = None
            policy_obs.projected_gravity = None
            policy_obs.velocity_commands = None
            policy_obs.joint_pos = None
            policy_obs.joint_vel = None
            policy_obs.actions = None
            policy_obs.height_scan = None

            policy_obs.cmd_vel_history = ObsTerm(func=canele_obs_cmd_vel_history)
            policy_obs.projected_gravity_history = ObsTerm(
                func=canele_obs_projected_gravity_history,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )
            policy_obs.ang_vel_history = ObsTerm(
                func=canele_obs_ang_vel_history,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )
            policy_obs.action_history = ObsTerm(func=canele_obs_action_history)
            print("[DEBUG] Replaced policy observations with IMU/action histories")

        actuated_asset_cfg = _make_actuated_asset_cfg()
        if getattr(self.events, "reset_robot_joints", None) is not None:
            self.events.reset_robot_joints.params["asset_cfg"] = actuated_asset_cfg
            self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        self.events.randomize_joint_drive_params = EventTerm(
            func=randomize_joint_drive_parameters,
            mode="reset",
            params={
                "asset_cfg": actuated_asset_cfg,
                "friction_range": JOINT_FRICTION_RANGE,
                "effort_limit_ratio_range": JOINT_EFFORT_LIMIT_RATIO_RANGE,
                "damping_range": JOINT_DAMPING_RANGE,
                "backlash_range_rad": BACKLASH_RANDOM_RANGE_RAD,
            },
        )

        event_base_com = getattr(self.events, "base_com", None)
        if event_base_com is not None and hasattr(event_base_com, "params"):
            if event_base_com.params is None:
                event_base_com.params = {}
            event_base_com.params["asset_cfg"] = SceneEntityCfg(
                "robot", body_names=[base_link_name], preserve_order=True
            )

        self.rewards.feet_slide.params["sensor_cfg"].body_names = ankle_names
        self.rewards.feet_slide.params["asset_cfg"].body_names = ankle_names

        self.events.physics_material.params["asset_cfg"].body_names = ankle_names
        self.events.physics_material.params["static_friction_range"] = (0.1, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.1, 1.0)

        self.events.add_base_mass.params["asset_cfg"].body_names = [base_link_name]
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 1.0)

        self.events.base_com.params["asset_cfg"].body_names = [base_link_name]
        self.events.base_com.params["com_range"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (-0.02, 0.02),
        }

        self.events.base_external_force_torque.params["asset_cfg"].body_names = [
            base_link_name
        ]
        self.events.base_external_force_torque.params["force_range"] = (-2.0, 2.0)
        self.events.base_external_force_torque.params["torque_range"] = (-0.8, 0.8)

        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
        }
        self.events.push_robot.interval_range_s = (5.0, 20.0)

        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.05),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.3, 0.3),
            },
        }

        self.rewards.dof_pos_limits = None
        self.rewards.lin_vel_z_l2 = RewTerm(
            func=canele_rewards_link.lin_vel_z_l2, weight=-0.2
        )
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2 = RewTerm(
            func=canele_rewards_link.flat_orientation_l2, weight=-1.0
        )
        self.rewards.ang_vel_xy_l2.weight = -0.01
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_acc_l2 = RewTerm(
            func=canele_rewards_joint.joint_action_acc_l2,
            weight=-1.0e-9,
            params={"dt": self.decimation * self.sim.dt},
        )
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                "left_hip_yaw",
                "left_hip_roll",
                "left_hip_pitch",
                "left_knee_pitch",
                "right_hip_yaw",
                "right_hip_roll",
                "right_hip_pitch",
                "right_knee_pitch",
            ],
        )

        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.2, 1.2)

        self.terminations.base_contact = None
        self.terminations.detect_fall = DoneTerm(
            func=canele_terminations.detect_fall,
            params={
                "limit_angle": 1.3,
                "asset_cfg": SceneEntityCfg("robot", body_names=[base_link_name]),
            },
            time_out=False,
        )

        body_and_ankle_names = [base_link_name] + ankle_names
        self.terminations.detect_height_too_low_relative = DoneTerm(
            func=canele_terminations.detect_height_too_low_relative,
            params={
                "min_height": 0.5,
                "asset_cfg": SceneEntityCfg("robot", body_names=body_and_ankle_names),
            },
            time_out=False,
        )
        self.terminations.detect_tilt = DoneTerm(
            func=canele_terminations.detect_tilt_too_high_any_link,
            params={
                "max_tilt": 1.5,
                "asset_cfg": SceneEntityCfg("robot", body_names=body_and_ankle_names),
            },
            time_out=False,
        )
        self.terminations.support_plane_tilt = DoneTerm(
            func=canele_terminations.detect_support_plane_tilt_too_high,
            params={
                "max_tilt": 1.3,
                "asset_cfg": SceneEntityCfg("robot", body_names=body_and_ankle_names),
            },
            time_out=False,
        )


@configclass
class CaneleRoughEnvCfg_PLAY(CaneleRoughEnvCfg):
    """Visualization-friendly settings."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.2, 1.2)

        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0, 0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.scene.height_scanner = None
        if hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.export_io_descriptors = True
