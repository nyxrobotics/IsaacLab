# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def torso_height_limit(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_height: float,
) -> torch.Tensor:
    """Penalize if torso (chest_link) gets too close to ankles.

    height = z(chest_link) - min(z(ankle_l_yaw_link), z(ankle_r_yaw_link))
    penalty = relu(min_height - height)
    """

    # robot articulation
    asset = env.scene[asset_cfg.name]  # usually "robot"

    # world positions of all bodies: (num_envs, num_bodies, 3)
    body_pos_w = asset.data.body_pos_w

    # indices resolved from body_names in SceneEntityCfg
    chest_id = asset_cfg.body_ids[0]
    ankle_ids = asset_cfg.body_ids[1:]

    chest_z = body_pos_w[:, chest_id, 2]                  # (num_envs,)
    ankles_z = body_pos_w[:, ankle_ids, 2]                # (num_envs, 2)
    min_ankle_z = ankles_z.min(dim=-1).values            # (num_envs,)

    # relative height (>= 0 が望ましい)
    rel_height = chest_z - min_ankle_z

    # しきい値より低い分だけペナルティ
    penalty = (min_height - rel_height).clamp(min=0.0)

    # RewardTerm の weight(<0) と掛け算されるので、ここでは正の値を返す
    return penalty

def feet_air_time_height_biped(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    desired_lift_time: float,
    desired_lift_height: float,
) -> torch.Tensor:
    """Reward for foot lift height * single-stance air-time for bipeds,
    with a time-varying target height that increases linearly until half of air_time
    and decreases linearly afterwards.
    """
    # --------------------------------------------------------
    # 0) commanded motion check (vx, vy, yaw)
    # --------------------------------------------------------
    cmd = env.command_manager.get_command(command_name)
    xy_speed = torch.norm(cmd[:, :2], dim=1)
    yaw_speed = torch.abs(cmd[:, 2])
    no_step_required = (xy_speed <= 0.001) & (yaw_speed <= 0.001)

    # reward tensors (per env)
    reward = torch.zeros_like(xy_speed)
    time_reward = torch.zeros_like(xy_speed)
    height_reward = torch.zeros_like(xy_speed)

    # If desired_lift_time is effectively zero, return zeros
    if desired_lift_time < 1.0e-6:
        return reward

    # --------------------------------------------------------
    # 1) air-time part
    # --------------------------------------------------------
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]      # (N, 2)
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]  # (N, 2)

    in_contact = contact_time > 0.0                                            # (N, 2)
    in_mode_time = torch.where(in_contact, contact_time, air_time)            # (N, 2)

    single_stance = torch.sum(in_contact.int(), dim=1) == 1                   # (N,)

    # normalized single-stance time in [0, 1]
    time_reward = torch.min(
        torch.where(
            single_stance.unsqueeze(-1),
            in_mode_time,
            torch.zeros_like(in_mode_time),
        ),
        dim=1,
    )[0]  # (N,)
    time_reward = torch.clamp(time_reward, max=desired_lift_time) / desired_lift_time

    # --------------------------------------------------------
    # 2) Time-varying target foot lift height (per foot)
    # --------------------------------------------------------
    # progress: 0 → 1 during desired_lift_time
    progress = torch.clamp(in_mode_time / desired_lift_time, max=1.0)          # (N, 2)
    # simple "parabolic" peak at progress=0.5: 1 - (2p-1)^2
    height_limit_scale = 1.0 - (2.0 * progress - 1.0) * (2.0 * progress - 1.0)
    height_limit_scale = torch.clamp(height_limit_scale, min=0.0)              # (N, 2)

    # target height over time for each foot
    required_height = desired_lift_height * height_limit_scale                 # (N, 2)

    # --------------------------------------------------------
    # 3) Actual foot height difference and height reward
    # --------------------------------------------------------
    robot = env.scene[asset_cfg.name]
    body_pos_w = robot.data.body_pos_w                                         # (N, num_bodies, 3)

    left_id, right_id = asset_cfg.body_ids
    left_z = body_pos_w[:, left_id, 2]                                         # (N,)
    right_z = body_pos_w[:, right_id, 2]                                       # (N,)
    foot_lift = torch.abs(left_z - right_z)                                    # (N,)

    # Expand to match (N, 2) so we can align with required_height per-foot
    foot_lift_expanded = foot_lift.unsqueeze(-1).expand_as(required_height)    # (N, 2)

    # Avoid division issues for very small required_height by using a mask
    eps = 1.0e-6
    active = (required_height > eps) & single_stance.unsqueeze(-1)             # (N, 2)

    raw_ratio = foot_lift_expanded / (required_height + eps)                   # (N, 2)
    height_ratio = torch.clamp(raw_ratio, max=1.0)                             # (N, 2)

    # Only count height where swing/stance pattern is valid
    height_ratio = torch.where(active, height_ratio, torch.zeros_like(height_ratio))

    # Reduce over feet dimension to get per-env height reward
    height_reward = torch.max(height_ratio, dim=1)[0]                          # (N,)

    # --------------------------------------------------------
    # 4) Combine time and height rewards
    # --------------------------------------------------------
    reward = time_reward * height_reward
    reward = torch.clamp(reward, min=0.0)
    reward = torch.sqrt(reward)

    # No reward if no step is required
    reward = torch.where(no_step_required, torch.zeros_like(reward), reward)

    return reward
