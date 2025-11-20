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
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import RigidObject
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

def torso_height_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_height: float,
) -> torch.Tensor:
    """Penalty version:
    - Returns 0 if torso height is sufficient.
    - Returns negative value proportional to height deficiency (meters).
    """

    # robot articulation
    asset = env.scene[asset_cfg.name]

    # world positions of all bodies: (num_envs, num_bodies, 3)
    body_pos_w = asset.data.body_pos_w

    chest_id = asset_cfg.body_ids[0]
    ankle_ids = asset_cfg.body_ids[1:]

    chest_z = body_pos_w[:, chest_id, 2]               # (num_envs,)
    ankles_z = body_pos_w[:, ankle_ids, 2]             # (num_envs,2)
    min_ankle_z = ankles_z.min(dim=-1).values          # (num_envs,)

    # torso height above lowest ankle
    rel_height = chest_z - min_ankle_z                 # (num_envs,)

    # height deficiency (meters)
    height_missing = torch.relu(min_height - rel_height)

    # negative penalty
    penalty = -height_missing

    return penalty


def feet_air_time_height_penalty(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    desired_lift_time: float,
    desired_lift_height: float,
) -> torch.Tensor:
    """Penalty version:
    0 when the foot lifts sufficiently.
    Negative when lift height or lift time is insufficient.
    """

    # --------------------------------------------------------
    # 0) commanded motion check (vx, vy, yaw)
    # --------------------------------------------------------
    cmd = env.command_manager.get_command(command_name)
    xy_speed = torch.norm(cmd[:, :2], dim=1)
    yaw_speed = torch.abs(cmd[:, 2])
    no_step_required = (xy_speed <= 0.001) & (yaw_speed <= 0.001)

    # output penalties (negative or zero)
    penalty = torch.zeros_like(xy_speed)

    if desired_lift_time < 1.0e-6:
        return penalty

    # --------------------------------------------------------
    # 1) air-time part
    # --------------------------------------------------------
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]      # (N, 2)
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]  # (N, 2)

    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)              # (N, 2)

    single_stance = torch.sum(in_contact.int(), dim=1) == 1                     # (N,)

    # actual single-stance time (N,)
    actual_time = torch.min(
        torch.where(
            single_stance.unsqueeze(-1),
            in_mode_time,
            torch.zeros_like(in_mode_time),
        ),
        dim=1,
    )[0]

    # desired_lift_time に対する不足量
    time_missing = torch.relu(desired_lift_time - actual_time)                  # (N,)
    time_penalty = -(time_missing / (desired_lift_time + 1e-6))                 # (N,)

    # --------------------------------------------------------
    # 2) Time-varying target foot lift height
    # --------------------------------------------------------
    progress = torch.clamp(in_mode_time / desired_lift_time, max=1.0)           # (N, 2)
    height_limit_scale = 1.0 - (2.0 * progress - 1.0) ** 2
    height_limit_scale = torch.clamp(height_limit_scale, min=0.0)
    required_height = desired_lift_height * height_limit_scale                  # (N, 2)

    # --------------------------------------------------------
    # 3) Actual foot height
    # --------------------------------------------------------
    robot = env.scene[asset_cfg.name]
    body_pos_w = robot.data.body_pos_w

    left_id, right_id = asset_cfg.body_ids
    left_z = body_pos_w[:, left_id, 2]
    right_z = body_pos_w[:, right_id, 2]
    foot_lift = torch.abs(left_z - right_z)                                     # (N,)
    foot_lift_exp = foot_lift.unsqueeze(-1).expand_as(required_height)          # (N, 2)

    eps = 1e-6
    active = (required_height > eps) & single_stance.unsqueeze(-1)              # (N, 2)

    # required_height と比較：不足していたら負
    height_missing = torch.relu(required_height - foot_lift_exp)                # (N, 2)
    height_penalty_foot = -(height_missing / (desired_lift_height + eps))       # (N, 2)

    # swing foot only
    height_penalty_foot = torch.where(active, height_penalty_foot, torch.zeros_like(height_penalty_foot))

    height_penalty = torch.min(height_penalty_foot, dim=1)[0]                   # (N,)

    # --------------------------------------------------------
    # 4) total penalty (more negative = worse)
    # --------------------------------------------------------
    penalty = time_penalty + height_penalty

    # no-step → no penalty
    penalty = torch.where(no_step_required, torch.zeros_like(penalty), penalty)

    return penalty

def track_lin_vel_xy_yaw_frame_linear_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Linear penalty for tracking linear velocity commands (x,y) in the yaw-aligned robot frame.

    - Perfect tracking → 0
    - Velocity mismatch → negative penalty proportional to L2 error
    """
    asset = env.scene[asset_cfg.name]

    # transform linear velocity into yaw-aligned base frame
    vel_yaw = quat_rotate_inverse(
        yaw_quat(asset.data.root_quat_w),
        asset.data.root_lin_vel_w[:, :3]
    )  # (N, 3)

    # commanded linear velocity (x, y) only
    cmd = env.command_manager.get_command(command_name)[:, :2]   # (N, 2)

    # linear L2 error
    error_vec = cmd - vel_yaw[:, :2]     # (N, 2)
    lin_vel_error = torch.linalg.norm(error_vec, ord=1, dim=1)
    # ※ L1（absの和）にしたい場合は ord=1、L2（二乗和ルート）なら ord=2

    # return negative penalty
    return -lin_vel_error

def track_ang_vel_z_world_linear_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Linear penalty for tracking yaw angular velocity (ang_vel_z) in world frame.

    - Perfect tracking: penalty = 0
    - Mismatch: negative penalty proportional to |cmd - actual|
    """
    asset = env.scene[asset_cfg.name]

    # commanded yaw rate
    cmd_yaw = env.command_manager.get_command(command_name)[:, 2]      # (N,)

    # actual yaw rate in world frame
    yaw_actual = asset.data.root_ang_vel_w[:, 2]                       # (N,)

    # absolute error
    ang_vel_error = torch.abs(cmd_yaw - yaw_actual)                    # (N,)

    # negative penalty
    penalty = -ang_vel_error

    return penalty


def command_ratio_alignment_penalty(
    env,
    command_name: str,
    asset_cfg=SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize mismatch between the ratio (direction) of command velocity
    and actual velocity: x, y, yaw components considered as one 3D vector.

    Perfect ratio match (same direction) → 0
    Larger mismatch → negative penalty
    """

    # -----------------------------
    # 1) Command vector c = [vx, vy, yaw]
    # -----------------------------
    cmd = env.command_manager.get_command(command_name)
    c = cmd[:, :3]   # (N, 3): vx, vy, yaw_cmd

    # -----------------------------
    # 2) Actual velocity vector v
    #     - linear vel xy in yaw frame
    #     - yaw rate in world frame
    # -----------------------------
    asset = env.scene[asset_cfg.name]

    # convert linear vel to yaw-aligned frame
    vel_yaw = quat_rotate_inverse(
        yaw_quat(asset.data.root_quat_w),
        asset.data.root_lin_vel_w[:, :3]
    )

    vx = vel_yaw[:, 0]
    vy = vel_yaw[:, 1]
    yaw_rate = asset.data.root_ang_vel_w[:, 2]

    v = torch.stack([vx, vy, yaw_rate], dim=1)

    # -----------------------------
    # 3) Cosine similarity
    # -----------------------------
    eps = 1e-6
    c_norm = torch.norm(c, dim=1) + eps
    v_norm = torch.norm(v, dim=1) + eps

    cos_sim = torch.sum(c * v, dim=1) / (c_norm * v_norm)  # (N,)

    # clamp to avoid numerical issues
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    # -----------------------------
    # 4) Penalty = negative mismatch
    # -----------------------------
    penalty = cos_sim - 1.0
    # cos_sim = 1 → penalty = 0
    # cos_sim = 0 → penalty = -1
    # cos_sim = -1 → penalty = -2

    return penalty

def alive_bonus_torso(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_height: float,
    max_tilt: float,
) -> torch.Tensor:
    """Alive bonus based on torso height and base tilt.

    - Returns 1.0 if:
        torso height above ankles >= min_height  AND
        base tilt (projected gravity xy L2) <= max_tilt
    - Returns 0.0 otherwise.
    """

    asset: RigidObject = env.scene[asset_cfg.name]

    # ---------------------------
    # 1) torso height above ankles
    # ---------------------------
    body_pos_w = asset.data.body_pos_w  # (N, num_bodies, 3)

    chest_id = asset_cfg.body_ids[0]
    ankle_ids = asset_cfg.body_ids[1:]

    chest_z = body_pos_w[:, chest_id, 2]          # (N,)
    ankles_z = body_pos_w[:, ankle_ids, 2]        # (N, 2)
    min_ankle_z = ankles_z.min(dim=-1).values     # (N,)

    rel_height = chest_z - min_ankle_z            # (N,)
    height_ok = rel_height >= min_height          # (N,)

    # ---------------------------
    # 2) base tilt using projected gravity
    # ---------------------------
    # projected_gravity_b : gravity expressed in base frame
    # ideally [0, 0, -1], so xy components measure tilt
    g_b = asset.data.projected_gravity_b          # (N, 3)
    tilt_l2 = torch.sum(g_b[:, :2] * g_b[:, :2], dim=1)  # (N,)
    tilt_ok = tilt_l2 <= max_tilt

    # ---------------------------
    # 3) Alive mask & bonus
    # ---------------------------
    alive = (height_ok & tilt_ok).float()         # 1.0 if alive, 0.0 otherwise

    return alive
