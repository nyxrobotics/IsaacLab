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
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0e-3
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

    両足接地中（single_stance=False）は常にペナルティ0。
    """

    # --------------------------------------------------------
    # 0) commanded motion check (vx, vy, yaw)
    # --------------------------------------------------------
    cmd = env.command_manager.get_command(command_name)
    xy_speed = torch.norm(cmd[:, :2], dim=1)
    yaw_speed = torch.abs(cmd[:, 2])
    no_step_required = (xy_speed <= 0.001) & (yaw_speed <= 0.001)

    penalty = torch.zeros_like(xy_speed)

    if desired_lift_time < 1.0e-6:
        return penalty

    # --------------------------------------------------------
    # 1) air-time part
    # --------------------------------------------------------
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]          # (N, 2)
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]  # (N, 2)

    in_contact = contact_time > 0.0                                                 # (N, 2)
    in_mode_time = torch.where(in_contact, contact_time, air_time)                 # (N, 2)

    # ちょうど片足だけ接地しているステップを single_stance とする
    single_stance = torch.sum(in_contact.int(), dim=1) == 1                        # (N,)

    # single_stance のときだけ実際の時間を使う。それ以外は desired_lift_time とみなして time_missing=0 にする
    actual_time_raw = torch.min(in_mode_time, dim=1)[0]                             # (N,)
    actual_time = torch.where(
        single_stance,
        actual_time_raw,
        torch.full_like(actual_time_raw, desired_lift_time),
    )

    # desired_lift_time に対する不足量（single_stance 以外は0になる）
    time_missing = torch.relu(desired_lift_time - actual_time)                      # (N,)
    time_missing = torch.where(single_stance, time_missing, torch.zeros_like(time_missing))
    time_penalty = -(time_missing / (desired_lift_time + 1e-6))                     # (N,)

    # --------------------------------------------------------
    # 2) Time-varying target foot lift height
    # --------------------------------------------------------
    progress = torch.clamp(in_mode_time / desired_lift_time, max=1.0)               # (N, 2)
    height_limit_scale = 1.0 - (2.0 * progress - 1.0) ** 2
    height_limit_scale = torch.clamp(height_limit_scale, min=0.0)
    required_height = desired_lift_height * height_limit_scale                      # (N, 2)

    # --------------------------------------------------------
    # 3) Actual foot height
    # --------------------------------------------------------
    robot = env.scene[asset_cfg.name]
    body_pos_w = robot.data.body_pos_w

    left_id, right_id = asset_cfg.body_ids
    left_z = body_pos_w[:, left_id, 2]
    right_z = body_pos_w[:, right_id, 2]
    foot_lift = torch.abs(left_z - right_z)                                         # (N,)
    foot_lift_exp = foot_lift.unsqueeze(-1).expand_as(required_height)              # (N, 2)

    eps = 1e-6
    active = (required_height > eps) & single_stance.unsqueeze(-1)                  # (N, 2)

    height_missing = torch.relu(required_height - foot_lift_exp)                    # (N, 2)
    height_penalty_foot = -(height_missing / (desired_lift_height + eps))           # (N, 2)

    # swing foot のときだけ高さペナルティを有効にする
    height_penalty_foot = torch.where(active, height_penalty_foot, torch.zeros_like(height_penalty_foot))

    # 両足のうち「より悪い方」を採用（必要なら max に変更）
    height_penalty = torch.min(height_penalty_foot, dim=1)[0]                       # (N,)

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
    height_decay_k: float = 1.0,
    tilt_decay_k: float = 1.0,
) -> torch.Tensor:
    """
    Continuous alive bonus in [0,1].

    Alive bonus is:
        alive = (height_term^height_decay_k) * (tilt_term^tilt_decay_k)

    - height_term : linear ratio rel_height / min_height (clamped 0→1)
    - tilt_term   : linear ratio 1 - tilt_l2 / max_tilt (clamped 0→1)

    Decay tuning:
        k = 1   → linear
        k > 1   → slower decay (more tolerant)
        k < 1   → sharper decay (stricter)
    """

    asset = env.scene[asset_cfg.name]

    # ---------------------------
    # 1) torso height above ankles
    # ---------------------------
    body_pos_w = asset.data.body_pos_w

    chest_id = asset_cfg.body_ids[0]
    ankle_ids = asset_cfg.body_ids[1:]

    chest_z = body_pos_w[:, chest_id, 2]
    ankles_z = body_pos_w[:, ankle_ids, 2]
    min_ankle_z = ankles_z.min(dim=-1).values

    rel_height = chest_z - min_ankle_z

    # height term (0～1)
    height_term = torch.clamp(rel_height / (min_height + 1e-6), 0.0, 1.0)

    # decay OR binary
    if height_decay_k == 0:
        height_bonus = (rel_height >= min_height).float()
    else:
        height_bonus = height_term ** height_decay_k

    # ---------------------------
    # 2) base tilt
    # ---------------------------
    g_b = asset.data.projected_gravity_b
    tilt_l2 = torch.sum(g_b[:, :2] * g_b[:, :2], dim=1)

    # tilt term (0～1)
    tilt_term = torch.clamp(1.0 - tilt_l2 / (max_tilt + 1e-6), 0.0, 1.0)

    # decay OR binary
    if tilt_decay_k == 0:
        tilt_bonus = (tilt_l2 <= max_tilt).float()
    else:
        tilt_bonus = tilt_term ** tilt_decay_k

    # ---------------------------
    # 3) final alive bonus
    # ---------------------------
    alive = height_bonus * tilt_bonus
    return alive


def step_reflex_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    vel_thresh: float,
    tilt_margin: float,
    min_air_time: float,
    max_stance_time: float,
    min_air_height: float,
    accel_margin: float,
) -> torch.Tensor:
    """
    Step-reflex penalty.

    0 is best, more negative is worse.

    When the robot is commanded to move or becomes tilted, this term:
      - penalizes double-stance that is too long
        (allowed time shrinks linearly with tilt)
      - penalizes swing air-time that is too short
      - penalizes swing height that is too low (world frame)
      - penalizes swing height that is too low (root-local frame)
      - penalizes swing foot velocity opposite to desired direction (world & root-local)
      - penalizes swing foot yaw velocity opposite to yaw command

    Assumptions:
      - asset_cfg.body_ids: [left_foot_id, right_foot_id]
      - torso frame: articulation root pose (root_pos_w, root_quat_w)
    """

    eps = 1e-6

    # -------------------------------------------------------
    # body IDs: [torso, left_foot, right_foot]
    # -------------------------------------------------------
    torso_id, left_id, right_id = asset_cfg.body_ids

    asset = env.scene[asset_cfg.name]

    # full body states
    body_pos = asset.data.body_pos_w          # (N, B, 3)
    body_quat = asset.data.body_quat_w        # (N, B, 4)
    body_vel = asset.data.body_lin_vel_w      # (N, B, 3)
    body_acc = asset.data.body_lin_acc_w      # (N, B, 3)

    # torso state
    torso_pos = body_pos[:, torso_id]
    torso_quat = body_quat[:, torso_id]
    torso_vel = body_vel[:, torso_id]
    torso_acc = body_acc[:, torso_id]

    # feet z
    left_z = body_pos[:, left_id, 2]
    right_z = body_pos[:, right_id, 2]

    # -------------------------------------------------------
    # 1) Command magnitude
    # -------------------------------------------------------
    cmd = env.command_manager.get_command(command_name)  # (N,3)
    cmd_xy = torch.norm(cmd[:, :2], dim=1)
    cmd_yaw = torch.abs(cmd[:, 2])
    cmd_speed = torch.sqrt(cmd_xy * cmd_xy + cmd_yaw * cmd_yaw)
    # --------------------------------------------------------
    # 2) Torso tilt (torso(body_link) frame)
    # --------------------------------------------------------
    asset = env.scene[asset_cfg.name]
    torso_g_b = asset.data.projected_gravity_b
    tilt_l2 = torch.sum(torso_g_b[:, :2] * torso_g_b[:, :2], dim=1)


    # tilt-dependent allowed double-stance time:
    tilt_norm = torch.clamp(tilt_l2 / (tilt_margin + eps), 0.0, 1.0)
    effective_max_stance_time = max_stance_time * (1.0 - tilt_norm)
    # near-level region: encourage alternating steps
    near_level = tilt_l2 <= tilt_margin
    
    need_to_step = (cmd_speed > vel_thresh) | (tilt_l2 > tilt_margin)

    # --------------------------------------------------------
    # 2b) forward acceleration-based stance shrink (torso-based)
    # --------------------------------------------------------
    v_xy = torso_vel[:, :2]
    a_xy = torso_acc[:, :2]

    v_norm = torch.norm(v_xy, dim=1)
    v_unit = v_xy / (v_norm.unsqueeze(-1) + 1e-6)

    a_forward = torch.sum(v_unit * a_xy, dim=1)
    a_forward_pos = torch.relu(a_forward)

    if accel_margin > 0:
        acc_ratio = torch.clamp(a_forward_pos / (accel_margin + eps), 0.0, 1.0)
        acc_scale = 1.0 - acc_ratio
    else:
        acc_scale = torch.ones_like(a_forward_pos)

    accelerating_forward = (a_forward_pos > 1.0e-2) & (v_norm > 1.0e-3)

    accel_scale_factor = torch.where(
        accelerating_forward, acc_scale, torch.ones_like(acc_scale)
    )

    effective_max_stance_time = effective_max_stance_time * accel_scale_factor

    need_to_step = need_to_step | accelerating_forward

    # -------------------------------------------------------
    # 3) Contact info
    # -------------------------------------------------------
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids[1:]]      # left,right
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids[1:]]

    in_contact = contact_time > 0
    left_contact = in_contact[:, 0]
    right_contact = in_contact[:, 1]
    double_stance_time = torch.min(contact_time, dim=1)[0]

    # -------------------------------------------------------
    # 4) Swing foot detection
    # -------------------------------------------------------
    left_higher = left_z > right_z
    right_higher = right_z > left_z

    left_stance_time = contact_time[:, 0]
    right_stance_time = contact_time[:, 1]

    right_stance_ok = torch.where(
        near_level,
        right_stance_time >= min_air_time,
        torch.ones_like(right_stance_time, dtype=torch.bool),
    )
    left_stance_ok = torch.where(
        near_level,
        left_stance_time >= min_air_time,
        torch.ones_like(left_stance_time, dtype=torch.bool),
    )

    left_swing = (
        left_higher & (~left_contact) & right_contact & right_stance_ok
    ) | ((~left_contact) & (~right_contact) & left_higher)

    right_swing = (
        right_higher & (~right_contact) & left_contact & left_stance_ok
    ) | ((~left_contact) & (~right_contact) & right_higher)

    is_swing = left_swing | right_swing
    reflex_single = need_to_step & is_swing
    reflex_double = need_to_step & (~is_swing)

    swing_air_time = torch.where(left_swing, air_time[:, 0], air_time[:, 1])
    swing_air_time = torch.where(is_swing, swing_air_time, torch.zeros_like(swing_air_time))

    swing_height = torch.abs(left_z - right_z)

    # -------------------------------------------------------
    # 5) Stance-time penalty
    # -------------------------------------------------------
    stance_missing = torch.relu(double_stance_time - effective_max_stance_time) / (max_stance_time + eps)
    stance_penalty = torch.where(reflex_double, -stance_missing, torch.zeros_like(stance_missing))

    # -------------------------------------------------------
    # 6) Swing time penalty
    # -------------------------------------------------------
    air_missing = torch.relu(min_air_time - swing_air_time) / (min_air_time + eps)
    air_penalty = torch.where(reflex_single, -air_missing, torch.zeros_like(air_missing))

    # -------------------------------------------------------
    # 7) Swing height penalty (world)
    # -------------------------------------------------------
    height_missing = torch.relu(min_air_height - swing_height) / (min_air_height + eps)
    height_penalty = torch.where(reflex_single, -height_missing, torch.zeros_like(height_missing))

    # -------------------------------------------------------
    # 8) Swing height penalty (local torso frame)
    # -------------------------------------------------------
    swing_pos = torch.where(left_swing.unsqueeze(-1), body_pos[:, left_id], body_pos[:, right_id])
    stance_pos = torch.where(left_higher.unsqueeze(-1), body_pos[:, right_id], body_pos[:, left_id])

    swing_local = quat_rotate_inverse(torso_quat, swing_pos - torso_pos)
    stance_local = quat_rotate_inverse(torso_quat, stance_pos - torso_pos)

    local_height_diff = swing_local[:, 2] - stance_local[:, 2]
    local_height_missing = torch.relu(min_air_height - local_height_diff) / (min_air_height + eps)
    local_height_penalty = torch.where(reflex_single, -local_height_missing, torch.zeros_like(local_height_missing))

    # -------------------------------------------------------
    # 9) Total
    # -------------------------------------------------------
    total_penalty = (
        stance_penalty
        + air_penalty
        + height_penalty
        + local_height_penalty
    )

    return torch.where(need_to_step, total_penalty, torch.zeros_like(total_penalty))



def feet_contact_angle_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    angle_limit_deg: float,
) -> torch.Tensor:
    """
    Penalize feet applying ground reaction forces at a large angle from gravity.
    Uses both sensor_cfg (force data) and asset_cfg (which foot links to include).
    """

    eps = 1e-6
    angle_limit_rad = angle_limit_deg * (3.14159265 / 180.0)

    # --------------------------------------------------------
    # 1) Contact forces from sensor
    # --------------------------------------------------------
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # forces_w_history: (N, history, sensor_bodies, 3)
    forces_w = contact_sensor.data.net_forces_w_history[:, -1, :, :]  # latest frame

    # sensor_cfg.body_ids → force array index (usually 0 or 1 for two feet)
    sensor_ids = sensor_cfg.body_ids

    # asset_cfg.body_ids → which robot foot links are evaluated
    asset_ids = asset_cfg.body_ids

    # センサーの foot index と asset_cfg の foot index が一致している前提で取り出す
    # もし一致していない場合は mapping 処理が必要（その場合は教えてください）
    forces_w = forces_w[:, sensor_ids, :]        # (N, num_feet, 3)

    force_norm = torch.norm(forces_w, dim=-1) + eps
    Fz = forces_w[..., 2]

    # detect contact
    in_contact = force_norm > 1e-3

    # --------------------------------------------------------
    # 2) Angle with gravity
    # --------------------------------------------------------
    cos_theta = (-Fz) / force_norm
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    # --------------------------------------------------------
    # 3) penalty
    # --------------------------------------------------------
    theta_excess = torch.relu(theta - angle_limit_rad)
    penalty_per_foot = -theta_excess

    # only apply when foot is in contact
    penalty_per_foot = torch.where(in_contact, penalty_per_foot, torch.zeros_like(penalty_per_foot))

    # sum over asset_cfg feet
    penalty = torch.sum(penalty_per_foot, dim=-1)

    return penalty


def track_lin_vel_xy_yaw_frame_linear_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Linear penalty for tracking commanded XY linear velocity,
    corrected by subtracting the torso's fall-induced translational velocity
    around the lower foot (stance foot).

    - stance foot = ankle with lower Z
    - fall-induced velocity: v_fall = ω × r
        ω = chest angular velocity
        r = chest_position - stance_foot_position

    - subtract v_fall **only** when fall direction is within 90° of command direction

    - both actual velocity and fall velocity are rotated into yaw frame
    """

    asset = env.scene[asset_cfg.name]

    body_pos = asset.data.body_pos_w         # (N, B, 3)
    body_quat = asset.data.body_quat_w       # (N, B, 4)
    body_vel = asset.data.body_lin_vel_w     # (N, B, 3)
    body_angvel = asset.data.body_ang_vel_w  # (N, B, 3)

    # body_ids = [chest_link, left_ankle, right_ankle]
    torso_id, left_id, right_id = asset_cfg.body_ids

    # torso state
    torso_pos = body_pos[:, torso_id]
    torso_angvel = body_angvel[:, torso_id]

    # ankle states
    left_pos = body_pos[:, left_id]
    right_pos = body_pos[:, right_id]

    left_z = left_pos[:, 2]
    right_z = right_pos[:, 2]

    # stance = lower foot
    left_lower = left_z < right_z
    stance_pos = torch.where(left_lower.unsqueeze(-1), left_pos, right_pos)

    # --------------------------------------------------------
    # 1) compute fall-induced velocity v_fall = ω × r
    # --------------------------------------------------------
    r = torso_pos - stance_pos                       # (N, 3)
    v_fall = torch.cross(torso_angvel, r, dim=1)     # (N, 3)
    v_fall_xy = v_fall[:, :2]                        # XY components

    # --------------------------------------------------------
    # 2) compute actual base linear velocity (root-free, from chest)
    # --------------------------------------------------------
    # "base velocity" = torso/chest velocity
    v_base_w = body_vel[:, torso_id]                 # (N,3)
    # rotate into yaw-aligned frame
    yaw_q = yaw_quat(body_quat[:, torso_id])         # use chest yaw only
    v_base_yaw = quat_rotate_inverse(yaw_q, v_base_w)[:, :2]   # (N,2)

    # --------------------------------------------------------
    # 3) rotate fall velocity into yaw frame
    # --------------------------------------------------------
    v_fall_yaw = quat_rotate_inverse(yaw_q, torch.cat(
        [v_fall_xy, torch.zeros_like(v_fall_xy[:, :1])], dim=1
    ))[:, :2]

    # --------------------------------------------------------
    # 4) angle test: subtract fall velocity only if within 90° of command direction
    # --------------------------------------------------------
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]   # (N,2)

    cmd_norm = torch.norm(cmd_xy, dim=1)
    cmd_dir = cmd_xy / (cmd_norm.unsqueeze(-1) + 1e-6)

    dot_cmd_fall = torch.sum(cmd_dir * v_fall_xy, dim=1)    # world XY basis

    # mask: fall is aiding or aligned with movement direction
    use_fall = dot_cmd_fall >= 0.0

    v_fall_yaw_used = torch.where(
        use_fall.unsqueeze(-1),
        v_fall_yaw,
        torch.zeros_like(v_fall_yaw),
    )

    # --------------------------------------------------------
    # 5) corrected velocity = actual − fall-induced
    # --------------------------------------------------------
    v_corrected = v_base_yaw - v_fall_yaw_used       # (N,2)

    # --------------------------------------------------------
    # 6) penalty: linear L1 or L2 mismatch
    # --------------------------------------------------------
    error_vec = cmd_xy - v_corrected
    lin_vel_error = torch.linalg.norm(error_vec, ord=1, dim=1)

    return -lin_vel_error
