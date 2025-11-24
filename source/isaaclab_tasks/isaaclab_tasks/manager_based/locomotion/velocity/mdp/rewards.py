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
from isaaclab.utils.math import quat_rotate, quat_rotate_inverse, yaw_quat
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


def feet_slide(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    margin: float = 0.0,
    gain: float = 1.0,
) -> torch.Tensor:
    """
    Penalize sliding feet during ground contact with a margin.

    - If foot speed <= margin → penalty = 0
    - If foot speed > margin → penalty = -gain * (speed - margin)
    """

    # Contact detection
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0] > 1.0e-3
    )  # (N, K) contact mask

    # Foot velocities (XY only)
    asset = env.scene[asset_cfg.name]
    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]   # (N, K, 2)
    foot_speed = foot_vel_xy.norm(dim=-1)                                # (N, K)

    # Margin-based excess velocity
    excess = torch.relu(foot_speed - margin)                             # (N, K)

    # Apply only when in contact
    penalty_per_foot = excess * contacts                                 # (N, K)

    # Sum across feet, apply gain, negative penalty
    penalty = -gain * torch.sum(penalty_per_foot, dim=1)                 # (N,)

    return -penalty


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
    target_height: float,
    margin: float,
    gain: float = 1.0,
) -> torch.Tensor:
    """
    Torso height penalty with margin.

    - rel_height = chest_z - min_ankle_z
    - If |rel_height - target_height| <= margin → penalty = 0
    - If above margin → penalty = -gain * (excess amount)

    Parameters
    ----------
    target_height : float
        Desired torso height above lowest ankle.
    margin : float
        Allowed deviation without penalty.
    gain : float
        Penalty gain applied to the amount exceeding the margin.
    """

    asset = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w

    chest_id = asset_cfg.body_ids[0]
    ankle_ids = asset_cfg.body_ids[1:]

    # Heights
    chest_z = body_pos_w[:, chest_id, 2]
    ankles_z = body_pos_w[:, ankle_ids, 2]
    min_ankle_z = ankles_z.min(dim=-1).values

    # Torso height above lowest ankle
    rel_height = chest_z - min_ankle_z

    # deviation
    diff = rel_height - target_height

    # margin excess (positive if outside ±margin)
    excess = torch.relu(torch.abs(diff) - margin)

    penalty = -gain * excess
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
    speed_scale: float = 0.0,
    margin: float = 0.0,   # radians
    gain: float = 1.0,
) -> torch.Tensor:
    """
    Penalize mismatch in direction between command velocity and actual velocity.

    - If angle <= margin → penalty = 0
    - If angle > margin:
        * speed_scale > 0 → penalty = -gain * (angle - margin) * |v| * speed_scale
        * speed_scale = 0 → penalty = -gain * (angle - margin)
    """

    # -----------------------------
    # 1) Command vector c = [vx, vy, yaw]
    # -----------------------------
    cmd = env.command_manager.get_command(command_name)
    c = cmd[:, :3]

    # -----------------------------
    # 2) Actual vector v
    # -----------------------------
    asset = env.scene[asset_cfg.name]

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

    cos_sim = torch.sum(c * v, dim=1) / (c_norm * v_norm)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    # angle
    angle = torch.acos(cos_sim)

    # margin handling
    excess = torch.relu(angle - margin)

    # -----------------------------
    # speed_scale による条件分岐
    # -----------------------------
    if speed_scale == 0.0:
        # 速度を使わない：角度超過のみでペナルティ
        penalty = -gain * excess
    else:
        # 従来通り：速度と掛ける
        penalty = -gain * excess * v_norm * speed_scale

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
    min_air_time: float,
    max_stance_time: float,
    min_air_height: float,
    tilt_margin: float,
    tilt_vel_scale: float,
    tilt_stance_time_scale: float,
    accel_margin: float,
    accel_vel_scale: float,
    accel_stance_time_scale: float,
    time_penalty_scale: float = 0.0,
    vel_penalty_scale: float = 0.0,
    height_penalty_scale: float = 0.0,
    body_offset_forward: float = 0.0,
) -> torch.Tensor:
    """
    Step-reflex penalty.

    0 is best, more negative is worse.

    When the robot is commanded to move or becomes tilted, this term:
      - penalizes double-stance that is too long
        (allowed stance time shrinks with tilt and forward acceleration)
      - penalizes swing air-time that is too short
      - penalizes swing height that is too low (world frame, relative to take-off)

    Assumptions:
      - asset_cfg.body_ids: [torso_id, left_foot_id, right_foot_id]
      - ContactSensor at env.scene[sensor_cfg.name] with feet order: [left_foot, right_foot]
      - torso frame: articulation root pose (torso_pos, torso_quat)
    """

    eps = 1e-6
    min_foot_up_speed = min_air_height / (min_air_time * 0.5)

    # -------------------------------------------------------
    # body IDs: [torso, left_foot, right_foot]
    # -------------------------------------------------------
    torso_id, left_id, right_id = asset_cfg.body_ids
    asset = env.scene[asset_cfg.name]
    body_pos = asset.data.body_pos_w          # (N, B, 3)
    body_quat = asset.data.body_quat_w        # (N, B, 4)
    body_vel = asset.data.body_lin_vel_w      # (N, B, 3)
    body_acc = asset.data.body_lin_acc_w      # (N, B, 3)

    torso_pos = body_pos[:, torso_id]
    torso_quat = body_quat[:, torso_id]
    torso_vel = body_vel[:, torso_id]
    torso_acc = body_acc[:, torso_id]

    left_foot_pos = body_pos[:, left_id]
    left_foot_quat = body_quat[:, left_id]
    left_foot_vel = body_vel[:, left_id]
    left_foot_acc = body_acc[:, left_id]

    right_foot_pos = body_pos[:, right_id]
    right_foot_quat = body_quat[:, right_id]
    right_foot_vel = body_vel[:, right_id]
    right_foot_acc = body_acc[:, right_id]

    # -------------------------------------------------------
    # Contact sensor data (feet: left, right)
    # -------------------------------------------------------
    sensor = env.scene[sensor_cfg.name]
    sdata = sensor.data
    left_idx, right_idx = sensor_cfg.body_ids

    left_is_air = sdata.current_air_time[:, left_idx] > 0.0
    left_air_time = sdata.current_air_time[:, left_idx]
    left_contact_time = sdata.current_contact_time[:, left_idx]
    left_air_start_z = sdata.air_start_z_w[:, left_idx]
    left_air_max_z = sdata.air_max_z_w[:, left_idx]
    left_contact_start_z = sdata.contact_start_z_w[:, left_idx]
    left_contact_min_z = sdata.contact_min_z_w[:, left_idx]

    right_is_air = sdata.current_air_time[:, right_idx] > 0.0
    right_air_time = sdata.current_air_time[:, right_idx]
    right_contact_time = sdata.current_contact_time[:, right_idx]
    right_air_start_z = sdata.air_start_z_w[:, right_idx]
    right_air_max_z = sdata.air_max_z_w[:, right_idx]
    right_contact_start_z = sdata.contact_start_z_w[:, right_idx]
    right_contact_min_z = sdata.contact_min_z_w[:, right_idx]

    # -------------------------------------------------------
    # 1) Command magnitude
    # -------------------------------------------------------
    cmd = env.command_manager.get_command(command_name)
    cmd_xy = torch.norm(cmd[:, :2], dim=1)
    cmd_yaw = torch.abs(cmd[:, 2])
    cmd_speed = torch.sqrt(cmd_xy * cmd_xy + cmd_yaw * cmd_yaw)
    need_to_step = (cmd_speed > vel_thresh)

    # --------------------------------------------------------
    # 2-1) Torso tilt (torso frame)
    #   projected_gravity_b ~ unit gravity in torso frame
    #   upright: g_b ≈ (0, 0, -1) → gx,gy ≈ 0 → tilt_torso ≈ 0
    #   tilted : gx,gy ≠ 0 → tilt_torso > 0
    # --------------------------------------------------------
    # gravity in torso frame: g_b = (gx, gy, gz), |g_b| ≈ 1
    torso_g_b = asset.data.projected_gravity_b           # (N, 3)
    # xy part in torso frame = "tilt direction & magnitude" before normalization
    g_xy = torso_g_b[:, :2]                              # (N, 2)
    # tilt magnitude = length of xy component = sin(tilt_angle)
    tilt_amount = torch.norm(g_xy, dim=1)                # (N,) in [0,1] if g_b is unit
    # unit direction of tilt in torso xy (胴体から見た傾き方向)
    #   upright → g_xy ≈ 0 → ベクトルは 0 にしておく
    tilt_dir_xy = torch.where(
        tilt_amount.unsqueeze(-1) > 0.0,
        g_xy / tilt_amount.unsqueeze(-1),
        torch.zeros_like(g_xy),
    )                                                     # (N, 2)                                                 # (N, 3)
    tilt_angle_rad = torch.asin(tilt_amount.clamp(0.0, 1.0))   # (N,)

    # --------------------------------------------------------
    # 2-2) Foot-plane tilt (torso + feet plane)
    #
    #  - v1 = torso→left, v2 = torso→right
    #  - n = v1 × v2  : foot plane normal (world)
    #  - z_world      : world up direction
    #
    #  upright (plane horizontal):
    #    n ⟂ z_world → |dot| = 0 → plane_tilt_angle_rad = 0
    #
    #  sideways (plane vertical):
    #    n || z_world → |dot| = 1 → plane_tilt_angle_rad = π/2
    #
    #  Outputs:
    #    plane_tilt_angle_rad : amount of tilt [rad]
    #    plane_tilt_dir_xy    : direction of tilt in torso XY (unit vector)
    # --------------------------------------------------------
    p_t = torso_pos  # (N,3)

    if body_offset_forward != 0.0:
        offset_local = torch.tensor(
            [body_offset_forward, 0.0, 0.0],
            device=torso_pos.device,
            dtype=torso_pos.dtype,
        ).view(1, 3).expand_as(p_t)
        offset_world = quat_rotate(torso_quat, offset_local)
        p_t = p_t + offset_world

    # feet positions
    p_l = body_pos[:, left_id]   # (N,3)
    p_r = body_pos[:, right_id]  # (N,3)

    # two edges on the foot plane
    v1 = p_l - p_t
    v2 = p_r - p_t

    # plane normal in world frame
    plane_n = torch.cross(v1, v2, dim=1)                        # (N,3)
    plane_n_unit = plane_n / (torch.norm(plane_n, dim=1, keepdim=True) + 1e-9)

    # world up
    z_world = torch.tensor(
        [0.0, 0.0, 1.0],
        device=torso_pos.device,
        dtype=torso_pos.dtype,
    ).view(1, 3)

    # --------------------------------------------------------
    # 傾き量: 足平面の法線と world z の「ずれ角」 [0, π/2]
    #  upright:  dot = 0  → angle = 0
    #  sideways: |dot|=1 → angle = π/2
    # --------------------------------------------------------
    dot_nz = torch.sum(plane_n_unit * z_world, dim=1).clamp(-1.0, 1.0)   # (N,)
    plane_tilt_angle_rad = torch.asin(torch.abs(dot_nz))                 # (N,)

    # --------------------------------------------------------
    # 傾き方向: 足平面が「どの横軸まわりに傾いているか」
    #
    # world で:
    #   tilt_axis_world = z_world × n   (回転軸は水平)
    #
    # torso から見た方向がほしいので:
    #   tilt_axis_torso = R(torso)^T * tilt_axis_world
    #   → その XY 成分を正規化
    # --------------------------------------------------------
    z_world_vec = z_world.expand_as(plane_n_unit)          # (N,3)
    tilt_axis_world = torch.cross(z_world_vec, plane_n_unit, dim=1)  # (N,3)

    # 軸ベクトルを torso frame へ
    tilt_axis_torso = quat_rotate_inverse(torso_quat, tilt_axis_world)    # (N,3)
    axis_xy = tilt_axis_torso[:, :2]                                      # (N,2)
    axis_xy_norm = torch.norm(axis_xy, dim=1)                             # (N,)

    # unit direction in torso XY (傾いている向き)
    plane_tilt_dir_xy = torch.where(
        axis_xy_norm.unsqueeze(-1) > 0.0,
        axis_xy / axis_xy_norm.unsqueeze(-1),
        torch.zeros_like(axis_xy),
    )  # (N,2)

    # --------------------------------------------------------
    # 2-3) Combine tilts and decide when tilt matters
    # --------------------------------------------------------
    torso_tilt_xy = tilt_angle_rad.unsqueeze(-1) * tilt_dir_xy
    plane_tilt_xy = plane_tilt_angle_rad.unsqueeze(-1) * plane_tilt_dir_xy
    combined_tilt_x = torch.where(
        torch.abs(torso_tilt_xy[:, 0]) >= torch.abs(plane_tilt_xy[:, 0]),
        torso_tilt_xy[:, 0],
        plane_tilt_xy[:, 0],
    )
    combined_tilt_y = torch.where(
        torch.abs(torso_tilt_xy[:, 1]) >= torch.abs(plane_tilt_xy[:, 1]),
        torso_tilt_xy[:, 1],
        plane_tilt_xy[:, 1],
    )
    combined_tilt_xy = torch.stack([combined_tilt_x, combined_tilt_y], dim=1)
    combined_tilt_norm = torch.norm(combined_tilt_xy, dim=1)
    tilt_scaled_vel = tilt_vel_scale * combined_tilt_xy

    # 傾きに応じて stance time を縮める
    excess_tilt = torch.clamp(combined_tilt_norm - tilt_margin, min=0.0)

    #   combined_tilt = tilt_margin → scale = 1
    #   combined_tilt 増えるほど scale ↓ （tilt_stance_softness で調整）
    tilt_scale_factor = 1.0 - tilt_stance_time_scale * excess_tilt
    tilt_scale_factor = torch.clamp(tilt_scale_factor, 0.0, 1.0)

    # 一定以上傾いたら「ステップすべき」
    need_to_step = need_to_step | (combined_tilt_norm > tilt_margin)

    # --------------------------------------------------------
    # 3) Acceleration-based stance shrink (torso-local)
    #
    # accel_margin:            許容したい前向き加速度 [m/s^2] のしきい値
    # accel_vel_scale:         加速度に応じた速度補正ベクトルのスケール
    # accel_stance_time_scale: 加速度超過に応じて stance time をどれだけ縮めるかの係数
    #
    # 出力:
    #   accel_amount        : 前向き加速度の大きさ (>=0)
    #   accel_dir_xy        : 前向き方向（torso xy, unit）
    #   accel_scaled_vel    : 速度補正ベクトル (N,2)
    #   accel_scale_factor  : stance time のスケール (0〜1)
    #   need_to_step        : 強い前向き加速時はステップが必要
    # --------------------------------------------------------
    # world → torso local
    torso_vel_local = quat_rotate_inverse(torso_quat, torso_vel)  # (N,3)
    torso_acc_local = quat_rotate_inverse(torso_quat, torso_acc)  # (N,3)

    v_xy = torso_vel_local[:, :2]   # (N,2) torso frame
    a_xy = torso_acc_local[:, :2]   # (N,2) torso frame

    v_norm = torch.norm(v_xy, dim=1)            # (N,)
    has_vel = v_norm > 1.0e-6

    # forward direction in torso XY (unit)
    v_dir_xy = torch.where(
        has_vel.unsqueeze(-1),
        v_xy / v_norm.unsqueeze(-1),
        torch.zeros_like(v_xy),
    )                                           # (N,2)

    # forward acceleration component (along torso forward direction)
    a_forward = torch.sum(v_dir_xy * a_xy, dim=1)   # (N,)
    accel_amount = torch.clamp(a_forward, min=0.0)  # forward accel only (>=0)

    # direction of acceleration effect = torso forward direction
    accel_dir_xy = v_dir_xy                         # (N,2)

    # velocity-scale vector based on acceleration
    accel_scaled_vel = accel_vel_scale * accel_amount.unsqueeze(-1) * accel_dir_xy  # (N,2)

    # stance time allowed shrinks when forward acceleration exceeds accel_margin
    excess_accel = torch.clamp(accel_amount - accel_margin, min=0.0)  # (N,)

    # accel_amount <= accel_margin  → accel_scale_factor = 1
    # accel_amount  > accel_margin  → 1 - accel_stance_time_scale * (超過分)
    accel_scale_factor = 1.0 - accel_stance_time_scale * excess_accel
    accel_scale_factor = torch.clamp(accel_scale_factor, 0.0, 1.0)    # (N,)

    # 強く前向きに加速しているときは「ステップすべき」とみなす
    need_to_step = need_to_step | (accel_amount > accel_margin)

    # --------------------------------------------------------
    # tilt + acceleration で stance time を決定
    # --------------------------------------------------------
    stance_scale = torch.minimum(tilt_scale_factor, accel_scale_factor)   # (N,)
    effective_max_stance_time = max_stance_time * stance_scale            # (N,)

    # -------------------------------------------------------
    # 4) Stance-time penalty
    # -------------------------------------------------------
    double_is_contact = ~left_is_air & ~right_is_air
    double_contact_time = torch.where(
        double_is_contact,
        torch.minimum(left_contact_time, right_contact_time),
        torch.zeros_like(left_contact_time),
    )  # (N,)
    stance_time_excess = torch.clamp(double_contact_time - effective_max_stance_time, min=0.0)
    stance_time_penalty = -stance_time_excess * time_penalty_scale

    # -------------------------------------------------------
    # 5) Swing time penalty
    #
    # swing_air_time < min_air_time * 0.5 のとき：
    #   浮いている足は
    #     - torso ローカル z 方向
    #     - world z 方向
    #   の両方で min_foot_up_speed 以上で上昇していなければならない。
    # min_air_time * 0.5 < swing_air_time のとき：
    #   浮いている足が(air_max_z - air_start_z) < min_air_heightのとき
    #     - torso ローカル z 方向
    #     - world z 方向
    #   の両方で min_foot_up_speed 以上で上昇していなければならない
    #
    # 両足浮きの場合：
    #   torso ローカル z が高い方の足のみを対象にする。
    #
    # 不足速度（local/world のうち大きい方）に対して
    # vel_penalty_scale をかけてペナルティとする。
    # -------------------------------------------------------

    half_air_time = min_air_time * 0.5

    # torso-local foot velocities
    left_foot_vel_local  = quat_rotate_inverse(torso_quat, left_foot_vel)   # (N,3)
    right_foot_vel_local = quat_rotate_inverse(torso_quat, right_foot_vel)

    left_up_speed_local  = left_foot_vel_local[:, 2]
    right_up_speed_local = right_foot_vel_local[:, 2]

    # world-frame z velocities
    left_up_speed_world  = left_foot_vel[:, 2]
    right_up_speed_world = right_foot_vel[:, 2]

    # air-time conditions
    left_short_air  = left_air_time  < half_air_time
    right_short_air = right_air_time < half_air_time

    # total lift since take-off (world)
    left_lift_total  = left_air_max_z  - left_air_start_z
    right_lift_total = right_air_max_z - right_air_start_z

    # long-air but still insufficient lift
    left_long_need  = (left_air_time  >= half_air_time) & (left_lift_total  < min_air_height)
    right_long_need = (right_air_time >= half_air_time) & (right_lift_total < min_air_height)

    # air-state flags
    # （上で left_is_air, right_is_air は定義済み）
    left_only_air  = left_is_air  & (~right_is_air)
    right_only_air = right_is_air & (~left_is_air)
    both_air = left_is_air & right_is_air

    # torso-local foot heights (for selecting higher one in both-air)
    left_height_local  = quat_rotate_inverse(torso_quat, left_foot_pos - torso_pos)[:, 2]
    right_height_local = quat_rotate_inverse(torso_quat, right_foot_pos - torso_pos)[:, 2]
    left_higher = left_height_local >= right_height_local

    # initialize penalty container
    swing_penalty = torch.zeros_like(left_air_time)

    # ---------------------------------------------------
    # 左足のみ浮いている場合
    # ---------------------------------------------------
    left_need_speed = left_only_air & (left_short_air | left_long_need)

    left_def_local  = torch.clamp(min_foot_up_speed - left_up_speed_local,  min=0.0)
    left_def_world  = torch.clamp(min_foot_up_speed - left_up_speed_world,  min=0.0)
    left_speed_deficit = torch.maximum(left_def_local, left_def_world)

    swing_penalty = swing_penalty + left_need_speed * left_speed_deficit * (-vel_penalty_scale)

    # ---------------------------------------------------
    # 右足のみ浮いている場合
    # ---------------------------------------------------
    right_need_speed = right_only_air & (right_short_air | right_long_need)

    right_def_local  = torch.clamp(min_foot_up_speed - right_up_speed_local,  min=0.0)
    right_def_world  = torch.clamp(min_foot_up_speed - right_up_speed_world,  min=0.0)
    right_speed_deficit = torch.maximum(right_def_local, right_def_world)

    swing_penalty = swing_penalty + right_need_speed * right_speed_deficit * (-vel_penalty_scale)

    # ---------------------------------------------------
    # 両足浮いている場合 → torso ローカル z が高い方のみ
    # ---------------------------------------------------
    both_need_speed = both_air & (
        (left_short_air | left_long_need) | (right_short_air | right_long_need)
    )

    # 左足が高い & 左条件が true
    left_higher_mask = both_need_speed & left_higher & (left_short_air | left_long_need)
    left_def_local_both  = torch.clamp(min_foot_up_speed - left_up_speed_local,  min=0.0)
    left_def_world_both  = torch.clamp(min_foot_up_speed - left_up_speed_world,  min=0.0)
    left_speed_deficit_both = torch.maximum(left_def_local_both, left_def_world_both)
    swing_penalty = swing_penalty + left_higher_mask * left_speed_deficit_both * (-vel_penalty_scale)

    # 右足が高い & 右条件が true
    right_higher_mask = both_need_speed & (~left_higher) & (right_short_air | right_long_need)
    right_def_local_both  = torch.clamp(min_foot_up_speed - right_up_speed_local,  min=0.0)
    right_def_world_both  = torch.clamp(min_foot_up_speed - right_up_speed_world,  min=0.0)
    right_speed_deficit_both = torch.maximum(right_def_local_both, right_def_world_both)
    swing_penalty = swing_penalty + right_higher_mask * right_speed_deficit_both * (-vel_penalty_scale)

    # 両足接地中は swing penalty を 0 にする
    air_time_penalty = swing_penalty * (~double_is_contact)

    # -------------------------------------------------------
    # 6) Swing height penalty
    # swing_air_time < min_air_time * 0.5 のとき：
    #   浮いている足は
    #     - torso ローカル z 方向
    #     - world z 方向
    #   の両方で min_air_height * (swing_air_time / (min_air_time * 0.5)) 以上の高さを維持しなければならない。
    # min_air_time * 0.5 < swing_air_time < min_air_timeのとき：
    #   浮いている足は
    #     - torso ローカル z 方向
    #     - world z 方向
    #   の両方で min_air_height * ((min_air_time - swing_air_time) / (min_air_time * 0.5)) 以上の高さを維持しなければならない。
    #
    # 両足浮きの場合：
    #   torso ローカル z が高い方の足のみを対象にする。
    #
    # 不足高度（local/world のうち大きい方）に対して
    # height_penalty_scale をかけてペナルティとする。
    # -------------------------------------------------------

    # world-frame absolute heights
    hL_w = left_foot_pos[:, 2]
    hR_w = right_foot_pos[:, 2]

    # torso-local absolute heights
    left_local_pos  = quat_rotate_inverse(torso_quat, left_foot_pos - torso_pos)
    right_local_pos = quat_rotate_inverse(torso_quat, right_foot_pos - torso_pos)
    hL_l = left_local_pos[:, 2]
    hR_l = right_local_pos[:, 2]

    # foot lift height = high - low
    world_lift_height = torch.abs(hL_w - hR_w)
    local_lift_height = torch.abs(hL_l - hR_l)

    # actual lift height = smaller of world/local
    lift_height = torch.minimum(world_lift_height, local_lift_height)  # (N,)

    # same lift height is used for whichever foot is considered “swing”
    left_rel_height  = lift_height
    right_rel_height = lift_height

    # required height profile (triangle 0 → min → 0)
    left_required_height  = torch.zeros_like(left_air_time)
    right_required_height = torch.zeros_like(right_air_time)

    left_phase0  = (left_air_time  > 0.0) & (left_air_time  < half_air_time)
    left_phase1  = (left_air_time  >= half_air_time) & (left_air_time  < min_air_time)

    right_phase0 = (right_air_time > 0.0) & (right_air_time < half_air_time)
    right_phase1 = (right_air_time >= half_air_time) & (right_air_time < min_air_time)

    # rising
    left_required_height[left_phase0] = (
        min_air_height * (left_air_time[left_phase0] / half_air_time)
    )
    right_required_height[right_phase0] = (
        min_air_height * (right_air_time[right_phase0] / half_air_time)
    )

    # falling
    left_required_height[left_phase1] = (
        min_air_height * ((min_air_time - left_air_time[left_phase1]) / half_air_time)
    )
    right_required_height[right_phase1] = (
        min_air_height * ((min_air_time - right_air_time[right_phase1]) / half_air_time)
    )

    # deficits
    left_height_deficit  = torch.clamp(left_required_height  - left_rel_height,  min=0.0)
    right_height_deficit = torch.clamp(right_required_height - right_rel_height, min=0.0)

    height_penalty = torch.zeros_like(left_air_time)

    # 片足だけ浮いている場合
    # left_only_air, right_only_air は 5) で定義済み
    left_need_height  = left_only_air  & (left_required_height  > 0.0)
    right_need_height = right_only_air & (right_required_height > 0.0)

    height_penalty = height_penalty + left_need_height  * left_height_deficit  * (-height_penalty_scale)
    height_penalty = height_penalty + right_need_height * right_height_deficit * (-height_penalty_scale)

    # 両足浮き → 5で決めた「高い方の足」だけを見る
    both_need_height = both_air & (
        (left_required_height > 0.0) | (right_required_height > 0.0)
    )

    left_higher_mask  = both_need_height & left_higher
    right_higher_mask = both_need_height & (~left_higher)

    height_penalty = height_penalty + left_higher_mask  * left_height_deficit  * (-height_penalty_scale)
    height_penalty = height_penalty + right_higher_mask * right_height_deficit * (-height_penalty_scale)

    air_height_penalty = height_penalty * (~double_is_contact)
    # -------------------------------------------------------
    # 7) Total
    # -------------------------------------------------------
    total_penalty = (
        stance_time_penalty
        + air_time_penalty
        + air_height_penalty
    )  # (N,)

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


def track_lin_vel_xy_compensated_penalty(
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

def drive_forward_foot_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    vel_thresh: float = 0.05,
    fall_gain: float = 1.0,
    stance_speed_gain: float = 1.0,
    swing_speed_gain: float = 1.0,
    comp_gain: float = 1.0,
) -> torch.Tensor:
    """
    Forward-drive penalty.

    0 is best, more negative is worse.

    前提:
      - command_name で与えられる XY コマンドは torso(body_link) ローカル座標系
      - asset_cfg.body_ids = [torso_id, left_foot_id, right_foot_id]
      - sensor_cfg.body_ids = [body_link, ankle_l, ankle_r]

    仕様(概要):
      - 目標並進速度 v_des_local = cmd_local + fall_gain * v_fall_local
      - torso の実速度が v_des_local に足りないときだけペナルティ発動
      - 接地足:
          * 片足接地: その足の前向き地面反力が不足していればペナルティ
          * 両足接地: 両足の合計前向き地面反力が不足していればペナルティ
      - 浮足:
          * 片足接地時のみ swing として扱い、
            command 速度 + speed_missing * swing_speed_gain 以上の前進速度を要求
      - 補償(位置):
          * stance foot が torso ローカルで「進行方向の反対側」にあるとき、
            swing foot は stance_local を進行方向に対して 180 度反転した位置
            (前方向成分で -stance_proj * comp_gain) より前に出すよう位置ペナルティ。
          * 両足接地中は補償も行わない。
    """

    eps = 1e-6
    asset = env.scene[asset_cfg.name]

    # -------------------------------------------------------
    # asset_cfg.body_ids = [torso, left_foot, right_foot]
    # -------------------------------------------------------
    torso_id, left_id, right_id = asset_cfg.body_ids

    body_pos = asset.data.body_pos_w
    body_vel = asset.data.body_lin_vel_w
    body_angvel = asset.data.body_ang_vel_w
    body_quat = asset.data.body_quat_w

    N = body_pos.shape[0]
    device = body_pos.device
    idx = torch.arange(N, device=device)

    # torso
    torso_pos = body_pos[:, torso_id]
    torso_vel_w = body_vel[:, torso_id]
    torso_ang_w = body_angvel[:, torso_id]
    torso_quat = body_quat[:, torso_id]

    # torso vel local
    torso_vel_local = quat_rotate_inverse(torso_quat, torso_vel_w)[:, :2]

    # foot pos
    left_pos = body_pos[:, left_id]
    right_pos = body_pos[:, right_id]
    left_z = left_pos[:, 2]
    right_z = right_pos[:, 2]

    # -------------------------------------------------------
    # 0) Command and fall-induced velocity
    # -------------------------------------------------------
    cmd_full = env.command_manager.get_command(command_name)
    cmd_local = cmd_full[:, :2]
    cmd_speed = torch.norm(cmd_local, dim=1)

    stance_is_left = left_z <= right_z
    stance_pos_w = torch.where(
        stance_is_left.unsqueeze(-1), left_pos, right_pos
    )

    r_w = torso_pos - stance_pos_w
    v_fall_w = torch.cross(torso_ang_w, r_w, dim=1)

    v_fall_local = quat_rotate_inverse(torso_quat, v_fall_w)[:, :2]

    v_des_local = cmd_local + fall_gain * v_fall_local
    des_speed_local = torch.norm(v_des_local, dim=1)

    des_dir_local = torch.zeros_like(v_des_local)
    valid_des = des_speed_local > 1e-4
    des_dir_local[valid_des] = v_des_local[valid_des] / (
        des_speed_local[valid_des].unsqueeze(-1) + eps
    )

    # -------------------------------------------------------
    # 1) torso forward deficiency
    # -------------------------------------------------------
    torso_forward_local = torch.sum(torso_vel_local * des_dir_local, dim=1)
    speed_missing = torch.relu(des_speed_local - torso_forward_local)

    need_drive = (des_speed_local > vel_thresh) & (speed_missing > 0.0)

    # -------------------------------------------------------
    # 2) Contact forces
    # -------------------------------------------------------
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_all = sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, :]

    F_left = forces_all[:, 1]
    F_right = forces_all[:, 2]

    left_contact = torch.norm(F_left, dim=1) > 1e-2
    right_contact = torch.norm(F_right, dim=1) > 1e-2

    both_contact = left_contact & right_contact
    left_only = left_contact & (~right_contact)
    right_only = right_contact & (~left_contact)
    both_in_air = (~left_contact) & (~right_contact)

    stance_sensor_idx = torch.where(
        stance_is_left, torch.full_like(idx, 1), torch.full_like(idx, 2)
    )
    F_stance = forces_all[idx, stance_sensor_idx]
    F_stance_xy = F_stance[:, :2]
    F_total_xy = F_left[:, :2] + F_right[:, :2]

    # -------------------------------------------------------
    # 2a) desired direction (world)
    # -------------------------------------------------------
    des_dir_local_3 = torch.zeros((N, 3), device=device)
    des_dir_local_3[:, :2] = des_dir_local
    des_dir_world_3 = quat_rotate(torso_quat, des_dir_local_3)
    des_dir_world_xy = des_dir_world_3[:, :2]

    des_dir_world_norm = torch.norm(des_dir_world_xy, dim=1)
    des_dir_world = torch.zeros_like(des_dir_world_xy)
    valid_world_dir = des_dir_world_norm > 1e-4
    des_dir_world[valid_world_dir] = des_dir_world_xy[valid_world_dir] / (
        des_dir_world_norm[valid_world_dir].unsqueeze(-1) + eps
    )

    # -------------------------------------------------------
    # 3) stance force penalty
    # -------------------------------------------------------
    required_forward = stance_speed_gain * speed_missing

    F_stance_forward = torch.sum(F_stance_xy * des_dir_world, dim=1)
    F_total_forward = torch.sum(F_total_xy * des_dir_world, dim=1)

    stance_force_missing_single = torch.relu(required_forward - F_stance_forward)
    stance_force_missing_both = torch.relu(required_forward - F_total_forward)

    stance_penalty = torch.zeros_like(speed_missing)
    stance_penalty = torch.where(
        left_only | right_only,
        -stance_force_missing_single,
        stance_penalty,
    )
    stance_penalty = torch.where(
        both_contact,
        -stance_force_missing_both,
        stance_penalty,
    )

    # -------------------------------------------------------
    # 4) swing foot penalty（修正版：両足浮き → 両足とも swing）
    # -------------------------------------------------------
    # 各足の local velocity
    left_vel_local = quat_rotate_inverse(torso_quat, body_vel[:, left_id])
    right_vel_local = quat_rotate_inverse(torso_quat, body_vel[:, right_id])

    left_forward_local = torch.sum(left_vel_local[:, :2] * des_dir_local, dim=1)
    right_forward_local = torch.sum(right_vel_local[:, :2] * des_dir_local, dim=1)

    required_swing_forward = cmd_speed + swing_speed_gain * speed_missing

    left_missing = torch.relu(required_swing_forward - left_forward_local)
    right_missing = torch.relu(required_swing_forward - right_forward_local)

    swing_missing = torch.zeros_like(speed_missing)

    # 片足接地 → 非接地側だけ
    swing_missing = torch.where(
        left_only,  # left stance → right swing
        right_missing,
        swing_missing,
    )
    swing_missing = torch.where(
        right_only,  # right stance → left swing
        left_missing,
        swing_missing,
    )

    # 両足浮き → 両足 swing → より悪い方を採用
    both_missing = torch.max(left_missing, right_missing)
    swing_missing = torch.where(
        both_in_air,
        both_missing,
        swing_missing,
    )

    # swing active 条件：片足接地 or 両足浮き / 両足接地では無効
    active_swing = left_only | right_only

    swing_penalty = torch.where(
        active_swing | both_in_air,
        -swing_missing,
        torch.zeros_like(swing_missing),
    )

    # -------------------------------------------------------
    # 5) compensatory penalty (位置ベース)
    #    stance foot が torso ローカルで進行方向の反対側にある場合、
    #    swing foot は stance_local を進行方向に対して 180 度反転した位置
    #    (前方向成分で -stance_proj * comp_gain) より前に出す。
    # -------------------------------------------------------
    compensatory_penalty = torch.zeros_like(speed_missing)

    stance_world = torch.where(
        stance_is_left.unsqueeze(-1),
        left_pos,
        right_pos,
    )
    swing_world = torch.where(
        (~stance_is_left).unsqueeze(-1),
        left_pos,
        right_pos,
    )

    stance_local = quat_rotate_inverse(torso_quat, stance_world - torso_pos)
    swing_local = quat_rotate_inverse(torso_quat, swing_world - torso_pos)

    stance_proj = torch.sum(stance_local[:, :2] * des_dir_local, dim=1)
    swing_proj = torch.sum(swing_local[:, :2] * des_dir_local, dim=1)

    stance_on_opposite_side = stance_proj < 0.0
    required_swing_proj = -stance_proj * comp_gain

    swing_proj_missing = torch.relu(required_swing_proj - swing_proj)

    compensatory_penalty = torch.where(
        active_swing & stance_on_opposite_side & valid_des,
        -swing_proj_missing,
        compensatory_penalty,
    )

    # -------------------------------------------------------
    # 6) 合成
    # -------------------------------------------------------
    total = (
        stance_penalty
        + swing_penalty
        + compensatory_penalty
    )

    total = torch.where(need_drive, total, torch.zeros_like(total))
    return total

def support_plane_tilt_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    tilt_margin: float,
    k: float = 1.0,
    body_offset_forward: float = 0.0,
) -> torch.Tensor:
    """
    Penalize the tilt of the support plane (torso + both feet) in radians.

    - asset_cfg.body_ids = [torso_id, left_foot_id, right_foot_id]
    - tilt_margin : allowable tilt angle [rad]
        (tilt_angle_rad <= tilt_margin → penalty = 0)
    - k           : proportional scale of penalty (k > 0).
        * k == 0 のときだけ、threshold を超えたら -1、超えなければ 0 の二値ペナルティ。
    """

    eps = 1e-6
    asset = env.scene[asset_cfg.name]

    body_pos  = asset.data.body_pos_w
    body_quat = asset.data.body_quat_w

    torso_id, left_id, right_id = asset_cfg.body_ids

    N      = body_pos.shape[0]
    device = body_pos.device
    dtype  = body_pos.dtype

    torso_pos  = body_pos[:, torso_id]
    torso_quat = body_quat[:, torso_id]

    # --------------------------------------------------------
    # torso point with optional forward offset
    # --------------------------------------------------------
    if body_offset_forward != 0.0:
        offset_local = torch.tensor(
            [body_offset_forward, 0.0, 0.0],
            device=device,
            dtype=dtype,
        ).view(1, 3).expand(N, 3)
        offset_world = quat_rotate(torso_quat, offset_local)
        p_t = torso_pos + offset_world
    else:
        p_t = torso_pos

    p_l = body_pos[:, left_id]
    p_r = body_pos[:, right_id]

    # --------------------------------------------------------
    # n: support plane 内の「前後軸」ベクトル
    #  upright: n はほぼ水平で Z と直交
    #  sideways: n が Z 方向に近づく
    # --------------------------------------------------------
    v1 = p_l - p_t
    v2 = p_r - p_t
    n = torch.cross(v1, v2, dim=1)
    n_unit = n / (torch.norm(n, dim=1, keepdim=True) + eps)

    # world Z
    z_world = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 3)

    # --------------------------------------------------------
    # tilt angle [rad]
    #
    #  cos_theta = dot(n_unit, z_world)
    #  upright:   n は水平 → cos ≈ 0      → angle ≈ 0
    #  sideways: |cos|→1     → angle→π/2
    #
    #  tilt_angle_rad = asin(|cos_theta|)
    # --------------------------------------------------------
    cos_theta = torch.sum(n_unit * z_world, dim=1).clamp(-1.0, 1.0)
    tilt_angle_rad = torch.asin(torch.abs(cos_theta))  # (N,) in [0, π/2]

    # --------------------------------------------------------
    # tilt_margin 以内はペナルティ 0
    # tilt_excess = max(tilt_angle_rad - tilt_margin, 0)
    # --------------------------------------------------------
    tilt_excess = torch.clamp(tilt_angle_rad - tilt_margin, min=0.0)

    if k == 0.0:
        # 二値ペナルティ: 超えたら -1, それ以外 0
        return torch.where(
            tilt_excess > 0.0,
            -torch.ones_like(tilt_excess),
            torch.zeros_like(tilt_excess),
        )

    # 比例ペナルティ: -k * (角度超過)
    penalty = -k * tilt_excess
    return penalty
