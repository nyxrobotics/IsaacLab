# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

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
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
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


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)

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



def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.stack((q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]), dim=-1)

def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )

def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)
    return _quat_mul(_quat_mul(q, qv), _quat_conjugate(q))[..., 1:]

def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _quat_rotate(_quat_conjugate(q), v)

def local_torso_height_penalty(
    env,
    asset_cfg,
    contact_sensor_cfg,
    target_height: float,
    margin: float,
    gain: float = 1.0,
    contact_force_threshold: float = 0.01,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    body_pos_w = asset.data.body_pos_w
    body_quat_w = asset.data.body_quat_w

    chest_id = asset_cfg.body_ids[0]
    ankle_ids = asset_cfg.body_ids[1:]

    chest_pos_w = body_pos_w[:, chest_id, :]
    chest_quat_w = body_quat_w[:, chest_id, :]
    ankles_pos_w = body_pos_w[:, ankle_ids, :]

    rel_vec_w = ankles_pos_w - chest_pos_w.unsqueeze(1)
    chest_quat_w_a = chest_quat_w.unsqueeze(1).expand(-1, rel_vec_w.shape[1], -1)
    rel_vec_b = _quat_rotate_inverse(chest_quat_w_a, rel_vec_w)

    foot_depths = -rel_vec_b[..., 2]  # (N, A) positive along torso -Z

    # Contact detection
    contact_sensor = env.scene.sensors[contact_sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids, :]  # (N, H, A, 3)
    in_contact = forces.norm(dim=-1).max(dim=1)[0] > contact_force_threshold                 # (N, A)

    num_contact = in_contact.sum(dim=-1)  # (N,)
    any_contact = num_contact > 0

    # If contact exists: use "highest foot" among contacting feet => min depth
    contact_depths = torch.where(in_contact, foot_depths, torch.full_like(foot_depths, 1e9))
    contact_depth = contact_depths.min(dim=-1).values  # (N,)

    # If flight: use lower foot => max depth
    flight_depth = foot_depths.max(dim=-1).values      # (N,)

    foot_depth = torch.where(any_contact, contact_depth, flight_depth)

    diff = foot_depth - target_height
    excess = torch.relu(torch.abs(diff) - margin)
    return -gain * excess



def feet_slide_with_yaw(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_weight: float = 1.0,
    yaw_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize feet sliding (XY) and twisting (yaw) while in contact.

    接地判定:
      - current_air_time < 1e-6
      - force_norm > 1e-6

    返り値は「スリップ量のコスト（正の値）」なので、
    報酬定義側では weight を負にして使う想定。
    """

    # --------------------------------------------------------
    # 1) Contact forces & contact state (same logic as feet_contact_angle_penalty)
    # --------------------------------------------------------
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # forces_w_history: (N, history, S, 3)
    forces_w_last = contact_sensor.data.net_forces_w_history[:, -1, :, :]  # (N, S, 3)

    # foot sensor indices (例: [left, right])
    sensor_ids = sensor_cfg.body_ids

    # 対象足の力ベクトルのみ取り出し: (N, F, 3)
    forces_feet = forces_w_last[:, sensor_ids, :]

    # 力のノルム: (N, F)
    force_norm = torch.norm(forces_feet, dim=-1)

    # air-time: (N, S) から対象足だけ取り出す → (N, F)
    air_time = contact_sensor.data.current_air_time[:, sensor_ids]

    # 空中でない（地面にいる） & 力がある
    grounded_by_air = air_time < 1e-6
    grounded_by_force = force_norm > 1e-6

    # 接地判定
    in_contact = grounded_by_air

    # --------------------------------------------------------
    # 2) Foot linear & angular velocities
    # --------------------------------------------------------
    asset = env.scene[asset_cfg.name]

    # body indices for the same feet (asset_cfg.body_ids と sensor_cfg.body_ids が対応している前提)
    body_ids = asset_cfg.body_ids  # (F,)

    # 線形速度（world）: (N, B, 3) → 対象足 (N, F, 3)
    body_lin_vel_w = asset.data.body_lin_vel_w[:, body_ids, :]  # (N, F, 3)
    lin_xy = body_lin_vel_w[..., :2]                            # (N, F, 2)
    lin_speed = lin_xy.norm(dim=-1)                             # (N, F)

    # 角速度（world）: (N, B, 3) → 対象足
    body_ang_vel_w = asset.data.body_ang_vel_w[:, body_ids, :]  # (N, F, 3)
    yaw_rate = torch.abs(body_ang_vel_w[..., 2])                # (N, F)  z軸まわり

    # --------------------------------------------------------
    # 3) Sliding + twisting cost (only when in contact)
    # --------------------------------------------------------
    # 基本コスト（接地・非接地に関係なくまず計算）
    slip_cost_per_foot = lin_weight * lin_speed + yaw_weight * yaw_rate  # (N, F)

    # 接地中だけカウント（bool → float に変換してマスク）
    slip_cost_per_foot = slip_cost_per_foot * in_contact.float()

    # 足ごとに合計 → (N,)
    penalty = torch.sum(slip_cost_per_foot, dim=1)

    return penalty



def feet_air_time_balanced_alternating_biped(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    *,
    # command scaling (linear, includes yaw)
    v_max: float = 1.0,          # m/s where linear scale reaches 1 for XY command
    yaw_max: float = 1.0,        # rad/s where linear scale reaches 1 for yaw command

    # swing constraints
    hold_min_air: float = 0.15,  # must stay in air at least this long to get swing credit
    hold_max_air: float = 0.45,  # if exceeded, apply "too long in air" penalty

    # anti-tap (air/contact)
    tap_air_threshold: float = 0.10,
    tap_air_penalty: float = 0.5,
    tap_contact_threshold: float = 0.10,
    tap_contact_penalty: float = 0.5,

    # alternation
    alternation_bonus: float = 0.25,

    # symmetry (EMA balance)
    ema_alpha: float = 0.02,
    balance_weight: float = 0.5,

    # avoid double flight
    double_flight_penalty: float = 1.0,

    # over-hold penalty (prevents "keep one foot up forever")
    overhold_penalty: float = 0.3,
    overhold_margin: float = 0.05,  # soft margin for smooth penalty ramp

    # command composition
    cmd_use_max: bool = True,     # True: max(lin,yaw). False: weighted sum
    w_lin: float = 1.0,           # used when cmd_use_max=False
    w_yaw: float = 1.0,           # used when cmd_use_max=False

    # command indices (typical: [vx, vy, yaw_rate])
    yaw_index: int = 2,
) -> torch.Tensor:
    """
    Balanced & alternating biped step reward.

    Key behavior:
      - Swing credit becomes ON after hold_min_air and does NOT increase with longer air time.
      - hold_max_air is used ONLY to penalize over-holding a foot in the air.
      - Double stance is allowed; double flight is penalized.
      - Brief air/contact taps are penalized.
      - Alternation is rewarded.
      - Long-horizon left/right asymmetry is penalized via EMA statistics.
      - Reward is linearly scaled by commanded motion magnitude (XY and yaw).
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_ids = sensor_cfg.body_ids
    assert len(foot_ids) == 2, "This reward is for bipeds (2 feet)."

    air_time = contact_sensor.data.current_air_time[:, foot_ids]          # (N,2)
    contact_time = contact_sensor.data.current_contact_time[:, foot_ids]  # (N,2)
    in_contact = contact_time > 0.0                                       # (N,2)

    n_env = in_contact.shape[0]
    device = in_contact.device

    # --- persistent buffer ---
    key = "_feet_balanced_alt_reward_buf"
    if not hasattr(env, key):
        buf = {}
        buf["prev_in_contact"] = in_contact.clone()
        buf["last_completed_air"] = torch.zeros((n_env, 2), device=device)
        buf["ema_air"] = torch.zeros((n_env, 2), device=device)
        buf["ema_contact"] = torch.zeros((n_env, 2), device=device)
        buf["last_swing_foot"] = torch.full((n_env,), -1, device=device, dtype=torch.long)
        setattr(env, key, buf)
    buf = getattr(env, key)

    prev_in_contact = buf["prev_in_contact"]
    last_completed_air = buf["last_completed_air"]
    ema_air = buf["ema_air"]
    ema_contact = buf["ema_contact"]
    last_swing_foot = buf["last_swing_foot"]

    # --- transitions ---
    lift_off = prev_in_contact & (~in_contact)    # contact -> air
    touch_down = (~prev_in_contact) & in_contact  # air -> contact

    # record completed swing time at touchdown (used for EMA)
    last_completed_air = torch.where(touch_down, air_time, last_completed_air)

    # --- linear command scaling (XY + yaw) ---
    cmd = env.command_manager.get_command(command_name)
    lin = torch.norm(cmd[:, :2], dim=1)  # |v_xy|

    if cmd.shape[1] > yaw_index:
        yaw = torch.abs(cmd[:, yaw_index])  # |omega_z|
    else:
        yaw = torch.zeros_like(lin)

    lin_scale = torch.clamp(lin / max(v_max, 1e-6), 0.0, 1.0)
    yaw_scale = torch.clamp(yaw / max(yaw_max, 1e-6), 0.0, 1.0)

    if cmd_use_max:
        cmd_scale = torch.maximum(lin_scale, yaw_scale)
    else:
        cmd_scale = torch.clamp(w_lin * lin_scale + w_yaw * yaw_scale, 0.0, 1.0)

    # --- swing credit: ON after hold_min_air, no extra reward for longer air time ---
    swing_ok = ((air_time >= hold_min_air) & (~in_contact)).float()  # (N,2)
    swing_reward = torch.sum(swing_ok, dim=1)                        # (N,) 0..2

    # --- over-hold penalty: punish keeping a foot in the air for too long ---
    # Smooth ramp from hold_max_air to hold_max_air+overhold_margin.
    if overhold_margin <= 0.0:
        overhold_amount = ((air_time - hold_max_air) > 0.0).float()
    else:
        overhold_amount = torch.clamp(
            (air_time - hold_max_air) / max(overhold_margin, 1e-6), 0.0, 1.0
        )

    # only penalize while foot is actually in air
    overhold_amount = overhold_amount * (~in_contact).float()
    penalty_overhold = overhold_amount.sum(dim=1) * overhold_penalty

    # --- double flight penalty ---
    double_flight = (torch.sum(in_contact.int(), dim=1) == 0)
    penalty_double_flight = double_flight.float() * double_flight_penalty

    # --- anti-tap: air (short swing) ---
    tap_air = touch_down & (air_time < tap_air_threshold)
    penalty_tap_air = tap_air.any(dim=1).float() * tap_air_penalty

    # --- anti-tap: contact (short stance) ---
    tap_contact = lift_off & (contact_time < tap_contact_threshold)
    penalty_tap_contact = tap_contact.any(dim=1).float() * tap_contact_penalty

    # --- alternation bonus (reward when swing leg alternates) ---
    lift_count = lift_off.int().sum(dim=1)
    lifted_foot = torch.where(
        lift_off[:, 0], 0,
        torch.where(lift_off[:, 1], 1, -1)
    ).long()

    valid_single_lift = lift_count == 1
    alternates = valid_single_lift & (last_swing_foot >= 0) & (lifted_foot != last_swing_foot)
    bonus_alternation = alternates.float() * alternation_bonus

    last_swing_foot = torch.where(valid_single_lift, lifted_foot, last_swing_foot)

    # --- EMA symmetry (long-horizon left/right balance) ---
    # EMA_t = (1 - alpha) * EMA_{t-1} + alpha * x_t
    ema_air = (1.0 - ema_alpha) * ema_air + ema_alpha * last_completed_air
    ema_contact = (1.0 - ema_alpha) * ema_contact + ema_alpha * contact_time

    air_imbalance = torch.abs(ema_air[:, 0] - ema_air[:, 1])
    contact_imbalance = torch.abs(ema_contact[:, 0] - ema_contact[:, 1])
    penalty_balance = balance_weight * (air_imbalance + contact_imbalance)

    # --- total reward ---
    reward = (
        swing_reward
        + bonus_alternation
        - penalty_overhold
        - penalty_double_flight
        - penalty_tap_air
        - penalty_tap_contact
        - penalty_balance
    )

    reward = reward * cmd_scale

    # --- write back buffers ---
    buf["prev_in_contact"] = in_contact
    buf["last_completed_air"] = last_completed_air
    buf["ema_air"] = ema_air
    buf["ema_contact"] = ema_contact
    buf["last_swing_foot"] = last_swing_foot

    return reward
