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


def local_torso_height_penalty_l2(
    env,
    asset_cfg,
    contact_sensor_cfg,
    target_height: float,
    margin: float,
    gain: float = 1.0,
    contact_force_threshold: float = 0.01,
    large_value: float = 1e9,
) -> torch.Tensor:
    """
    L2 version of local_torso_height_penalty.

    - Computes foot depth along the torso local -Z axis (positive means "below the torso").
    - If any foot is in contact: uses the highest contacting foot (min depth) as support reference.
    - If in flight: uses the lower foot (max depth) as reference.
    - Penalizes deviation outside a dead-zone (margin) with a squared (L2) penalty:
        penalty = -gain * (max(|diff| - margin, 0))^2
    """
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

    # Contact detection (use peak force over history)
    contact_sensor = env.scene.sensors[contact_sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids, :]  # (N, H, A, 3)
    in_contact = forces.norm(dim=-1).amax(dim=1) > contact_force_threshold                   # (N, A)

    any_contact = in_contact.any(dim=-1)  # (N,)

    # If contact exists: use "highest foot" among contacting feet => min depth
    contact_depths = torch.where(in_contact, foot_depths, torch.full_like(foot_depths, large_value))
    contact_depth = contact_depths.min(dim=-1).values  # (N,)

    # If flight: use lower foot => max depth
    flight_depth = foot_depths.max(dim=-1).values      # (N,)

    foot_depth = torch.where(any_contact, contact_depth, flight_depth)

    diff = foot_depth - target_height
    excess = torch.relu(torch.abs(diff) - margin)

    # L2 penalty
    return -gain * excess * excess


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
    # gating thresholds (set < 0 to disable each gate)
    linear_cmd_threshold: float = 1.0,    # m/s, |v_xy| where linear gate reaches 1
    angular_cmd_threshold: float = 1.0,   # rad/s, |yaw_rate| where angular gate reaches 1
    body_tilt_threshold: float = 0.35,    # rad, torso tilt angle where gate reaches 1

    # swing constraints
    hold_min_air: float = 0.15,
    hold_max_air: float = 0.45,  # TRUE swing timeout: beyond this, the current swing is treated as failed

    # timeout penalty (applied while a swing is timed out and the foot is still in the air)
    swing_timeout_penalty: float = 1.0,

    # step completion bonus (reward at touchdown for a valid step)
    step_complete_bonus: float = 1.0,

    # anti-tap (air/contact)
    tap_air_threshold: float = 0.10,
    tap_air_penalty: float = 0.5,
    tap_contact_threshold: float = 0.10,
    tap_contact_penalty: float = 0.5,

    # symmetry (EMA balance)
    ema_alpha: float = 0.02,
    balance_weight: float = 0.5,

    # avoid double flight
    double_flight_penalty: float = 2.0,

    # positive shaping weights during swing (keep modest if you want completion to dominate)
    swing_in_air_reward: float = 0.3,
    support_in_contact_reward: float = 0.3,
) -> torch.Tensor:
    """
    Balanced alternating biped reward.

    Gate logic (linear, optional):
      - linear_cmd_threshold >= 0:
          lin_scale = clamp(|v_xy| / linear_cmd_threshold, 0..1)
      - angular_cmd_threshold >= 0:
          yaw_scale = clamp(|yaw_rate| / angular_cmd_threshold, 0..1)
      - body_tilt_threshold >= 0:
          tilt_angle = asin(||projected_gravity_b.xy||)   [rad]
          tilt_scale = clamp(tilt_angle / body_tilt_threshold, 0..1)

      gate_scale = max(lin_scale, yaw_scale, tilt_scale)

      If all three thresholds are < 0:
          gate_scale = 1 (reward always enabled).

    TRUE swing timeout (hold_max_air):
      - If the active swing exceeds hold_max_air, the swing is latched as timed-out (failed).
      - While timed out: step rewards are disabled and a timeout penalty is applied.
      - The timed-out swing is not counted as a valid step (no completion bonus, no prev-step update).
      - The latch clears only when the swing foot touches down.

    EMA = Exponential Moving Average.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_ids = sensor_cfg.body_ids
    assert len(foot_ids) == 2, "This reward is for bipeds (2 feet)."

    # Sensor state
    air_time = contact_sensor.data.current_air_time[:, foot_ids]          # (N,2)
    contact_time = contact_sensor.data.current_contact_time[:, foot_ids]  # (N,2)
    in_contact = contact_time > 0.0                                       # (N,2)

    n_env = in_contact.shape[0]
    device = in_contact.device

    # ---------------- Persistent buffers ----------------
    key = "_feet_balanced_alt_reward_buf"
    if not hasattr(env, key):
        buf = {
            "prev_in_contact": in_contact.clone(),
            "last_completed_air": torch.zeros((n_env, 2), device=device),
            "ema_air": torch.zeros((n_env, 2), device=device),
            "ema_contact": torch.zeros((n_env, 2), device=device),
            "active_swing_foot": torch.full((n_env,), -1, device=device, dtype=torch.long),     # -1/0/1
            "prev_step_swing_foot": torch.full((n_env,), -1, device=device, dtype=torch.long),  # -1/0/1
            "swing_timed_out": torch.zeros((n_env,), device=device, dtype=torch.bool),
        }
        setattr(env, key, buf)

    buf = getattr(env, key)
    prev_in_contact = buf["prev_in_contact"]
    last_completed_air = buf["last_completed_air"]
    ema_air = buf["ema_air"]
    ema_contact = buf["ema_contact"]
    active_swing_foot = buf["active_swing_foot"]
    prev_step_swing_foot = buf["prev_step_swing_foot"]
    swing_timed_out = buf["swing_timed_out"]

    # ---------------- Reset per-env buffers on episode reset ----------------
    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is not None:
        reset_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            last_completed_air[reset_ids] = 0.0
            ema_air[reset_ids] = 0.0
            ema_contact[reset_ids] = 0.0
            active_swing_foot[reset_ids] = -1
            prev_step_swing_foot[reset_ids] = -1
            swing_timed_out[reset_ids] = False
            prev_in_contact[reset_ids] = in_contact[reset_ids]

    # ---------------- Transitions ----------------
    lift_off = prev_in_contact & (~in_contact)    # contact -> air
    touch_down = (~prev_in_contact) & in_contact  # air -> contact

    # Record completed swing time at touchdown (used for EMA)
    last_completed_air = torch.where(touch_down, air_time, last_completed_air)

    # ---------------- Gate scales ----------------
    cmd = env.command_manager.get_command(command_name)
    lin_mag = torch.norm(cmd[:, :2], dim=1)  # |v_xy|
    yaw_mag = torch.abs(cmd[:, 2]) if cmd.shape[1] > 2 else torch.zeros_like(lin_mag)  # |omega_z|

    lin_scale = torch.zeros_like(lin_mag)
    if linear_cmd_threshold >= 0.0:
        lin_scale = torch.clamp(lin_mag / max(linear_cmd_threshold, 1e-6), 0.0, 1.0)

    yaw_scale = torch.zeros_like(lin_mag)
    if angular_cmd_threshold >= 0.0:
        yaw_scale = torch.clamp(yaw_mag / max(angular_cmd_threshold, 1e-6), 0.0, 1.0)

    tilt_scale = torch.zeros_like(lin_mag)
    if body_tilt_threshold >= 0.0:
        # A: using projected_gravity_b
        # NOTE: Replace "robot" with your articulation name if needed.
        robot = env.scene.articulations["robot"]
        proj_g = robot.data.projected_gravity_b  # (N,3), gravity direction expressed in base frame
        sin_tilt = torch.norm(proj_g[:, :2], dim=1).clamp(0.0, 1.0)  # ~= sin(tilt)
        tilt_angle = torch.asin(sin_tilt)  # [rad]
        tilt_scale = torch.clamp(tilt_angle / max(body_tilt_threshold, 1e-6), 0.0, 1.0)

    all_gates_disabled = (linear_cmd_threshold < 0.0) and (angular_cmd_threshold < 0.0) and (body_tilt_threshold < 0.0)
    if all_gates_disabled:
        gate_scale = torch.ones_like(lin_mag)
    else:
        gate_scale = torch.maximum(torch.maximum(lin_scale, yaw_scale), tilt_scale)

    # ---------------- Contact pattern ----------------
    contact_count = in_contact.int().sum(dim=1)
    single_support = contact_count == 1
    double_flight = contact_count == 0

    # ---------------- Active swing bookkeeping (liftoff -> touchdown) ----------------
    lift_count = lift_off.int().sum(dim=1)
    lifted_foot = torch.where(lift_off[:, 0], 0, torch.where(lift_off[:, 1], 1, -1)).long()

    start_swing = (active_swing_foot < 0) & (lift_count == 1) & single_support & (~swing_timed_out)
    active_swing_foot = torch.where(start_swing, lifted_foot, active_swing_foot)

    # One-time liftoff reward for the very first step (no extra parameters; uses swing_in_air_reward)
    first_step = prev_step_swing_foot < 0
    reward_first_liftoff = (start_swing & first_step).float() * swing_in_air_reward

    end_swing = ((active_swing_foot == 0) & touch_down[:, 0]) | ((active_swing_foot == 1) & touch_down[:, 1])

    active_air = torch.where(
        active_swing_foot == 0, air_time[:, 0],
        torch.where(active_swing_foot == 1, air_time[:, 1], torch.zeros_like(lin_mag))
    )

    # ---------------- TRUE swing timeout latch using hold_max_air ----------------
    timeout_now = (active_swing_foot >= 0) & (active_air > hold_max_air)
    swing_timed_out = swing_timed_out | timeout_now
    swing_timed_out = torch.where(end_swing, torch.zeros_like(swing_timed_out), swing_timed_out)

    valid_swing_end = end_swing & (active_air >= hold_min_air) & (~swing_timed_out)

    # Update prev step only on valid completion (enforces alternation on real steps only)
    prev_step_swing_foot = torch.where(valid_swing_end, active_swing_foot, prev_step_swing_foot)

    # Clear active swing foot on touchdown
    active_swing_foot = torch.where(end_swing, torch.full_like(active_swing_foot, -1), active_swing_foot)

    # ---------------- Alternation gating (unknown_prev removed) ----------------
    has_prev_step = prev_step_swing_foot >= 0
    step_reward_enabled = (
        (active_swing_foot >= 0)
        & has_prev_step
        & (active_swing_foot != prev_step_swing_foot)
        & (~swing_timed_out)
    )

    # ---------------- Swing shaping rewards (small, during swing) ----------------
    swing_air_ok = torch.zeros((n_env,), device=device)
    swing_air_ok = torch.where((active_swing_foot == 0) & (air_time[:, 0] >= hold_min_air), 1.0, swing_air_ok)
    swing_air_ok = torch.where((active_swing_foot == 1) & (air_time[:, 1] >= hold_min_air), 1.0, swing_air_ok)

    reward_swing_air = swing_in_air_reward * swing_air_ok * step_reward_enabled.float() * single_support.float()
    reward_support_contact = support_in_contact_reward * step_reward_enabled.float() * single_support.float()

    # Touchdown bonus
    reward_step_complete = valid_swing_end.float() * step_complete_bonus

    # ---------------- Penalties ----------------
    penalty_double_flight = double_flight.float() * double_flight_penalty
    penalty_timeout = swing_timed_out.float() * swing_timeout_penalty

    # Penalize short air-time (touchdown immediately after liftoff)
    tap_air = touch_down & (air_time < tap_air_threshold)
    penalty_tap_air = tap_air.any(dim=1).float() * tap_air_penalty

    # Penalize short contact-time (liftoff immediately after touchdown)
    tap_contact = lift_off & (contact_time < tap_contact_threshold)
    penalty_tap_contact = tap_contact.any(dim=1).float() * tap_contact_penalty

    # ---------------- EMA symmetry ----------------
    # EMA = Exponential Moving Average.
    # EMA_t = (1 - alpha) * EMA_{t-1} + alpha * x_t
    ema_air = (1.0 - ema_alpha) * ema_air + ema_alpha * last_completed_air
    ema_contact = (1.0 - ema_alpha) * ema_contact + ema_alpha * contact_time

    air_imbalance = torch.abs(ema_air[:, 0] - ema_air[:, 1])
    contact_imbalance = torch.abs(ema_contact[:, 0] - ema_contact[:, 1])
    penalty_balance = balance_weight * (air_imbalance + contact_imbalance)

    # ---------------- Total reward ----------------
    reward = (
        reward_first_liftoff
        + reward_swing_air
        + reward_support_contact
        + reward_step_complete
        - penalty_double_flight
        - penalty_timeout
        - penalty_tap_air
        - penalty_tap_contact
        - penalty_balance
    )

    reward = reward * gate_scale

    # ---------------- Write back buffers ----------------
    buf["prev_in_contact"] = in_contact
    buf["last_completed_air"] = last_completed_air
    buf["ema_air"] = ema_air
    buf["ema_contact"] = ema_contact
    buf["active_swing_foot"] = active_swing_foot
    buf["prev_step_swing_foot"] = prev_step_swing_foot
    buf["swing_timed_out"] = swing_timed_out

    return reward


def prefer_foot_contact_only(
    env,
    foot_sensor_cfg: SceneEntityCfg,
    other_sensor_cfg: SceneEntityCfg,
    *,
    contact_force_threshold: float = 0.01,
    foot_contact_reward: float = 1.0,
    nonfoot_contact_penalty: float = 1.0,
) -> torch.Tensor:
    """
    Encourage states where at least one foot link is in contact with the ground,
    and discourage states where any non-foot link is in contact.

    - Reward if ANY foot link is in contact.
    - Penalize if ANY non-foot link is in contact.
    - Contact counts do NOT matter (existence check only).

    Contact detection logic follows other reward functions in this file:
      - Uses peak contact force over history from ContactSensor.
    """

    # --------------------------------------------------------
    # Foot contact detection
    # --------------------------------------------------------
    foot_sensor: ContactSensor = env.scene.sensors[foot_sensor_cfg.name]
    foot_forces = foot_sensor.data.net_forces_w_history[
        :, :, foot_sensor_cfg.body_ids, :
    ]  # (N, H, F, 3)

    foot_in_contact = (
        foot_forces.norm(dim=-1).amax(dim=1) > contact_force_threshold
    )  # (N, F)

    any_foot_contact = foot_in_contact.any(dim=1)  # (N,)

    # --------------------------------------------------------
    # Non-foot contact detection
    # --------------------------------------------------------
    other_sensor: ContactSensor = env.scene.sensors[other_sensor_cfg.name]
    other_forces = other_sensor.data.net_forces_w_history[
        :, :, other_sensor_cfg.body_ids, :
    ]  # (N, H, K, 3)

    other_in_contact = (
        other_forces.norm(dim=-1).amax(dim=1) > contact_force_threshold
    )  # (N, K)

    any_nonfoot_contact = other_in_contact.any(dim=1)  # (N,)

    # --------------------------------------------------------
    # Reward composition
    # --------------------------------------------------------
    reward = torch.zeros_like(any_foot_contact, dtype=torch.float)

    reward += any_foot_contact.float() * foot_contact_reward
    reward -= any_nonfoot_contact.float() * nonfoot_contact_penalty

    return reward


def links_z_axis_parallel_penalty_relative(
    env,
    asset_cfg: "SceneEntityCfg",
) -> "torch.Tensor":
    """Penalize squared misalignment angles of multiple link Z-axes
    relative to the first link.

    The penalty is the sum of squared tilt angles (in radians^2) between
    each link's local +Z axis and the first link's local +Z axis.
    Only relative alignment is evaluated; the world frame orientation is irrelevant.

    Returns
    -------
    torch.Tensor
        Positive penalty (sum of squared angles in radians^2).
    """
    asset = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if len(body_ids) < 2:
        raise ValueError("asset_cfg.body_ids must contain at least 2 links.")

    # (N, K, 4)
    body_quat_w = asset.data.body_quat_w[:, body_ids, :]

    # Local +Z axis
    z_local = torch.zeros(
        (body_quat_w.shape[0], body_quat_w.shape[1], 3),
        device=body_quat_w.device,
        dtype=body_quat_w.dtype,
    )
    z_local[..., 2] = 1.0

    # Rotate local Z axes into world frame
    z_w = _quat_rotate(body_quat_w, z_local)  # (N, K, 3)

    z0 = z_w[:, 0, :].unsqueeze(1)  # (N, 1, 3)
    zi = z_w[:, 1:, :]              # (N, K-1, 3)

    # Cosine between axes
    cos_theta = torch.sum(zi * z0, dim=-1)  # (N, K-1)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # Angle in radians
    angles = torch.acos(cos_theta)  # (N, K-1)

    # Squared-angle penalty
    return torch.sum(angles * angles, dim=1)
