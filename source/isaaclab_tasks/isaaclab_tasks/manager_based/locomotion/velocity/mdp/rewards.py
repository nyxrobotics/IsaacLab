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
    # --- gating thresholds (set < 0 to disable each gate) ---
    linear_cmd_threshold: float = 1.0,    # m/s, |v_xy| where linear command fully activates the reward
    angular_cmd_threshold: float = 1.0,   # rad/s, |yaw_rate| where angular command fully activates the reward
    body_tilt_threshold: float = 0.35,    # rad, torso tilt angle where tilt fully activates the reward

    # --- air/contact hold constraints ---
    hold_min_air: float = 0.15,           # minimum continuous air duration for a valid air phase
    hold_max_air: float = 0.45,           # maximum allowed air duration before the air phase is treated as failed
    hold_min_contact: float = 0.10,       # minimum continuous contact duration required before liftoff

    # --- penalties / bonuses ---
    air_timeout_penalty: float = 1.0,     # penalty applied while an air phase exceeds hold_max_air
    step_complete_bonus: float = 1.0,     # bonus given at touchdown after a valid air phase

    # --- symmetry control using EMA ---
    ema_alpha: float = 0.02,              # EMA update rate (smaller = longer memory)
    balance_weight: float = 0.5,           # strength of left/right imbalance penalty

    # --- contact pattern penalties ---
    double_flight_penalty: float = 2.0,    # penalty when both feet are in the air

    # --- shaping rewards ---
    air_hold_reward: float = 0.3,          # reward for holding a valid air phase
    contact_hold_reward: float = 0.3,      # reward for stable single-foot contact
) -> torch.Tensor:
    """
    Reward function for bipedal locomotion based on air/contact phases.

    The reward encourages the following behavior:
      - When commands are large or the body is tilted, create an air phase (lift one foot).
      - Once lifted, keep the foot in the air for at least hold_min_air.
      - Do not keep the foot in the air indefinitely (hold_max_air).
      - Touch down to complete a step and receive a step completion bonus.
      - Alternate left and right feet between successive valid steps.
      - Maintain long-term symmetry between left and right feet using EMA statistics.

    Terminology:
      - "air phase": a foot is not in contact with the ground.
      - "contact phase": a foot is in contact with the ground.
    """

    # ------------------------------------------------------------------
    # Sensor access
    # ------------------------------------------------------------------
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_ids = sensor_cfg.body_ids
    assert len(foot_ids) == 2, "This reward function assumes a biped (two feet)."

    # Per-foot timers provided by the contact sensor
    air_time = contact_sensor.data.current_air_time[:, foot_ids]          # (N, 2)
    contact_time = contact_sensor.data.current_contact_time[:, foot_ids]  # (N, 2)
    in_contact = contact_time > 0.0                                       # (N, 2)

    n_env = in_contact.shape[0]
    device = in_contact.device

    # ------------------------------------------------------------------
    # Persistent per-environment buffers
    # ------------------------------------------------------------------
    # These buffers track the phase history and long-term statistics
    key = "_feet_air_contact_reward_buf"
    if not hasattr(env, key):
        buf = {
            # Contact state from the previous step (used to detect transitions)
            "prev_in_contact": in_contact.clone(),

            # Air duration recorded at the moment of touchdown (per foot)
            "last_completed_air": torch.zeros((n_env, 2), device=device),

            # Exponential Moving Averages of air/contact durations
            "ema_air": torch.zeros((n_env, 2), device=device),
            "ema_contact": torch.zeros((n_env, 2), device=device),

            # Index of the foot currently in the air phase (-1 if none)
            "active_air_foot": torch.full((n_env,), -1, device=device, dtype=torch.long),

            # Index of the foot used in the previous valid step (-1 if none yet)
            "prev_step_air_foot": torch.full((n_env,), -1, device=device, dtype=torch.long),

            # Latched flag indicating that the current air phase exceeded hold_max_air
            "air_timed_out": torch.zeros((n_env,), device=device, dtype=torch.bool),
        }
        setattr(env, key, buf)

    buf = getattr(env, key)
    prev_in_contact = buf["prev_in_contact"]
    last_completed_air = buf["last_completed_air"]
    ema_air = buf["ema_air"]
    ema_contact = buf["ema_contact"]
    active_air_foot = buf["active_air_foot"]
    prev_step_air_foot = buf["prev_step_air_foot"]
    air_timed_out = buf["air_timed_out"]

    # ------------------------------------------------------------------
    # Episode reset handling
    # ------------------------------------------------------------------
    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is not None:
        reset_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            last_completed_air[reset_ids] = 0.0
            ema_air[reset_ids] = 0.0
            ema_contact[reset_ids] = 0.0
            active_air_foot[reset_ids] = -1
            prev_step_air_foot[reset_ids] = -1
            air_timed_out[reset_ids] = False
            prev_in_contact[reset_ids] = in_contact[reset_ids]

    # ------------------------------------------------------------------
    # Detect contact transitions
    # ------------------------------------------------------------------
    lift_off = prev_in_contact & (~in_contact)    # contact -> air
    touch_down = (~prev_in_contact) & in_contact  # air -> contact

    # Record air duration at touchdown for symmetry statistics
    last_completed_air = torch.where(touch_down, air_time, last_completed_air)

    # ------------------------------------------------------------------
    # Command / tilt gating
    # ------------------------------------------------------------------
    # The entire reward is scaled by gate_scale.
    # When commands are small and the body is upright, the reward is suppressed.
    cmd = env.command_manager.get_command(command_name)
    lin_mag = torch.norm(cmd[:, :2], dim=1)
    yaw_mag = torch.abs(cmd[:, 2]) if cmd.shape[1] > 2 else torch.zeros_like(lin_mag)

    lin_scale = torch.zeros_like(lin_mag)
    if linear_cmd_threshold >= 0.0:
        lin_scale = torch.clamp(lin_mag / max(linear_cmd_threshold, 1e-6), 0.0, 1.0)

    yaw_scale = torch.zeros_like(lin_mag)
    if angular_cmd_threshold >= 0.0:
        yaw_scale = torch.clamp(yaw_mag / max(angular_cmd_threshold, 1e-6), 0.0, 1.0)

    tilt_scale = torch.zeros_like(lin_mag)
    if body_tilt_threshold >= 0.0:
        robot = env.scene.articulations["robot"]
        proj_g = robot.data.projected_gravity_b
        sin_tilt = torch.norm(proj_g[:, :2], dim=1).clamp(0.0, 1.0)
        tilt_angle = torch.asin(sin_tilt)
        tilt_scale = torch.clamp(tilt_angle / max(body_tilt_threshold, 1e-6), 0.0, 1.0)

    if linear_cmd_threshold < 0 and angular_cmd_threshold < 0 and body_tilt_threshold < 0:
        gate_scale = torch.ones_like(lin_mag)
    else:
        gate_scale = torch.maximum(torch.maximum(lin_scale, yaw_scale), tilt_scale)

    # ------------------------------------------------------------------
    # Contact pattern classification
    # ------------------------------------------------------------------
    contact_count = in_contact.int().sum(dim=1)
    single_contact = contact_count == 1
    double_flight = contact_count == 0

    # ------------------------------------------------------------------
    # Air phase start (liftoff)
    # ------------------------------------------------------------------
    lift_count = lift_off.int().sum(dim=1)
    lifted_foot = torch.where(lift_off[:, 0], 0, torch.where(lift_off[:, 1], 1, -1))

    lifted_contact_time = torch.where(
        lifted_foot == 0, contact_time[:, 0],
        torch.where(lifted_foot == 1, contact_time[:, 1], torch.zeros_like(lin_mag))
    )
    liftoff_contact_ok = lifted_contact_time >= hold_min_contact

    start_air_phase = (
        (active_air_foot < 0) &
        (lift_count == 1) &
        single_contact &
        (~air_timed_out) &
        (lifted_foot >= 0) &
        liftoff_contact_ok
    )

    active_air_foot = torch.where(start_air_phase, lifted_foot, active_air_foot)

    # Liftoff is rewarded when it alternates with the previous valid step.
    has_prev_step = prev_step_air_foot >= 0
    liftoff_alternation_ok = (~has_prev_step) | (lifted_foot != prev_step_air_foot)
    reward_liftoff = start_air_phase.float() * liftoff_alternation_ok.float() * air_hold_reward

    # ------------------------------------------------------------------
    # Air phase end (touchdown)
    # ------------------------------------------------------------------
    end_air_phase = (
        ((active_air_foot == 0) & touch_down[:, 0]) |
        ((active_air_foot == 1) & touch_down[:, 1])
    )

    active_air_time = torch.where(
        active_air_foot == 0, air_time[:, 0],
        torch.where(active_air_foot == 1, air_time[:, 1], torch.zeros_like(lin_mag))
    )

    timeout_now = (active_air_foot >= 0) & (active_air_time > hold_max_air)
    air_timed_out = air_timed_out | timeout_now
    air_timed_out = torch.where(end_air_phase, torch.zeros_like(air_timed_out), air_timed_out)

    valid_step_complete = end_air_phase & (active_air_time >= hold_min_air) & (~air_timed_out)

    prev_step_air_foot = torch.where(valid_step_complete, active_air_foot, prev_step_air_foot)
    active_air_foot = torch.where(end_air_phase, torch.full_like(active_air_foot, -1), active_air_foot)

    # ------------------------------------------------------------------
    # Shaping rewards during the air/contact phase
    # ------------------------------------------------------------------
    alternation_ok = (active_air_foot >= 0) & has_prev_step & (active_air_foot != prev_step_air_foot)
    phase_reward_enabled = alternation_ok & (~air_timed_out)

    air_hold_ok = torch.zeros((n_env,), device=device)
    air_hold_ok = torch.where((active_air_foot == 0) & (air_time[:, 0] >= hold_min_air), 1.0, air_hold_ok)
    air_hold_ok = torch.where((active_air_foot == 1) & (air_time[:, 1] >= hold_min_air), 1.0, air_hold_ok)

    reward_air_hold = air_hold_reward * air_hold_ok * phase_reward_enabled.float() * single_contact.float()
    reward_contact_hold = contact_hold_reward * phase_reward_enabled.float() * single_contact.float()
    reward_step_complete = valid_step_complete.float() * step_complete_bonus

    # ------------------------------------------------------------------
    # Penalties
    # ------------------------------------------------------------------
    penalty_double_flight = double_flight.float() * double_flight_penalty
    penalty_timeout = air_timed_out.float() * air_timeout_penalty

    # ------------------------------------------------------------------
    # Symmetry penalty using EMA
    # ------------------------------------------------------------------
    # EMA = Exponential Moving Average.
    # These statistics penalize long-term imbalance between left and right feet.
    ema_air = (1.0 - ema_alpha) * ema_air + ema_alpha * last_completed_air
    ema_contact = (1.0 - ema_alpha) * ema_contact + ema_alpha * contact_time

    air_imbalance = torch.abs(ema_air[:, 0] - ema_air[:, 1])
    contact_imbalance = torch.abs(ema_contact[:, 0] - ema_contact[:, 1])
    penalty_balance = balance_weight * (air_imbalance + contact_imbalance)

    # ------------------------------------------------------------------
    # Total reward
    # ------------------------------------------------------------------
    reward = (
        reward_liftoff
        + reward_air_hold
        + reward_contact_hold
        + reward_step_complete
        - penalty_double_flight
        - penalty_timeout
        - penalty_balance
    )

    reward = reward * gate_scale

    # ------------------------------------------------------------------
    # Write back buffers
    # ------------------------------------------------------------------
    buf["prev_in_contact"] = in_contact
    buf["last_completed_air"] = last_completed_air
    buf["ema_air"] = ema_air
    buf["ema_contact"] = ema_contact
    buf["active_air_foot"] = active_air_foot
    buf["prev_step_air_foot"] = prev_step_air_foot
    buf["air_timed_out"] = air_timed_out

    return reward


def prefer_foot_contact(
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



def both_feet_flight_time_penalty_and_grounded_time_reward(
    env,
    foot_sensor_cfg: SceneEntityCfg,
    *,
    contact_force_threshold: float = 0.01,
    penalty_scale: float = 1.0,
    reward_scale: float = 1.0,
    max_reward: float = 1.0,
    dt: float | None = None,
) -> torch.Tensor:
    """
    - Penalty proportional to the continuous duration where BOTH feet are airborne.
    - If not airborne (i.e., penalty not active), reward proportional to the continuous duration
      where at least one foot is in contact, clipped to avoid divergence.

    Contact detection matches existing functions in this file:
      - Uses peak contact force over history from ContactSensor.
    """

    if dt is None:
        dt = env.step_dt

    contact_sensor: ContactSensor = env.scene.sensors[foot_sensor_cfg.name]
    foot_forces = contact_sensor.data.net_forces_w_history[:, :, foot_sensor_cfg.body_ids, :]  # (N, H, F, 3)
    foot_in_contact = foot_forces.norm(dim=-1).amax(dim=1) > contact_force_threshold          # (N, F)

    any_foot_contact = foot_in_contact.any(dim=1)  # (N,)
    both_feet_air = ~any_foot_contact              # (N,)

    n_env = both_feet_air.shape[0]
    device = both_feet_air.device

    # Persistent buffers on env to track continuous durations across steps.
    key = "_both_feet_flight_time_buf"
    if not hasattr(env, key):
        buf = {
            "flight_time": torch.zeros((n_env,), device=device),
            "grounded_time": torch.zeros((n_env,), device=device),
        }
        setattr(env, key, buf)

    buf = getattr(env, key)
    flight_time = buf["flight_time"]
    grounded_time = buf["grounded_time"]

    # Episode reset handling (if available)
    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is not None:
        reset_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            flight_time[reset_ids] = 0.0
            grounded_time[reset_ids] = 0.0

    # Update continuous duration counters
    flight_time = torch.where(both_feet_air, flight_time + dt, torch.zeros_like(flight_time))
    grounded_time = torch.where(~both_feet_air, grounded_time + dt, torch.zeros_like(grounded_time))

    buf["flight_time"] = flight_time
    buf["grounded_time"] = grounded_time

    # Penalty and reward
    penalty = -penalty_scale * flight_time
    max_time = max_reward / max(reward_scale, 1e-6)
    bonus = reward_scale * torch.clamp(grounded_time, max=max_time)

    return torch.where(both_feet_air, penalty, bonus)


def nonfoot_contact_time_penalty_and_clear_time_reward(
    env,
    other_sensor_cfg: SceneEntityCfg,
    *,
    contact_force_threshold: float = 0.01,
    penalty_scale: float = 1.0,
    reward_scale: float = 1.0,
    max_reward: float = 1.0,
    dt: float | None = None,
) -> torch.Tensor:
    """
    - Penalty proportional to the continuous duration where ANY non-foot body is in contact.
    - If no non-foot contact (i.e., penalty not active), reward proportional to the continuous
      "clear" duration, clipped to avoid divergence.

    Contact detection matches existing functions in this file:
      - Uses peak contact force over history from ContactSensor.
    """

    if dt is None:
        dt = env.step_dt

    contact_sensor: ContactSensor = env.scene.sensors[other_sensor_cfg.name]
    other_forces = contact_sensor.data.net_forces_w_history[:, :, other_sensor_cfg.body_ids, :]  # (N, H, K, 3)
    other_in_contact = other_forces.norm(dim=-1).amax(dim=1) > contact_force_threshold          # (N, K)

    any_nonfoot_contact = other_in_contact.any(dim=1)  # (N,)

    n_env = any_nonfoot_contact.shape[0]
    device = any_nonfoot_contact.device

    # Persistent buffers on env to track continuous durations across steps.
    key = "_nonfoot_contact_time_buf"
    if not hasattr(env, key):
        buf = {
            "bad_time": torch.zeros((n_env,), device=device),
            "clear_time": torch.zeros((n_env,), device=device),
        }
        setattr(env, key, buf)

    buf = getattr(env, key)
    bad_time = buf["bad_time"]
    clear_time = buf["clear_time"]

    # Episode reset handling (if available)
    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is not None:
        reset_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            bad_time[reset_ids] = 0.0
            clear_time[reset_ids] = 0.0

    # Update continuous duration counters
    bad_time = torch.where(any_nonfoot_contact, bad_time + dt, torch.zeros_like(bad_time))
    clear_time = torch.where(~any_nonfoot_contact, clear_time + dt, torch.zeros_like(clear_time))

    buf["bad_time"] = bad_time
    buf["clear_time"] = clear_time

    # Penalty and reward
    penalty = -penalty_scale * bad_time
    max_time = max_reward / max(reward_scale, 1e-6)
    bonus = reward_scale * torch.clamp(clear_time, max=max_time)

    return torch.where(any_nonfoot_contact, penalty, bonus)
