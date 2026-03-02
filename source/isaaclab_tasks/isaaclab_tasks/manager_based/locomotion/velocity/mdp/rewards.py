# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils.math import yaw_quat
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg,
                  threshold: float) -> torch.Tensor:
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


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor:
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
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]),
                              dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(env, command_name: str, std: float,
                              asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(env,
                                   command_name: str,
                                   command_threshold: float = 0.06,
                                   asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')) -> torch.Tensor:
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
    in_contact = forces.norm(dim=-1).max(dim=1)[0] > contact_force_threshold  # (N, A)

    num_contact = in_contact.sum(dim=-1)  # (N,)
    any_contact = num_contact > 0

    # If contact exists: use "highest foot" among contacting feet => min depth
    contact_depths = torch.where(in_contact, foot_depths, torch.full_like(foot_depths, 1e9))
    contact_depth = contact_depths.min(dim=-1).values  # (N,)

    # If flight: use lower foot => max depth
    flight_depth = foot_depths.max(dim=-1).values  # (N,)

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
    in_contact = forces.norm(dim=-1).amax(dim=1) > contact_force_threshold  # (N, A)

    any_contact = in_contact.any(dim=-1)  # (N,)

    # If contact exists: use "highest foot" among contacting feet => min depth
    contact_depths = torch.where(in_contact, foot_depths, torch.full_like(foot_depths, large_value))
    contact_depth = contact_depths.min(dim=-1).values  # (N,)

    # If flight: use lower foot => max depth
    flight_depth = foot_depths.max(dim=-1).values  # (N,)

    foot_depth = torch.where(any_contact, contact_depth, flight_depth)

    diff = foot_depth - target_height
    excess = torch.relu(torch.abs(diff) - margin)

    # L2 penalty
    return -gain * excess * excess


def feet_slide_with_yaw(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg('robot'),
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
    lin_xy = body_lin_vel_w[..., :2]  # (N, F, 2)
    lin_speed = lin_xy.norm(dim=-1)  # (N, F)

    # 角速度（world）: (N, B, 3) → 対象足
    body_ang_vel_w = asset.data.body_ang_vel_w[:, body_ids, :]  # (N, F, 3)
    yaw_rate = torch.abs(body_ang_vel_w[..., 2])  # (N, F)  z軸まわり

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


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) v by quaternion(s) q.
    q: (..., 4) in (x, y, z, w)
    v: (..., 3)
    """
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q_w * t + torch.cross(q_xyz, t, dim=-1)


def feet_slide_keep_flat(
        env,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg('robot'),
        lin_weight: float = 1.0,
        tilt_weight: float = 1.0,
        yaw_weight: float = 1.0,
        allow_uprighting_tilt: bool = True,  # default ON
        air_time_eps: float = 1e-3,
        force_eps: float = 1e-6,
        local_up_axis: torch.Tensor | None = None,  # default: +Z
) -> torch.Tensor:
    """Penalize XY sliding and orientation change while the foot is in contact.

    Tilt handling:
      - allow_uprighting_tilt=True:
          Only penalize tilt motion that increases tilt (moves away from world up).
      - allow_uprighting_tilt=False:
          Penalize total tilt rate ||omega_xy|| as usual.

    Returns positive cost (penalty).
    """
    # --------------------------------------------------------
    # 1) Contact state
    # --------------------------------------------------------
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    forces_w_last = contact_sensor.data.net_forces_w_history[:, -1, :, :]  # (N, S, 3)
    sensor_ids = sensor_cfg.body_ids
    forces_feet = forces_w_last[:, sensor_ids, :]  # (N, F, 3)
    force_norm = torch.norm(forces_feet, dim=-1)  # (N, F)

    air_time = contact_sensor.data.current_air_time[:, sensor_ids]  # (N, F)

    grounded_by_air = air_time < air_time_eps
    grounded_by_force = force_norm > force_eps

    # Match your reference behavior:
    in_contact = grounded_by_air | grounded_by_force  # (N, F) bool

    # --------------------------------------------------------
    # 2) Foot kinematics (world)
    # --------------------------------------------------------
    asset = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids  # (F,)

    body_lin_vel_w = asset.data.body_lin_vel_w[:, body_ids, :]  # (N, F, 3)
    lin_speed_xy = torch.norm(body_lin_vel_w[..., :2], dim=-1)  # (N, F)

    body_ang_vel_w = asset.data.body_ang_vel_w[:, body_ids, :]  # (N, F, 3)
    omega_xy_norm = torch.norm(body_ang_vel_w[..., :2], dim=-1)  # (N, F)
    yaw_rate = torch.abs(body_ang_vel_w[..., 2])  # (N, F)

    # --------------------------------------------------------
    # 3) Tilt cost: optionally allow "uprighting" direction
    # --------------------------------------------------------
    if local_up_axis is None:
        local_up_axis = torch.tensor([0.0, 0.0, 1.0], device=body_ang_vel_w.device, dtype=body_ang_vel_w.dtype)

    if allow_uprighting_tilt:
        # Need foot orientation in world to get u = R * local_up
        # Isaac Lab typically provides body_quat_w as (N, B, 4) with (x,y,z,w).
        body_quat_w = asset.data.body_quat_w[:, body_ids, :]  # (N, F, 4)

        local_up = local_up_axis.view(1, 1, 3).expand(body_quat_w.shape[0], body_quat_w.shape[1], 3)
        u = _quat_apply(body_quat_w, local_up)  # (N, F, 3) foot up in world

        z = torch.zeros_like(u)
        z[..., 2] = 1.0  # world up

        # s = d/dt(u·z) = ω · (u × z)
        # If s < 0 => u·z decreases => tilt gets worse => penalize (-s)
        u_cross_z = torch.cross(u, z, dim=-1)  # (N, F, 3)
        s = torch.sum(body_ang_vel_w * u_cross_z, dim=-1)  # (N, F)
        tilt_cost = torch.relu(-s)  # only "tilt-worsening" component
    else:
        tilt_cost = omega_xy_norm  # penalize all tilt rate

    # --------------------------------------------------------
    # 4) Total cost (contact-masked)
    # --------------------------------------------------------
    cost_per_foot = (lin_weight * lin_speed_xy + tilt_weight * tilt_cost + yaw_weight * yaw_rate)

    cost_per_foot = cost_per_foot * in_contact.float()
    penalty = torch.sum(cost_per_foot, dim=1)  # (N,)

    return penalty


def feet_air_time_balanced_alternating_biped(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    *,
    # --- gating thresholds (set < 0 to disable each gate) ---
    linear_cmd_threshold: float = 1.0,
    angular_cmd_threshold: float = 1.0,
    body_tilt_threshold: float = 0.35,

    # --- air/contact time constraints ---
    air_min_time: float = 0.15,
    air_max_time: float = 0.45,
    min_contact_time: float = 0.10,

    # --- symmetry control using EMA (multiplicative downscaler) ---
    ema_alpha: float = 0.02,
    balance_weight: float = 0.5,

    # --- shaping rewards ---
    air_reward: float = 1.0,
    contact_reward: float = 1.0,

    # --- alternation rewards ---
    alternation_reward_scale: float = 1.0,
    non_alternation_reward_scale: float = 0.2,
) -> torch.Tensor:
    """
    Pure reward-scaling design:
      - All terms are non-negative.
      - Air reward ramps to 1 at air_min_time, then stays constant until air_max_time.
      - Contact reward ramps to 1 at min_contact_time, then stays constant.
      - Double flight and timeout zero the reward.
      - Symmetry imbalance smoothly downscales reward.

    Step-state machine completed:
      - updates prev_step_air_foot on touchdown of the active foot
      - clears active_air_foot on touchdown of the active foot
      - clears air_timed_out when a step completes (touchdown_active)
      - stores last_completed_air for the completed (active) foot on touchdown_active

    Change requested:
      - reward_contact_hold also applies during double_contact (both feet in contact),
        using stance_time = min(contact_time_left, contact_time_right).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_ids = sensor_cfg.body_ids
    assert len(foot_ids) == 2

    air_time = contact_sensor.data.current_air_time[:, foot_ids]
    contact_time = contact_sensor.data.current_contact_time[:, foot_ids]
    in_contact = contact_time > 0.0

    n_env = in_contact.shape[0]
    device = in_contact.device

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------
    key = '_feet_air_contact_reward_buf'
    if not hasattr(env, key):
        setattr(
            env, key, {
                'prev_in_contact': in_contact.clone(),
                'last_completed_air': torch.zeros((n_env, 2), device=device),
                'ema_air': torch.zeros((n_env, 2), device=device),
                'ema_contact': torch.zeros((n_env, 2), device=device),
                'active_air_foot': torch.full((n_env,), -1, device=device, dtype=torch.long),
                'prev_step_air_foot': torch.full((n_env,), -1, device=device, dtype=torch.long),
                'air_timed_out': torch.zeros((n_env,), device=device, dtype=torch.bool),
            })

    buf = getattr(env, key)
    prev_in_contact = buf['prev_in_contact']
    last_completed_air = buf['last_completed_air']
    ema_air = buf['ema_air']
    ema_contact = buf['ema_contact']
    active_air_foot = buf['active_air_foot']
    prev_step_air_foot = buf['prev_step_air_foot']
    air_timed_out = buf['air_timed_out']

    # ------------------------------------------------------------------
    # Contact transitions
    # ------------------------------------------------------------------
    lift_off = prev_in_contact & (~in_contact)
    touch_down = (~prev_in_contact) & in_contact

    # (Kept as-is; step-completion storage is handled later using touchdown_active.)
    last_completed_air = torch.where(touch_down, air_time, last_completed_air)

    # ------------------------------------------------------------------
    # Gating
    # ------------------------------------------------------------------
    cmd = env.command_manager.get_command(command_name)
    lin_mag = torch.norm(cmd[:, :2], dim=1)
    yaw_mag = torch.abs(cmd[:, 2]) if cmd.shape[1] > 2 else torch.zeros_like(lin_mag)

    lin_scale = torch.clamp(lin_mag / max(linear_cmd_threshold, 1e-6), 0, 1) if linear_cmd_threshold >= 0 else 0
    yaw_scale = torch.clamp(yaw_mag / max(angular_cmd_threshold, 1e-6), 0, 1) if angular_cmd_threshold >= 0 else 0

    tilt_scale = torch.zeros_like(lin_mag)
    if body_tilt_threshold >= 0:
        robot = env.scene.articulations['robot']
        proj_g = robot.data.projected_gravity_b
        sin_tilt = torch.norm(proj_g[:, :2], dim=1).clamp(0, 1)
        tilt_angle = torch.asin(sin_tilt)
        tilt_scale = torch.clamp(tilt_angle / max(body_tilt_threshold, 1e-6), 0, 1)

    if linear_cmd_threshold < 0 and angular_cmd_threshold < 0 and body_tilt_threshold < 0:
        gate_scale = torch.ones_like(lin_mag)
    else:
        gate_scale = torch.maximum(torch.maximum(lin_scale, yaw_scale), tilt_scale)

    # ------------------------------------------------------------------
    # Contact classification
    # ------------------------------------------------------------------
    contact_count = in_contact.int().sum(dim=1)
    single_contact = contact_count == 1
    double_contact = contact_count == 2
    double_flight = contact_count == 0

    # ------------------------------------------------------------------
    # Air phase logic
    # ------------------------------------------------------------------
    lift_count = lift_off.int().sum(dim=1)
    lifted_foot = torch.where(lift_off[:, 0], 0, torch.where(lift_off[:, 1], 1, -1))

    lifted_contact_time = torch.where(lifted_foot == 0, contact_time[:, 0],
                                      torch.where(lifted_foot == 1, contact_time[:, 1], torch.zeros_like(lin_mag)))

    liftoff_contact_ok = lifted_contact_time >= min_contact_time

    start_air_phase = ((active_air_foot < 0) & (lift_count == 1) & single_contact & (~air_timed_out) &
                       (lifted_foot >= 0) & liftoff_contact_ok)

    active_air_foot = torch.where(start_air_phase, lifted_foot, active_air_foot)

    has_prev_step = prev_step_air_foot >= 0
    liftoff_alternation_ok = (~has_prev_step) | (lifted_foot != prev_step_air_foot)

    reward_liftoff = air_reward * start_air_phase.float() * (
        non_alternation_reward_scale +
        (alternation_reward_scale - non_alternation_reward_scale) * liftoff_alternation_ok.float())

    # ------------------------------------------------------------------
    # Air progress (ramp → constant)
    # ------------------------------------------------------------------
    active_air_time = torch.where(active_air_foot == 0, air_time[:, 0],
                                  torch.where(active_air_foot == 1, air_time[:, 1], torch.zeros_like(lin_mag)))

    air_progress = torch.clamp(active_air_time / max(air_min_time, 1e-6), 0, 1)

    timeout_now = (active_air_foot >= 0) & (active_air_time > air_max_time)
    air_timed_out = air_timed_out | timeout_now

    # ------------------------------------------------------------------
    # Contact progress (ramp → constant)
    #
    # - single_contact: stance_time = contact_time of the stance (opposite) foot
    # - double_contact: stance_time = min(contact_time_left, contact_time_right)
    # ------------------------------------------------------------------
    stance_time_single = torch.where(active_air_foot == 0, contact_time[:, 1],
                                     torch.where(active_air_foot == 1, contact_time[:, 0], torch.zeros_like(lin_mag)))

    stance_time_double = torch.minimum(contact_time[:, 0], contact_time[:, 1])
    stance_time = torch.where(double_contact, stance_time_double, stance_time_single)

    contact_progress = torch.clamp(stance_time / max(min_contact_time, 1e-6), 0, 1)

    reward_air_hold = air_reward * air_progress * single_contact.float()

    # ★ requested change: apply also on double_contact
    reward_contact_hold = contact_reward * contact_progress * (single_contact | double_contact).float()

    # ------------------------------------------------------------------
    # Alternation shaping
    # ------------------------------------------------------------------
    alternation_ok = (active_air_foot >= 0) & has_prev_step & (active_air_foot != prev_step_air_foot)
    alternation_scale = torch.where(alternation_ok, torch.full_like(lin_mag, alternation_reward_scale),
                                    torch.full_like(lin_mag, non_alternation_reward_scale))

    reward_alternation = alternation_scale * single_contact.float()

    # ------------------------------------------------------------------
    # Downscaling (no negatives)
    # ------------------------------------------------------------------
    double_flight_scale = (~double_flight).float()
    timeout_scale = (~air_timed_out).float()

    ema_air = (1 - ema_alpha) * ema_air + ema_alpha * last_completed_air
    ema_contact = (1 - ema_alpha) * ema_contact + ema_alpha * contact_time

    imbalance = torch.abs(ema_air[:, 0] - ema_air[:, 1]) + torch.abs(ema_contact[:, 0] - ema_contact[:, 1])
    balance_scale = torch.exp(-balance_weight * imbalance).clamp_min(0)

    # ------------------------------------------------------------------
    # Total reward
    # ------------------------------------------------------------------
    base_reward = reward_liftoff + reward_air_hold + reward_contact_hold + reward_alternation
    reward = base_reward * gate_scale * double_flight_scale * timeout_scale * balance_scale

    # ------------------------------------------------------------------
    # STEP COMPLETION: close the state machine on touchdown of active foot
    # ------------------------------------------------------------------
    active_is_0 = active_air_foot == 0
    active_is_1 = active_air_foot == 1
    touchdown_active = (active_is_0 & touch_down[:, 0]) | (active_is_1 & touch_down[:, 1])

    completed_air_value = torch.where(active_is_0, air_time[:, 0],
                                      torch.where(active_is_1, air_time[:, 1], torch.zeros_like(lin_mag)))

    mask = touchdown_active & (active_air_foot >= 0)
    if mask.any():
        idx = active_air_foot.clamp(min=0)
        last_completed_air = last_completed_air.clone()
        last_completed_air[mask, idx[mask]] = completed_air_value[mask]

    prev_step_air_foot = torch.where(touchdown_active, active_air_foot, prev_step_air_foot)
    active_air_foot = torch.where(touchdown_active, torch.full_like(active_air_foot, -1), active_air_foot)

    air_timed_out = torch.where(touchdown_active, torch.zeros_like(air_timed_out), air_timed_out)

    # ------------------------------------------------------------------
    # Save buffers
    # ------------------------------------------------------------------
    buf['prev_in_contact'] = in_contact
    buf['last_completed_air'] = last_completed_air
    buf['ema_air'] = ema_air
    buf['ema_contact'] = ema_contact
    buf['active_air_foot'] = active_air_foot
    buf['prev_step_air_foot'] = prev_step_air_foot
    buf['air_timed_out'] = air_timed_out

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
    foot_in_contact = foot_forces.norm(dim=-1).amax(dim=1) > contact_force_threshold  # (N, F)

    any_foot_contact = foot_in_contact.any(dim=1)  # (N,)
    both_feet_air = ~any_foot_contact  # (N,)

    n_env = both_feet_air.shape[0]
    device = both_feet_air.device

    # Persistent buffers on env to track continuous durations across steps.
    key = '_both_feet_flight_time_buf'
    if not hasattr(env, key):
        buf = {
            'flight_time': torch.zeros((n_env,), device=device),
            'grounded_time': torch.zeros((n_env,), device=device),
        }
        setattr(env, key, buf)

    buf = getattr(env, key)
    flight_time = buf['flight_time']
    grounded_time = buf['grounded_time']

    # Episode reset handling (if available)
    reset_buf = getattr(env, 'reset_buf', None)
    if reset_buf is not None:
        reset_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            flight_time[reset_ids] = 0.0
            grounded_time[reset_ids] = 0.0

    # Update continuous duration counters
    flight_time = torch.where(both_feet_air, flight_time + dt, torch.zeros_like(flight_time))
    grounded_time = torch.where(~both_feet_air, grounded_time + dt, torch.zeros_like(grounded_time))

    buf['flight_time'] = flight_time
    buf['grounded_time'] = grounded_time

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
    other_in_contact = other_forces.norm(dim=-1).amax(dim=1) > contact_force_threshold  # (N, K)

    any_nonfoot_contact = other_in_contact.any(dim=1)  # (N,)

    n_env = any_nonfoot_contact.shape[0]
    device = any_nonfoot_contact.device

    # Persistent buffers on env to track continuous durations across steps.
    key = '_nonfoot_contact_time_buf'
    if not hasattr(env, key):
        buf = {
            'bad_time': torch.zeros((n_env,), device=device),
            'clear_time': torch.zeros((n_env,), device=device),
        }
        setattr(env, key, buf)

    buf = getattr(env, key)
    bad_time = buf['bad_time']
    clear_time = buf['clear_time']

    # Episode reset handling (if available)
    reset_buf = getattr(env, 'reset_buf', None)
    if reset_buf is not None:
        reset_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            bad_time[reset_ids] = 0.0
            clear_time[reset_ids] = 0.0

    # Update continuous duration counters
    bad_time = torch.where(any_nonfoot_contact, bad_time + dt, torch.zeros_like(bad_time))
    clear_time = torch.where(~any_nonfoot_contact, clear_time + dt, torch.zeros_like(clear_time))

    buf['bad_time'] = bad_time
    buf['clear_time'] = clear_time

    # Penalty and reward
    penalty = -penalty_scale * bad_time
    max_time = max_reward / max(reward_scale, 1e-6)
    bonus = reward_scale * torch.clamp(clear_time, max=max_time)

    return torch.where(any_nonfoot_contact, penalty, bonus)


def prevent_both_feet_airborne(
    env,
    sensor_cfg: SceneEntityCfg,
    weight: float = 1.0,
    contact_time_eps: float = 1e-3,
    force_eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize states where no foot qualifies as 'in contact'.

    A foot is considered in contact only if:
      - current_contact_time > contact_time_eps
      - AND contact force norm > force_eps

    This discourages lifting a foot immediately after touching down, because
    very short contacts won't count as support.

    Returns:
      penalty (N,) positive only when zero feet qualify as in contact.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    sensor_ids = sensor_cfg.body_ids  # feet sensor indices (F,)

    forces_w_last = contact_sensor.data.net_forces_w_history[:, -1, :, :]  # (N, S, 3)
    forces_feet = forces_w_last[:, sensor_ids, :]  # (N, F, 3)
    force_norm = torch.norm(forces_feet, dim=-1)  # (N, F)

    contact_time = contact_sensor.data.current_contact_time[:, sensor_ids]  # (N, F)

    grounded_by_time = contact_time > contact_time_eps
    grounded_by_force = force_norm > force_eps
    in_contact = grounded_by_time & grounded_by_force  # (N, F)

    any_contact = torch.any(in_contact, dim=1)  # (N,)
    penalty = weight * torch.logical_not(any_contact).float()
    return penalty


def prevent_both_feet_airborne_linear(
    env,
    sensor_cfg: SceneEntityCfg,
    contact_time_eps: float = 1e-3,
    progress_reward_weight: float = 1.0,
    airborne_penalty_weight: float = 1.0,
    squared: bool = False,
) -> torch.Tensor:
    """
    Linear shaping using only current_contact_time.

    (A) Progress reward (linear for 0 <= contact_time < contact_time_eps)
    (B) Airborne penalty (linear, only when no support foot exists)

    If squared=True:
        Final penalty is squared (nonlinear amplification).

    Returns:
        penalty (N,) positive.
        (Progress is internally subtracted as negative penalty.)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    sensor_ids = sensor_cfg.body_ids

    contact_time = contact_sensor.data.current_contact_time[:, sensor_ids]

    # ---------------------------
    # (A) Linear progress reward
    # ---------------------------
    progress = torch.clamp(contact_time / contact_time_eps, 0.0, 1.0)
    progress = progress * (contact_time < contact_time_eps).float()
    progress_term = -progress_reward_weight * torch.sum(progress, dim=1)

    # ---------------------------
    # (B) Linear airborne penalty
    # ---------------------------
    support_foot = contact_time >= contact_time_eps
    has_support = torch.any(support_foot, dim=1)

    missing = torch.clamp(
        (contact_time_eps - contact_time) / contact_time_eps,
        0.0,
        1.0,
    )
    airborne_amount = torch.sum(missing, dim=1)

    airborne_term = airborne_penalty_weight * airborne_amount * (~has_support).float()

    penalty = progress_term + airborne_term

    if squared:
        penalty = penalty * penalty

    return penalty
