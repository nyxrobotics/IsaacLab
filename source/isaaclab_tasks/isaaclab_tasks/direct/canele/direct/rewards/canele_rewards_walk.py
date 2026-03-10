from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg


def _resolve_sensor_ids(sensor_cfg: SceneEntityCfg | None = None, sensor_ids: list[int] | None = None):
    if sensor_ids is not None:
        return torch.as_tensor(sensor_ids, dtype=torch.long)
    if sensor_cfg is None or getattr(sensor_cfg, 'body_ids', None) is None:
        raise ValueError('sensor ids are required for direct rewards')
    return torch.as_tensor(sensor_cfg.body_ids, dtype=torch.long)


def _resolve_body_ids(asset_cfg: SceneEntityCfg | None = None, body_ids: list[int] | None = None):
    if body_ids is not None:
        return torch.as_tensor(body_ids, dtype=torch.long)
    if asset_cfg is None or getattr(asset_cfg, 'body_ids', None) is None:
        raise ValueError('body ids are required for direct rewards')
    return torch.as_tensor(asset_cfg.body_ids, dtype=torch.long)


def _quat_conjugate_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.stack((q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]), dim=-1)


def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2), dim=-1)


def _quat_rotate_inverse_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)
    return _quat_mul_wxyz(_quat_mul_wxyz(_quat_conjugate_wxyz(q), qv), q)[..., 1:]


def _yaw_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    out = torch.zeros_like(q)
    out[..., 0] = torch.cos(half)
    out[..., 3] = torch.sin(half)
    return out


def _any_foot_in_contact(env, sensor_cfg: SceneEntityCfg | None = None, sensor_ids: list[int] | None = None, contact_time_eps: float = 1e-3, force_eps: float = 1e-6) -> torch.Tensor:
    contact_sensor = env.scene.sensors['contact_forces'] if sensor_cfg is None else env.scene.sensors[sensor_cfg.name]
    ids = _resolve_sensor_ids(sensor_cfg, sensor_ids).to(env.device)
    forces_w_last = contact_sensor.data.net_forces_w_history[:, -1].index_select(1, ids)
    force_norm = torch.norm(forces_w_last, dim=-1)
    contact_time = contact_sensor.data.current_contact_time.index_select(1, ids)
    grounded_by_time = contact_time > contact_time_eps
    grounded_by_force = force_norm > force_eps
    return torch.any(grounded_by_time & grounded_by_force, dim=1)


def track_lin_vel_xy_yaw_frame_exp_no_flight(env, std: float, command_name: str, sensor_cfg: SceneEntityCfg | None = None, asset_cfg: SceneEntityCfg | None = None, contact_time_eps: float = 1e-3, force_eps: float = 1e-6, sensor_ids: list[int] | None = None) -> torch.Tensor:
    root_quat_w = env.robot.data.root_quat_w
    root_lin_vel_w = getattr(env.robot.data, 'root_lin_vel_w', None)
    if root_lin_vel_w is None:
        root_lin_vel_w = getattr(env.robot.data, 'root_com_vel_w')
    vel_yaw = _quat_rotate_inverse_wxyz(_yaw_quat_wxyz(root_quat_w), root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1)
    reward = torch.exp(-lin_vel_error / max(std * std, 1e-12))
    any_contact = _any_foot_in_contact(env, sensor_cfg=sensor_cfg, sensor_ids=sensor_ids, contact_time_eps=contact_time_eps, force_eps=force_eps)
    return reward * any_contact.float()


def track_ang_vel_z_world_exp_no_flight(env, command_name: str, std: float, sensor_cfg: SceneEntityCfg | None = None, asset_cfg: SceneEntityCfg | None = None, contact_time_eps: float = 1e-3, force_eps: float = 1e-6, sensor_ids: list[int] | None = None) -> torch.Tensor:
    root_ang_vel_w = getattr(env.robot.data, 'root_ang_vel_w', None)
    if root_ang_vel_w is None:
        root_ang_vel_w = env.robot.data.body_ang_vel_w[:, 0, :]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / max(std * std, 1e-12))
    any_contact = _any_foot_in_contact(env, sensor_cfg=sensor_cfg, sensor_ids=sensor_ids, contact_time_eps=contact_time_eps, force_eps=force_eps)
    return reward * any_contact.float()


def feet_slide_keep_flat(env, sensor_cfg: SceneEntityCfg | None = None, asset_cfg: SceneEntityCfg | None = None, lin_weight: float = 1.0, tilt_weight: float = 1.0, yaw_weight: float = 1.0, allow_uprighting_tilt: bool = True, air_time_eps: float = 1e-3, force_eps: float = 1e-6, local_up_axis: torch.Tensor | None = None, sensor_ids: list[int] | None = None, body_ids: list[int] | None = None) -> torch.Tensor:
    contact_sensor = env.scene.sensors['contact_forces'] if sensor_cfg is None else env.scene.sensors[sensor_cfg.name]
    sids = _resolve_sensor_ids(sensor_cfg, sensor_ids).to(env.device)
    bids = _resolve_body_ids(asset_cfg, body_ids).to(env.device)
    forces_w_last = contact_sensor.data.net_forces_w_history[:, -1].index_select(1, sids)
    force_norm = torch.norm(forces_w_last, dim=-1)
    air_time = contact_sensor.data.current_air_time.index_select(1, sids)
    in_contact = (air_time < air_time_eps) | (force_norm > force_eps)
    body_lin_vel_w = env.robot.data.body_lin_vel_w.index_select(1, bids)
    lin_speed_xy = torch.norm(body_lin_vel_w[..., :2], dim=-1)
    body_ang_vel_w = env.robot.data.body_ang_vel_w.index_select(1, bids)
    yaw_rate = torch.abs(body_ang_vel_w[..., 2])
    if local_up_axis is None:
        local_up_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=body_ang_vel_w.dtype)
    if allow_uprighting_tilt:
        body_quat_w = env.robot.data.body_quat_w.index_select(1, bids)
        local_up = local_up_axis.view(1, 1, 3).expand(body_quat_w.shape[0], body_quat_w.shape[1], 3)
        u = quat_apply_wxyz(body_quat_w, local_up)
        z = torch.zeros_like(u)
        z[..., 2] = 1.0
        s = torch.sum(body_ang_vel_w * torch.cross(u, z, dim=-1), dim=-1)
        tilt_cost = torch.relu(-s)
    else:
        tilt_cost = torch.norm(body_ang_vel_w[..., :2], dim=-1)
    cost_per_foot = (lin_weight * lin_speed_xy + tilt_weight * tilt_cost + yaw_weight * yaw_rate) * in_contact.float()
    return torch.sum(cost_per_foot, dim=1)


def feet_air_time_alternating_biped(env, command_name: str, sensor_cfg: SceneEntityCfg | None = None, *, dsp_ratio: float = 0.0, linear_cmd_threshold: float = 0.0, angular_cmd_threshold: float = 0.0, body_tilt_threshold: float = 0.0, air_min_time: float = 0.1, air_max_time: float = 1.0, min_contact_time: float = 0.1, ema_alpha: float = 0.02, air_balance_weight: float = 1.0, contact_balance_weight: float = 0.0, air_reward: float = 1.0, contact_reward: float = 1.0, alternation_reward_scale: float = 1.0, non_alternation_reward_scale: float = 0.2, sensor_ids: list[int] | None = None) -> torch.Tensor:
    contact_sensor = env.scene.sensors['contact_forces'] if sensor_cfg is None else env.scene.sensors[sensor_cfg.name]
    foot_ids = _resolve_sensor_ids(sensor_cfg, sensor_ids).to(env.device)
    assert foot_ids.numel() == 2
    air_time = contact_sensor.data.current_air_time.index_select(1, foot_ids)
    contact_time = contact_sensor.data.current_contact_time.index_select(1, foot_ids)
    in_contact = contact_time > 0.0
    n_env = in_contact.shape[0]
    device = in_contact.device
    key = '_feet_air_contact_reward_buf'
    if not hasattr(env, key):
        setattr(env, key, {'prev_in_contact': in_contact.clone(), 'last_completed_air': torch.zeros((n_env, 2), device=device), 'ema_air': torch.zeros((n_env, 2), device=device), 'ema_contact': torch.zeros((n_env, 2), device=device), 'active_air_foot': torch.full((n_env,), -1, device=device, dtype=torch.long), 'prev_step_air_foot': torch.full((n_env,), -1, device=device, dtype=torch.long), 'air_timed_out': torch.zeros((n_env,), device=device, dtype=torch.bool)})
    buf = getattr(env, key)
    prev_in_contact = buf['prev_in_contact']
    last_completed_air = buf['last_completed_air']
    ema_air = buf['ema_air']
    ema_contact = buf['ema_contact']
    active_air_foot = buf['active_air_foot']
    prev_step_air_foot = buf['prev_step_air_foot']
    air_timed_out = buf['air_timed_out']
    lift_off = prev_in_contact & (~in_contact)
    touch_down = (~prev_in_contact) & in_contact
    last_completed_air = torch.where(touch_down, air_time, last_completed_air)
    cmd = env.command_manager.get_command(command_name)
    lin_mag = torch.norm(cmd[:, :2], dim=1)
    yaw_mag = torch.abs(cmd[:, 2]) if cmd.shape[1] > 2 else torch.zeros_like(lin_mag)
    lin_scale = torch.clamp(lin_mag / max(linear_cmd_threshold, 1e-6), 0, 1) if linear_cmd_threshold >= 0 else torch.zeros_like(lin_mag)
    yaw_scale = torch.clamp(yaw_mag / max(angular_cmd_threshold, 1e-6), 0, 1) if angular_cmd_threshold >= 0 else torch.zeros_like(lin_mag)
    tilt_scale = torch.zeros_like(lin_mag)
    if body_tilt_threshold >= 0:
        proj_g = env.robot.data.projected_gravity_b
        sin_tilt = torch.norm(proj_g[:, :2], dim=1).clamp(0, 1)
        tilt_angle = torch.asin(sin_tilt)
        tilt_scale = torch.clamp(tilt_angle / max(body_tilt_threshold, 1e-6), 0, 1)
    gate_scale = torch.ones_like(lin_mag) if (linear_cmd_threshold < 0 and angular_cmd_threshold < 0 and body_tilt_threshold < 0) else torch.maximum(torch.maximum(lin_scale, yaw_scale), tilt_scale)
    contact_count = in_contact.int().sum(dim=1)
    single_contact = contact_count == 1
    double_contact = contact_count == 2
    double_flight = contact_count == 0
    lift_count = lift_off.int().sum(dim=1)
    lifted_foot = torch.where(lift_off[:, 0], 0, torch.where(lift_off[:, 1], 1, -1))
    lifted_contact_time = torch.where(lifted_foot == 0, contact_time[:, 0], torch.where(lifted_foot == 1, contact_time[:, 1], torch.zeros_like(lin_mag)))
    liftoff_contact_ok = lifted_contact_time >= min_contact_time
    has_prev_step = prev_step_air_foot >= 0
    prev_air_duration = torch.where(prev_step_air_foot == 0, last_completed_air[:, 0], torch.where(prev_step_air_foot == 1, last_completed_air[:, 1], torch.zeros_like(lin_mag)))
    prev_contact_since_touchdown = torch.where(prev_step_air_foot == 0, contact_time[:, 0], torch.where(prev_step_air_foot == 1, contact_time[:, 1], torch.zeros_like(lin_mag)))
    dsp_required_time = dsp_ratio * prev_air_duration
    dsp_ok = (dsp_ratio <= 0.0) | (~has_prev_step) | (prev_contact_since_touchdown >= dsp_required_time)
    start_air_phase = (active_air_foot < 0) & (lift_count == 1) & single_contact & (~air_timed_out) & (lifted_foot >= 0) & liftoff_contact_ok & dsp_ok
    active_air_foot = torch.where(start_air_phase, lifted_foot, active_air_foot)
    liftoff_alternation_ok = (~has_prev_step) | (lifted_foot != prev_step_air_foot)
    reward_liftoff = air_reward * start_air_phase.float() * (non_alternation_reward_scale + (alternation_reward_scale - non_alternation_reward_scale) * liftoff_alternation_ok.float())
    active_air_time = torch.where(active_air_foot == 0, air_time[:, 0], torch.where(active_air_foot == 1, air_time[:, 1], torch.zeros_like(lin_mag)))
    air_progress = torch.clamp(active_air_time / max(air_min_time, 1e-6), 0, 1)
    timeout_now = (active_air_foot >= 0) & (active_air_time > air_max_time)
    air_timed_out = air_timed_out | timeout_now
    stance_time_single = torch.where(active_air_foot == 0, contact_time[:, 1], torch.where(active_air_foot == 1, contact_time[:, 0], torch.zeros_like(lin_mag)))
    stance_time_double = torch.minimum(contact_time[:, 0], contact_time[:, 1])
    stance_time = torch.where(double_contact, stance_time_double, stance_time_single)
    contact_progress = torch.clamp(stance_time / max(min_contact_time, 1e-6), 0, 1)
    reward_air_hold = air_reward * air_progress * single_contact.float()
    reward_contact_hold = contact_reward * contact_progress * (single_contact | double_contact).float()
    alternation_ok = (active_air_foot >= 0) & has_prev_step & (active_air_foot != prev_step_air_foot)
    alternation_scale = torch.where(alternation_ok, torch.full_like(lin_mag, alternation_reward_scale), torch.full_like(lin_mag, non_alternation_reward_scale))
    reward_alternation = alternation_scale * single_contact.float()
    double_flight_scale = (~double_flight).float()
    timeout_scale = (~air_timed_out).float()
    ema_air = (1 - ema_alpha) * ema_air + ema_alpha * last_completed_air
    ema_contact = (1 - ema_alpha) * ema_contact + ema_alpha * contact_time
    imbalance = air_balance_weight * torch.abs(ema_air[:, 0] - ema_air[:, 1]) + contact_balance_weight * torch.abs(ema_contact[:, 0] - ema_contact[:, 1])
    balance_scale = torch.exp(-imbalance).clamp_min(0)
    reward = (reward_liftoff + reward_air_hold + reward_contact_hold + reward_alternation) * gate_scale * double_flight_scale * timeout_scale * balance_scale
    active_is_0 = active_air_foot == 0
    active_is_1 = active_air_foot == 1
    touchdown_active = (active_is_0 & touch_down[:, 0]) | (active_is_1 & touch_down[:, 1])
    completed_air_value = torch.where(active_is_0, air_time[:, 0], torch.where(active_is_1, air_time[:, 1], torch.zeros_like(lin_mag)))
    mask = touchdown_active & (active_air_foot >= 0)
    if mask.any():
        idx = active_air_foot.clamp(min=0)
        last_completed_air = last_completed_air.clone()
        last_completed_air[mask, idx[mask]] = completed_air_value[mask]
    prev_step_air_foot = torch.where(touchdown_active, active_air_foot, prev_step_air_foot)
    active_air_foot = torch.where(touchdown_active, torch.full_like(active_air_foot, -1), active_air_foot)
    air_timed_out = torch.where(touchdown_active, torch.zeros_like(air_timed_out), air_timed_out)
    buf['prev_in_contact'] = in_contact
    buf['last_completed_air'] = last_completed_air
    buf['ema_air'] = ema_air
    buf['ema_contact'] = ema_contact
    buf['active_air_foot'] = active_air_foot
    buf['prev_step_air_foot'] = prev_step_air_foot
    buf['air_timed_out'] = air_timed_out
    return reward


def quat_apply_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_xyz = q[..., 1:4]
    q_w = q[..., 0:1]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q_w * t + torch.cross(q_xyz, t, dim=-1)
