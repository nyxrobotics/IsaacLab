from __future__ import annotations

import torch

def _yaw_only_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q is xyzw. extract yaw-only inverse rotation.
    x, y, z, w = q.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cy = torch.cos(-yaw)
    sy = torch.sin(-yaw)
    out = v.clone()
    out_x = cy * v[..., 0] - sy * v[..., 1]
    out_y = sy * v[..., 0] + cy * v[..., 1]
    out[..., 0] = out_x
    out[..., 1] = out_y
    return out

def _any_foot_in_contact(env, sensor_ids: list[int]) -> torch.Tensor:
    contact_sensor = env.scene.sensors["contact_forces"]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_ids]
    return (contact_time > 0.0).any(dim=1)

def track_lin_vel_xy_yaw_frame_exp_no_flight(env, std: float, command_name: str, sensor_ids: list[int]) -> torch.Tensor:
    any_contact = _any_foot_in_contact(env, sensor_ids)
    cmd = env.command_manager.get_command(command_name)
    root_vel = env.robot.data.root_com_vel_w[:, :3]
    root_quat = env.robot.data.root_quat_w
    vel_yaw = _yaw_only_inverse(root_quat, root_vel)[:, :2]
    err = torch.sum(torch.square(cmd[:, :2] - vel_yaw), dim=1)
    reward = torch.exp(-err / max(std * std, 1e-12))
    return reward * any_contact.float()

def track_ang_vel_z_world_exp_no_flight(env, command_name: str, std: float, sensor_ids: list[int]) -> torch.Tensor:
    any_contact = _any_foot_in_contact(env, sensor_ids)
    cmd = env.command_manager.get_command(command_name)
    yaw_cmd = cmd[:, 2]
    yaw_vel = env.robot.data.root_ang_vel_w[:, 2]
    err = torch.square(yaw_cmd - yaw_vel)
    reward = torch.exp(-err / max(std * std, 1e-12))
    return reward * any_contact.float()

def feet_air_time_alternating_biped(
    env, command_name: str, sensor_ids: list[int], *,
    linear_cmd_threshold: float = 0.0, angular_cmd_threshold: float = 0.0, body_tilt_threshold: float = 0.0,
    air_min_time: float = 0.1, air_max_time: float = 1.0, min_contact_time: float = 0.1, ema_alpha: float = 0.02,
    air_balance_weight: float = 1.0, contact_balance_weight: float = 0.0, air_reward: float = 1.0, contact_reward: float = 1.0
) -> torch.Tensor:
    contact_sensor = env.scene.sensors["contact_forces"]
    air_time = contact_sensor.data.current_air_time[:, sensor_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_ids]
    in_contact = contact_time > 0.0

    cmd = env.command_manager.get_command(command_name)
    lin_mag = torch.norm(cmd[:, :2], dim=1)
    yaw_mag = torch.abs(cmd[:, 2])
    gate = torch.maximum(
        torch.clamp(lin_mag / max(linear_cmd_threshold, 1e-6), 0, 1) if linear_cmd_threshold >= 0 else torch.zeros_like(lin_mag),
        torch.clamp(yaw_mag / max(angular_cmd_threshold, 1e-6), 0, 1) if angular_cmd_threshold >= 0 else torch.zeros_like(lin_mag),
    )
    if linear_cmd_threshold < 0 and angular_cmd_threshold < 0:
        gate = torch.ones_like(lin_mag)

    single_contact = in_contact.int().sum(dim=1) == 1
    alternating = in_contact[:, 0] ^ in_contact[:, 1]
    air_ok = torch.clamp((air_time - air_min_time) / max(air_max_time - air_min_time, 1e-6), 0, 1)
    air_score = torch.max(air_ok, dim=1).values * air_reward
    stance_score = torch.clamp(torch.min(contact_time[:, 0], contact_time[:, 1]) / max(min_contact_time, 1e-6), 0, 1) * contact_reward
    reward = torch.where(single_contact & alternating, air_score, stance_score)
    return reward * gate

def feet_slide_keep_flat(env, sensor_ids: list[int], body_ids: list[int], air_time_eps: float = 0.02) -> torch.Tensor:
    contact_sensor = env.scene.sensors["contact_forces"]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_ids]
    in_contact = contact_time > air_time_eps
    vel_xy = env.robot.data.body_lin_vel_w[:, body_ids, :2]
    slide = torch.norm(vel_xy, dim=-1) * in_contact.float()
    return torch.sum(slide, dim=1)
