from __future__ import annotations

import torch

def quaternion_to_euler(quat: torch.Tensor) -> torch.Tensor:
    quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * (torch.pi / 2), torch.asin(sinp))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=-1)

def detect_fall(env, body_id: int, limit_angle: float = 1.3, max_lin_vel: float = 1e3, max_ang_vel: float = 1e3) -> torch.Tensor:
    quat = env.robot.data.body_quat_w[:, body_id]
    lin = env.robot.data.body_lin_vel_w[:, body_id]
    ang = env.robot.data.body_ang_vel_w[:, body_id]
    euler = quaternion_to_euler(quat)
    tilt = torch.linalg.norm(euler[:, :2], dim=1)
    bad = (tilt > limit_angle) | (~torch.isfinite(tilt))
    bad = bad | (torch.linalg.norm(lin, dim=1) > max_lin_vel)
    bad = bad | (torch.linalg.norm(ang, dim=1) > max_ang_vel)
    return bad

def detect_height_too_low_relative(env, base_id: int, foot_ids: list[int], min_height: float = 0.5) -> torch.Tensor:
    torso_z = env.robot.data.body_pos_w[:, base_id, 2]
    max_foot_z = env.robot.data.body_pos_w[:, foot_ids, 2].max(dim=1).values
    return (torso_z - max_foot_z) < min_height

def detect_tilt_too_high_any_link(env, body_ids: list[int], max_tilt: float = 1.5) -> torch.Tensor:
    q = env.robot.data.body_quat_w[:, body_ids]
    e = quaternion_to_euler(q.reshape(-1, 4)).reshape(q.shape[0], q.shape[1], 3)
    tilt = torch.linalg.norm(e[..., :2], dim=-1)
    return (tilt > max_tilt).any(dim=1)

def detect_support_plane_tilt_too_high(env, foot_ids: list[int], max_tilt: float = 1.3) -> torch.Tensor:
    feet_pos_w = env.robot.data.body_pos_w[:, foot_ids]
    dz = torch.abs(feet_pos_w[:, 0, 2] - feet_pos_w[:, 1, 2])
    dxy = torch.linalg.norm(feet_pos_w[:, 0, :2] - feet_pos_w[:, 1, :2], dim=1).clamp_min(1e-6)
    tilt = torch.atan2(dz, dxy)
    return tilt > max_tilt
