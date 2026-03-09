from __future__ import annotations

import torch

def joint_action_deviation_l1(env, joint_ids: list[int]) -> torch.Tensor:
    ids = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)
    action = env.action_manager.action.index_select(1, ids)
    return torch.sum(torch.abs(action), dim=1)

def joint_action_acc_l2(env, dt: float) -> torch.Tensor:
    acc = (env.action_manager.action - 2.0 * env.action_manager.prev_action + env.action_manager.prev_prev_action) / max(dt * dt, 1e-12)
    return torch.sum(torch.square(acc), dim=1)

def flat_orientation_links_l2(env, body_ids: list[int], margin: float = 0.0, gain: float = 1.0) -> torch.Tensor:
    q = env.robot.data.body_quat_w[:, body_ids]
    gravity = torch.zeros((q.shape[0], q.shape[1], 3), device=env.device, dtype=q.dtype)
    gravity[..., 2] = -1.0
    g_b = quat_apply_inverse_xyzw(q, gravity)
    tilt = torch.sum(torch.square(g_b[..., :2]), dim=-1)
    reward = torch.clamp(gain * (margin - tilt), min=0.0)
    return torch.mean(reward, dim=1)

def ang_vel_xy_links_l2(env, body_ids: list[int]) -> torch.Tensor:
    ang = env.robot.data.body_ang_vel_w[:, body_ids, :2]
    return torch.sum(torch.square(ang), dim=(1, 2))

@torch.jit.script
def quat_apply_inverse_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v - q_w * t + torch.cross(q_xyz, t, dim=-1)
