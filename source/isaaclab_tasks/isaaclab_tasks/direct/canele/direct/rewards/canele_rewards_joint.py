from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg


def _get_asset(env, asset_cfg: SceneEntityCfg | None):
    if asset_cfg is None:
        return env.robot
    return env.scene[asset_cfg.name]


def _resolve_action_indices(env, asset_cfg: SceneEntityCfg | None = None, joint_ids: list[int] | torch.Tensor | None = None):
    if joint_ids is not None:
        return torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)
    if asset_cfg is None:
        return None
    resolved = getattr(asset_cfg, 'joint_ids', None)
    if resolved is None:
        joint_names = getattr(asset_cfg, 'joint_names', None)
        if joint_names is None:
            return None
        name_to_idx = {name: i for i, name in enumerate(env.action_manager.get_term('joint_position')._joint_names)}
        resolved = [name_to_idx[str(name)] for name in joint_names]
        asset_cfg.joint_ids = list(resolved)
    return torch.as_tensor(resolved, device=env.device, dtype=torch.long)


def joint_action_deviation_l1(env, asset_cfg: SceneEntityCfg | None = None, default_action: torch.Tensor | None = None, joint_ids: list[int] | None = None) -> torch.Tensor:
    action_ids = _resolve_action_indices(env, asset_cfg, joint_ids)
    action = env.action_manager.action
    if action_ids is not None:
        action = action.index_select(1, action_ids)
    if default_action is None:
        default_sel = torch.zeros_like(action)
    else:
        if default_action.dim() == 1:
            default_sel = default_action.unsqueeze(0)
            if action_ids is not None and default_action.numel() == env.action_manager.action.shape[1]:
                default_sel = default_action.index_select(0, action_ids).unsqueeze(0)
        else:
            default_sel = default_action
            if action_ids is not None and default_action.shape[1] == env.action_manager.action.shape[1]:
                default_sel = default_action.index_select(1, action_ids)
    return torch.sum(torch.abs(action - default_sel), dim=1)


def joint_action_acc_l2(env, dt: float, asset_cfg: SceneEntityCfg | None = None, joint_ids: list[int] | None = None) -> torch.Tensor:
    action_ids = _resolve_action_indices(env, asset_cfg, joint_ids)
    action = env.action_manager.action
    prev = env.action_manager.prev_action
    prev_prev = env.action_manager.prev_prev_action
    if action_ids is not None:
        action = action.index_select(1, action_ids)
        prev = prev.index_select(1, action_ids)
        prev_prev = prev_prev.index_select(1, action_ids)
    acc = (action - 2.0 * prev + prev_prev) / max(dt * dt, 1e-12)
    return torch.sum(torch.square(acc), dim=1)


def _resolve_body_ids(asset_cfg: SceneEntityCfg | None = None, body_ids: list[int] | torch.Tensor | None = None):
    if body_ids is not None:
        return torch.as_tensor(body_ids, dtype=torch.long)
    if asset_cfg is None:
        raise ValueError('body_ids or asset_cfg is required')
    resolved = getattr(asset_cfg, 'body_ids', None)
    if resolved is None:
        raise ValueError('asset_cfg.body_ids must be resolved before use in direct rewards')
    return torch.as_tensor(resolved, dtype=torch.long)


def flat_orientation_links_l2(env, asset_cfg: SceneEntityCfg | None = None, body_ids: list[int] | None = None, margin: float = 0.3, gain: float = 1.0) -> torch.Tensor:
    ids = _resolve_body_ids(asset_cfg, body_ids).to(env.device)
    q = env.robot.data.body_quat_w.index_select(1, ids)
    g_w = torch.zeros((q.shape[0], q.shape[1], 3), device=env.device, dtype=q.dtype)
    g_w[..., 2] = -1.0
    g_b = quat_apply_inverse_xyzw(q, g_w)
    l2_per_body = torch.sum(g_b[..., :2] * g_b[..., :2], dim=-1)
    excess_per_body = torch.relu(torch.sqrt(l2_per_body) - margin)
    penalty_per_body = -gain * excess_per_body
    return torch.sum(penalty_per_body, dim=1)


def ang_vel_xy_links_l2(env, asset_cfg: SceneEntityCfg | None = None, body_ids: list[int] | None = None) -> torch.Tensor:
    ids = _resolve_body_ids(asset_cfg, body_ids).to(env.device)
    ang = env.robot.data.body_ang_vel_w.index_select(1, ids)[..., :2]
    return torch.sum(torch.square(ang), dim=(1, 2))


@torch.jit.script
def quat_apply_inverse_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v - q_w * t + torch.cross(q_xyz, t, dim=-1)
