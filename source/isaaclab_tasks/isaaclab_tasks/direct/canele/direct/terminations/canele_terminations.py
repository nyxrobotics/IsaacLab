from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg


def _resolve_body_ids(asset_cfg: SceneEntityCfg | None = None, body_id: int | None = None, body_ids: list[int] | None = None) -> torch.Tensor:
    if body_id is not None:
        return torch.tensor([body_id], dtype=torch.long)
    if body_ids is not None:
        return torch.as_tensor(body_ids, dtype=torch.long)
    if asset_cfg is None or getattr(asset_cfg, 'body_ids', None) is None:
        raise ValueError('body ids are required for direct terminations')
    return torch.as_tensor(asset_cfg.body_ids, dtype=torch.long)


def _has_non_finite(x: torch.Tensor) -> torch.Tensor:
    reduce_dims = tuple(range(1, x.ndim))
    return ~torch.isfinite(x).all(dim=reduce_dims)


def detect_fall(env, limit_angle: float, max_lin_vel: float = 1e3, max_ang_vel: float = 1e3, max_lin_acc: float = 1e6, max_ang_acc: float = 1e6, asset_cfg: SceneEntityCfg | None = None, body_id: int | None = None) -> torch.Tensor:
    root_ids = _resolve_body_ids(asset_cfg, body_id=body_id).to(env.device)
    root_id = int(root_ids[0].item())

    proj_g = env.robot.data.projected_gravity_b
    acos_arg = torch.clamp(-proj_g[:, 2], -1.0, 1.0)
    tilt = torch.acos(acos_arg).abs()
    bad_tilt = (tilt > limit_angle) | (~torch.isfinite(tilt))

    body_pos = env.robot.data.body_pos_w
    body_quat = env.robot.data.body_quat_w
    body_lin_vel = env.robot.data.body_lin_vel_w
    body_ang_vel = env.robot.data.body_ang_vel_w
    if hasattr(env.robot.data, 'body_acc_w'):
        body_lin_acc = env.robot.data.body_acc_w[..., 0:3]
        body_ang_acc = env.robot.data.body_acc_w[..., 3:6]
    else:
        z = torch.zeros_like(body_lin_vel)
        body_lin_acc = z
        body_ang_acc = z

    root_pose = torch.cat((env.robot.data.body_pos_w[:, root_id, :], env.robot.data.body_quat_w[:, root_id, :]), dim=1)
    root_vel = torch.cat((env.robot.data.body_lin_vel_w[:, root_id, :], env.robot.data.body_ang_vel_w[:, root_id, :]), dim=1)

    bad_numeric = (
        _has_non_finite(body_pos)
        | _has_non_finite(body_quat)
        | _has_non_finite(body_lin_vel)
        | _has_non_finite(body_ang_vel)
        | _has_non_finite(body_lin_acc)
        | _has_non_finite(body_ang_acc)
        | _has_non_finite(root_pose)
        | _has_non_finite(root_vel)
        | _has_non_finite(proj_g)
    )

    lin_vel_norm = torch.norm(body_lin_vel, dim=-1)
    ang_vel_norm = torch.norm(body_ang_vel, dim=-1)
    lin_acc_norm = torch.norm(body_lin_acc, dim=-1)
    ang_acc_norm = torch.norm(body_ang_acc, dim=-1)
    bad_norms = _has_non_finite(lin_vel_norm) | _has_non_finite(ang_vel_norm) | _has_non_finite(lin_acc_norm) | _has_non_finite(ang_acc_norm)
    bad_velocity = torch.any(lin_vel_norm > max_lin_vel, dim=1) | torch.any(ang_vel_norm > max_ang_vel, dim=1)
    bad_acceleration = torch.any(lin_acc_norm > max_lin_acc, dim=1) | torch.any(ang_acc_norm > max_ang_acc, dim=1)

    root_lin_vel_norm = torch.norm(env.robot.data.body_lin_vel_w[:, root_id, :], dim=-1)
    root_ang_vel_norm = torch.norm(env.robot.data.body_ang_vel_w[:, root_id, :], dim=-1)
    bad_root_norms = (~torch.isfinite(root_lin_vel_norm)) | (~torch.isfinite(root_ang_vel_norm))
    bad_velocity = bad_velocity | (root_lin_vel_norm > max_lin_vel) | (root_ang_vel_norm > max_ang_vel)

    return bad_tilt | bad_numeric | bad_norms | bad_root_norms | bad_velocity | bad_acceleration


def detect_height_too_low_relative(env, min_height: float, asset_cfg: SceneEntityCfg | None = None, base_id: int | None = None, foot_ids: list[int] | None = None) -> torch.Tensor:
    if asset_cfg is not None:
        ids = _resolve_body_ids(asset_cfg).to(env.device)
        if ids.numel() != 3:
            raise ValueError('detect_height_too_low_relative expects [torso, right_foot, left_foot]')
        torso_id, rfoot_id, lfoot_id = [int(v.item()) for v in ids]
    else:
        if base_id is None or foot_ids is None or len(foot_ids) != 2:
            raise ValueError('provide asset_cfg or base_id+foot_ids')
        torso_id, rfoot_id, lfoot_id = base_id, foot_ids[0], foot_ids[1]
    torso_z = env.robot.data.body_pos_w[:, torso_id, 2]
    rfoot_z = env.robot.data.body_pos_w[:, rfoot_id, 2]
    lfoot_z = env.robot.data.body_pos_w[:, lfoot_id, 2]
    dz = torch.maximum(torso_z - rfoot_z, torso_z - lfoot_z)
    return dz < min_height


def detect_tilt_too_high_any_link(env, max_tilt: float, asset_cfg: SceneEntityCfg | None = None, body_ids: list[int] | None = None) -> torch.Tensor:
    ids = _resolve_body_ids(asset_cfg, body_ids=body_ids).to(env.device)
    q = env.robot.data.body_quat_w.index_select(1, ids)
    g_w = torch.tensor([0.0, 0.0, -1.0], device=env.device, dtype=q.dtype).view(1, 1, 3).expand(q.shape[0], q.shape[1], 3)
    proj_g_b = quat_apply_inverse_wxyz(q, g_w)
    acos_arg = torch.clamp(-proj_g_b[..., 2], -1.0, 1.0)
    tilt = torch.acos(acos_arg).abs()
    return torch.any((tilt > max_tilt) | (~torch.isfinite(tilt)), dim=1)


def detect_support_plane_tilt_too_high(env, max_tilt: float, asset_cfg: SceneEntityCfg | None = None, foot_ids: list[int] | None = None) -> torch.Tensor:
    ids = _resolve_body_ids(asset_cfg, body_ids=foot_ids).to(env.device)
    if ids.numel() != 3:
        raise ValueError('detect_support_plane_tilt_too_high expects [torso, right_foot, left_foot]')
    torso_id, rfoot_id, lfoot_id = [int(v.item()) for v in ids]
    p_t = env.robot.data.body_pos_w[:, torso_id, :]
    p_r = env.robot.data.body_pos_w[:, rfoot_id, :]
    p_l = env.robot.data.body_pos_w[:, lfoot_id, :]
    a = p_r - p_t
    b = p_l - p_t
    n = torch.cross(a, b, dim=-1)
    n_norm = torch.linalg.norm(n, dim=-1)
    eps = 1e-8
    sin_theta = torch.abs(n[:, 2]) / torch.clamp(n_norm, min=eps)
    sin_theta = torch.clamp(sin_theta, -1.0, 1.0)
    tilt = torch.asin(sin_theta)
    return (tilt > max_tilt) | (~torch.isfinite(tilt)) | (n_norm < eps)


def quat_apply_inverse_wxyz(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qw = q_wxyz[..., 0:1]
    qx = q_wxyz[..., 1:2]
    qy = q_wxyz[..., 2:3]
    qz = q_wxyz[..., 3:4]
    cx = -qx
    cy = -qy
    cz = -qz
    tx = 2.0 * (cy * v[..., 2:3] - cz * v[..., 1:2])
    ty = 2.0 * (cz * v[..., 0:1] - cx * v[..., 2:3])
    tz = 2.0 * (cx * v[..., 1:2] - cy * v[..., 0:1])
    cxt = cy * tz - cz * ty
    cyt = cz * tx - cx * tz
    czt = cx * ty - cy * tx
    return torch.cat([v[..., 0:1] + qw * tx + cxt, v[..., 1:2] + qw * ty + cyt, v[..., 2:3] + qw * tz + czt], dim=-1)
