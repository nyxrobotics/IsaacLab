from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg


def _has_non_finite(x: torch.Tensor) -> torch.Tensor:
    reduce_dims = tuple(range(1, x.ndim))
    return ~torch.isfinite(x).all(dim=reduce_dims)


def detect_fall(
    env,
    limit_angle: float,
    max_lin_vel: float = 1e3,
    max_ang_vel: float = 1e3,
    max_lin_acc: float = 1e6,
    max_ang_acc: float = 1e6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    proj_g = asset.data.projected_gravity_b
    acos_arg = torch.clamp(-proj_g[:, 2], -1.0, 1.0)
    tilt = torch.acos(acos_arg).abs()
    bad_tilt = (tilt > limit_angle) | (~torch.isfinite(tilt))

    body_pos = asset.data.body_pos_w
    body_quat = asset.data.body_quat_w
    body_vel = asset.data.body_vel_w
    body_acc = asset.data.body_acc_w
    root_pose = asset.data.root_link_pose_w
    root_vel = asset.data.root_link_vel_w

    bad_numeric = _has_non_finite(body_pos) | _has_non_finite(body_quat) | _has_non_finite(body_vel) | _has_non_finite(body_acc) | _has_non_finite(root_pose) | _has_non_finite(root_vel) | _has_non_finite(proj_g)

    lin_vel = body_vel[..., 0:3]
    ang_vel = body_vel[..., 3:6]
    lin_acc = body_acc[..., 0:3]
    ang_acc = body_acc[..., 3:6]
    lin_vel_norm = torch.norm(lin_vel, dim=-1)
    ang_vel_norm = torch.norm(ang_vel, dim=-1)
    lin_acc_norm = torch.norm(lin_acc, dim=-1)
    ang_acc_norm = torch.norm(ang_acc, dim=-1)
    bad_norms = _has_non_finite(lin_vel_norm) | _has_non_finite(ang_vel_norm) | _has_non_finite(lin_acc_norm) | _has_non_finite(ang_acc_norm)

    bad_velocity = torch.any(lin_vel_norm > max_lin_vel, dim=1) | torch.any(ang_vel_norm > max_ang_vel, dim=1)
    bad_acceleration = torch.any(lin_acc_norm > max_lin_acc, dim=1) | torch.any(ang_acc_norm > max_ang_acc, dim=1)
    root_lin_vel_norm = torch.norm(root_vel[:, 0:3], dim=-1)
    root_ang_vel_norm = torch.norm(root_vel[:, 3:6], dim=-1)
    bad_root_norms = (~torch.isfinite(root_lin_vel_norm)) | (~torch.isfinite(root_ang_vel_norm))
    bad_velocity = bad_velocity | (root_lin_vel_norm > max_lin_vel) | (root_ang_vel_norm > max_ang_vel)
    return bad_tilt | bad_numeric | bad_norms | bad_root_norms | bad_velocity | bad_acceleration


def detect_height_too_low_relative(
    env,
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if body_ids is None or isinstance(body_ids, slice) or len(body_ids) != 3:
        raise ValueError("detect_height_too_low_relative expects 3 body ids [torso, r_foot, l_foot].")
    torso_id, rfoot_id, lfoot_id = body_ids
    torso_z = asset.data.body_pos_w[:, torso_id, 2]
    rfoot_z = asset.data.body_pos_w[:, rfoot_id, 2]
    lfoot_z = asset.data.body_pos_w[:, lfoot_id, 2]
    dz = torch.maximum(torso_z - rfoot_z, torso_z - lfoot_z)
    return dz < min_height


def detect_tilt_too_high_any_link(
    env,
    max_tilt: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids if asset_cfg.body_ids is not None else slice(None)
    q = asset.data.body_quat_w[:, body_ids, :]

    def _quat_rotate_inverse_wxyz(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
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

    g_w = torch.tensor([0.0, 0.0, -1.0], device=q.device, dtype=q.dtype).view(1, 1, 3).expand(q.shape[0], q.shape[1], 3)
    proj_g_b = _quat_rotate_inverse_wxyz(q, g_w)
    acos_arg = torch.clamp(-proj_g_b[..., 2], -1.0, 1.0)
    tilt = torch.acos(acos_arg).abs()
    bad = (tilt > max_tilt) | (~torch.isfinite(tilt))
    return torch.any(bad, dim=1)


def detect_support_plane_tilt_too_high(
    env,
    max_tilt: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if body_ids is None or isinstance(body_ids, slice) or len(body_ids) != 3:
        raise ValueError("detect_support_plane_tilt_too_high expects 3 body ids [torso, r_foot, l_foot].")
    torso_id, rfoot_id, lfoot_id = body_ids
    p_t = asset.data.body_pos_w[:, torso_id, :]
    p_r = asset.data.body_pos_w[:, rfoot_id, :]
    p_l = asset.data.body_pos_w[:, lfoot_id, :]
    a = p_r - p_t
    b = p_l - p_t
    n = torch.cross(a, b, dim=-1)
    n_norm = torch.linalg.norm(n, dim=-1)
    eps = 1e-8
    sin_theta = torch.abs(n[:, 2]) / torch.clamp(n_norm, min=eps)
    sin_theta = torch.clamp(sin_theta, -1.0, 1.0)
    tilt = torch.asin(sin_theta)
    return (tilt > max_tilt) | (~torch.isfinite(tilt)) | (n_norm < eps)
