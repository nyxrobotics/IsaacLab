from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse


def lin_vel_z_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def flat_orientation_links_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    margin: float = 0.3,
    gain: float = 1.0,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids if asset_cfg.body_ids is not None else slice(None)

    body_quat_w = asset.data.body_quat_w
    g_w = torch.zeros_like(asset.data.body_pos_w)
    g_w[..., 2] = -1.0
    g_b = quat_apply_inverse(body_quat_w, g_w)
    g_sel = g_b[:, body_ids, :2]
    l2_per_body = torch.sum(g_sel * g_sel, dim=-1)
    excess_per_body = torch.relu(torch.sqrt(l2_per_body) - margin)
    penalty_per_body = -gain * excess_per_body
    return torch.sum(penalty_per_body, dim=1)


def ang_vel_xy_links_l2(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids if asset_cfg.body_ids is not None else slice(None)
    ang_vel_w = asset.data.body_ang_vel_w[:, body_ids, :2]
    l2_per_body = torch.sum(ang_vel_w * ang_vel_w, dim=-1)
    return torch.sum(l2_per_body, dim=1)
