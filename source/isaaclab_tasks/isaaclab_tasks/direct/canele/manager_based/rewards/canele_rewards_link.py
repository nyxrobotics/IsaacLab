# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


from isaaclab.assets.rigid_object.rigid_object import RigidObject
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.assets import RigidObject
from isaaclab.utils.math import quat_apply_inverse
import torch

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_z_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def flat_orientation_links_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    margin: float = 0.3,
    gain: float = 1.0,
) -> torch.Tensor:
    """
    Penalize non-flat orientation for multiple links with a tolerance margin.

    - Project gravity into body frames
    - Compute L2 norm of xy components (tilt magnitude)
    - If tilt <= margin → penalty = 0
    - If tilt > margin → penalty = -gain * (tilt - margin)
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # body orientations
    body_quat_w = asset.data.body_quat_w  # (N, B, 4)

    # world gravity expanded to match shape (N, B, 3)
    g_w = torch.zeros_like(asset.data.body_pos_w)
    g_w[..., 2] = -1.0

    # gravity in body frame
    g_b = quat_apply_inverse(body_quat_w, g_w)  # (N, B, 3)

    # select target bodies
    body_ids = asset_cfg.body_ids
    g_sel = g_b[:, body_ids, :2]  # (N, K, 2)

    # tilt magnitude: L2 norm of xy gravity components
    l2_per_body = torch.sum(g_sel * g_sel, dim=-1)  # (N, K)
    excess_per_body = torch.relu(torch.sqrt(l2_per_body) - margin)  # (N, K)
    # margin-based penalty
    penalty_per_body = -gain * excess_per_body  # (N, K)
    penalty_sum = torch.sum(penalty_per_body, dim=1)  # (N,)

    return penalty_sum


def ang_vel_xy_links_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize xy-axis angular velocity for multiple links using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :2]  # (N, K, 2)
    l2_per_body = torch.sum(ang_vel_w * ang_vel_w, dim=-1)  # (N, K)
    l2_sum = torch.sum(l2_per_body, dim=1)  # (N,)
    return l2_sum
