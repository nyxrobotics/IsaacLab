# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Common functions that can be used to activate certain terminations.

Pure direct-environment termination helpers for the Canele task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
import torch



def detect_fall(
    env: "object",
    limit_angle: float,
    max_lin_vel: float = 1e3,
    max_ang_vel: float = 1e3,
    max_lin_acc: float = 1e6,
    max_ang_acc: float = 1e6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot is in an unstable or exploded state.

    Policy:
    - Tilt is evaluated only from the first link (root) using projected_gravity_b.
    - bad_numeric checks NaN/Inf on:
        - all-link pose/velocity/acceleration tensors (body_*)
        - root_link_pose_w and root_link_vel_w
        - projected_gravity_b (used for tilt, and also numeric-critical)
    - Magnitude checks (velocity/acceleration) are done over all links, and root is explicitly included.
    - Additional guards:
        - non-finite tilt / norm results are treated as terminal
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # -----------------------------
    # Helper: "any NaN/Inf" over all non-env dimensions
    # -----------------------------
    def _has_non_finite(x: torch.Tensor) -> torch.Tensor:
        reduce_dims = tuple(range(1, x.ndim))
        return ~torch.isfinite(x).all(dim=reduce_dims)

    # -----------------------------
    # Tilt (root-only)
    # -----------------------------
    proj_g = asset.data.projected_gravity_b  # (N, 3)

    # acos argument should be in [-1, 1], but if proj_g is non-finite, result can be NaN.
    acos_arg = torch.clamp(-proj_g[:, 2], -1.0, 1.0)
    tilt = torch.acos(acos_arg).abs()

    # Treat non-finite tilt as a terminal condition as well.
    bad_tilt = (tilt > limit_angle) | (~torch.isfinite(tilt))

    # -----------------------------
    # All-link tensors
    # -----------------------------
    body_pos = asset.data.body_pos_w  # (N, B, 3)
    body_quat = asset.data.body_quat_w  # (N, B, 4)
    body_vel = asset.data.body_vel_w  # (N, B, 6) [lin_vel(3), ang_vel(3)]
    body_acc = asset.data.body_acc_w  # (N, B, 6) [lin_acc(3), ang_acc(3)]

    # -----------------------------
    # Root tensors (explicitly included)
    # -----------------------------
    root_pose = asset.data.root_link_pose_w  # (N, 7) [pos(3), quat(4)]
    root_vel = asset.data.root_link_vel_w  # (N, 6) [lin_vel(3), ang_vel(3)]

    # -----------------------------
    # bad_numeric: NaN / Inf checks (raw tensors)
    # -----------------------------
    bad_numeric = (
        _has_non_finite(body_pos)
        | _has_non_finite(body_quat)
        | _has_non_finite(body_vel)
        | _has_non_finite(body_acc)
        | _has_non_finite(root_pose)
        | _has_non_finite(root_vel)
        | _has_non_finite(proj_g)
    )

    # -----------------------------
    # Magnitude checks (all links)
    # -----------------------------
    lin_vel = body_vel[..., 0:3]
    ang_vel = body_vel[..., 3:6]
    lin_acc = body_acc[..., 0:3]
    ang_acc = body_acc[..., 3:6]

    lin_vel_norm = torch.norm(lin_vel, dim=-1)
    ang_vel_norm = torch.norm(ang_vel, dim=-1)
    lin_acc_norm = torch.norm(lin_acc, dim=-1)
    ang_acc_norm = torch.norm(ang_acc, dim=-1)

    # If norm results are non-finite, terminate as well.
    bad_norms = (
        _has_non_finite(lin_vel_norm)
        | _has_non_finite(ang_vel_norm)
        | _has_non_finite(lin_acc_norm)
        | _has_non_finite(ang_acc_norm)
    )

    bad_velocity = torch.any(lin_vel_norm > max_lin_vel, dim=1) | torch.any(
        ang_vel_norm > max_ang_vel, dim=1
    )

    bad_acceleration = torch.any(lin_acc_norm > max_lin_acc, dim=1) | torch.any(
        ang_acc_norm > max_ang_acc, dim=1
    )

    # Root velocity magnitude (explicitly included)
    root_lin_vel_norm = torch.norm(root_vel[:, 0:3], dim=-1)
    root_ang_vel_norm = torch.norm(root_vel[:, 3:6], dim=-1)

    bad_root_norms = (~torch.isfinite(root_lin_vel_norm)) | (
        ~torch.isfinite(root_ang_vel_norm)
    )

    bad_velocity = (
        bad_velocity
        | (root_lin_vel_norm > max_lin_vel)
        | (root_ang_vel_norm > max_ang_vel)
    )

    return (
        bad_tilt
        | bad_numeric
        | bad_norms
        | bad_root_norms
        | bad_velocity
        | bad_acceleration
    )


def detect_height_too_low(
    env: "object",
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot's root height is below the minimum height."""
    asset: RigidObject = env.scene[asset_cfg.name]
    root_height = asset.data.root_pos_w[:, 2]
    return root_height < min_height


def detect_height_too_low_relative(
    env: "object",
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the torso is too low relative to the feet.

    asset_cfg.body_names must specify exactly 3 bodies in this order:
      [torso_link, right_foot_tip, left_foot_tip]

    We compute:
      dz_r = torso_z - right_foot_z
      dz_l = torso_z - left_foot_z
      dz   = max(dz_r, dz_l)   # "the larger deviation w.r.t. both feet"

    If dz < min_height => torso is considered too low.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    if asset_cfg.body_ids is None:
        raise ValueError(
            "asset_cfg.body_ids is None. Please set asset_cfg.body_names to "
            "[torso_link, right_foot_tip, left_foot_tip] so body_ids can be resolved."
        )

    # Expect exactly 3 ids: torso, right foot, left foot
    if isinstance(asset_cfg.body_ids, slice):
        raise ValueError(
            "detect_height_too_low_relative requires exactly 3 body ids (torso, r_foot, l_foot); "
            "got slice(None). Please specify asset_cfg.body_names explicitly."
        )

    if len(asset_cfg.body_ids) != 3:
        raise ValueError(
            f"detect_height_too_low_relative expects 3 bodies (torso, r_foot, l_foot) "
            f"but got {len(asset_cfg.body_ids)}."
        )

    torso_id, rfoot_id, lfoot_id = asset_cfg.body_ids

    torso_z = asset.data.body_pos_w[:, torso_id, 2]
    rfoot_z = asset.data.body_pos_w[:, rfoot_id, 2]
    lfoot_z = asset.data.body_pos_w[:, lfoot_id, 2]

    dz_r = torso_z - rfoot_z
    dz_l = torso_z - lfoot_z
    dz = torch.maximum(dz_r, dz_l)  # larger deviation vs the two feet

    return dz < min_height


def detect_tilt_too_high(
    env: "object",
    max_tilt: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot's tilt angle is above the maximum tilt angle."""
    asset: RigidObject = env.scene[asset_cfg.name]
    proj_g = asset.data.projected_gravity_b  # (N, 3)

    # acos argument should be in [-1, 1], but if proj_g is non-finite, result can be NaN.
    acos_arg = torch.clamp(-proj_g[:, 2], -1.0, 1.0)
    tilt = torch.acos(acos_arg).abs()

    return tilt > max_tilt


def detect_tilt_too_high_any_link(
    env: "object",
    max_tilt: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when ANY specified link tilt angle is above max_tilt.

    Tilt per link is computed from link orientation (body_quat_w) by rotating world gravity
    into the link frame, similar to how projected_gravity_b is used for the root.
    Uses asset_cfg.body_ids (typically resolved from asset_cfg.body_names).
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # If body_ids wasn't resolved, fall back to "all links".
    if asset_cfg.body_ids is None:
        asset_cfg.body_ids = slice(None)

    # Select link quaternions (N, K, 4)
    q = asset.data.body_quat_w[:, asset_cfg.body_ids, :]

    # --- helper: rotate a vector by inverse quaternion (q assumed as wxyz) ---
    # v_b = q_conj * v_w * q
    def _quat_rotate_inverse_wxyz(
        q_wxyz: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        qw = q_wxyz[..., 0:1]
        qx = q_wxyz[..., 1:2]
        qy = q_wxyz[..., 2:3]
        qz = q_wxyz[..., 3:4]

        # conjugate
        cx = -qx
        cy = -qy
        cz = -qz

        # t = 2 * cross(q_vec_conj, v)
        # where q_vec_conj = (cx, cy, cz)
        tx = 2.0 * (cy * v[..., 2:3] - cz * v[..., 1:2])
        ty = 2.0 * (cz * v[..., 0:1] - cx * v[..., 2:3])
        tz = 2.0 * (cx * v[..., 1:2] - cy * v[..., 0:1])

        # v' = v + qw * t + cross(q_vec_conj, t)
        cxt = cy * tz - cz * ty
        cyt = cz * tx - cx * tz
        czt = cx * ty - cy * tx

        return torch.cat(
            [
                v[..., 0:1] + qw * tx + cxt,
                v[..., 1:2] + qw * ty + cyt,
                v[..., 2:3] + qw * tz + czt,
            ],
            dim=-1,
        )

    # World gravity direction (unit): same convention as projected_gravity_b upright -> [0,0,-1] in body frame
    g_w = torch.tensor([0.0, 0.0, -1.0], device=q.device, dtype=q.dtype).view(1, 1, 3)
    g_w = g_w.expand(q.shape[0], q.shape[1], 3)  # (N, K, 3)

    proj_g_b = _quat_rotate_inverse_wxyz(q, g_w)  # (N, K, 3)

    # acos argument should be in [-1, 1], but if proj_g_b is non-finite, result can be NaN.
    acos_arg = torch.clamp(-proj_g_b[..., 2], -1.0, 1.0)  # (N, K)
    tilt = torch.acos(acos_arg).abs()  # (N, K)

    # If any specified link exceeds max_tilt OR becomes non-finite -> terminate.
    bad = (tilt > max_tilt) | (~torch.isfinite(tilt))
    return torch.any(bad, dim=1)


def detect_support_plane_tilt_too_high(
    env: "object",
    max_tilt: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when support plane tilt becomes too large.

    Definition:
      - Upright: plane normal is horizontal (parallel to XY plane) -> tilt ≈ 0
      - Fallen:  plane normal is parallel to Z axis                -> tilt ≈ pi/2

    tilt = asin( |n_z| / ||n|| )

    Terminate if tilt > max_tilt.
    asset_cfg.body_names must specify exactly:
      [torso_link, right_foot_tip, left_foot_tip]
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    if asset_cfg.body_ids is None or isinstance(asset_cfg.body_ids, slice):
        raise ValueError(
            "Please set asset_cfg.body_names to "
            "[torso_link, right_foot_tip, left_foot_tip]."
        )

    if len(asset_cfg.body_ids) != 3:
        raise ValueError("Exactly 3 body names required: torso, r_foot, l_foot.")

    torso_id, rfoot_id, lfoot_id = asset_cfg.body_ids

    p_t = asset.data.body_pos_w[:, torso_id, :]
    p_r = asset.data.body_pos_w[:, rfoot_id, :]
    p_l = asset.data.body_pos_w[:, lfoot_id, :]

    a = p_r - p_t
    b = p_l - p_t

    n = torch.cross(a, b, dim=-1)
    n_norm = torch.linalg.norm(n, dim=-1)

    eps = 1e-8

    # tilt = asin(|n_z| / ||n||)
    sin_theta = torch.abs(n[:, 2]) / torch.clamp(n_norm, min=eps)
    sin_theta = torch.clamp(sin_theta, -1.0, 1.0)

    tilt = torch.asin(sin_theta)

    bad = (tilt > max_tilt) | (~torch.isfinite(tilt)) | (n_norm < eps)
    return bad
