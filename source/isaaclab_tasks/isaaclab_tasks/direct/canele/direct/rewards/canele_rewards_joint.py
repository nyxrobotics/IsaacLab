# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Common functions that can be used to enable reward functions.

Pure direct-environment reward helpers for the Canele task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
import torch



"""
Action penalties.
"""


def _build_joint_name_to_action_index(env: "object") -> dict[str, int]:
    """Build mapping from joint name -> global action index.

    Priority:
      1) term.IO_descriptor.extras["joint_names"] if available
      2) term._joint_names / term.joint_names if available
    The mapping is cached on the environment instance.
    """
    cache_attr = "_joint_name_to_action_index_cache"
    cached = getattr(env, cache_attr, None)
    if cached is not None:
        return cached

    name_to_idx: dict[str, int] = {}

    offset = 0
    for term_name in env.action_manager.active_terms:
        term = env.action_manager.get_term(term_name)
        term_dim = int(getattr(term, "action_dim", 0))

        joint_names = None

        # 1) Try IO descriptor extras
        try:
            iod = getattr(term, "IO_descriptor", None)
            extras = getattr(iod, "extras", None)
            if isinstance(extras, dict):
                joint_names = extras.get("joint_names", None)
        except Exception:
            joint_names = None

        # 2) Fallback: term internal fields (e.g., JointPositionAction has _joint_names)
        if joint_names is None:
            if hasattr(term, "_joint_names"):
                joint_names = getattr(term, "_joint_names")
            elif hasattr(term, "joint_names"):
                joint_names = getattr(term, "joint_names")

        # Register mapping
        if joint_names is not None:
            for local_i, jn in enumerate(list(joint_names)):
                name_to_idx[str(jn)] = offset + int(local_i)

        offset += term_dim

    setattr(env, cache_attr, name_to_idx)
    return name_to_idx


def _debug_print_action_joint_mapping(env: "object") -> None:
    """Print joint-name mapping that exists on the action side for debugging."""
    name_to_idx = _build_joint_name_to_action_index(env)

    # Print in a stable order (by action index)
    items = sorted(name_to_idx.items(), key=lambda kv: kv[1])
    joint_names_sorted = [name for name, _ in items]

    print("\n[RewardDebug] Action joint mapping (extras['joint_names']):")
    print(f"[RewardDebug] total_mapped_joints={len(joint_names_sorted)}")
    if joint_names_sorted:
        print("[RewardDebug] joint_names=")
        for name in joint_names_sorted:
            print(f"  - {name}")
    else:
        print(
            "[RewardDebug] joint_names is EMPTY. No action term exported extras['joint_names']."
        )

    # Also show action dimensionality for reference
    try:
        total_action_dim = int(env.action_manager.action.shape[1])
        print(f"[RewardDebug] env.action_manager.action_dim={total_action_dim}")
    except Exception:
        pass
    print("")


def _debug_print_action_terms(env: "object") -> None:
    """Print detailed info about action terms to figure out joint/action ordering."""
    print("\n[RewardDebug] ===== Action terms debug =====")
    try:
        action_dim_total = int(env.action_manager.action.shape[1])
        print(f"[RewardDebug] total_action_dim={action_dim_total}")
    except Exception:
        pass

    try:
        term_names = list(env.action_manager.active_terms)
    except Exception:
        term_names = []
    print(f"[RewardDebug] active_terms={term_names}")

    offset = 0
    for term_name in term_names:
        term = env.action_manager.get_term(term_name)
        term_dim = int(getattr(term, "action_dim", -1))
        cls_name = term.__class__.__name__
        print(
            f"\n[RewardDebug] term='{term_name}' class={cls_name} offset={offset} action_dim={term_dim}"
        )

        # Try common places where joint info is stored
        candidates = [
            "joint_names",
            "_joint_names",
            "joint_ids",
            "_joint_ids",
            "dof_names",
            "_dof_names",
            "dof_ids",
            "_dof_ids",
        ]
        for key in candidates:
            if hasattr(term, key):
                val = getattr(term, key)
                try:
                    if isinstance(val, (list, tuple)):
                        print(f"[RewardDebug]   {key} (len={len(val)}): {val}")
                    elif hasattr(val, "shape"):
                        print(
                            f"[RewardDebug]   {key} (tensor shape={tuple(val.shape)}): {val}"
                        )
                    else:
                        print(f"[RewardDebug]   {key}: {val}")
                except Exception:
                    print(f"[RewardDebug]   {key}: <unprintable>")

        # Print cfg / asset_cfg if present
        if hasattr(term, "cfg"):
            cfg = getattr(term, "cfg")
            print(f"[RewardDebug]   has cfg: {type(cfg).__name__}")
            if hasattr(cfg, "asset_cfg"):
                ac = getattr(cfg, "asset_cfg")
                print(f"[RewardDebug]   cfg.asset_cfg: {ac}")
                for k in ["name", "joint_names", "joint_ids", "body_names", "body_ids"]:
                    if hasattr(ac, k):
                        print(f"[RewardDebug]     asset_cfg.{k}: {getattr(ac, k)}")

        # IO descriptor if present
        if hasattr(term, "IO_descriptor"):
            iod = getattr(term, "IO_descriptor")
            print(f"[RewardDebug]   has IO_descriptor: {type(iod).__name__}")
            extras = getattr(iod, "extras", None)
            print(f"[RewardDebug]     IO_descriptor.extras: {extras}")

        offset += max(term_dim, 0)

    print("\n[RewardDebug] ===== End action terms debug =====\n")


def _resolve_action_indices(
    env: "object", asset_cfg: SceneEntityCfg | None
) -> torch.Tensor | None:
    """Resolve action indices.
    - asset_cfg is None -> use all action dimensions (return None)
    - asset_cfg is not None -> match joint names and return indices
    """
    if asset_cfg is None:
        return None

    asset: Articulation = env.scene[asset_cfg.name]

    # get articulation joint names
    if hasattr(asset, "joint_names"):
        all_joint_names = list(asset.joint_names)
    elif hasattr(asset.data, "joint_names"):
        all_joint_names = list(asset.data.joint_names)
    else:
        raise AttributeError("Cannot access articulation joint names.")

    target_joint_names = [str(all_joint_names[jid]) for jid in asset_cfg.joint_ids]

    name_to_action_idx = _build_joint_name_to_action_index(env)
    missing = [jn for jn in target_joint_names if jn not in name_to_action_idx]
    if missing:
        _debug_print_action_terms(env)
        raise KeyError(
            "Some joints in asset_cfg were not found in action joint_names mapping. "
            f"Missing: {missing}. "
            "Make sure your action term exports joint names or we extract them from the term."
        )
    return torch.tensor(
        [name_to_action_idx[jn] for jn in target_joint_names],
        device=env.device,
        dtype=torch.long,
    )


def joint_action_deviation_l1(
    env: "object",
    asset_cfg: SceneEntityCfg | None = None,
    default_action: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize action deviation using L1-kernel.
    - default_action is None -> zero action is used as reference
    - asset_cfg is None -> all action dimensions are used
    """
    action_ids = _resolve_action_indices(env, asset_cfg)

    action = env.action_manager.action
    if action_ids is not None:
        action = action.index_select(1, action_ids)

    # default_action handling
    if default_action is None:
        default_sel = torch.zeros_like(action)
    else:
        if default_action.dim() == 1:
            if action_ids is None:
                default_sel = default_action.unsqueeze(0)
            else:
                if default_action.numel() == env.action_manager.action.shape[1]:
                    default_sel = default_action.index_select(0, action_ids).unsqueeze(
                        0
                    )
                else:
                    # assume already in selected-joint order
                    default_sel = default_action.unsqueeze(0)
        else:
            if action_ids is None:
                default_sel = default_action
            else:
                if default_action.shape[1] == env.action_manager.action.shape[1]:
                    default_sel = default_action.index_select(1, action_ids)
                else:
                    default_sel = default_action

    deviation = action - default_sel
    return torch.sum(torch.abs(deviation), dim=1)


def joint_action_deviation_l2(
    env: "object",
    asset_cfg: SceneEntityCfg | None = None,
    default_action: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize action deviation using L2 squared kernel."""
    action_ids = _resolve_action_indices(env, asset_cfg)

    action = env.action_manager.action
    if action_ids is not None:
        action = action.index_select(1, action_ids)

    # default_action handling
    if default_action is None:
        default_sel = torch.zeros_like(action)
    else:
        if default_action.dim() == 1:
            if action_ids is None:
                default_sel = default_action.unsqueeze(0)
            else:
                if default_action.numel() == env.action_manager.action.shape[1]:
                    default_sel = default_action.index_select(0, action_ids).unsqueeze(
                        0
                    )
                else:
                    # assume already in selected-joint order
                    default_sel = default_action.unsqueeze(0)
        else:
            if action_ids is None:
                default_sel = default_action
            else:
                if default_action.shape[1] == env.action_manager.action.shape[1]:
                    default_sel = default_action.index_select(1, action_ids)
                else:
                    default_sel = default_action

    deviation = action - default_sel
    return torch.sum(torch.square(deviation), dim=1)


def joint_action_vel_l1(
    env: "object", dt: float, asset_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """Penalize action velocity using L1-kernel.
    If asset_cfg is None, all action dimensions are used.
    """
    action_ids = _resolve_action_indices(env, asset_cfg)

    action = env.action_manager.action
    prev = env.action_manager.prev_action
    if prev is None:
        prev = action

    if action_ids is not None:
        action = action.index_select(1, action_ids)
        prev = prev.index_select(1, action_ids)

    action_vel = (action - prev) / dt
    return torch.sum(torch.abs(action_vel), dim=1)


def joint_action_vel_l2(
    env: "object", dt: float, asset_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """Penalize action velocity using L2 squared kernel."""
    action_ids = _resolve_action_indices(env, asset_cfg)

    action = env.action_manager.action
    prev = env.action_manager.prev_action
    if prev is None:
        prev = action

    if action_ids is not None:
        action = action.index_select(1, action_ids)
        prev = prev.index_select(1, action_ids)

    action_vel = (action - prev) / dt
    return torch.sum(torch.square(action_vel), dim=1)


def joint_action_acc_l1(
    env: "object", dt: float, asset_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """Penalize action acceleration using L1 kernel."""
    action_ids = _resolve_action_indices(env, asset_cfg)

    action = env.action_manager.action
    prev = env.action_manager.prev_action
    prev_prev = env.action_manager.prev_prev_action

    if prev is None:
        prev = action
    if prev_prev is None:
        prev_prev = prev

    if action_ids is not None:
        action = action.index_select(1, action_ids)
        prev = prev.index_select(1, action_ids)
        prev_prev = prev_prev.index_select(1, action_ids)

    action_acc = (action - 2.0 * prev + prev_prev) / (dt * dt)
    return torch.sum(torch.abs(action_acc), dim=1)


def joint_action_acc_l2(
    env: "object", dt: float, asset_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """Penalize action acceleration using L2 squared kernel."""
    action_ids = _resolve_action_indices(env, asset_cfg)

    action = env.action_manager.action
    prev = env.action_manager.prev_action
    prev_prev = env.action_manager.prev_prev_action

    if prev is None:
        prev = action
    if prev_prev is None:
        prev_prev = prev

    if action_ids is not None:
        action = action.index_select(1, action_ids)
        prev = prev.index_select(1, action_ids)
        prev_prev = prev_prev.index_select(1, action_ids)

    action_acc = (action - 2.0 * prev + prev_prev) / (dt * dt)
    return torch.sum(torch.square(action_acc), dim=1)
