from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def _build_joint_name_to_action_index(env) -> dict[str, int]:
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
        try:
            iod = getattr(term, "IO_descriptor", None)
            extras = getattr(iod, "extras", None)
            if isinstance(extras, dict):
                joint_names = extras.get("joint_names", None)
        except Exception:
            joint_names = None
        if joint_names is None:
            if hasattr(term, "_joint_names"):
                joint_names = getattr(term, "_joint_names")
            elif hasattr(term, "joint_names"):
                joint_names = getattr(term, "joint_names")
        if joint_names is not None:
            for local_i, jn in enumerate(list(joint_names)):
                name_to_idx[str(jn)] = offset + int(local_i)
        offset += term_dim

    setattr(env, cache_attr, name_to_idx)
    return name_to_idx


def _resolve_action_indices(env, asset_cfg: SceneEntityCfg | None) -> torch.Tensor | None:
    if asset_cfg is None:
        return None

    target_joint_names = None
    if getattr(asset_cfg, "joint_names", None):
        target_joint_names = [str(name) for name in asset_cfg.joint_names]
    else:
        asset: Articulation = env.scene[asset_cfg.name]
        all_joint_names = list(asset.joint_names)
        target_joint_names = [str(all_joint_names[jid]) for jid in asset_cfg.joint_ids]

    name_to_action_idx = _build_joint_name_to_action_index(env)
    missing = [jn for jn in target_joint_names if jn not in name_to_action_idx]
    if missing:
        raise KeyError(f"Missing action indices for joints: {missing}")

    return torch.tensor(
        [name_to_action_idx[jn] for jn in target_joint_names],
        device=env.device,
        dtype=torch.long,
    )


def joint_action_deviation_l1(
    env,
    asset_cfg: SceneEntityCfg | None = None,
    default_action: torch.Tensor | None = None,
) -> torch.Tensor:
    action_ids = _resolve_action_indices(env, asset_cfg)
    action = env.action_manager.action
    if action_ids is not None:
        action = action.index_select(1, action_ids)

    if default_action is None:
        default_sel = torch.zeros_like(action)
    else:
        if default_action.dim() == 1:
            if action_ids is None:
                default_sel = default_action.unsqueeze(0)
            elif default_action.numel() == env.action_manager.action.shape[1]:
                default_sel = default_action.index_select(0, action_ids).unsqueeze(0)
            else:
                default_sel = default_action.unsqueeze(0)
        else:
            if action_ids is None:
                default_sel = default_action
            elif default_action.shape[1] == env.action_manager.action.shape[1]:
                default_sel = default_action.index_select(1, action_ids)
            else:
                default_sel = default_action

    return torch.sum(torch.abs(action - default_sel), dim=1)


def joint_action_acc_l2(
    env, dt: float, asset_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
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
