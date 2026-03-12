# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import fnmatch
import os

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import torch
from isaaclab.utils import configclass

# Canele articulation config
from ..assets.canele_cfg import CANELE_MINIMAL_CFG
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg
from .terminations import canele_terminations
from .rewards import canele_rewards_env
from .rewards import canele_rewards_walk
from .rewards import canele_rewards_joint
from .rewards import canele_rewards_link
from .io_descriptors import history_observation_descriptor

# For USD prim inspection
from pxr import Usd


# ---------------------------------------------------------------------
# Utility function: find prims in USD by matching last component
# ---------------------------------------------------------------------
def find_prim_paths(usd_path, pattern):
    """
    Find USD prim paths whose LAST ELEMENT matches fnmatch pattern.

    Example:
        find_prim_paths(path, "ankle_*_yaw_link")
    """
    stage = Usd.Stage.Open(usd_path)
    results = []
    for prim in stage.Traverse():
        name = prim.GetPath().name
        if fnmatch.fnmatch(name, pattern):
            results.append(str(prim.GetPath()))
    return results


# ---------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------
def _get_history_buffer(env, attr_name: str, feature_dim: int) -> torch.Tensor:
    """Return or lazily create a [num_envs, 4, feature_dim] history buffer on the env."""
    hist = getattr(env, attr_name, None)
    if (
        hist is None
        or hist.shape[0] != env.num_envs
        or hist.shape[1] != 4
        or hist.shape[2] != feature_dim
        or hist.device != env.device
    ):
        hist = torch.zeros(env.num_envs, 4, feature_dim, device=env.device)
        setattr(env, attr_name, hist)
    return hist


def _update_history(
    env, attr_name: str, values: torch.Tensor, *, init_with_current: bool
) -> torch.Tensor:
    """Append current values to a 4-step per-env history buffer."""
    hist = _get_history_buffer(env, attr_name, int(values.shape[-1]))
    env_step_count = getattr(env, "episode_length_buf", None)
    if env_step_count is None:
        reset_mask = torch.zeros(
            values.shape[0], dtype=torch.bool, device=values.device
        )
    else:
        reset_mask = env_step_count == 0

    non_reset_mask = ~reset_mask
    if torch.any(non_reset_mask):
        hist[non_reset_mask, :-1] = hist[non_reset_mask, 1:].clone()
        hist[non_reset_mask, -1] = values[non_reset_mask]

    if torch.any(reset_mask):
        if init_with_current:
            hist[reset_mask] = values[reset_mask].unsqueeze(1).repeat(1, 4, 1)
        else:
            hist[reset_mask] = 0.0

    return hist.reshape(values.shape[0], -1)


def _normalize_joint_positions(
    joint_pos: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    denom = (upper - lower).clamp_min(1.0e-6)
    return 2.0 * (joint_pos - lower) / denom - 1.0


@history_observation_descriptor(
    observation_type="CommandHistory",
    terms_per_step=3,
    history_length=4,
    units="normalized",
    source="base_velocity command [lin_vel_x, lin_vel_y, ang_vel_z]",
    normalization="minmax_to_minus1_plus1",
)
def canele_obs_cmd_vel_history(env) -> torch.Tensor:
    """Flattened 4-step history of normalized base velocity commands [vx, vy, wz]."""
    commands = env.command_manager.get_command("base_velocity")[:, :3]
    cmd_cfg = env.cfg.commands.base_velocity.ranges
    cmd_min = torch.tensor(
        [cmd_cfg.lin_vel_x[0], cmd_cfg.lin_vel_y[0], cmd_cfg.ang_vel_z[0]],
        device=env.device,
        dtype=commands.dtype,
    )
    cmd_max = torch.tensor(
        [cmd_cfg.lin_vel_x[1], cmd_cfg.lin_vel_y[1], cmd_cfg.ang_vel_z[1]],
        device=env.device,
        dtype=commands.dtype,
    )
    cmd_norm = 2.0 * (commands - cmd_min) / (cmd_max - cmd_min).clamp_min(1.0e-6) - 1.0
    return _update_history(
        env, "_canele_cmd_vel_hist", cmd_norm, init_with_current=True
    )


@history_observation_descriptor(
    observation_type="IMUHistory",
    terms_per_step=2,
    history_length=4,
    units="normalized",
    axes=["gravity_x", "gravity_y"],
    source="projected_gravity_b[:2]",
    normalization="clamp_to_minus1_plus1",
)
def canele_obs_projected_gravity_history(
    env, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Flattened 4-step history of projected gravity x/y components in the body frame."""
    asset = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b[:, :2]
    gravity_xy = projected_gravity.clamp(-1.0, 1.0)
    return _update_history(
        env,
        "_canele_projected_gravity_hist",
        gravity_xy,
        init_with_current=False,
    )


@history_observation_descriptor(
    observation_type="IMUHistory",
    terms_per_step=3,
    history_length=4,
    units="normalized",
    axes=["wx", "wy", "wz"],
    source="root_ang_vel_b",
    normalization="clamp(-2,2)/2",
)
def canele_obs_ang_vel_history(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Flattened 4-step history of normalized base angular velocity [wx, wy, wz]."""
    asset = env.scene[asset_cfg.name]
    angular_vel = asset.data.root_ang_vel_b[:, :3]
    angular_vel = angular_vel.clamp(-2.0, 2.0) / 2.0
    return _update_history(
        env,
        "_canele_ang_vel_hist",
        angular_vel,
        init_with_current=False,
    )


@history_observation_descriptor(
    observation_type="ActionHistory",
    terms_per_step=13,
    history_length=4,
    units="normalized",
    source="env.action_manager.action for the actuated joints",
    normalization="policy_action_clamped_to_minus1_plus1",
    include_joint_names=True,
)
def canele_obs_action_history(env) -> torch.Tensor:
    """Flattened 4-step history of previous policy actions."""
    action = env.action_manager.action
    action = action.clamp(-1.0, 1.0)
    return _update_history(env, "_canele_action_hist", action, init_with_current=True)


# ---------------------------------------------------------------------
# Reward config
# ---------------------------------------------------------------------
@configclass
class CaneleRewards(RewardsCfg):
    """Reward terms for the MDP (Canele)."""

    termination_penalty = RewTerm(
        func=canele_rewards_env.is_terminated,
        weight=-200.0,
    )
    track_lin_vel_xy_exp = RewTerm(
        func=canele_rewards_walk.track_lin_vel_xy_yaw_frame_exp_no_flight,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "right_toe_link",
                    "left_toe_link",
                ],
            ),
            "contact_time_eps": 1.0e-3,
            "force_eps": 1.0e-3,
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=canele_rewards_walk.track_ang_vel_z_world_exp_no_flight,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "right_toe_link",
                    "left_toe_link",
                ],
            ),
            "contact_time_eps": 1.0e-3,
            "force_eps": 1.0e-3,
        },
    )
    feet_air_time = RewTerm(
        func=canele_rewards_walk.feet_air_time_alternating_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "right_toe_link",
                    "left_toe_link",
                ],
            ),
            "linear_cmd_threshold": 0.0,
            "angular_cmd_threshold": 0.0,
            "body_tilt_threshold": 0.0,
            "air_min_time": 0.1,
            "air_max_time": 1.0,
            "min_contact_time": 0.1,
            "ema_alpha": 0.02,
            "air_balance_weight": 1.0,
            "contact_balance_weight": 0.0,
            "air_reward": 1.0,
            "contact_reward": 1.0,
        },
    )
    feet_slide = RewTerm(
        func=canele_rewards_walk.feet_slide_keep_flat,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "right_toe_link",
                    "left_toe_link",
                ],
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "right_toe_link",
                    "left_toe_link",
                ],
            ),
            "air_time_eps": 0.02,
        },
    )
    joint_deviation_hip = RewTerm(
        func=canele_rewards_joint.joint_action_deviation_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_hip_roll",
                    "left_hip_pitch",
                    "right_hip_roll",
                    "right_hip_pitch",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=canele_rewards_joint.joint_action_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["left_hip_yaw", "right_hip_yaw", "torso_yaw"]
            )
        },
    )
    flat_toe_penalty = RewTerm(
        func=canele_rewards_link.flat_orientation_links_l2,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["right_toe_link", "left_toe_link"]
            ),
            "margin": 0.0,
            "gain": 1.0,
        },
    )


# ---------------------------------------------------------------------
# Main environment config
# ---------------------------------------------------------------------
@configclass
class CaneleRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: CaneleRewards = CaneleRewards()

    def __post_init__(self):
        super().__post_init__()

        # -----------------------------------------------------------
        # 1. Load USD and discover prims
        # -----------------------------------------------------------
        usd_path = CANELE_MINIMAL_CFG.spawn.usd_path
        print("[DEBUG] Loading USD:", usd_path)

        base_paths = find_prim_paths(usd_path, "body_link")
        print("[DEBUG] Found base_link prims:", base_paths)

        if not base_paths:
            raise RuntimeError("body_link not found in USD!")

        base_link_full = base_paths[0]
        base_link_name = os.path.basename(base_link_full)

        print("[DEBUG] base_link_full:", base_link_full)
        print("[DEBUG] base_link_name:", base_link_name)

        ankle_paths = find_prim_paths(usd_path, "*_toe_link")
        print("[DEBUG] Found ankle yaw prims:", ankle_paths)

        if not ankle_paths:
            raise RuntimeError("*_toe_link not found in USD!")

        ankle_names = [os.path.basename(p) for p in ankle_paths]
        print("[DEBUG] ankle yaw link names:", ankle_names)

        # -----------------------------------------------------------
        # 2. Apply robot config into scene
        # -----------------------------------------------------------
        self.scene.robot = CANELE_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        try:
            robot_prim_resolved = self.scene.robot.prim_path
        except Exception:
            robot_prim_resolved = "{ENV_REGEX_NS}/Robot"

        print("[DEBUG] Detected robot prim path BEFORE spawn:", robot_prim_resolved)

        # -----------------------------------------------------------
        # 3. Disable synthetic height scanner and its observation
        # -----------------------------------------------------------
        self.scene.height_scanner = None
        if hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None

        # -----------------------------------------------------------
        # 4. Restrict action joints to actuated leg/body joints only
        # -----------------------------------------------------------
        actuated_joint_names: list[str] = []
        try:
            for actuator_cfg in self.scene.robot.actuators.values():
                actuated_joint_names.extend(list(actuator_cfg.joint_names_expr))
        except Exception as e:
            print(
                "[DEBUG] Failed to collect actuated joints from self.scene.robot.actuators:",
                e,
            )

        _seen = set()
        actuated_joint_names = [
            j for j in actuated_joint_names if not (j in _seen or _seen.add(j))
        ]

        print("[DEBUG] Actuated joint names count:", len(actuated_joint_names))
        print("[DEBUG] Actuated joint names:", actuated_joint_names)

        arm_joints = [
            "left_shoulder_yaw",
            "left_shoulder_pitch",
            "left_shoulder_roll",
            "left_elbow_yaw",
            "left_elbow_pitch",
            "left_wrist_yaw",
            "left_wrist_roll",
            "left_wrist_pitch",
            "right_shoulder_yaw",
            "right_shoulder_pitch",
            "right_shoulder_roll",
            "right_elbow_yaw",
            "right_elbow_pitch",
            "right_wrist_yaw",
            "right_wrist_roll",
            "right_wrist_pitch",
        ]

        print("[DEBUG] Arm joints to exclude:", arm_joints)
        actuated_joint_names = [j for j in actuated_joint_names if j not in arm_joints]
        print(
            "[DEBUG] Actuated joint names after excluding arm joints:",
            actuated_joint_names,
        )

        def _make_actuated_asset_cfg() -> SceneEntityCfg:
            return SceneEntityCfg(
                "robot",
                joint_names=actuated_joint_names,
                preserve_order=True,
            )

        if hasattr(self, "actions") and hasattr(self.actions, "joint_pos"):
            action_term = self.actions.joint_pos
            if hasattr(action_term, "joint_names"):
                action_term.joint_names = actuated_joint_names
                print("[DEBUG] Restricted action term 'joint_pos' via joint_names")

        # -----------------------------------------------------------
        # 5. Replace policy observations completely
        # -----------------------------------------------------------
        if hasattr(self.observations, "policy"):
            policy_obs = self.observations.policy

            policy_obs.enable_corruption = True
            policy_obs.concatenate_terms = True
            policy_obs.concatenate_dim = -1
            policy_obs.history_length = None
            policy_obs.flatten_history_dim = True

            policy_obs.base_lin_vel = None
            policy_obs.base_ang_vel = None
            policy_obs.projected_gravity = None
            policy_obs.velocity_commands = None
            policy_obs.joint_pos = None
            policy_obs.joint_vel = None
            policy_obs.actions = None
            policy_obs.height_scan = None

            policy_obs.cmd_vel_history = ObsTerm(func=canele_obs_cmd_vel_history)
            policy_obs.projected_gravity_history = ObsTerm(
                func=canele_obs_projected_gravity_history,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )
            policy_obs.ang_vel_history = ObsTerm(
                func=canele_obs_ang_vel_history,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )
            policy_obs.action_history = ObsTerm(
                func=canele_obs_action_history,
            )
            print(
                "[DEBUG] Replaced policy observations with cmd_vel/projected_gravity/ang_vel/action 4-step histories"
            )

        # -----------------------------------------------------------
        # 6. Update reset config
        # -----------------------------------------------------------
        if getattr(self.events, "reset_robot_joints", None) is not None:
            self.events.reset_robot_joints.params["asset_cfg"] = (
                _make_actuated_asset_cfg()
            )
            print(
                "[DEBUG] Updated 'reset_robot_joints' event to use actuated asset_cfg"
            )

        # -----------------------------------------------------------
        # 7. Fix startup event base body name
        # -----------------------------------------------------------
        if hasattr(self, "events") and getattr(self, "events", None) is not None:
            event_base_com = getattr(self.events, "base_com", None)
            if event_base_com is not None and hasattr(event_base_com, "params"):
                if event_base_com.params is None:
                    event_base_com.params = {}
                event_base_com.params["asset_cfg"] = SceneEntityCfg(
                    "robot",
                    body_names=[base_link_name],
                    preserve_order=True,
                )
                print(
                    f"[DEBUG] Patched startup event term 'base_com' to use body '{base_link_name}'"
                )

        # -----------------------------------------------------------
        # 8. Rewards & terminations use short names only
        # -----------------------------------------------------------
        self.rewards.feet_slide.params["sensor_cfg"].body_names = ankle_names
        self.rewards.feet_slide.params["asset_cfg"].body_names = ankle_names

        print("[DEBUG] Feet link names for reward:", ankle_names)
        print("[DEBUG] Base contact link:", base_link_name)

        # -----------------------------------------------------------
        # 9. Reduce terrain size if terrain generator exists
        # -----------------------------------------------------------
        if hasattr(self.scene, "terrain") and self.scene.terrain is not None:
            terrain_cfg = self.scene.terrain
            terrain_gen = getattr(terrain_cfg, "terrain_generator", None)
            if terrain_gen is not None:
                print("[DEBUG] Original terrain generator:", terrain_gen)

                terrain_scale = 0.5

                for attr in [
                    "size",
                    "border_width",
                    "horizontal_scale",
                    "vertical_scale",
                ]:
                    if hasattr(terrain_gen, attr):
                        val = getattr(terrain_gen, attr)
                        if isinstance(val, (int, float)):
                            setattr(terrain_gen, attr, val * terrain_scale)
                        elif isinstance(val, tuple) and len(val) == 2:
                            setattr(
                                terrain_gen,
                                attr,
                                (val[0] * terrain_scale, val[1] * terrain_scale),
                            )

                sub_terrains = getattr(terrain_gen, "sub_terrains", None)
                if sub_terrains:
                    for _, cfg in sub_terrains.items():
                        for attr in [
                            "step_height_range",
                            "platform_width",
                            "platform_width_range",
                            "stone_width_range",
                            "stone_distance_range",
                            "noise_range",
                        ]:
                            if hasattr(cfg, attr):
                                val = getattr(cfg, attr)
                                if isinstance(val, (int, float)):
                                    setattr(cfg, attr, val * terrain_scale)
                                elif isinstance(val, tuple) and len(val) == 2:
                                    lo, hi = val
                                    setattr(
                                        cfg,
                                        attr,
                                        (lo * terrain_scale, hi * terrain_scale),
                                    )

        # -----------------------------------------------------------
        # 10. Randomize events
        # -----------------------------------------------------------
        self.events.physics_material.params["asset_cfg"].body_names = ankle_names
        self.events.physics_material.params["static_friction_range"] = (0.1, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.1, 1.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = [base_link_name]
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 1.0)
        self.events.base_com.params["asset_cfg"].body_names = [base_link_name]
        self.events.base_com.params["com_range"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (-0.02, 0.02),
        }
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [
            base_link_name
        ]
        self.events.base_external_force_torque.params["force_range"] = (-2.0, 2.0)
        self.events.base_external_force_torque.params["torque_range"] = (-0.8, 0.8)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
        }
        self.events.push_robot.interval_range_s = (5.0, 20.0)

        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.05),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.3, 0.3),
            },
        }

        # Rewards
        self.rewards.dof_pos_limits = None
        self.rewards.lin_vel_z_l2 = RewTerm(
            func=canele_rewards_link.lin_vel_z_l2, weight=-0.2
        )
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2 = RewTerm(
            func=canele_rewards_link.flat_orientation_l2, weight=-1.0
        )
        self.rewards.ang_vel_xy_l2.weight = -0.01
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_acc_l2 = RewTerm(
            func=canele_rewards_joint.joint_action_acc_l2,
            weight=-1.0e-9,
            params={"dt": self.decimation * self.sim.dt},
        )
        self.rewards.dof_vel_l2 = RewTerm(
            func=canele_rewards_joint.joint_action_vel_l2,
            weight=-1.0e-6,
            params={"dt": self.decimation * self.sim.dt},
        )

        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                "left_hip_yaw",
                "left_hip_roll",
                "left_hip_pitch",
                "left_knee_pitch",
                "left_ankle_pitch",
                "left_ankle_roll",
                "right_hip_yaw",
                "right_hip_roll",
                "right_hip_pitch",
                "right_knee_pitch",
                "right_ankle_pitch",
                "right_ankle_roll",
            ],
        )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.2, 1.2)

        # terminations
        self.terminations.base_contact = None

        self.terminations.detect_fall = DoneTerm(
            func=canele_terminations.detect_fall,
            params={
                "limit_angle": 1.3,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[base_link_name],
                ),
            },
            time_out=False,
        )

        body_and_ankle_names = [base_link_name] + ankle_names
        self.terminations.detect_height_too_low_relative = DoneTerm(
            func=canele_terminations.detect_height_too_low_relative,
            params={
                "min_height": 0.5,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=body_and_ankle_names,
                ),
            },
            time_out=False,
        )

        self.terminations.detect_tilt = DoneTerm(
            func=canele_terminations.detect_tilt_too_high_any_link,
            params={
                "max_tilt": 1.5,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=body_and_ankle_names,
                ),
            },
            time_out=False,
        )

        self.terminations.support_plane_tilt = DoneTerm(
            func=canele_terminations.detect_support_plane_tilt_too_high,
            params={
                "max_tilt": 1.3,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=body_and_ankle_names,
                ),
            },
            time_out=False,
        )


# ---------------------------------------------------------------------
# PLAY config
# ---------------------------------------------------------------------
@configclass
class CaneleRoughEnvCfg_PLAY(CaneleRoughEnvCfg):
    """Visualization-friendly settings."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.2, 1.2)
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0, 0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.scene.height_scanner = None
        if hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None

        self.observations.policy.enable_corruption = False

        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.export_io_descriptors = True
