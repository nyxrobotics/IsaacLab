# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

# Kuroko articulation config
from isaaclab_assets.robots.kuroko.kuroko_cfg import KUROKO_MINIMAL_CFG

# For USD prim inspection
from pxr import Usd
import fnmatch
import os


# ---------------------------------------------------------------------
# Utility: Traverse USD and find prims whose "name" matches pattern
# ---------------------------------------------------------------------
def find_prim_paths(usd_path, pattern):
    """
    Find USD prim paths whose LAST ELEMENT matches fnmatch pattern.
    Example:
        pattern = "ankle_*_yaw_link"
    """
    stage = Usd.Stage.Open(usd_path)
    results = []
    for prim in stage.Traverse():
        name = prim.GetPath().name  # Extract last component
        if fnmatch.fnmatch(name, pattern):
            results.append(str(prim.GetPath()))
    return results


# ---------------------------------------------------------------------
# Reward class
# ---------------------------------------------------------------------
@configclass
class KurokoRewards(RewardsCfg):
    """Reward terms for the MDP (Kuroko)."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # Feet terms (body names will be filled dynamically in env_cfg)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[]),
            "threshold": 0.4,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[]),
        },
    )

    # Joint limit / deviation penalties
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["ankle_.*_roll", "ankle_.*_yaw"],
            )
        },
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_.*"])},
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_.*", "elbow_.*"])
        },
    )

    joint_deviation_fingers = None  # Kuroko has no finger joints

    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["chest"])},
    )


# ---------------------------------------------------------------------
# Main environment config
# ---------------------------------------------------------------------
@configclass
class KurokoRoughLowEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Rough terrain locomotion config for Kuroko."""

    rewards: KurokoRewards = KurokoRewards()

    def __post_init__(self):
        super().__post_init__()

        # -----------------------------------------------------
        # 1. Get USD file path
        # -----------------------------------------------------
        usd_path = KUROKO_MINIMAL_CFG.spawn.usd_path
        print("[DEBUG] Loading USD:", usd_path)

        # -----------------------------------------------------
        # 2. Find base_link path in USD
        # -----------------------------------------------------
        base_link_paths = find_prim_paths(usd_path, "body_link")
        print("[DEBUG] Found base_link prims:", base_link_paths)

        if not base_link_paths:
            raise RuntimeError("body_link not found in USD!")

        # Full path in USD (ex: /Root/kuroko/body_link)
        base_link_full = base_link_paths[0]

        # Short name (ex: body_link)
        base_link_name = os.path.basename(base_link_full)
        print("[DEBUG] base_link name:", base_link_name)

        # -----------------------------------------------------
        # 3. Find ankle yaw links (feet)
        # -----------------------------------------------------
        ankle_paths = find_prim_paths(usd_path, "ankle_*_yaw_link")
        print("[DEBUG] Found ankle yaw prims:", ankle_paths)

        if not ankle_paths:
            raise RuntimeError("ankle_*_yaw_link not found in USD!")

        ankle_names = [os.path.basename(p) for p in ankle_paths]
        print("[DEBUG] ankle yaw link names:", ankle_names)

        # -----------------------------------------------------
        # 4. Set robot spawn prim
        # -----------------------------------------------------
        self.scene.robot = KUROKO_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # -----------------------------------------------------
        # 5. Ray caster (height scanner) needs USD full path
        # -----------------------------------------------------
        # ex: "{ENV}/Robot/Root/kuroko/body_link"
        self.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot{base_link_full}"
        print("[DEBUG] height_scanner prim_path:", self.scene.height_scanner.prim_path)

        # -----------------------------------------------------
        # 6. Rewards and terminations need SHORT LINK NAMES
        # -----------------------------------------------------
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ankle_names
        self.rewards.feet_slide.params["sensor_cfg"].body_names = ankle_names
        self.rewards.feet_slide.params["asset_cfg"].body_names = ankle_names

        # External force
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [base_link_name]

        # Termination
        self.terminations.base_contact.params["sensor_cfg"].body_names = base_link_name

        print("[DEBUG] feet body_names:", ankle_names)
        print("[DEBUG] base_contact body_name:", base_link_name)

        # -----------------------------------------------------
        # 7. Terrain and randomization settings (same as original)
        # -----------------------------------------------------
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0, 0.0001)

        self.events.push_robot = None
        self.events.add_base_mass = None

        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005

        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["hip_.*", "shin_.*", "thigh_.*", "ankle_.*"]
        )

        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["hip_.*", "shin_.*", "thigh_.*", "ankle_.*"]
        )

        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.4)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-4.0, 4.0)


# ---------------------------------------------------------------------
# PLAY version
# ---------------------------------------------------------------------
@configclass
class KurokoRoughLowEnvCfg_PLAY(KurokoRoughLowEnvCfg):
    """Smaller scene / no randomization for visualization."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
