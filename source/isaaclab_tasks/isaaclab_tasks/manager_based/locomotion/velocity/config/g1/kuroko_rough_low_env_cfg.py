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

    # Feet-related terms (Kuroko: ankle_l_roll_link, ankle_r_roll_link)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names="ankle_.*_roll_link",  # e.g. ankle_l_roll_link, ankle_r_roll_link
            ),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names="ankle_.*_roll_link",
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names="ankle_.*_roll_link",
            ),
        },
    )

    # Penalize ankle joint limits
    # Kuroko joints: ankle_l_roll, ankle_l_yaw, ankle_r_roll, ankle_r_yaw
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

    # Hip joints: hip_l_pitch, hip_l_roll, hip_r_pitch, hip_r_roll
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_.*"])},
    )

    # Arm joints: shoulder_l_*, shoulder_r_*, elbow_l_*, elbow_r_*
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["shoulder_.*", "elbow_.*"],
            )
        },
    )

    # Kuroko には finger 関連の関節がないので無効化
    joint_deviation_fingers = None

    # Torso-like joint: "chest"
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["chest"])},
    )


@configclass
class KurokoRoughLowEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Rough terrain, low difficulty locomotion config for Kuroko."""

    rewards: KurokoRewards = KurokoRewards()

    def __post_init__(self):
        # parent init
        super().__post_init__()

        # Scene / robot
        self.scene.robot = KUROKO_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # height scanner should attach to Kuroko base link (body_link)
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/body_link"

        # Terrain height scaling
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0, 0.0001)

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # Apply external forces on base link
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["body_link"]

        # Base pose / velocity ranges
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

        # Rewards (mostly same as元ファイルだが target joints/link をKurokoに合わせてある)
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005

        # Use only main leg joints for acceleration / torque penalties
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=["hip_.*", "shin_.*", "thigh_.*", "ankle_.*"],
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=["hip_.*", "shin_.*", "thigh_.*", "ankle_.*"],
        )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.4)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-4.0, 4.0)

        # Termination: base link contact
        self.terminations.base_contact.params["sensor_cfg"].body_names = "body_link"


@configclass
class KurokoRoughLowEnvCfg_PLAY(KurokoRoughLowEnvCfg):
    """Smaller scene / no randomization for play/visualization."""

    def __post_init__(self):
        super().__post_init__()

        # smaller scene
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        # spawn robot randomly in grid
        self.scene.terrain.max_init_terrain_level = None

        # reduce terrain count
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # keep command ranges same
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.4)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-4.0, 4.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
