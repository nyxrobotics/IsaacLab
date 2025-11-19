# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
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
# Reward config
# ---------------------------------------------------------------------
@configclass
class KurokoRewards(RewardsCfg):
    """Reward terms for the MDP (Kuroko)."""

    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-1000.0,
    )

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=8.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=46.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["chest"])},
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_height_biped,
        weight=10.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[]),
            "desired_lift_time": 0.3,
            "desired_lift_height": 0.01,
        },
    )

    torso_height_limit = RewTerm(
        func=mdp.torso_height_limit,
        weight= -30.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "body_link",
                    "ankle_l_yaw_link",
                    "ankle_r_yaw_link",
                ],
            ),
            "min_height": 0.32,
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

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                "hip_l_pitch",
                "hip_r_pitch"])},
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                "shoulder_l_pitch",
                "shoulder_l_roll",
                "shoulder_r_pitch",
                "shoulder_r_roll",
                "elbow_l_front",
                "elbow_l_rear",
                "elbow_r_front",
                "elbow_r_rear",])},
    )

    joint_acc_arms = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                "shoulder_l_pitch",
                "shoulder_l_roll",
                "shoulder_r_pitch",
                "shoulder_r_roll",
                "elbow_l_front",
                "elbow_l_rear",
                "elbow_r_front",
                "elbow_r_rear",])},
    )



# ---------------------------------------------------------------------
# Main environment config
# ---------------------------------------------------------------------
@configclass
class KurokoRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: KurokoRewards = KurokoRewards()

    def __post_init__(self):
        super().__post_init__()

        # -----------------------------------------------------------
        # 1. Load USD and discover prims
        # -----------------------------------------------------------
        usd_path = KUROKO_MINIMAL_CFG.spawn.usd_path
        print("[DEBUG] Loading USD:", usd_path)

        base_paths = find_prim_paths(usd_path, "body_link")
        print("[DEBUG] Found base_link prims:", base_paths)

        if not base_paths:
            raise RuntimeError("body_link not found in USD!")

        base_link_full = base_paths[0]  # /Root/kuroko/body_link
        base_link_name = os.path.basename(base_link_full)  # body_link

        print("[DEBUG] base_link_full:", base_link_full)
        print("[DEBUG] base_link_name:", base_link_name)

        # Feet: ankle yaw links
        ankle_paths = find_prim_paths(usd_path, "ankle_*_yaw_link")
        print("[DEBUG] Found ankle yaw prims:", ankle_paths)

        if not ankle_paths:
            raise RuntimeError("ankle_*_yaw_link not found in USD!")

        ankle_names = [os.path.basename(p) for p in ankle_paths]
        print("[DEBUG] ankle yaw link names:", ankle_names)

        # -----------------------------------------------------------
        # 2. Apply robot config into scene
        # -----------------------------------------------------------
        self.scene.robot = KUROKO_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ⚠ IsaacLab spawns robot under scene, but its actual prim_path
        #    may be /World/envs/env_0/Robot  OR /World/envs/env_0/Root
        # → We MUST wait until the scene is constructed to know real path.
        robot_prim_resolved = None

        # -----------------------------------------------------------
        # 3. Ask Scene to tell us the actual robot prim path
        #    (resolve happens after super().__post_init__)
        # -----------------------------------------------------------
        try:
            robot_prim_resolved = self.scene.robot.prim_path
        except Exception:
            # fallback when not resolved yet
            robot_prim_resolved = "{ENV_REGEX_NS}/Robot"

        print("[DEBUG] Detected robot prim path BEFORE spawn:", robot_prim_resolved)

        # -----------------------------------------------------------
        # 4. Disable synthetic height scanner and its observation.
        # -----------------------------------------------------------
        self.scene.height_scanner = None
        if hasattr(self.observations, "policy") and hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None
        if hasattr(self.observations.policy, "contact_forces"):
            self.observations.policy.contact_forces = None

        # -----------------------------------------------------------
        # 5. Rewards & terminations use short names only
        # -----------------------------------------------------------
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ankle_names
        self.rewards.feet_air_time.params["asset_cfg"].body_names = ankle_names
        self.rewards.feet_slide.params["sensor_cfg"].body_names = ankle_names
        self.rewards.feet_slide.params["asset_cfg"].body_names = ankle_names

        self.terminations.base_contact.params["sensor_cfg"].body_names = [base_link_name]

        print("[DEBUG] Feet link names for reward:", ankle_names)
        print("[DEBUG] Base contact link:", base_link_name)

        # -----------------------------------------------------------
        # 6. Explosion termination: use base_link as reference body
        # -----------------------------------------------------------
        if hasattr(self.terminations, "robot_exploded"):
            self.terminations.robot_exploded.params["asset_cfg"] = SceneEntityCfg(
                "robot",
                body_names=[base_link_name],
            )

        # -----------------------------------------------------------
        # Remaining default settings
        # -----------------------------------------------------------
        if self.scene.terrain.terrain_generator is not None:
            tg = self.scene.terrain.terrain_generator
            tg.difficulty_range = (0, 1.0)
            terrain_scale = 0.1

            # ★ 全ての段差の高さを 0.1 倍にスケールする処理 ★
            tg.vertical_scale *= terrain_scale

            for cfg in tg.sub_terrains.values():
                # Mesh 系 stair: step_height_range
                if hasattr(cfg, "step_height_range"):
                    lo, hi = cfg.step_height_range
                    cfg.step_height_range = (lo * terrain_scale, hi * terrain_scale)

                # Mesh 系 blocks: grid_height_range
                elif hasattr(cfg, "grid_height_range"):
                    lo, hi = cfg.grid_height_range
                    cfg.grid_height_range = (lo * terrain_scale, hi * terrain_scale)

                # HeightField 系: noise_range のように height を含むパラメータにも適用（必要なら）
                elif hasattr(cfg, "noise_range"):
                    lo, hi = cfg.noise_range
                    cfg.noise_range = (lo * terrain_scale, hi * terrain_scale)

                # 他にも "height" を含むパラメータ名があれば自動的に 0.1 倍
                else:
                    for attr in dir(cfg):
                        if "height" in attr and isinstance(getattr(cfg, attr), (float, tuple)):
                            val = getattr(cfg, attr)
                            if isinstance(val, float):
                                setattr(cfg, attr, val * terrain_scale)
                            elif isinstance(val, tuple) and len(val) == 2:
                                lo, hi = val
                                setattr(cfg, attr, (lo * terrain_scale, hi * terrain_scale))


        # Randomize initial joint angles
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (-0.25 * math.pi, 0.25 * math.pi)
        self.events.reset_robot_joints.params["velocity_range"] = (-0.25 * math.pi, 0.25 * math.pi)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [base_link_name]
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

        self.rewards.lin_vel_z_l2 = None
        self.rewards.lin_acc_z_l2 = None
        self.rewards.dof_pos_limits = None
        self.rewards.undesired_contacts = None
        self.rewards.ang_vel_xy_l2 = None
        self.rewards.flat_orientation_l2.weight = -40.0
        self.rewards.action_rate_l2.weight = -0.001

        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                "shin_l_active",
                "shin_r_active",
                "shoulder_l_roll",
                "shoulder_r_roll",
                "thigh_l_active",
                "thigh_r_active",
                "ankle_l_roll",
                "ankle_l_yaw",
                "ankle_r_roll",
                "ankle_r_yaw",
                "chest",
                "elbow_l_front",
                "elbow_l_rear",
                "elbow_r_front",
                "elbow_r_rear",
                "hip_l_pitch",
                "hip_l_roll",
                "hip_r_pitch",
                "hip_r_roll",
                "shoulder_l_pitch",
                "shoulder_r_pitch",
            ],
        )

        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                "shin_l_active",
                "shin_r_active",
                "shoulder_l_roll",
                "shoulder_r_roll",
                "thigh_l_active",
                "thigh_r_active",
                "ankle_l_roll",
                "ankle_l_yaw",
                "ankle_r_roll",
                "ankle_r_yaw",
                "chest",
                "elbow_l_front",
                "elbow_l_rear",
                "elbow_r_front",
                "elbow_r_rear",
                "hip_l_pitch",
                "hip_l_roll",
                "hip_r_pitch",
                "hip_r_roll",
                "shoulder_l_pitch",
                "shoulder_r_pitch",
            ],
        )

        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 0.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # -----------------------------------------------------------
        # FIX: physics_material の body_names/body_ids 衝突を解消
        # -----------------------------------------------------------
        if hasattr(self, "physics_material") and self.physics_material is not None:

            # asset_cfg が無ければ新しく作る
            if self.physics_material.asset_cfg is None:
                self.physics_material.asset_cfg = SceneEntityCfg(
                    "robot",
                    body_names=[".*"],
                )
            else:
                # body_ids があれば削除
                if hasattr(self.physics_material.asset_cfg, "body_ids"):
                    # body_ids フィールドが存在する（SceneEntityCfg仕様）
                    if getattr(self.physics_material.asset_cfg, "body_ids") not in (None, [], ()):
                        print("[DEBUG] Removing physics_material.asset_cfg.body_ids (conflict fix)")
                        self.physics_material.asset_cfg.body_ids = None

                # body_names は .* に強制上書き（最も安全）
                self.physics_material.asset_cfg.body_names = [".*"]


# ---------------------------------------------------------------------
# PLAY config
# ---------------------------------------------------------------------
@configclass
class KurokoRoughEnvCfg_PLAY(KurokoRoughEnvCfg):
    """Visualization-friendly settings."""

    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
