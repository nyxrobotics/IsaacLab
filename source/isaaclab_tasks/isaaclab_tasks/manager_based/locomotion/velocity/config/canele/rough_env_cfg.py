# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

# Canele articulation config
from isaaclab_assets.robots.canele.canele_cfg import CANELE_MINIMAL_CFG

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
class CaneleRewards(RewardsCfg):
    """Reward terms for the MDP (Canele)."""

    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-200.0,
    )

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5}
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[
                "right_toe_link",
                "left_toe_link",
            ]),
            "threshold": 0.4,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[
                "right_toe_link",
                "left_toe_link",
            ]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[
                "right_toe_link",
                "left_toe_link",
            ]),
        },
    )

    flat_toe_penalty = RewTerm(
        func=mdp.flat_orientation_links_l2,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[
                "right_toe_link",
                "left_toe_link"]),
                "margin": 0.0,
                "gain": 1.0, },
    )

    torso_height = RewTerm(
        func=mdp.local_torso_height_penalty_l2,
        weight=1.0,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=[
                "right_toe_link",
                "left_toe_link",
            ]),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "body_link",
                    "right_toe_link",
                    "left_toe_link",
                ],
            ),
            "target_height": 0.8,
            "margin": 0.0,
            "gain": 1.0,
        },
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_action_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
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
                ])},
    )

    joint_deviation_torso = RewTerm(
        func=mdp.joint_action_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["torso_yaw"])},
    )

    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_action_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                "right_hip_yaw",
                "left_hip_yaw",])},
    )

    joint_deviation_hip_roll = RewTerm(
        func=mdp.joint_action_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                "right_hip_roll",
                "left_hip_roll",])},
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

        base_link_full = base_paths[0]  # /Root/canele/body_link
        base_link_name = os.path.basename(base_link_full)  # body_link

        print("[DEBUG] base_link_full:", base_link_full)
        print("[DEBUG] base_link_name:", base_link_name)

        # Feet: ankle yaw links
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

        if hasattr(self.observations, "policy"):
            if hasattr(self.observations.policy, "base_lin_vel"):
                self.observations.policy.base_lin_vel = None

            if hasattr(self.observations.policy, "projected_gravity"):
                self.observations.policy.projected_gravity = None

            if hasattr(self.observations.policy, "height_scan"):
                self.observations.policy.height_scan = None

            # Add accelerometer-like linear acceleration (IMU-style) with uniform noise
            self.observations.policy.base_lin_acc_sens = ObsTerm(
                func=mdp.base_lin_acc_sens,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "gravity_mag": 9.81,
                },
                noise=Unoise(
                    n_min=-0.05,
                    n_max=0.05,
                ),
            )

        # -----------------------------------------------------------
        # 4b. Restrict action/observation joints to actuated joints only
        # -----------------------------------------------------------
        # Collect actuated joint names from the configured actuators.
        actuated_joint_names: list[str] = []
        try:
            for actuator_cfg in self.scene.robot.actuators.values():
                actuated_joint_names.extend(list(actuator_cfg.joint_names_expr))
        except Exception as e:
            print("[DEBUG] Failed to collect actuated joints from self.scene.robot.actuators:", e)

        # De-duplicate while preserving order
        _seen = set()
        actuated_joint_names = [j for j in actuated_joint_names if not (j in _seen or _seen.add(j))]

        print("[DEBUG] Actuated joint names count:", len(actuated_joint_names))
        print("[DEBUG] Actuated joint names:", actuated_joint_names)

        # Build an asset cfg that only exposes actuated joints.
        # Important: explicitly clear joint_ids. Isaac Lab errors if both joint_names and joint_ids
        # are set but not consistent (order-sensitive).
        def _make_actuated_asset_cfg() -> SceneEntityCfg:
            return SceneEntityCfg(
                "robot",
                joint_names=actuated_joint_names,
                joint_ids=slice(None),
                preserve_order=True,
            )

        # ---- Actions: override the joint selection of the joint position action term
        if hasattr(self, "actions") and hasattr(self.actions, "joint_pos"):
            action_term = self.actions.joint_pos
            if hasattr(action_term, "joint_names"):
                action_term.joint_names = actuated_joint_names
                print("[DEBUG] Restricted action term 'joint_pos' via joint_names")
            else:
                print("[DEBUG] Action term 'joint_pos' has no joint_names field; cannot restrict.")
        else:
            print("[DEBUG] No actions.joint_pos term found; cannot restrict action dimension.")

        # ---- Observations: force-inject asset_cfg into joint_pos/joint_vel terms
        if hasattr(self.observations, "policy"):
            for obs_name in ("joint_pos", "joint_vel"):
                obs_term = getattr(self.observations.policy, obs_name, None)
                if obs_term is None or not hasattr(obs_term, "params"):
                    continue
                if obs_term.params is None:
                    obs_term.params = {}
                # Use a fresh cfg per term to avoid cross-term mutation during resolve.
                obs_term.params["asset_cfg"] = _make_actuated_asset_cfg()
                print(f"[DEBUG] Injected actuated asset_cfg into observation term '{obs_name}'")

        # --- Fix: base_com observation expects 'base' in parent cfg, but Canele uses 'body_link'
        if hasattr(self.observations, "policy"):
            base_com_term = getattr(self.observations.policy, "base_com", None)
            if base_com_term is not None and hasattr(base_com_term, "params"):
                if base_com_term.params is None:
                    base_com_term.params = {}
                base_com_term.params["asset_cfg"] = SceneEntityCfg(
                    "robot",
                    body_names=[base_link_name],
                    preserve_order=True,
                )
                print(f"[DEBUG] Patched observation term 'base_com' to use body '{base_link_name}'")

        # --- Fix: base_com STARTUP EVENT expects 'base' in parent cfg, but Canele uses 'body_link'
        if hasattr(self, "events") and getattr(self, "events", None) is not None:
            event_base_com = getattr(self.events, "base_com", None)
            if event_base_com is not None and hasattr(event_base_com, "params"):
                if event_base_com.params is None:
                    event_base_com.params = {}
                # Some configs use key 'asset_cfg' (not nested in params for events); unify here.
                if "asset_cfg" in event_base_com.params and event_base_com.params["asset_cfg"] is not None:
                    try:
                        event_base_com.params["asset_cfg"].body_names = [base_link_name]
                        event_base_com.params["asset_cfg"].preserve_order = True
                    except Exception:
                        event_base_com.params["asset_cfg"] = SceneEntityCfg(
                            "robot",
                            body_names=[base_link_name],
                            preserve_order=True,
                        )
                else:
                    event_base_com.params["asset_cfg"] = SceneEntityCfg(
                        "robot",
                        body_names=[base_link_name],
                        preserve_order=True,
                    )
                print(f"[DEBUG] Patched startup event term 'base_com' to use body '{base_link_name}'")

        # -----------------------------------------------------------
        # 5. Rewards & terminations use short names only
        # -----------------------------------------------------------
        self.rewards.feet_slide.params["sensor_cfg"].body_names = ankle_names
        self.rewards.feet_slide.params["asset_cfg"].body_names = ankle_names
        
        print("[DEBUG] Feet link names for reward:", ankle_names)
        print("[DEBUG] Base contact link:", base_link_name)

        # -----------------------------------------------------------
        # Remaining default settings
        # -----------------------------------------------------------
        if self.scene.terrain.terrain_generator is not None:
            tg = self.scene.terrain.terrain_generator
            tg.difficulty_range = (0, 1.0)
            terrain_scale = 0.01

            # ★ 全ての段差の高さをスケールする処理 ★
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

        # Set PhysicsScene params
        self.sim.dt = 0.002  # Simulation: 500 Hz 
        self.decimation = 10  # Control: 50 Hz
        self.sim.render_interval = 4  # Rendering: 120Hz
        self.episode_length_s = 50.0
        self.sim.physx.max_position_iteration_count = 4
        self.sim.physx.min_position_iteration_count = 4
        self.sim.physx.max_velocity_iteration_count = 4
        self.sim.physx.min_velocity_iteration_count = 4
        # Slover type: PGS
        self.sim.physx.solver_type = 0
        # TODO: enableGPUDynamics = 0, broadphaseType = "MBP"

        # Randomize events
        self.events.physics_material.params["asset_cfg"].body_names = ankle_names
        self.events.physics_material.params["static_friction_range"] = (0.1, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.1, 1.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = [base_link_name]
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 1.0)
        self.events.base_com.params["asset_cfg"].body_names = [base_link_name]
        self.events.base_com.params["com_range"] = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)}
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [base_link_name]
        self.events.base_external_force_torque.params["force_range"] = (-2.0, 2.0)
        self.events.base_external_force_torque.params["torque_range"] = (-0.8, 0.8)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}
        self.events.push_robot.interval_range_s = (5.0, 20.0)

        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.05),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.3, 0.3),
            },
        }

        self.rewards.lin_vel_z_l2 = None
        self.rewards.dof_pos_limits = None
        self.rewards.undesired_contacts = None
        self.rewards.ang_vel_xy_l2.weight = -1.0
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.action_l1 = RewTerm(
            func=mdp.action_l1,
            weight=-0.001,
        )

        # self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_acc_l2 = None
        self.rewards.dof_acc_l2 = RewTerm(
            func=mdp.joint_action_acc_l2,
            weight=-1.0e-15,
            params={"dt": self.sim.dt},
        )
        self.rewards.action_acceleration_l1 = RewTerm(
            func=mdp.joint_action_acc_l1,
            weight=-5.0e-8,
            params={"dt": self.sim.dt},
        )
        # self.rewards.dof_torques_l2 = None
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[
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
            "right_ankle_roll", ])

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # terminations
        self.terminations.base_contact = None
        # self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        self.terminations.detect_fall = DoneTerm( # type: ignore
            func=mdp.detect_fall,
            params={
                "limit_angle": 1.3,
                "asset_cfg": SceneEntityCfg("robot", body_names=[base_link_name],),
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

        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
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
        self.observations.policy.height_scan = None

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Enable IO descriptor export at env startup
        self.export_io_descriptors = True
        