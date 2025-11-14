from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

# Kuroko config
from isaaclab_assets.robots.kuroko.kuroko_cfg import KUROKO_MINIMAL_CFG

# Added for USD inspection
from pxr import Usd
import fnmatch


def find_prim_paths(usd_path, pattern):
    """Search prims matching wildcard pattern (ex: *ankle*_yaw_link)."""
    stage = Usd.Stage.Open(usd_path)
    results = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if fnmatch.fnmatch(path.split("/")[-1], pattern):
            results.append(path)
    return results


@configclass
class KurokoRewards(RewardsCfg):
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

    # Placeholder (real paths will be set dynamically in env_cfg)
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

    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["ankle_.*"])},
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_.*"])},
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_.*", "elbow_.*"])},
    )

    joint_deviation_fingers = None

    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["chest"])},
    )


@configclass
class KurokoRoughLowEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: KurokoRewards = KurokoRewards()

    def __post_init__(self):
        super().__post_init__()

        # 1. Locate USD file
        usd_path = KUROKO_MINIMAL_CFG.spawn.usd_path
        print("[DEBUG] Loading USD:", usd_path)

        # 2. Find base link dynamically
        base_links = find_prim_paths(usd_path, "body_link")
        print("[DEBUG] Found base_link prims:", base_links)

        if not base_links:
            raise RuntimeError("body_link not found in USD!")

        base_link = base_links[0]  # /Root/kuroko/body_link

        # 3. Find feet (ankle yaw only)
        ankle_yaws = find_prim_paths(usd_path, "ankle_*_yaw_link")
        print("[DEBUG] Found ankle yaw prims:", ankle_yaws)

        if not ankle_yaws:
            raise RuntimeError("ankle_*_yaw_link not found in USD!")

        # 4. Inject dynamic paths into cfg
        self.scene.robot = KUROKO_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot{base_link}"

        # Rewards feet paths
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ankle_yaws
        self.rewards.feet_slide.params["sensor_cfg"].body_names = ankle_yaws
        self.rewards.feet_slide.params["asset_cfg"].body_names = ankle_yaws

        # External force & termination
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [base_link]
        self.terminations.base_contact.params["sensor_cfg"].body_names = base_link

        # Debug print of final resolved paths
        print("[DEBUG] height_scanner path:", self.scene.height_scanner.prim_path)
        print("[DEBUG] feet paths:", ankle_yaws)
        print("[DEBUG] base_contact:", base_link)

        # Keep rest same as original
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0, 0.0001)

        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]},
        }

        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005

        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.4)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-4.0, 4.0)


@configclass
class KurokoRoughLowEnvCfg_PLAY(KurokoRoughLowEnvCfg):
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
