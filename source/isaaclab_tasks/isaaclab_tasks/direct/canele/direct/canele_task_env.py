# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

"""Canele direct RL environment with Bimo-style observations and rough-env rewards."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, Imu, ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners import RigidBodyMaterialCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, gaussian_noise

from ..assets.canele_cfg import CANELE_MINIMAL_CFG
from .terminations import canele_terminations
from .rewards import canele_rewards_env
from .rewards import canele_rewards_walk
from .rewards import canele_rewards_joint
from .rewards import canele_rewards_link

LOWER_BODY_JOINTS = [
    "torso_yaw",
    "left_hip_yaw",
    "left_hip_pitch",
    "left_hip_roll",
    "left_knee_pitch",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_yaw",
    "right_hip_pitch",
    "right_hip_roll",
    "right_knee_pitch",
    "right_ankle_pitch",
    "right_ankle_roll",
]

HIP_JOINTS = [
    "left_hip_roll",
    "left_hip_pitch",
    "right_hip_roll",
    "right_hip_pitch",
]

TORSO_JOINTS = [
    "left_hip_yaw",
    "right_hip_yaw",
    "torso_yaw",
]

TORQUE_JOINTS = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee_pitch",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee_pitch",
]

RIGHT_FOOT = "right_toe_link"
LEFT_FOOT = "left_toe_link"
BASE_LINK = "body_link"


class _DirectCommandManager:
    def __init__(self, env: "CaneleEnv"):
        self._env = env

    def get_command(self, command_name: str) -> torch.Tensor:
        if command_name != "base_velocity":
            raise KeyError(f"Unsupported command name: {command_name}")
        return self._env.commands


class _ActionTerm:
    def __init__(self, joint_names: list[str]):
        self.action_dim = len(joint_names)
        self._joint_names = list(joint_names)
        self.IO_descriptor = SimpleNamespace(extras={"joint_names": list(joint_names)})


class _DirectActionManager:
    def __init__(self, env: "CaneleEnv", joint_names: list[str]):
        self._env = env
        self.active_terms = ["joint_position"]
        self._term = _ActionTerm(joint_names)
        self.total_action_dim = int(env.cfg.action_space)
        self.action = torch.zeros(
            env.scene.num_envs, len(joint_names), device=env.device
        )
        self.prev_action = torch.zeros_like(self.action)
        self.prev_prev_action = torch.zeros_like(self.action)

    def get_term(self, term_name: str):
        if term_name != "joint_position":
            raise KeyError(f"Unsupported action term: {term_name}")
        return self._term


class _DirectTerminationManager:
    def __init__(self, env: "CaneleEnv"):
        self._env = env
        self.terminated = torch.zeros(
            env.scene.num_envs, dtype=torch.bool, device=env.device
        )
        self.time_outs = torch.zeros_like(self.terminated)

    def get_term(self, term: str) -> torch.Tensor:
        if term == "terminated":
            return self.terminated.float()
        raise KeyError(f"Unsupported termination term: {term}")


@configclass
class CaneleEnvCfg(DirectRLEnvCfg):
    decimation = 10
    episode_length_s = 20.0
    observation_space = 20 + 4 * len(LOWER_BODY_JOINTS)
    action_space = len(LOWER_BODY_JOINTS)
    state_space = 0
    dt = 0.005

    sim: SimulationCfg = SimulationCfg(dt=dt)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        env_spacing=2.5, replicate_physics=True
    )
    robot_cfg: ArticulationCfg = CANELE_MINIMAL_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    imu: ImuCfg = ImuCfg(
        prim_path=f"/World/envs/env_.*/Robot/{BASE_LINK}",
        debug_vis=False,
        update_period=dt * decimation,
    )

    contact: ContactSensorCfg = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/.*toe_link",
        history_length=3,
        update_period=dt,
        debug_vis=False,
        track_pose=True,
        track_air_time=True,
        force_threshold=0.001,
    )


class CaneleEnv(DirectRLEnv):
    def __init__(self, cfg: CaneleEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        self.joint_names = list(LOWER_BODY_JOINTS)
        self.joint_ids = [self._joint_name_to_id(name) for name in self.joint_names]
        self.base_id = self._body_name_to_id(BASE_LINK)
        self.foot_ids = [
            self._body_name_to_id(RIGHT_FOOT),
            self._body_name_to_id(LEFT_FOOT),
        ]
        self.base_and_feet_ids = [self.base_id] + self.foot_ids

        self.hip_joint_ids = [
            self.joint_ids[self.joint_names.index(joint_name)]
            for joint_name in HIP_JOINTS
        ]
        self.torso_joint_ids = [
            self.joint_ids[self.joint_names.index(joint_name)]
            for joint_name in TORSO_JOINTS
        ]
        self.torque_joint_ids = [
            self.joint_ids[self.joint_names.index(joint_name)]
            for joint_name in TORQUE_JOINTS
        ]

        lower_limits = self.robot.data.soft_joint_pos_limits[:, self.joint_ids, 0]
        upper_limits = self.robot.data.soft_joint_pos_limits[:, self.joint_ids, 1]
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits

        self.base_pose = torch.rad2deg(
            self.robot.data.default_joint_pos[:, self.joint_ids]
        ).clone()
        self.cmd_actions = self.base_pose.clone()
        self.noisy_act = self.base_pose.clone()

        self.orient_noise = GaussianNoiseCfg(mean=0.0, std=0.015, operation="add")
        self.gyro_noise = GaussianNoiseCfg(mean=0.0, std=0.01, operation="add")
        self.actuator_noise = GaussianNoiseCfg(mean=0.0, std=0.01, operation="add")

        self.orient_h = torch.zeros(self.scene.num_envs, 4, 3, device=self.device)
        self.gyro_h = torch.zeros(self.scene.num_envs, 4, 3, device=self.device)

        norm_base = normalize_actions(
            self.base_pose, self.lower_limits, self.upper_limits
        )
        self.act_hist = norm_base.unsqueeze(1).repeat(1, 4, 1)

        self.commands = torch.zeros(self.scene.num_envs, 3, device=self.device)

        self.command_manager = _DirectCommandManager(self)
        self.action_manager = _DirectActionManager(self, self.joint_names)
        self.termination_manager = _DirectTerminationManager(self)

        num_contact_bodies = int(self.contact.data.current_contact_time.shape[1])
        if num_contact_bodies < 2:
            raise RuntimeError(
                f"ContactSensor matched only {num_contact_bodies} body/bodies for {self.cfg.contact.prim_path}"
            )
        self.contact_sensor_ids = [0, 1]
        self.foot_ids = self.foot_ids

        self.base_body_cfg = self._make_body_cfg("robot", [BASE_LINK], [self.base_id])
        self.feet_body_cfg = self._make_body_cfg("robot", [RIGHT_FOOT, LEFT_FOOT], self.foot_ids)
        self.base_and_feet_body_cfg = self._make_body_cfg(
            "robot", [BASE_LINK, RIGHT_FOOT, LEFT_FOOT], self.base_and_feet_ids
        )
        self.contact_sensor_cfg = self._make_body_cfg(
            "contact_forces", [RIGHT_FOOT, LEFT_FOOT], self.contact_sensor_ids
        )
        self.hip_action_cfg = self._make_joint_cfg("robot", HIP_JOINTS)
        self.torso_action_cfg = self._make_joint_cfg("robot", TORSO_JOINTS)
        self.torque_action_cfg = self._make_joint_cfg("robot", TORQUE_JOINTS)

    def _joint_name_to_id(self, joint_name: str) -> int:
        matches = self.robot.find_joints(joint_name)
        if isinstance(matches, tuple):
            return int(matches[0][0])
        return int(matches[0])

    def _body_name_to_id(self, body_name: str) -> int:
        matches = self.robot.find_bodies(body_name)
        if isinstance(matches, tuple):
            return int(matches[0][0])
        return int(matches[0])

    def _make_body_cfg(
        self, name: str, body_names: list[str], body_ids: list[int]
    ) -> SceneEntityCfg:
        cfg = SceneEntityCfg(name, body_names=list(body_names), preserve_order=True)
        cfg.body_ids = list(body_ids)
        return cfg

    def _make_joint_cfg(self, name: str, joint_names: list[str]) -> SceneEntityCfg:
        joint_ids = [self._joint_name_to_id(jn) for jn in joint_names]
        cfg = SceneEntityCfg(name, joint_names=list(joint_names), preserve_order=True)
        cfg.joint_ids = list(joint_ids)
        return cfg

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        self.imu = Imu(self.cfg.imu)
        self.scene.sensors["imu"] = self.imu

        self.contact = ContactSensor(self.cfg.contact)
        self.scene.sensors["contact"] = self.contact
        self.scene.sensors["contact_forces"] = self.contact

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        ground_cfg = RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=0.7,
            restitution=0.0,
            friction_combine_mode="average",
        )
        spawn_ground_plane(
            prim_path="/World/ground", cfg=GroundPlaneCfg(physics_material=ground_cfg)
        )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.cfg.viewer.eye = (5.0, -5.0, 3.5)
        self.cfg.viewer.lookat = (0.0, 0.0, 1.0)

    def update_imu_history(self, new_orient, new_gyro):
        self.orient_h[:, :-1] = self.orient_h[:, 1:].clone()
        self.gyro_h[:, :-1] = self.gyro_h[:, 1:].clone()
        self.orient_h[:, -1] = new_orient
        self.gyro_h[:, -1] = new_gyro

    def _get_observations(self):
        imu_data = self.scene.sensors["imu"].data
        orient = quaternion_to_euler(imu_data.quat_w)
        orient = gaussian_noise(orient, self.orient_noise)
        angular_vel = gaussian_noise(imu_data.ang_vel_b, self.gyro_noise)

        orient = scale_value(orient, -1.0, 1.0)
        angular_vel = scale_value(angular_vel, -2.0, 2.0)

        self.update_imu_history(orient, angular_vel)

        imu_hist = torch.cat((self.orient_h[:, :, :2], self.gyro_h), dim=2).reshape(
            self.scene.num_envs, 20
        )

        cmd_act_norm = normalize_actions(
            self.cmd_actions, self.lower_limits, self.upper_limits
        )
        self.act_hist[:, :-1] = self.act_hist[:, 1:].clone()
        self.act_hist[:, -1] = cmd_act_norm
        act_hist = self.act_hist.reshape(self.scene.num_envs, 4 * len(self.joint_names))

        obs_buffer = torch.cat((imu_hist, act_hist), dim=1)
        obs_buffer = torch.round(obs_buffer, decimals=4)

        return {"policy": obs_buffer}

    def _pre_physics_step(self, actions):
        self.action_manager.prev_prev_action.copy_(self.action_manager.prev_action)
        self.action_manager.prev_action.copy_(self.action_manager.action)
        self.action_manager.action.copy_(torch.clamp(actions, -1.0, 1.0))

        self.cmd_actions = denormalize_actions(
            self.action_manager.action, self.lower_limits, self.upper_limits
        )
        self.noisy_act = gaussian_noise(self.cmd_actions, self.actuator_noise)
        self.noisy_act = torch.max(
            torch.min(self.noisy_act, self.upper_limits), self.lower_limits
        )

    def _apply_action(self):
        self.robot.set_joint_position_target(self.noisy_act, joint_ids=self.joint_ids)

    def _compute_terminated(self) -> torch.Tensor:
        return (
            canele_terminations.detect_fall(
                self,
                limit_angle=1.3,
                asset_cfg=self.base_body_cfg,
            )
            | canele_terminations.detect_height_too_low_relative(
                self,
                min_height=0.5,
                asset_cfg=self.base_and_feet_body_cfg,
            )
            | canele_terminations.detect_tilt_too_high_any_link(
                self,
                max_tilt=1.5,
                asset_cfg=self.base_and_feet_body_cfg,
            )
            | canele_terminations.detect_support_plane_tilt_too_high(
                self,
                max_tilt=1.3,
                asset_cfg=self.base_and_feet_body_cfg,
            )
        )

    def _joint_torque_l2(self, joint_ids: list[int]) -> torch.Tensor:
        ids = torch.as_tensor(joint_ids, device=self.device, dtype=torch.long)
        # Prefer applied torque if available; otherwise fall back to computed torque.
        if hasattr(self.robot.data, "applied_torque"):
            torques = self.robot.data.applied_torque.index_select(1, ids)
        elif hasattr(self.robot.data, "computed_torque"):
            torques = self.robot.data.computed_torque.index_select(1, ids)
        elif hasattr(self.robot.data, "joint_torques"):
            torques = self.robot.data.joint_torques.index_select(1, ids)
        else:
            return torch.zeros(self.scene.num_envs, device=self.device)
        return torch.sum(torch.square(torques), dim=1)

    def _action_rate_l2(self) -> torch.Tensor:
        delta = self.action_manager.action - self.action_manager.prev_action
        return torch.sum(torch.square(delta), dim=1)

    def _get_rewards(self):
        terminated = self._compute_terminated()
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        self.termination_manager.terminated = terminated
        self.termination_manager.time_outs = truncated

        reward = torch.zeros(self.scene.num_envs, device=self.device)

        reward += -200.0 * canele_rewards_env.is_terminated(self)
        reward += 1.0 * canele_rewards_walk.track_lin_vel_xy_yaw_frame_exp_no_flight(
            self,
            std=0.5,
            command_name="base_velocity",
            sensor_cfg=self.contact_sensor_cfg,
        )
        reward += 2.0 * canele_rewards_walk.track_ang_vel_z_world_exp_no_flight(
            self,
            command_name="base_velocity",
            std=0.5,
            sensor_cfg=self.contact_sensor_cfg,
        )
        reward += 1.0 * canele_rewards_walk.feet_air_time_alternating_biped(
            self,
            command_name="base_velocity",
            sensor_cfg=self.contact_sensor_cfg,
            linear_cmd_threshold=0.0,
            angular_cmd_threshold=0.0,
            body_tilt_threshold=0.0,
            air_min_time=0.1,
            air_max_time=1.0,
            min_contact_time=0.1,
            ema_alpha=0.02,
            air_balance_weight=1.0,
            contact_balance_weight=0.0,
            air_reward=1.0,
            contact_reward=1.0,
        )
        reward += -0.1 * canele_rewards_walk.feet_slide_keep_flat(
            self,
            sensor_cfg=self.contact_sensor_cfg,
            asset_cfg=self.feet_body_cfg,
            air_time_eps=0.02,
        )
        reward += -0.01 * canele_rewards_joint.joint_action_deviation_l1(
            self,
            asset_cfg=self.hip_action_cfg,
        )
        reward += -0.1 * canele_rewards_joint.joint_action_deviation_l1(
            self,
            asset_cfg=self.torso_action_cfg,
        )
        reward += 0.1 * canele_rewards_link.flat_orientation_links_l2(
            self,
            asset_cfg=self.feet_body_cfg,
            margin=0.0,
            gain=1.0,
        )

        reward += -0.2 * canele_rewards_link.lin_vel_z_l2(self)
        reward += -1.0 * canele_rewards_link.flat_orientation_l2(self)
        reward += -0.01 * canele_rewards_link.ang_vel_xy_l2(self)
        reward += -0.01 * self._action_rate_l2()
        reward += -1.0e-9 * canele_rewards_joint.joint_action_acc_l2(
            self,
            dt=self.cfg.decimation * self.cfg.sim.dt,
        )
        reward += -2.0e-6 * self._joint_torque_l2(self.torque_joint_ids)

        return reward

    def _get_dones(self):
        terminated = self._compute_terminated()
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        self.termination_manager.terminated = terminated
        self.termination_manager.time_outs = truncated
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.orient_h[env_ids] = 0.0
        self.gyro_h[env_ids] = 0.0
        self.cmd_actions[env_ids] = self.base_pose[env_ids]
        self.noisy_act[env_ids] = self.base_pose[env_ids]

        reset_norm = normalize_actions(
            self.base_pose[env_ids],
            self.lower_limits[env_ids],
            self.upper_limits[env_ids],
        )
        self.action_manager.action[env_ids] = reset_norm
        self.action_manager.prev_action[env_ids] = reset_norm
        self.action_manager.prev_prev_action[env_ids] = reset_norm
        self.act_hist[env_ids] = reset_norm.unsqueeze(1).repeat(1, 4, 1)

        self.termination_manager.terminated[env_ids] = False
        self.termination_manager.time_outs[env_ids] = False

        self.commands[env_ids, 0] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(-0.6, 0.6)
        self.commands[env_ids, 1] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(-0.6, 0.6)
        self.commands[env_ids, 2] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(-1.2, 1.2)


@torch.jit.script
def quaternion_to_euler(quat: torch.Tensor):
    quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.tensor(torch.pi / 2, device=quat.device),
        torch.asin(sinp),
    )
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=1)


@torch.jit.script
def scale_value(value: torch.Tensor, min_val: float, max_val: float):
    return torch.clamp((value - min_val) / (max_val - min_val) * 2.0 - 1.0, -1.0, 1.0)


@torch.jit.script
def normalize_actions(
    value_deg: torch.Tensor, lower_deg: torch.Tensor, upper_deg: torch.Tensor
):
    denom = (upper_deg - lower_deg).clamp_min(1e-6)
    return torch.clamp((value_deg - lower_deg) / denom * 2.0 - 1.0, -1.0, 1.0)


@torch.jit.script
def denormalize_actions(
    value_norm: torch.Tensor, lower_deg: torch.Tensor, upper_deg: torch.Tensor
):
    return lower_deg + (value_norm + 1.0) * 0.5 * (upper_deg - lower_deg)


@torch.jit.script
def quat_apply_inverse_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v - q_w * t + torch.cross(q_xyz, t, dim=-1)
