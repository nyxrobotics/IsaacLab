# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

"""Canele direct RL environment with Bimo-style observations."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, Imu, ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners import RigidBodyMaterialCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, gaussian_noise

from .canele_cfg import CANELE_MINIMAL_CFG


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

LEFT_FOOT = "left_toe_link"
RIGHT_FOOT = "right_toe_link"
BASE_LINK = "body_link"


@configclass
class CaneleEnvCfg(DirectRLEnvCfg):
    decimation = 10
    episode_length_s = 20.0
    observation_space = 72
    action_space = len(LOWER_BODY_JOINTS)
    state_space = 0
    dt = 0.005

    obj = "walk"

    sim: SimulationCfg = SimulationCfg(dt=dt)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(env_spacing=2.5, replicate_physics=True)
    robot_cfg: ArticulationCfg = CANELE_MINIMAL_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    imu: ImuCfg = ImuCfg(
        prim_path=f"/World/envs/env_.*/Robot/{BASE_LINK}",
        debug_vis=False,
        update_period=dt * decimation,
    )

    contact: ContactSensorCfg = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/.*toe_link",
        history_length=0,
        update_period=dt,
        debug_vis=False,
        track_pose=True,
        track_air_time=True,
        force_threshold=0.001,
    )


class CaneleEnv(DirectRLEnv):
    def __init__(self, cfg: CaneleEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.obj = self.cfg.obj
        self.joint_names = list(LOWER_BODY_JOINTS)
        self.num_actions = len(self.joint_names)
        self.joint_ids = [self.robot_joint_name_to_id(name) for name in self.joint_names]

        default_joint_pos = self.robot.data.default_joint_pos[:, self.joint_ids]
        self.base_pose = torch.rad2deg(default_joint_pos).clone()
        self.cmd_actions = self.base_pose.clone()
        self.noisy_act = self.base_pose.clone()

        self.orient_noise = GaussianNoiseCfg(mean=0.0, std=0.015, operation="add")
        self.gyro_noise = GaussianNoiseCfg(mean=0.0, std=0.01, operation="add")
        self.actuator_noise = GaussianNoiseCfg(mean=0.0, std=0.25, operation="add")

        self.orient_h = torch.zeros(self.scene.num_envs, 4, 3, device=self.device)
        self.gyro_h = torch.zeros(self.scene.num_envs, 4, 3, device=self.device)
        self.act_hist = torch.zeros(self.scene.num_envs, 4, self.num_actions, device=self.device)

        self.lower_limits = torch.rad2deg(self.robot.data.soft_joint_pos_limits[:, self.joint_ids, 0]).clone()
        self.upper_limits = torch.rad2deg(self.robot.data.soft_joint_pos_limits[:, self.joint_ids, 1]).clone()
        start_norm = normalize_actions(self.base_pose, self.lower_limits, self.upper_limits)
        self.act_hist[:] = start_norm.unsqueeze(1)

    def robot_joint_name_to_id(self, joint_name: str) -> int:
        matches = self.robot.find_joints(joint_name)
        if isinstance(matches, tuple):
            return int(matches[0][0])
        return int(matches[0])

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        self.imu = Imu(self.cfg.imu)
        self.scene.sensors["imu"] = self.imu

        self.contact = ContactSensor(self.cfg.contact)
        self.scene.sensors["contact"] = self.contact

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        ground_cfg = RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=0.7,
            restitution=0.0,
            friction_combine_mode="average",
        )
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(physics_material=ground_cfg))

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
        imu_obs = torch.cat((self.orient_h[:, :, :2], self.gyro_h), dim=2).reshape(self.scene.num_envs, 20)

        cmd_act = normalize_actions(self.cmd_actions, self.lower_limits, self.upper_limits)
        self.act_hist[:, :-1] = self.act_hist[:, 1:].clone()
        self.act_hist[:, -1] = cmd_act
        act_obs = self.act_hist.reshape(self.scene.num_envs, 4 * self.num_actions)

        obs_buffer = torch.cat((imu_obs, act_obs), dim=1)
        obs_buffer = torch.round(obs_buffer, decimals=4)
        return {"policy": obs_buffer}

    def _pre_physics_step(self, actions):
        delta_deg = torch.clamp(actions, -3.0, 3.0) * 0.5
        self.cmd_actions = torch.clamp(self.cmd_actions + delta_deg, self.lower_limits, self.upper_limits)
        self.noisy_act = torch.clamp(
            gaussian_noise(self.cmd_actions, self.actuator_noise),
            self.lower_limits,
            self.upper_limits,
        )

    def _apply_action(self):
        target = torch.deg2rad(self.noisy_act)
        self.robot.set_joint_position_target(target, joint_ids=self.joint_ids)

    def _get_rewards(self):
        root_pos = self.robot.data.root_pos_w
        root_lin_vel = self.robot.data.root_com_vel_w[:, :3]
        root_ang_vel = self.robot.data.root_com_vel_w[:, 3:]
        contact_data = self.scene.sensors["contact"].data
        air_time = contact_data.current_air_time
        foot_pos = contact_data.pos_w
        joint_error = torch.abs(self.cmd_actions - self.base_pose)

        height_target = 0.95
        height_reward = torch.exp(-10.0 * torch.square(root_pos[:, 2] - height_target))
        upright_reward = torch.exp(-2.0 * torch.sum(torch.square(quaternion_to_euler(self.robot.data.root_quat_w)[:, :2]), dim=1))
        forward_reward = torch.clamp(root_lin_vel[:, 0], min=0.0, max=0.8)
        yaw_penalty = torch.abs(root_ang_vel[:, 2])
        pose_penalty = torch.mean(joint_error / 30.0, dim=1)
        feet_reward = feet_height_reward(air_time, foot_pos, 0.04, 100.0)

        reward = (
            1.0 * height_reward
            + 1.0 * upright_reward
            + 1.0 * forward_reward
            + 0.5 * feet_reward
            - 0.2 * yaw_penalty
            - 0.1 * pose_penalty
        )
        return reward

    def _get_dones(self):
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        root_pos = self.robot.data.root_pos_w
        euler_angles = quaternion_to_euler(self.robot.data.root_quat_w)
        fallen = (root_pos[:, 2] < 0.45) | (torch.abs(euler_angles[:, 0]) > 1.2) | (torch.abs(euler_angles[:, 1]) > 1.2)
        return fallen, truncated

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
        start_norm = normalize_actions(self.base_pose[env_ids], self.lower_limits[env_ids], self.upper_limits[env_ids])
        self.act_hist[env_ids] = start_norm.unsqueeze(1)


@torch.jit.script
def quaternion_to_euler(quat: torch.Tensor):
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * (torch.pi / 2), torch.asin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=1)


@torch.jit.script
def scale_value(value: torch.Tensor, min_val: float, max_val: float):
    return torch.clamp((value - min_val) / (max_val - min_val) * 2 - 1, -1, 1)


@torch.jit.script
def normalize_actions(value: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor):
    return torch.clamp((value - min_val) / (max_val - min_val) * 2 - 1, -1, 1)


@torch.jit.script
def feet_height_reward(air_time, feet_pos, target_h: float, scale: float = 25.0):
    in_air = air_time > 0
    num_in_air = in_air.sum(dim=1)
    both_in_air = num_in_air == 2
    both_on_ground = num_in_air == 0

    z_pos = feet_pos[..., 2]
    z_err = torch.abs(z_pos - target_h)
    reward_per_leg = torch.where(z_pos >= target_h, torch.ones_like(z_pos), torch.exp(-scale * z_err)) * in_air.float()
    reward = reward_per_leg.sum(dim=1)
    reward = torch.where(both_in_air | both_on_ground, torch.zeros_like(reward), reward)
    return reward
