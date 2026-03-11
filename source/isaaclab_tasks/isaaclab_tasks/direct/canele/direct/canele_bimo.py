# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

"""Canele direct RL environment with Bimo-style action, observation, and reward.

All angles in this file are expressed in radians.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, Imu, ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners import RigidBodyMaterialCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, gaussian_noise

from ..assets.canele_cfg import CANELE_MINIMAL_CFG
from .terminations import canele_terminations

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

RIGHT_FOOT = "right_toe_link"
LEFT_FOOT = "left_toe_link"
BASE_LINK = "body_link"

ACTION_SCALE_RAD = math.radians(2.0 / 3.0)
BACKLASH_RAD = math.radians(1.6)
ACTUATOR_NOISE_STD_RAD = math.radians(0.5)


@configclass
class CaneleEnvCfg(DirectRLEnvCfg):
    # Environment settings
    dt = 0.005  # Physics: 200 Hz (decimation of 4 for control at 50 Hz)
    decimation = 4  # Control: 50Hz
    episode_length_s = 20.0
    observation_space = 20 + 4 * len(LOWER_BODY_JOINTS)  # +1 when obj == "turn"
    action_space = len(LOWER_BODY_JOINTS)
    state_space = 0

    # Training objective: walk | turn | stop
    obj = "walk"

    # Reward weights
    # [orientation, height, joint pos, joint pos sigmoid, feet height, velocity, deviation]
    weights = {
        "stop": [1, 1, 1, 1, 1, 0, 0],
        "walk": [1, 1, 1, 0, 2, 1, 1],
        "turn": [1, 1, 1, 1, 2, 1, 2],
    }

    # Canele-specific reward targets
    body_height_target = 0.95
    feet_height_target = 0.1

    # COM randomization: sampled per episode, kept constant during the episode
    com_shift_max = 0.03

    # Actuator settings (all radians)
    actuator_delay_max = 4  # physics steps
    actuator_delay_min = 1  # physics steps
    backlash = BACKLASH_RAD
    action_scale = ACTION_SCALE_RAD
    actuator_noise_std = ACTUATOR_NOISE_STD_RAD

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=dt)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        env_spacing=2.5,
        replicate_physics=True,
    )

    # Robot configuration
    robot_cfg: ArticulationCfg = CANELE_MINIMAL_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # Sensors configuration
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

        self.obj = self.cfg.obj
        self.weights = torch.tensor(
            self.cfg.weights[self.obj],
            device=self.device,
            dtype=torch.float32,
        ).repeat(self.scene.num_envs, 1)

        self.joint_names = list(LOWER_BODY_JOINTS)
        self.joint_ids = [self._joint_name_to_id(name) for name in self.joint_names]
        self.base_id = self._body_name_to_id(BASE_LINK)
        self.foot_ids = [
            self._body_name_to_id(RIGHT_FOOT),
            self._body_name_to_id(LEFT_FOOT),
        ]
        self.base_and_feet_ids = [self.base_id] + self.foot_ids

        self.base_body_cfg = self._make_body_cfg("robot", [BASE_LINK], [self.base_id])
        self.base_and_feet_body_cfg = self._make_body_cfg(
            "robot",
            [BASE_LINK, RIGHT_FOOT, LEFT_FOOT],
            self.base_and_feet_ids,
        )

        self.lower_limits = self.robot.data.soft_joint_pos_limits[:, self.joint_ids, 0].clone()
        self.upper_limits = self.robot.data.soft_joint_pos_limits[:, self.joint_ids, 1].clone()

        # Initial posture comes from CANELE_MINIMAL_CFG (already in radians).
        self.base_pose = self.robot.data.default_joint_pos[:, self.joint_ids].clone()
        self.position_reward_max_diff = torch.maximum(
            self.upper_limits - self.base_pose,
            self.base_pose - self.lower_limits,
        ).clamp_min(1e-6)

        # Buffers for joint targets
        self.cmd_actions = self.base_pose.clone()  # commanded by the policy (rad)
        self.last_direction = torch.zeros_like(self.base_pose)
        self.gear_position = self.base_pose.clone()  # applied joint targets before noise (rad)
        self.noisy_act = self.base_pose.clone()  # final actuator target (rad)

        # Action direction: turn left (-1) / right (+1)
        half = self.scene.num_envs // 2
        self.act_direction = torch.cat(
            (
                torch.ones(half, device=self.device),
                -torch.ones(self.scene.num_envs - half, device=self.device),
            ),
            dim=0,
        )

        # Noise settings
        self.orient_noise = GaussianNoiseCfg(mean=0.0, std=0.015, operation="add")
        self.gyro_noise = GaussianNoiseCfg(mean=0.0, std=0.01, operation="add")
        self.actuator_noise = GaussianNoiseCfg(
            mean=0.0,
            std=self.cfg.actuator_noise_std,
            operation="add",
        )

        # Actuator delay
        self.act_timer = 0
        self.act_delay = 0

        # History buffers
        self.orient_h = torch.zeros(self.scene.num_envs, 4, 3, device=self.device)
        self.gyro_h = torch.zeros(self.scene.num_envs, 4, 3, device=self.device)

        normalized_base = normalize_actions(
            self.base_pose,
            self.lower_limits,
            self.upper_limits,
        )
        self.act_hist = normalized_base.unsqueeze(1).repeat(1, 4, 1)

        # COM shift buffers
        self.default_coms: torch.Tensor | None = None
        self.com_shift_episode = torch.zeros(self.scene.num_envs, 3, device=self.device)

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
        self,
        name: str,
        body_names: list[str],
        body_ids: list[int],
    ) -> SceneEntityCfg:
        cfg = SceneEntityCfg(name, body_names=list(body_names), preserve_order=True)
        cfg.body_ids = list(body_ids)
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
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(physics_material=ground_cfg),
        )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.cfg.viewer.eye = (5.0, -5.0, 3.5)
        self.cfg.viewer.lookat = (0.0, 0.0, 1.0)

    def _get_observations(self):
        # Get IMU data, add noise and scale to [-1, 1]
        imu_data = self.scene.sensors["imu"].data
        orient = quaternion_to_euler(imu_data.quat_w)
        orient = gaussian_noise(orient, self.orient_noise)
        angular_vel = gaussian_noise(imu_data.ang_vel_b, self.gyro_noise)

        orient = scale_value(orient, -1.0, 1.0)
        angular_vel = scale_value(angular_vel, -2.0, 2.0)

        # Update IMU history and arrange for observations
        self.update_imu_history(orient, angular_vel)
        imu_hist = torch.cat((self.orient_h[:, :, :2], self.gyro_h), dim=2)
        imu_hist = imu_hist.reshape(self.scene.num_envs, 20)

        # Get last commanded position and scale to [-1, 1]
        cmd_act = normalize_actions(
            self.cmd_actions,
            self.lower_limits,
            self.upper_limits,
        )

        # Update action history and arrange for observations
        self.act_hist[:, :-1] = self.act_hist[:, 1:].clone()
        self.act_hist[:, -1] = cmd_act
        proc_act = self.act_hist.reshape(self.scene.num_envs, 4 * len(self.joint_names))

        # Create observation buffer
        if self.obj != "turn":
            obs_buffer = torch.cat((imu_hist, proc_act), dim=1)
        else:
            obs_buffer = torch.cat(
                (self.act_direction.unsqueeze(1), imu_hist, proc_act),
                dim=1,
            )

        obs_buffer = torch.round(obs_buffer, decimals=4)
        return {"policy": obs_buffer}

    def _pre_physics_step(self, actions):
        # Calculates action delta in radians (Bimo-style, but fully radian-based)
        actions_cpy = torch.clamp(actions.clone(), -3.0, 3.0)
        self.cmd_actions += actions_cpy * self.cfg.action_scale
        self.cmd_actions = torch.max(
            torch.min(self.cmd_actions, self.upper_limits),
            self.lower_limits,
        )

        # Simulates backlash
        delta = self.cmd_actions - self.gear_position
        direction = torch.sign(delta)
        direction_changed = (direction != self.last_direction) & (self.last_direction != 0.0)

        movement = torch.where(
            direction_changed,
            torch.clamp(torch.abs(delta) - self.cfg.backlash, min=0.0) * direction,
            delta,
        )

        self.gear_position += movement
        self.gear_position = torch.max(
            torch.min(self.gear_position, self.upper_limits),
            self.lower_limits,
        )
        self.last_direction = torch.where(delta != 0.0, direction, self.last_direction)

        # Adds noise
        self.noisy_act = gaussian_noise(self.gear_position, self.actuator_noise)
        self.noisy_act = torch.max(
            torch.min(self.noisy_act, self.upper_limits),
            self.lower_limits,
        )

        # Action delay: 5 ms - 20 ms (1-4 physics steps at dt=0.005)
        self.act_timer = 0
        self.act_delay = torch.randint(
            low=self.cfg.actuator_delay_min,
            high=self.cfg.actuator_delay_max + 1,
            size=(1,),
            device=self.device,
        ).item()

    def _apply_action(self):
        # Applies policy action
        self.act_timer += 1
        if self.act_timer >= self.act_delay:
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

    def _get_rewards(self):
        # Get data for reward
        imu_data = self.scene.sensors["imu"].data
        contact_data = self.scene.sensors["contact"].data

        euler_imu_orient = quaternion_to_euler(imu_data.quat_w)
        root_pos = self.robot.data.root_pos_w
        lin_vel = self.robot.data.root_com_vel_w
        contact_pos = contact_data.pos_w
        air_time = contact_data.current_air_time

        # Compute reward components (same structure as bimo_task_env.py)
        orientation_rew = orientation_reward(euler_imu_orient, self.obj)
        height_rew = height_reward(root_pos, self.cfg.body_height_target)
        position_rew = joint_position_reward(
            self.cmd_actions,
            self.base_pose,
            self.position_reward_max_diff,
        )
        sig_extra = sigmoid_extra(self.cmd_actions, self.base_pose)
        vel_rew = velocity_reward(lin_vel, self.act_direction, self.obj)
        feet_h_rew = feet_height_reward(
            air_time,
            contact_pos,
            self.cfg.feet_height_target,
            150.0,
        )
        dev_rew = deviation_reward(self.scene.env_origins, root_pos, self.obj)

        # Compute weighted reward
        w = self.weights / torch.sum(self.weights, dim=1, keepdim=True)
        total_reward = (
            orientation_rew * w[:, 0]
            + height_rew * w[:, 1]
            + position_rew * w[:, 2]
            + sig_extra * w[:, 3]
            + feet_h_rew * w[:, 4]
            + vel_rew * w[:, 5]
            + dev_rew * w[:, 6]
        )

        return total_reward

    def _get_dones(self):
        terminated = self._compute_terminated()
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.flatten().long()

        super()._reset_idx(env_ids)

        self._randomize_episode_com_shift(env_ids)

        # Get default root pose and add env origin position (for spacing)
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]

        # Get default joint positions and velocities from CANELE_MINIMAL_CFG init_state
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # Write data to sim
        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset buffers
        self.orient_h[env_ids] = 0.0
        self.gyro_h[env_ids] = 0.0

        reset_norm = normalize_actions(
            self.base_pose[env_ids],
            self.lower_limits[env_ids],
            self.upper_limits[env_ids],
        )
        self.act_hist[env_ids] = reset_norm.unsqueeze(1).repeat(1, 4, 1)
        self.cmd_actions[env_ids] = self.base_pose[env_ids]
        self.noisy_act[env_ids] = self.base_pose[env_ids]
        self.gear_position[env_ids] = self.base_pose[env_ids]
        self.last_direction[env_ids] = 0.0

    def _randomize_episode_com_shift(self, env_ids: torch.Tensor):
        physx_view = self.robot.root_physx_view
        current_coms = physx_view.get_coms()
        if self.default_coms is None:
            self.default_coms = current_coms.clone()

        # PhysX expects the same global COM tensor shape as get_coms(),
        # even when indices are provided to set_coms().
        full_coms = current_coms.clone()
        base_coms = self.default_coms
        com_device = full_coms.device
        env_ids_com = env_ids.to(device=com_device, dtype=torch.long).flatten()

        num_reset = env_ids_com.numel()
        if self.cfg.com_shift_max > 0.0:
            directions = torch.randn((num_reset, 3), device=com_device)
            directions = directions / torch.norm(directions, dim=1, keepdim=True).clamp_min(1e-6)
            magnitudes = torch.rand((num_reset, 1), device=com_device) * float(self.cfg.com_shift_max)
            shifts = directions * magnitudes
        else:
            shifts = torch.zeros((num_reset, 3), device=com_device)

        if full_coms.dim() == 2:
            full_coms[env_ids_com, :3] = base_coms[env_ids_com, :3] + shifts
        else:
            body_index = self.base_id if self.base_id < full_coms.shape[1] else 0
            full_coms[env_ids_com, body_index, :3] = (
                base_coms[env_ids_com, body_index, :3] + shifts
            )

        physx_view.set_coms(full_coms, indices=env_ids.cpu())
        self.com_shift_episode[env_ids] = shifts.to(self.device)

    def update_imu_history(self, new_orient, new_gyro):
        self.orient_h[:, :-1] = self.orient_h[:, 1:].clone()
        self.gyro_h[:, :-1] = self.gyro_h[:, 1:].clone()

        self.orient_h[:, -1] = new_orient
        self.gyro_h[:, -1] = new_gyro


@torch.jit.script
def quaternion_to_euler(quat: torch.Tensor):
    quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)

    # Extract quaternion components
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1.0,
        torch.sign(sinp) * torch.tensor(torch.pi / 2.0, device=quat.device),
        torch.asin(sinp),
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=1)


@torch.jit.script
def scale_value(value: torch.Tensor, min_val: float, max_val: float):
    return torch.clamp((value - min_val) / (max_val - min_val) * 2.0 - 1.0, -1.0, 1.0)


@torch.jit.script
def normalize_actions(value: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor):
    denom = (upper - lower).clamp_min(1e-6)
    return torch.clamp((value - lower) / denom * 2.0 - 1.0, -1.0, 1.0)


@torch.jit.script
def orientation_reward(euler_imu_orient: torch.Tensor, action: str):
    # Calculate the sum of absolute Euler angles
    if action == "walk":
        angle_sums = torch.sum(torch.abs(euler_imu_orient), dim=1)
    else:
        # Excludes Z from reward to aid in turning
        angle_sums = torch.sum(torch.abs(euler_imu_orient[:, :2]), dim=1)

    # Calculate the reward
    orientation_rew = torch.where(
        angle_sums <= 0.95,
        1.0 - torch.sqrt(angle_sums / 0.95),
        torch.ones_like(angle_sums) * -1.0,
    )

    return orientation_rew


@torch.jit.script
def deviation_reward(og_pose: torch.Tensor, curr_pose: torch.Tensor, action: str = "walk"):
    # X, Y distance deviation reward
    x_dev = torch.abs(og_pose[:, 0] - curr_pose[:, 0])
    y_dev = torch.abs(og_pose[:, 1] - curr_pose[:, 1])

    if action == "walk":
        # Y distance only for walking
        reward = torch.where(
            y_dev <= 0.3,
            1.0 - torch.sqrt(y_dev / 0.3),
            torch.ones_like(y_dev) * -1.0,
        )
    else:
        # X + Y distance for turning / stop
        dist = x_dev + y_dev
        reward = torch.where(
            dist <= 0.3,
            1.0 - torch.sqrt(dist / 0.3),
            torch.ones_like(dist) * -1.0,
        )

    return reward


@torch.jit.script
def height_reward(root_pos: torch.Tensor, ideal_height: float):
    """Reward based on root height."""
    heights = root_pos[:, 2]

    max_deviation = 0.3
    height_diff = torch.abs(heights - ideal_height)
    clipped_diff = torch.clamp(height_diff, 0.0, max_deviation)
    height_rew = scale_value(clipped_diff, max_deviation, 0.0)

    # Scale to [0, 1]
    height_rew = (height_rew + 1.0) / 2.0
    return height_rew


@torch.jit.script
def joint_position_reward(
    pos_buff: torch.Tensor,
    start_pos: torch.Tensor,
    max_diff: torch.Tensor,
):
    """Calculates how far the joint position is from the initial pose."""
    diff = torch.abs(pos_buff - start_pos)
    diff_scaled = 1.0 - torch.sqrt(torch.clamp(diff / max_diff, 0.0, 1.0))

    # Calculate mean for each environment
    pos_rew = torch.mean(diff_scaled, dim=1)

    # Scale reward from [0, 1] to [-1, 1]
    pos_rew = pos_rew * 2.0 - 1.0
    return pos_rew


@torch.jit.script
def velocity_reward(vel_data: torch.Tensor, direction: torch.Tensor, action: str = "walk"):
    """Calculates reward based on linear and angular velocities."""
    reward = torch.zeros_like(direction)

    if action == "walk":
        vx = vel_data[:, 0]
        vy = torch.abs(vel_data[:, 1])

        rew_lin = torch.where(
            vx > 0.0,
            torch.clamp(vx / (vx + vy + 1e-8), 0.0, 1.0),
            torch.zeros_like(vx),
        )

        rew_ang = torch.clamp(-torch.abs(vel_data[:, 4]) / 2.0, -1.0, 0.0)
        reward = 0.5 * rew_lin + 0.5 * rew_ang
    else:
        z_ang_vel = vel_data[:, 5]

        # -1 = turn left, +1 = turn right | turning left angular velocity is +Z
        correct_vel = torch.sign(z_ang_vel) != torch.sign(direction)

        rew_ang = torch.where(
            correct_vel,
            torch.clamp(torch.abs(z_ang_vel) / 0.2, 0.0, 1.0),
            torch.ones_like(z_ang_vel) * -1.0,
        )

        rew_ang_penalty = torch.clamp(-torch.abs(vel_data[:, 4]) / 2.0, -1.0, 0.0)
        reward = 0.5 * rew_ang + 0.5 * rew_ang_penalty

    return reward


@torch.jit.script
def sigmoid_extra(pos_buff: torch.Tensor, start_pos: torch.Tensor):
    """Extra reward when all actuators stay close to the initial pose."""
    diff = torch.abs(pos_buff - start_pos)
    greatest_diff, _ = torch.max(diff, dim=1)
    sigmoid_values = 1.0 / (1.0 + torch.exp(45.836623610465864 * greatest_diff - 6.0))
    return sigmoid_values


@torch.jit.script
def feet_height_reward(
    air_time: torch.Tensor,
    feet_pos: torch.Tensor,
    target_h: float,
    scale: float = 25.0,
):
    """Feet clearance reward."""
    in_air = air_time > 0.0
    num_in_air = in_air.sum(dim=1)

    both_in_air = num_in_air == 2
    both_on_ground = num_in_air == 0

    # Foot Z positions
    z_pos = feet_pos[..., 2]
    z_err = torch.abs(z_pos - target_h)

    # Reward is 1.0 if z >= threshold, else exponential decay
    reward_per_leg = torch.where(
        z_pos >= target_h,
        torch.ones_like(z_pos),
        torch.exp(-scale * z_err),
    ) * in_air.float()

    reward = reward_per_leg.sum(dim=1)

    # If both feet are in air or both on the ground, set reward to 0
    reward = torch.where(
        both_in_air | both_on_ground,
        torch.zeros_like(reward),
        reward,
    )

    return reward