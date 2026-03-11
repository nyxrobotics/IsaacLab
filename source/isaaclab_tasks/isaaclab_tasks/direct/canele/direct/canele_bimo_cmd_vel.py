# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

"""Canele direct RL environment with Bimo-style action/observation and cmd_vel tracking.

Design intent:
- keep the original canele_bimo actuator path as much as possible
  (lower-body-only control, incremental joint targets in radians,
   backlash, actuator delay, actuator noise, IMU history, action history)
- add cmd_vel = [v_x, v_y, yaw_rate] to observations
- train the policy to track cmd_vel with dense rewards
- keep the original Canele termination logic
- keep all angles in radians
- resample cmd_vel during the episode using configurable min/max intervals
- sample an exact zero cmd_vel with configurable probability at each resample
  to learn quiet standing without foot lift while still balancing
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
    dt = 0.005
    decimation = 4
    episode_length_s = 20.0
    observation_space = 3 + 20 + 4 * len(LOWER_BODY_JOINTS)
    action_space = len(LOWER_BODY_JOINTS)
    state_space = 0

    # Reward weights
    # [orientation, height, joint pos, joint pos sigmoid, feet height, vel tracking]
    reward_weights = [1.0, 1.0, 1.0, 1.0, 4.0, 1.0]

    # Canele-specific posture/foot targets
    body_height_target = 0.95
    feet_height_target = 0.2

    # Command ranges [m/s, m/s, rad/s]
    command_x_range = (-0.6, 0.6)
    command_y_range = (-0.6, 0.6)
    command_yaw_range = (-1.2, 1.2)

    # Command sampling / conditioning
    command_resample_time_min_s = 5.0
    command_resample_time_max_s = 20.0
    zero_command_probability = 0.20
    lin_cmd_deadzone = 0.05
    ang_cmd_deadzone = 0.10

    # Tracking reward scales
    lin_vel_tracking_std = 0.5
    ang_vel_tracking_std = 0.5

    # COM randomization: sampled per episode, kept constant during the episode
    com_shift_max = 0.03

    # Actuator settings (all radians)
    actuator_delay_max = 4
    actuator_delay_min = 1
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

        self.reward_weights = torch.tensor(
            self.cfg.reward_weights, device=self.device, dtype=torch.float32
        )

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

        # Initial posture from CANELE_MINIMAL_CFG (already in radians).
        self.base_pose = self.robot.data.default_joint_pos[:, self.joint_ids].clone()
        self.position_reward_max_diff = torch.maximum(
            self.upper_limits - self.base_pose,
            self.base_pose - self.lower_limits,
        ).clamp_min(1e-6)

        # Bimo-style actuator path buffers.
        self.cmd_actions = self.base_pose.clone()
        self.gear_position = self.base_pose.clone()
        self.noisy_act = self.base_pose.clone()
        self.last_direction = torch.zeros_like(self.base_pose)

        # Noise settings.
        self.orient_noise = GaussianNoiseCfg(mean=0.0, std=0.015, operation="add")
        self.gyro_noise = GaussianNoiseCfg(mean=0.0, std=0.01, operation="add")
        self.actuator_noise = GaussianNoiseCfg(
            mean=0.0,
            std=self.cfg.actuator_noise_std,
            operation="add",
        )

        # Actuator delay.
        self.act_timer = 0
        self.act_delay = 0

        # History buffers.
        self.orient_h = torch.zeros(self.scene.num_envs, 4, 3, device=self.device)
        self.gyro_h = torch.zeros(self.scene.num_envs, 4, 3, device=self.device)

        normalized_base = normalize_actions(
            self.base_pose,
            self.lower_limits,
            self.upper_limits,
        )
        self.act_hist = normalized_base.unsqueeze(1).repeat(1, 4, 1)

        # Command buffers.
        self.commands = torch.zeros(self.scene.num_envs, 3, device=self.device)
        self.command_min = torch.tensor(
            [
                self.cfg.command_x_range[0],
                self.cfg.command_y_range[0],
                self.cfg.command_yaw_range[0],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.command_max = torch.tensor(
            [
                self.cfg.command_x_range[1],
                self.cfg.command_y_range[1],
                self.cfg.command_yaw_range[1],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        env_step_dt = float(self.cfg.decimation * self.cfg.sim.dt)
        self.command_resample_min_steps = max(
            1,
            int(round(float(self.cfg.command_resample_time_min_s) / env_step_dt)),
        )
        self.command_resample_max_steps = max(
            self.command_resample_min_steps,
            int(round(float(self.cfg.command_resample_time_max_s) / env_step_dt)),
        )
        self.next_command_resample_step = torch.zeros(
            self.scene.num_envs,
            dtype=torch.long,
            device=self.device,
        )

        # COM shift buffers.
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
        imu_data = self.scene.sensors["imu"].data
        orient = quaternion_to_euler(imu_data.quat_w)
        orient = gaussian_noise(orient, self.orient_noise)
        angular_vel = gaussian_noise(imu_data.ang_vel_b, self.gyro_noise)

        orient = scale_value(orient, -1.0, 1.0)
        angular_vel = scale_value(angular_vel, -2.0, 2.0)

        self.update_imu_history(orient, angular_vel)
        imu_hist = torch.cat((self.orient_h[:, :, :2], self.gyro_h), dim=2)
        imu_hist = imu_hist.reshape(self.scene.num_envs, 20)

        cmd_act = normalize_actions(
            self.cmd_actions,
            self.lower_limits,
            self.upper_limits,
        )

        self.act_hist[:, :-1] = self.act_hist[:, 1:].clone()
        self.act_hist[:, -1] = cmd_act
        proc_act = self.act_hist.reshape(self.scene.num_envs, 4 * len(self.joint_names))

        cmd_vel_obs = normalize_values(
            self.commands,
            self.command_min,
            self.command_max,
        )

        obs_buffer = torch.cat((cmd_vel_obs, imu_hist, proc_act), dim=1)
        obs_buffer = torch.round(obs_buffer, decimals=4)
        return {"policy": obs_buffer}

    def _pre_physics_step(self, actions):
        self._resample_commands_if_needed()

        # Bimo-style delta action in radians.
        actions_cpy = torch.clamp(actions.clone(), -3.0, 3.0)
        self.cmd_actions += actions_cpy * self.cfg.action_scale
        self.cmd_actions = torch.max(
            torch.min(self.cmd_actions, self.upper_limits),
            self.lower_limits,
        )

        # Backlash model.
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

        # Actuator noise.
        self.noisy_act = gaussian_noise(self.gear_position, self.actuator_noise)
        self.noisy_act = torch.max(
            torch.min(self.noisy_act, self.upper_limits),
            self.lower_limits,
        )

        # Action delay: 5-20 ms (1-4 physics steps at dt=0.005).
        self.act_timer = 0
        self.act_delay = torch.randint(
            low=self.cfg.actuator_delay_min,
            high=self.cfg.actuator_delay_max + 1,
            size=(1,),
            device=self.device,
        ).item()

    def _apply_action(self):
        self.act_timer += 1
        if self.act_timer >= self.act_delay:
            self.robot.set_joint_position_target(
                self.noisy_act, joint_ids=self.joint_ids
            )

    def _get_rewards(self):
        imu_data = self.scene.sensors["imu"].data
        contact_data = self.scene.sensors["contact"].data

        euler_imu_orient = quaternion_to_euler(imu_data.quat_w)
        root_pos = self.robot.data.root_pos_w
        root_vel_w = self.robot.data.root_com_vel_w
        contact_pos = contact_data.pos_w
        air_time = contact_data.current_air_time

        orientation_rew = orientation_reward(euler_imu_orient)
        height_rew = height_reward(root_pos, self.cfg.body_height_target)
        position_rew = joint_position_reward(
            self.cmd_actions,
            self.base_pose,
            self.position_reward_max_diff,
        )

        stop_mask = stop_command_mask(
            self.commands,
            self.cfg.lin_cmd_deadzone,
            self.cfg.ang_cmd_deadzone,
        )
        motion_mask = 1.0 - stop_mask

        sig_extra = sigmoid_extra(self.cmd_actions, self.base_pose) * stop_mask

        feet_h_rew = (
            feet_height_reward(
                air_time,
                contact_pos,
                self.cfg.feet_height_target,
                150.0,
            )
            * motion_mask
        )

        base_vel_yaw = world_xy_to_yaw_frame(root_vel_w[:, :2], euler_imu_orient[:, 2])
        support_mask = support_phase_mask(air_time)

        vel_track_rew = (
            velocity_tracking_reward(
                base_vel_yaw,
                root_vel_w[:, 5],
                self.commands,
                self.cfg.lin_vel_tracking_std,
                self.cfg.ang_vel_tracking_std,
            )
            * support_mask
        )

        w = self.reward_weights
        total_reward = (
            orientation_rew * w[0]
            + height_rew * w[1]
            + position_rew * w[2]
            + sig_extra * w[3]
            + feet_h_rew * w[4]
            + vel_track_rew * w[5]
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
        self._sample_commands(env_ids)
        self._schedule_next_command_resample(env_ids)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

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
            directions = directions / torch.norm(
                directions, dim=1, keepdim=True
            ).clamp_min(1e-6)
            magnitudes = torch.rand((num_reset, 1), device=com_device) * float(
                self.cfg.com_shift_max
            )
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

    def _resample_commands_if_needed(self):
        env_ids = torch.nonzero(
            self.episode_length_buf >= self.next_command_resample_step,
            as_tuple=False,
        ).squeeze(-1)
        if env_ids.numel() == 0:
            return
        self._sample_commands(env_ids)
        self._schedule_next_command_resample(env_ids)

    def _schedule_next_command_resample(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        intervals = torch.randint(
            low=self.command_resample_min_steps,
            high=self.command_resample_max_steps + 1,
            size=(env_ids.numel(),),
            device=self.device,
        )
        self.next_command_resample_step[env_ids] = (
            self.episode_length_buf[env_ids] + intervals
        )

    def _sample_commands(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        zero_mask = torch.rand(env_ids.numel(), device=self.device) < float(
            self.cfg.zero_command_probability
        )
        zero_env_ids = env_ids[zero_mask]
        move_env_ids = env_ids[~zero_mask]

        if zero_env_ids.numel() > 0:
            self.commands[zero_env_ids] = 0.0

        if move_env_ids.numel() > 0:
            self._sample_nonzero_commands(move_env_ids)

    def _sample_nonzero_commands(self, env_ids: torch.Tensor):
        x = torch.empty(env_ids.numel(), device=self.device).uniform_(
            self.cfg.command_x_range[0],
            self.cfg.command_x_range[1],
        )
        y = torch.empty(env_ids.numel(), device=self.device).uniform_(
            self.cfg.command_y_range[0],
            self.cfg.command_y_range[1],
        )
        yaw = torch.empty(env_ids.numel(), device=self.device).uniform_(
            self.cfg.command_yaw_range[0],
            self.cfg.command_yaw_range[1],
        )

        for _ in range(16):
            small_mask = (
                (torch.abs(x) < float(self.cfg.lin_cmd_deadzone))
                & (torch.abs(y) < float(self.cfg.lin_cmd_deadzone))
                & (torch.abs(yaw) < float(self.cfg.ang_cmd_deadzone))
            )
            if not torch.any(small_mask):
                break
            n_small = int(torch.sum(small_mask).item())
            x[small_mask] = torch.empty(n_small, device=self.device).uniform_(
                self.cfg.command_x_range[0],
                self.cfg.command_x_range[1],
            )
            y[small_mask] = torch.empty(n_small, device=self.device).uniform_(
                self.cfg.command_y_range[0],
                self.cfg.command_y_range[1],
            )
            yaw[small_mask] = torch.empty(n_small, device=self.device).uniform_(
                self.cfg.command_yaw_range[0],
                self.cfg.command_yaw_range[1],
            )

        small_mask = (
            (torch.abs(x) < float(self.cfg.lin_cmd_deadzone))
            & (torch.abs(y) < float(self.cfg.lin_cmd_deadzone))
            & (torch.abs(yaw) < float(self.cfg.ang_cmd_deadzone))
        )
        if torch.any(small_mask):
            n_small = int(torch.sum(small_mask).item())
            signs = torch.where(
                torch.rand(n_small, device=self.device) < 0.5,
                -torch.ones(n_small, device=self.device),
                torch.ones(n_small, device=self.device),
            )
            yaw[small_mask] = signs * float(self.cfg.ang_cmd_deadzone)

        self.commands[env_ids, 0] = x
        self.commands[env_ids, 1] = y
        self.commands[env_ids, 2] = yaw

    def update_imu_history(self, new_orient, new_gyro):
        self.orient_h[:, :-1] = self.orient_h[:, 1:].clone()
        self.gyro_h[:, :-1] = self.gyro_h[:, 1:].clone()

        self.orient_h[:, -1] = new_orient
        self.gyro_h[:, -1] = new_gyro

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


@torch.jit.script
def quaternion_to_euler(quat: torch.Tensor):
    quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1.0,
        torch.sign(sinp) * torch.tensor(torch.pi / 2.0, device=quat.device),
        torch.asin(sinp),
    )

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
def normalize_values(value: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor):
    denom = (upper - lower).clamp_min(1e-6)
    return torch.clamp((value - lower) / denom * 2.0 - 1.0, -1.0, 1.0)


@torch.jit.script
def orientation_reward(euler_imu_orient: torch.Tensor):
    # For cmd_vel tracking, do not penalize yaw heading itself.
    roll_pitch_sum = torch.sum(torch.abs(euler_imu_orient[:, :2]), dim=1)
    return torch.where(
        roll_pitch_sum <= 0.95,
        1.0 - torch.sqrt(roll_pitch_sum / 0.95),
        torch.ones_like(roll_pitch_sum) * -1.0,
    )


@torch.jit.script
def height_reward(root_pos: torch.Tensor, ideal_height: float):
    heights = root_pos[:, 2]
    max_deviation = 0.3
    height_diff = torch.abs(heights - ideal_height)
    clipped_diff = torch.clamp(height_diff, 0.0, max_deviation)
    height_rew = scale_value(clipped_diff, max_deviation, 0.0)
    return (height_rew + 1.0) / 2.0


@torch.jit.script
def joint_position_reward(
    pos_buff: torch.Tensor,
    start_pos: torch.Tensor,
    max_diff: torch.Tensor,
):
    diff = torch.abs(pos_buff - start_pos)
    diff_scaled = 1.0 - torch.sqrt(torch.clamp(diff / max_diff, 0.0, 1.0))
    pos_rew = torch.mean(diff_scaled, dim=1)
    return pos_rew * 2.0 - 1.0


@torch.jit.script
def sigmoid_extra(pos_buff: torch.Tensor, start_pos: torch.Tensor):
    diff = torch.abs(pos_buff - start_pos)
    greatest_diff, _ = torch.max(diff, dim=1)
    return 1.0 / (1.0 + torch.exp(45.836623610465864 * greatest_diff - 6.0))


@torch.jit.script
def feet_height_reward(
    air_time: torch.Tensor,
    feet_pos: torch.Tensor,
    target_h: float,
    scale: float = 25.0,
):
    in_air = air_time > 0.0
    num_in_air = in_air.sum(dim=1)

    both_in_air = num_in_air == 2
    both_on_ground = num_in_air == 0

    z_pos = feet_pos[..., 2]
    z_err = torch.abs(z_pos - target_h)

    reward_per_leg = (
        torch.where(
            z_pos >= target_h,
            torch.ones_like(z_pos),
            torch.exp(-scale * z_err),
        )
        * in_air.float()
    )

    reward = reward_per_leg.sum(dim=1)
    reward = torch.where(
        both_in_air | both_on_ground,
        torch.zeros_like(reward),
        reward,
    )
    return reward


@torch.jit.script
def world_xy_to_yaw_frame(vel_xy_world: torch.Tensor, yaw: torch.Tensor):
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    vel_x = cos_yaw * vel_xy_world[:, 0] + sin_yaw * vel_xy_world[:, 1]
    vel_y = -sin_yaw * vel_xy_world[:, 0] + cos_yaw * vel_xy_world[:, 1]
    return torch.stack((vel_x, vel_y), dim=1)


@torch.jit.script
def support_phase_mask(air_time: torch.Tensor):
    in_air = air_time > 0.0
    both_in_air = torch.sum(in_air, dim=1) == 2
    return (~both_in_air).float()


@torch.jit.script
def stop_command_mask(
    commands: torch.Tensor, lin_deadzone: float, ang_deadzone: float
):
    lin_mag = torch.norm(commands[:, :2], dim=1)
    ang_mag = torch.abs(commands[:, 2])
    return ((lin_mag <= lin_deadzone) & (ang_mag <= ang_deadzone)).float()


@torch.jit.script
def velocity_tracking_reward(
    vel_xy_yaw: torch.Tensor,
    yaw_rate: torch.Tensor,
    commands: torch.Tensor,
    lin_std: float,
    ang_std: float,
):
    lin_err = torch.sum(torch.square(vel_xy_yaw - commands[:, :2]), dim=1)
    ang_err = torch.square(yaw_rate - commands[:, 2])

    lin_rew = torch.exp(-lin_err / (lin_std * lin_std))
    ang_rew = torch.exp(-ang_err / (ang_std * ang_std))
    return 0.6 * lin_rew + 0.4 * ang_rew