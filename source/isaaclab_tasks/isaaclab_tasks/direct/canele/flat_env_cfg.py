# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import CaneleRoughEnvCfg


@configclass
class CaneleFlatEnvCfg(CaneleRoughEnvCfg):
    """Flat-terrain locomotion config for Canele.

    This is the flat version of CaneleRoughEnvCfg, similar in spirit to
    G1FlatEnvCfg vs G1RoughEnvCfg.
    """

    def __post_init__(self):
        # Initialize base rough config first
        super().__post_init__()

        # ------------------------------------------------------------------
        # Terrain: switch to infinite flat plane, no terrain generator
        # ------------------------------------------------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # ------------------------------------------------------------------
        # Disable height scanner and height-scan observations
        # ------------------------------------------------------------------
        self.scene.height_scanner = None
        if hasattr(self.observations, "policy") and hasattr(
            self.observations.policy, "height_scan"
        ):
            self.observations.policy.height_scan = None

        # ------------------------------------------------------------------
        # Disable terrain curriculum (no levels on a flat plane)
        # ------------------------------------------------------------------
        if hasattr(self, "curriculum") and hasattr(self.curriculum, "terrain_levels"):
            self.curriculum.terrain_levels = None

        # ------------------------------------------------------------------
        # Reward / command tweaks for flat terrain
        # (picked conservative values; feel free to tune)
        # ------------------------------------------------------------------

        # Change the command range
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.2, 1.2)


@configclass
class CaneleFlatEnvCfg_PLAY(CaneleFlatEnvCfg):
    """Visualization-friendly flat env settings for Canele."""

    def __post_init__(self):
        # Inherit flat settings
        super().__post_init__()

        # Make a smaller, more lightweight scene for interactive play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Shorter episodes for quick inspection
        self.episode_length_s = 40.0

        # Disable observation corruption for clean visuals
        if hasattr(self.observations, "policy"):
            self.observations.policy.enable_corruption = False

        # Disable resampling (effectively never resample within an episode)
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
        # Remove random external pushes for stable visualization
        self.events.base_external_force_torque = None
        self.events.push_robot = None
