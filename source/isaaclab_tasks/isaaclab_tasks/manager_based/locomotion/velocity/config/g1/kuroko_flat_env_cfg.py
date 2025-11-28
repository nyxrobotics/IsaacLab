# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .kuroko_rough_env_cfg import KurokoRoughEnvCfg

@configclass
class KurokoFlatEnvCfg(KurokoRoughEnvCfg):
    """Flat-terrain locomotion config for Kuroko.

    This is the flat version of KurokoRoughEnvCfg, similar in spirit to
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
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.4)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)


@configclass
class KurokoFlatEnvCfg_PLAY(KurokoFlatEnvCfg):
    """Visualization-friendly flat env settings for Kuroko."""

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

        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.4)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
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
