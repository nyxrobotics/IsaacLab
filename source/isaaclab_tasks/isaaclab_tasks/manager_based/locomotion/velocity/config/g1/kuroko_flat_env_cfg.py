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

        # Angular velocity tracking is slightly down-weighted on flat terrain
        self.rewards.track_ang_vel_z_exp.weight = 1.0

        # Penalize vertical motion of the base a bit instead of 0.0
        if self.rewards.lin_vel_z_l2 is not None:
            self.rewards.lin_vel_z_l2.weight = -0.2

        # Keep the action rate / joint regularization from rough cfg
        # (already set in KurokoRoughEnvCfg.__post_init__)

        # Narrow the command range a bit for easier training on flat terrain
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-3.0, 3.0)


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

        # Remove random external pushes for stable visualization
        self.events.base_external_force_torque = None
        self.events.push_robot = None
