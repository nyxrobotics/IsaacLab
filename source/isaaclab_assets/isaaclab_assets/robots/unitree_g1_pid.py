# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Configuration for the Unitree G1 humanoid robot (29 DoF).

This file is a slimmed-down extract of the original Unitree robot configuration module,
containing only :obj:`G1_PID_CFG`.

Reference: https://github.com/unitreerobotics/unitree_ros
"""

from .unitree import G1_CFG  # isort: skip

G1_PID_CFG = G1_CFG.copy()
