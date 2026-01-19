# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree GO1 configuration using explicit PID position control."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.actuators import PIDActuatorCfg

##
# Actuator configuration (PID)
##

GO1_PID_ACTUATOR_CFG = PIDActuatorCfg(
    # PID gains (example values – tune as needed)
    kp=60.0,
    ki=0.0,
    kd=2.0,

    # Motor limits (Unitree GO1 nominal values, SI units)
    # Hip / thigh / calf motors are typically similar
    effort_limit=33.5,          # [N*m] stall torque
    velocity_limit=21.0,        # [rad/s] no-load speed (~200 rpm)

    # Damping behavior
    # True  : use PhysX viscous damping
    # False : internal torque-speed saturation
    use_physx_damping=False,
)

##
# Articulation configuration
##

UNITREE_GO1_PID_CFG = ArticulationCfg(
    prim_path="/World/Go1",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/Go1/go1.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Default standing pose (same as standard GO1 configs)
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.8,
            ".*_calf_joint": -1.6,
        },
        pos=(0.0, 0.0, 0.42),
    ),
    actuators={
        "legs": GO1_PID_ACTUATOR_CFG.replace(
            joint_names_expr=[
                ".*_hip_joint",
                ".*_thigh_joint",
                ".*_calf_joint",
            ],
        )
    },
    soft_joint_pos_limit_factor=0.95,
)
