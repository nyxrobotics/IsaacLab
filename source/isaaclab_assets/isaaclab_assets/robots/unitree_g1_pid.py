# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Unitree G1 configuration using explicit PID position control.

This is a lightweight companion to :obj:`G1_29DOF_CFG` from ``unitree.py`` that replaces
the default actuator setup (DC-motor / implicit PD) with an explicit PID controller.

Notes:
    * The PID gains/limits below are derived from the PD-style stiffness/damping and
      motor limits used in :obj:`G1_29DOF_CFG`.
    * The values are reasonable defaults, but you should still tune them for your task.
"""

from __future__ import annotations

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.actuators import PIDActuatorCfg

from .unitree import G1_MINIMAL_CFG  # isort: skip

##
# Actuator configuration (PID)
##

# Legs
G1_HIP_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=670.0,
    ki=0.0,
    kd=3.4,
    effort_limit=78.1,
    velocity_limit=16.1,
    use_physx_damping=True,
    viscous_friction=1.0,
)

G1_KNEE_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=670.0,
    ki=0.0,
    kd=3.4,
    effort_limit=78.1,
    velocity_limit=16.1,
    use_physx_damping=True,
    viscous_friction=1.0,
)

# Feet
G1_ANKLE_PITCH_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=1174.0,
    ki=0.0,
    kd=6.0,
    effort_limit=45.5,
    velocity_limit=18.0,
    use_physx_damping=True,
    viscous_friction=1.0,
)

G1_ANKLE_ROLL_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=1174.0,
    ki=0.0,
    kd=6.0,
    effort_limit=45.5,
    velocity_limit=18.0,
    use_physx_damping=True,
    viscous_friction=1.0,
)

# Waist
G1_WAIST_YAW_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=265.0,
    ki=0.0,
    kd=12.4,
    effort_limit=30.4,
    velocity_limit=11.5,
    use_physx_damping=True,
    viscous_friction=1.0,
)

G1_WAIST_ROLL_PITCH_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=1174.0,
    ki=0.0,
    kd=6.0,
    effort_limit=45.5,
    velocity_limit=18.0,
    use_physx_damping=True,
    viscous_friction=1.0,
)

# Arms and hands
G1_ARMS_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=265.0,
    ki=0.0,
    kd=12.4,
    effort_limit=30.4,
    velocity_limit=11.5,
    use_physx_damping=True,
    viscous_friction=1.0,
)

G1_HANDS_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=20.0,
    ki=0.0,
    kd=2.0,
    effort_limit=3.0,
    velocity_limit=1.0,
    use_physx_damping=True,
    viscous_friction=1.0,
)

##
# Articulation configuration
##

G1_PID_CFG = G1_MINIMAL_CFG.copy()

# Replace actuator setup with explicit PID controllers.

G1_PID_CFG.actuators = {
    'hip':
        G1_HIP_PID_ACTUATOR_CFG.replace(joint_names_expr=[
            '.*_hip_yaw_joint',
            '.*_hip_roll_joint',
            '.*_hip_pitch_joint',
        ],),
    'knee':
        G1_KNEE_PID_ACTUATOR_CFG.replace(joint_names_expr=[
            '.*_knee_joint',
        ],),
    'ankle_pitch':
        G1_ANKLE_PITCH_PID_ACTUATOR_CFG.replace(joint_names_expr=[
            '.*_ankle_pitch_joint',
        ],),
    'ankle_roll':
        G1_ANKLE_ROLL_PID_ACTUATOR_CFG.replace(joint_names_expr=[
            '.*_ankle_roll_joint',
        ],),
    'waist_yaw':
        G1_WAIST_YAW_PID_ACTUATOR_CFG.replace(joint_names_expr=[
            'torso_joint',
        ],),
    'arms':
        G1_ARMS_PID_ACTUATOR_CFG.replace(joint_names_expr=[
            '.*_shoulder_pitch_joint',
            '.*_shoulder_roll_joint',
            '.*_shoulder_yaw_joint',
            '.*_elbow_pitch_joint',
            '.*_elbow_roll_joint',
        ],),
    'hands':
        ImplicitActuatorCfg(
            joint_names_expr=[
                '.*_five_joint',
                '.*_three_joint',
                '.*_six_joint',
                '.*_four_joint',
                '.*_zero_joint',
                '.*_one_joint',
                '.*_two_joint',
            ],
            effort_limit_sim=300,
            stiffness=40.0,
            damping=10.0,
            armature={
                '.*_five_joint': 0.001,
                '.*_three_joint': 0.001,
                '.*_six_joint': 0.001,
                '.*_four_joint': 0.001,
                '.*_zero_joint': 0.001,
                '.*_one_joint': 0.001,
                '.*_two_joint': 0.001,
            },
        ),
}
