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

from isaaclab.actuators import PIDActuatorCfg

from .unitree import G1_MINIMAL_CFG  # isort: skip

##
# Actuator configuration (PID)
##

# Legs
G1_HIP_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=100.0,
    ki=0.0,
    kd=2.5,
    effort_limit=88.0,
    velocity_limit=32.0,
    use_physx_damping=True,
    # Default viscous friction consistent with stall/no-load (tau = tau_stall - b*w).
    viscous_friction=0.03,
)

G1_KNEE_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=200.0,
    ki=0.0,
    kd=5.0,
    effort_limit=139.0,
    velocity_limit=20.0,
    use_physx_damping=True,
    viscous_friction=0.03,
)

# Feet
G1_ANKLE_PITCH_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=200.0,
    ki=0.0,
    kd=0.2,
    effort_limit=50.0,
    velocity_limit=37.0,
    use_physx_damping=True,
    viscous_friction=0.03,
)

G1_ANKLE_ROLL_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=200.0,
    ki=0.0,
    kd=0.1,
    effort_limit=50.0,
    velocity_limit=37.0,
    use_physx_damping=True,
    viscous_friction=0.03,
)

# Waist
G1_WAIST_YAW_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=500.0,
    ki=0.0,
    kd=5.0,
    effort_limit=88.0,
    velocity_limit=32.0,
    use_physx_damping=True,
    viscous_friction=0.03,
)

G1_WAIST_ROLL_PITCH_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=500.0,
    ki=0.0,
    kd=5.0,
    effort_limit=50.0,
    velocity_limit=37.0,
    use_physx_damping=True,
    viscous_friction=0.03,
)

# Arms and hands
G1_ARMS_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=300.0,
    ki=0.0,
    kd=10.0,
    effort_limit=300.0,
    velocity_limit=100.0,
    use_physx_damping=True,
    viscous_friction=0.03,
)

G1_HANDS_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=200.0,
    ki=0.0,
    kd=2.0,
    effort_limit=300.0,
    velocity_limit=100.0,
    use_physx_damping=True,
    viscous_friction=0.03,
)

##
# Articulation configuration
##

G1_PID_CFG = G1_MINIMAL_CFG.copy()

# Replace actuator setup with explicit PID controllers.
G1_PID_CFG.actuators = {
    # Legs
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
    # Feet
    'ankle_pitch':
        G1_ANKLE_PITCH_PID_ACTUATOR_CFG.replace(joint_names_expr=[
            '.*_ankle_pitch_joint',
        ],),
    'ankle_roll':
        G1_ANKLE_ROLL_PID_ACTUATOR_CFG.replace(joint_names_expr=[
            '.*_ankle_roll_joint',
        ],),
    # Waist
    'waist_yaw':
        G1_WAIST_YAW_PID_ACTUATOR_CFG.replace(joint_names_expr=[
            'torso_joint',
        ],),
    # Arms / wrists
    'arms':
        G1_ARMS_PID_ACTUATOR_CFG.replace(joint_names_expr=[
            '.*_shoulder_pitch_joint',
            '.*_shoulder_roll_joint',
            '.*_shoulder_yaw_joint',
            '.*_elbow_pitch_joint',
            '.*_elbow_roll_joint',
        ],),
    # Hands / fingers
    'hands':
        G1_HANDS_PID_ACTUATOR_CFG.replace(joint_names_expr=[
            '.*_five_joint',
            '.*_three_joint',
            '.*_six_joint',
            '.*_four_joint',
            '.*_zero_joint',
            '.*_one_joint',
            '.*_two_joint',
        ],),
}
