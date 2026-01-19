# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Sequence

from isaaclab.utils import configclass

@configclass
class PIDActuatorCfg:
    """Configuration for PID position actuator.

    This actuator performs explicit PID position control and outputs joint efforts.
    All internal units are SI (rad, rad/s, N*m).
    """
    # Required by Isaac Lab actuator plumbing to map joints
    joint_names_expr: Sequence[str] = ()
    
    # ---------------------------------------------------------------------
    # PID gains
    # ---------------------------------------------------------------------
    kp: float | Sequence[float] = 0.0
    ki: float | Sequence[float] = 0.0
    kd: float | Sequence[float] = 0.0

    # ---------------------------------------------------------------------
    # Motor limits (SI units)
    # effort_limit   : stall torque
    # velocity_limit : no-load speed
    # ---------------------------------------------------------------------
    effort_limit: float | Sequence[float] | None = None
    velocity_limit: float | Sequence[float] | None = None

    # ---------------------------------------------------------------------
    # Damping handling mode
    #
    # True  : viscous friction (= effort_limit / velocity_limit) is written
    #         into PhysX articulation and NOT used inside the controller.
    # False : viscous friction is used internally to compute velocity-
    #         dependent torque limits (DC motor-like saturation).
    # ---------------------------------------------------------------------
    use_physx_damping: bool = False

    # ---------------------------------------------------------------------
    # Optional compatibility fields
    # (user does NOT need to set these)
    # They will be automatically synchronized with effort_limit / velocity_limit
    # inside PIDActuator.
    # ---------------------------------------------------------------------
    effort_limit_sim: float | Sequence[float] | None = None
    velocity_limit_sim: float | Sequence[float] | None = None

    # ---------------------------------------------------------------------
    # Optional controller timestep [s]
    # If not provided, integral action is disabled and derivative falls back
    # to -joint_velocity.
    # ---------------------------------------------------------------------
    dt: float | None = None
