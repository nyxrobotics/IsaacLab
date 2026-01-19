# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

from isaaclab.utils.types import ArticulationActions

from .actuator_base import ActuatorBase

if TYPE_CHECKING:
    from .actuator_pid_cfg import PIDActuatorCfg


class PIDActuator(ActuatorBase):
    """Explicit PID position controller actuator (goal position -> effort output).

    - Internal units are SI (rad, rad/s, N*m).
    - effort_limit is treated as stall torque.
    - velocity_limit is treated as no-load speed.
    - viscous_friction is NOT taken as an input parameter:
        viscous_friction := effort_limit / velocity_limit  (virtual back-emf slope)

      Mode 1 (use_physx_damping == True):
        - Set viscous_friction into articulation (via ActuatorBase/Articulation pipeline).
        - Do NOT use viscous_friction inside the controller (torque limits are +/- effort_limit).

      Mode 2 (use_physx_damping == False):
        - Use viscous_friction internally to compute velocity-dependent torque limits (DC motor-like).
        - Respect rotation direction (four-quadrant torque-speed saturation).
    """

    cfg: PIDActuatorCfg

    def __init__(
        self,
        cfg: PIDActuatorCfg,
        joint_names: list[str],
        joint_ids: Sequence[int],
        num_envs: int,
        device: str,
        # Keep signature compatible with other actuators, but we do not use stiffness/damping.
        stiffness: torch.Tensor | float = 0.0,
        damping: torch.Tensor | float = 0.0,
        armature: torch.Tensor | float = 0.0,
        friction: torch.Tensor | float = 0.0,
        dynamic_friction: torch.Tensor | float = 0.0,
        viscous_friction: torch.Tensor | float = 0.0,  # ignored (derived internally)
        effort_limit: torch.Tensor | float = torch.inf,
        velocity_limit: torch.Tensor | float = torch.inf,
    ):
        # ---- Backward/compat behavior for "*_sim" parameters (user doesn't need to set them) ----
        # We want: effort_limit_sim == effort_limit, velocity_limit_sim == velocity_limit
        # without requiring the user to specify *_sim.
        if getattr(cfg, "effort_limit_sim", None) is None and getattr(cfg, "effort_limit", None) is not None:
            cfg.effort_limit_sim = cfg.effort_limit
        elif getattr(cfg, "effort_limit_sim", None) is not None and getattr(cfg, "effort_limit", None) is None:
            cfg.effort_limit = cfg.effort_limit_sim
        elif getattr(cfg, "effort_limit_sim", None) is not None and getattr(cfg, "effort_limit", None) is not None:
            if cfg.effort_limit_sim != cfg.effort_limit:
                raise ValueError(
                    "Both 'effort_limit_sim' and 'effort_limit' are set with different values "
                    f"{cfg.effort_limit_sim} != {cfg.effort_limit}. Please set only 'effort_limit'."
                )

        if getattr(cfg, "velocity_limit_sim", None) is None and getattr(cfg, "velocity_limit", None) is not None:
            cfg.velocity_limit_sim = cfg.velocity_limit
        elif getattr(cfg, "velocity_limit_sim", None) is not None and getattr(cfg, "velocity_limit", None) is None:
            cfg.velocity_limit = cfg.velocity_limit_sim
        elif getattr(cfg, "velocity_limit_sim", None) is not None and getattr(cfg, "velocity_limit", None) is not None:
            if cfg.velocity_limit_sim != cfg.velocity_limit:
                raise ValueError(
                    "Both 'velocity_limit_sim' and 'velocity_limit' are set with different values "
                    f"{cfg.velocity_limit_sim} != {cfg.velocity_limit}. Please set only 'velocity_limit'."
                )

        # Ensure kp/ki/kd exist with default 0.0
        if not hasattr(cfg, "kp") or cfg.kp is None:
            cfg.kp = 0.0
        if not hasattr(cfg, "ki") or cfg.ki is None:
            cfg.ki = 0.0
        if not hasattr(cfg, "kd") or cfg.kd is None:
            cfg.kd = 0.0
        if not hasattr(cfg, "use_physx_damping") or cfg.use_physx_damping is None:
            cfg.use_physx_damping = False

        # ---- Derive viscous friction (virtual back-emf slope) ----
        # viscous_friction := effort_limit / velocity_limit
        # Handle inf/0 robustly.
        eff_lim = cfg.effort_limit if cfg.effort_limit is not None else effort_limit
        vel_lim = cfg.velocity_limit if cfg.velocity_limit is not None else velocity_limit

        # Convert to tensors later; for now keep python-level checks
        self._use_physx_damping = bool(cfg.use_physx_damping)

        # If velocity limit is not provided, we cannot form a slope. We fallback to 0 viscous friction.
        if vel_lim is None:
            omni.log.warn(
                "PIDActuator: 'velocity_limit' is None. "
                "Torque-speed saturation and viscous friction derivation will be disabled."
            )
            derived_viscous = 0.0
        else:
            # If vel_lim is inf or 0 -> slope 0
            try:
                if float(vel_lim) == 0.0 or float(vel_lim) == float("inf"):
                    derived_viscous = 0.0
                else:
                    derived_viscous = eff_lim / vel_lim
            except Exception:
                # tensor or other types handled after super().__init__
                derived_viscous = eff_lim / vel_lim

        # ---- IMPORTANT: stiffness/damping are not used (set to zero) ----
        stiffness = 0.0
        damping = 0.0

        # Pass derived viscous friction to the base so the articulation can receive it (Mode 1).
        super().__init__(
            cfg,
            joint_names,
            joint_ids,
            num_envs,
            device,
            stiffness,
            damping,
            armature,
            friction,
            dynamic_friction,
            derived_viscous,
            eff_lim,
            vel_lim,
        )
        
        if self._use_physx_damping:
            self._viscous_friction = torch.as_tensor(derived_viscous, device=self._device)
            self._viscous_friction = self._expand_to_shape(self._viscous_friction, self.computed_effort.shape)
            # Set the viscous friction into the articulation
            self.viscous_friction = self._viscous_friction

        # Gains (broadcasting-friendly): shape (num_envs, num_joints) via ActuatorBase helpers if needed.
        # We keep them as tensors on device for fast compute.
        self.kp = torch.as_tensor(cfg.kp, device=self._device).reshape(1, -1) if isinstance(cfg.kp, (list, tuple)) else torch.as_tensor(cfg.kp, device=self._device)
        self.ki = torch.as_tensor(cfg.ki, device=self._device).reshape(1, -1) if isinstance(cfg.ki, (list, tuple)) else torch.as_tensor(cfg.ki, device=self._device)
        self.kd = torch.as_tensor(cfg.kd, device=self._device).reshape(1, -1) if isinstance(cfg.kd, (list, tuple)) else torch.as_tensor(cfg.kd, device=self._device)

        # Expand scalars to joint dimension if necessary (best-effort).
        # computed_effort exists after base init; use its shape as reference.
        ref_shape = self.computed_effort.shape  # (num_envs, num_joints)
        self.kp = self._expand_to_shape(self.kp, ref_shape)
        self.ki = self._expand_to_shape(self.ki, ref_shape)
        self.kd = self._expand_to_shape(self.kd, ref_shape)

        # PID states
        self._integral = torch.zeros_like(self.computed_effort)
        self._prev_error = torch.zeros_like(self.computed_effort)

        # Cache derived viscous friction as tensor (torque per rad/s)
        self._viscous_friction = torch.as_tensor(derived_viscous, device=self._device)
        self._viscous_friction = self._expand_to_shape(self._viscous_friction, ref_shape)

        # dt (best-effort):
        # If ActuatorBase exposes dt, we use it; otherwise fall back to cfg.dt if present; else None.
        self._dt = getattr(self, "dt", None)
        if self._dt is None:
            self._dt = getattr(cfg, "dt", None)
        if self._dt is not None:
            self._dt = float(self._dt)

    def _expand_to_shape(self, x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        """Broadcast a scalar/1D tensor to (num_envs, num_joints) if possible."""
        if x.ndim == 0:
            return x.expand(shape)
        if x.ndim == 1:
            # assume joints
            if x.numel() == shape[1]:
                return x.view(1, -1).expand(shape)
            # if envs
            if x.numel() == shape[0]:
                return x.view(-1, 1).expand(shape)
        if x.shape == shape:
            return x
        # fallback: rely on torch broadcasting in operations
        return x

    def reset(self, env_ids: Sequence[int]):
        # Clear PID state for selected envs.
        if env_ids is None or env_ids == slice(None):
            self._integral.zero_()
            self._prev_error.zero_()
        else:
            self._integral[env_ids].zero_()
            self._prev_error[env_ids].zero_()

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        """Compute effort from desired joint positions (PID)."""
        if control_action.joint_positions is None:
            raise ValueError("PIDActuator requires control_action.joint_positions (desired position).")

        # Position error
        error = control_action.joint_positions - joint_pos

        # Derivative of error
        if self._dt is not None and self._dt > 0.0:
            d_error = (error - self._prev_error) / self._dt
        else:
            # fallback: if desired position is piecewise constant, d(error)/dt ≈ -joint_vel
            d_error = -joint_vel

        # Integral
        if self._dt is not None and self._dt > 0.0:
            self._integral = self._integral + error * self._dt
        else:
            # If dt is unknown, do not integrate to avoid timestep-dependent behavior.
            # This keeps ki usable only when dt is provided.
            pass

        # Simple anti-windup: clamp integral so that |ki * integral| <= effort_limit (per joint)
        # (Only meaningful when ki != 0 and effort_limit is finite.)
        eps = 1e-12
        ki_abs = torch.clamp(torch.abs(self.ki), min=0.0)
        finite_eff = torch.isfinite(self.effort_limit)
        if torch.any(ki_abs > 0.0) and torch.any(finite_eff):
            max_int = torch.where(ki_abs > eps, self.effort_limit / ki_abs, torch.full_like(self._integral, float("inf")))
            self._integral = torch.clamp(self._integral, min=-max_int, max=max_int)

        # PID effort
        self.computed_effort = self.kp * error + self.kd * d_error + self.ki * self._integral + control_action.joint_efforts

        # Apply limits:
        if self._use_physx_damping:
            # Mode 1: viscous friction handled by PhysX; here we only apply stall torque clamp.
            self.applied_effort = torch.clamp(self.computed_effort, min=-self.effort_limit, max=self.effort_limit)
        else:
            # Mode 2: internal torque-speed saturation (four quadrant).
            self.applied_effort = self._clip_effort_torque_speed(self.computed_effort, joint_vel)

        # Update state
        self._prev_error[:] = error

        # Output effort only (goal position input -> effort output)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def _clip_effort_torque_speed(self, effort: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
        """Torque-speed saturation with:
        - effort_limit: stall torque (at 0 speed)
        - velocity_limit: no-load speed (at 0 torque)
        This is equivalent to linear back-emf model:
            tau_max(v) = tau_stall * (1 - v / v_nl)
            tau_min(v) = tau_stall * (-1 - v / v_nl)
        """
        # If velocity limit is not finite, fallback to simple clamp.
        if not torch.isfinite(self.velocity_limit).all():
            return torch.clamp(effort, min=-self.effort_limit, max=self.effort_limit)

        # Avoid division by zero
        vel_lim = torch.clamp(self.velocity_limit, min=1e-12)

        tau_stall = self.effort_limit

        # Four-quadrant limits (same structure as DCMotor but stall == continuous here)
        tau_speed_top = tau_stall * (1.0 - joint_vel / vel_lim)
        tau_speed_bottom = tau_stall * (-1.0 - joint_vel / vel_lim)

        # When stall == continuous, these clips are effectively no-ops but kept for clarity.
        max_effort = torch.minimum(tau_speed_top, tau_stall)
        min_effort = torch.maximum(tau_speed_bottom, -tau_stall)

        return torch.clamp(effort, min=min_effort, max=max_effort)
