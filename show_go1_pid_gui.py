from __future__ import annotations

import math


def main():
    # IMPORTANT: Do not import omni/pxr/isaaclab modules before SimulationApp is created.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=False)
    sim_app = app_launcher.app

    # Heavy imports must come AFTER SimulationApp exists.
    import torch
    from isaaclab.assets import Articulation
    from isaaclab.utils.types import ArticulationActions
    import isaaclab.sim as sim_utils
    from isaaclab_assets.robots.unitree_go1_pid import UNITREE_GO1_PID_CFG

    sim_cfg = sim_utils.SimulationCfg(dt=0.001)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(8, 0, 4), target=(0.0, 0.0, 0.0))

    robot = Articulation(cfg=UNITREE_GO1_PID_CFG)

    sim.reset()
    robot.reset()

    device = sim.device
    num_dof = robot.num_joints

    action = ArticulationActions(
        joint_positions=torch.zeros((1, num_dof), device=device),
        joint_velocities=None,
        joint_efforts=torch.zeros((1, num_dof), device=device),
    )

    # Use current pose as the baseline target so it stands still first.
    q0 = robot.data.joint_pos.clone()
    action.joint_positions[:] = q0

    # Pick one joint to wiggle (find by name)
    joint_name = "FL_thigh_joint"
    try:
        idx = robot.find_joints(joint_name)[0]
    except Exception:
        # fallback: just wiggle the first dof
        idx = 0
        print(f"[WARN] Joint '{joint_name}' not found. Wiggling joint index 0 instead.")

    t = 0.0
    while sim_app.is_running():
        sim_dt = sim.get_physics_dt()
        sim.step(render=True)
        robot.update(sim_dt)

        amp = 0.25  # rad
        freq = 0.7  # Hz

        target = q0.clone()
        target[0, idx] = q0[0, idx] + amp * math.sin(2.0 * math.pi * freq * t)

        action.joint_positions[:] = target
        action.joint_efforts.zero_()

        # robot.apply_action(action)
        robot.pre_physics_step(action)
        robot.apply_action()  

        t += sim.dt


if __name__ == "__main__":
    main()
