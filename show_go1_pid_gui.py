from __future__ import annotations

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import math
import torch

from isaaclab.app import AppLauncher
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils.types import ArticulationActions

from isaaclab_assets.robots.unitree_go1_pid import UNITREE_GO1_PID_CFG


def main():
    # Launch Isaac Sim with GUI
    app_launcher = AppLauncher(headless=False)
    sim_app = app_launcher.app

    # Simulation
    sim = SimulationContext(dt=1.0 / 60.0)
    sim.set_camera_view([2.0, 2.0, 1.2], [0.0, 0.0, 0.4])

    # Spawn robot
    robot = Articulation(cfg=UNITREE_GO1_PID_CFG)
    robot.spawn("/World/Go1")

    sim.reset()
    robot.reset()

    # Build a nominal standing pose (rad)
    # unitree_go1_pid.py の init_state と同じ
    # joint順はrobot側のDOF順なので、ここでは「現在姿勢をベース」にして特定関節だけ動かすのが安全
    device = sim.device
    num_dof = robot.num_joints
    action = ArticulationActions(
        joint_positions=torch.zeros((1, num_dof), device=device),
        joint_velocities=None,
        joint_efforts=torch.zeros((1, num_dof), device=device),
    )

    # “初期姿勢”をターゲットのベースにする
    # reset直後の姿勢を読み出してそのまま目標に入れると、まず安定して立ちます
    q0 = robot.data.joint_pos.clone()  # shape (1, num_dof)
    action.joint_positions[:] = q0

    t = 0.0
    while sim_app.is_running():
        sim.step(render=True)
        robot.update(sim.dt)

        # 例: 前脚の thigh 関節だけを軽くサイン波で振る（見た目で動作確認）
        # joint名から index を引けるならそれが一番安全。
        # Isaac Labの版によってAPI差があるので、両対応っぽく書きます。
        try:
            idx = robot.find_joints("FL_thigh_joint")[0]
        except Exception:
            # 名前が違うUSDの場合はここを調整
            idx = 0

        amp = 0.25  # rad
        freq = 0.7  # Hz
        target = q0.clone()
        target[0, idx] = q0[0, idx] + amp * math.sin(2.0 * math.pi * freq * t)

        action.joint_positions[:] = target
        action.joint_efforts.zero_()

        robot.set_joint_position_target(action.joint_positions)
        # 重要: PIDActuatorは「目標位置→effort」を計算してくれる想定なので、
        # 通常は articulation の apply_action を呼ぶだけでOKな構成が多いです。
        # もしあなたの実装が Actuator 経由で effort を適用するなら、下の apply_action を使ってください。
        robot.apply_action(action)

        t += sim.dt


if __name__ == "__main__":
    main()
