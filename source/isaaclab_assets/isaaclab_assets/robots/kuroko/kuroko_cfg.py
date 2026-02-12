# source/isaaclab_assets/isaaclab_assets/robots/kuroko/kuroko_cfg.py

from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import PIDActuatorCfg
import os

_KUROKO_DIR = os.path.dirname(__file__)
_KUROKO_USD_PATH = os.path.join(_KUROKO_DIR, "kuroko.usda")

XM540_W150_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=42.41,
    ki=0.0,
    kd=0.0,
    effort_limit=8.9,
    velocity_limit=6.9,
    use_physx_damping=True,
)

XH430_W210_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=16.62,
    ki=0.0,
    kd=0.0,
    effort_limit=3.1,
    velocity_limit=6.5,
    use_physx_damping=True,
)


# Minimal robot config for kuroko
KUROKO_MINIMAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_KUROKO_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=10000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.0005,
            stabilization_threshold=0.0001,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.36),  # Initial pose
        joint_pos={
            "ankle_l_roll": -0.17453292519943295,
            "ankle_l_yaw": 0.0,
            "ankle_r_roll": 0.17453292519943295,
            "ankle_r_yaw": 0.0,
            "chest": 0.0,
            "elbow_l_front": -1.57,
            "elbow_l_rear": -1.57,
            "elbow_r_front": -1.57,
            "elbow_r_rear": -1.57,
            "hip_l_pitch": 0.0,
            "hip_l_roll": 0.17453292519943295,
            "hip_r_pitch": 0.0,
            "hip_r_roll": -0.17453292519943295,
            "shin_l_active": 0.0,
            "shin_r_active": 0.0,
            "shoulder_l_pitch": -1.57,
            "shoulder_l_roll": -1.4311699866353502,
            "shoulder_r_pitch": -1.57,
            "shoulder_r_roll": 1.4311699866353502,
            "thigh_l_active": 0.0,
            "thigh_r_active": 0.0
        },
        joint_vel={".*": 0.0},
    ),

    actuators={
        # XM540-W150
        "xm540_w150": XM540_W150_PID_ACTUATOR_CFG.replace(
            joint_names_expr=[
                "shin_l_active",
                "shin_r_active",
                "shoulder_l_roll",
                "shoulder_r_roll",
                "thigh_l_active",
                "thigh_r_active",
            ],
        ),

        # XH430-W210
        "xh430_w210": XH430_W210_PID_ACTUATOR_CFG.replace(
            joint_names_expr=[
                "ankle_l_roll",
                "ankle_l_yaw",
                "ankle_r_roll",
                "ankle_r_yaw",
                "chest",
                "elbow_l_front",
                "elbow_l_rear",
                "elbow_r_front",
                "elbow_r_rear",
                "hip_l_pitch",
                "hip_l_roll",
                "hip_r_pitch",
                "hip_r_roll",
                "shoulder_l_pitch",
                "shoulder_r_pitch",
            ],
        ),
    },
)
