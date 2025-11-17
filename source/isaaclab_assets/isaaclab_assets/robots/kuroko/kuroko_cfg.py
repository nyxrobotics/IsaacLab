# source/isaaclab_assets/isaaclab_assets/robots/kuroko/kuroko_cfg.py

from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
import os

_KUROKO_DIR = os.path.dirname(__file__)
_KUROKO_USD_PATH = os.path.join(_KUROKO_DIR, "kuroko.usda")

# Minimal robot config for kuroko
KUROKO_MINIMAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_KUROKO_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.32),  # Initial Pose
        joint_pos={
            "ankle_l_roll": -0.17453292519943295,
            "ankle_l_yaw": 0.0,
            "ankle_r_roll": 0.17453292519943295,
            "ankle_r_yaw": 0.0,
            "chest": 0.0,
            "elbow_l_front": -2.007128639793479,
            "elbow_l_rear": -2.007128639793479,
            "elbow_r_front": -2.007128639793479,
            "elbow_r_rear": -2.007128639793479,
            "hip_l_pitch": 0.0,
            "hip_l_roll": 0.17453292519943295,
            "hip_r_pitch": 0.0,
            "hip_r_roll": -0.17453292519943295,
            "shin_l_active": 0.0,
            "shin_r_active": 0.0,
            "shoulder_l_pitch": -1.1344640137963142,
            "shoulder_l_roll": -1.4311699866353502,
            "shoulder_r_pitch": -1.1344640137963142,
            "shoulder_r_roll": 1.4311699866353502,
            "thigh_l_active": 0.0,
            "thigh_r_active": 0.0
        },
        joint_vel={".*": 0.0},
    ),

    actuators={
        "all": ImplicitActuatorCfg(
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
                "shin_l_active",
                "shin_r_active",
                "shoulder_l_pitch",
                "shoulder_l_roll",
                "shoulder_r_pitch",
                "shoulder_r_roll",
                "thigh_l_active",
                "thigh_r_active"
            ],   # Actuators
            effort_limit_sim=150.0,
            stiffness=60.0,
            damping=3.0,
        ),
    },
)
