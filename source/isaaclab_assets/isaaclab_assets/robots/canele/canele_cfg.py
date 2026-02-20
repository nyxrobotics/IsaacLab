# source/isaaclab_assets/isaaclab_assets/robots/canele/canele_cfg.py

import os

from isaaclab.actuators import PIDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

_CANELE_DIR = os.path.dirname(__file__)
_CANELE_USD_PATH = os.path.join(_CANELE_DIR, 'canele.usd')

X8_120_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=670.0,
    ki=0.0,
    kd=0.34,
    effort_limit=78.0,
    velocity_limit=16.0,
    use_physx_damping=True,
    viscous_friction=1.2,
)

X6_60_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=1200.0,
    ki=0.0,
    kd=0.60,
    effort_limit=45.0,
    velocity_limit=18.0,
    use_physx_damping=True,
    viscous_friction=0.63,
)

X6_P36_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=270.0,
    ki=0.0,
    kd=1.2,
    effort_limit=30.0,
    velocity_limit=11.0,
    use_physx_damping=True,
    viscous_friction=0.66,
)

DM_J4340_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=200.0,
    ki=0.0,
    kd=5.0,
    effort_limit=20.0,
    velocity_limit=11.0,
    use_physx_damping=True,
    viscous_friction=0.44,
)

KNEE_PITCH_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=670.0,
    ki=0.0,
    kd=3.4,
    effort_limit=78.0,
    velocity_limit=16.0,
    use_physx_damping=True,
    viscous_friction=1.2,
)

ANKLE_PITCH_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=1200.0,
    ki=0.0,
    kd=6.0,
    effort_limit=45.0,
    velocity_limit=18.0,
    use_physx_damping=True,
    viscous_friction=0.63,
)

ELBOW_YAW_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=270.0,
    ki=0.0,
    kd=0.12,
    effort_limit=30.0,
    velocity_limit=11.0,
    use_physx_damping=True,
    viscous_friction=0.66,
)

WRIST_YAW_PID_ACTUATOR_CFG = PIDActuatorCfg(
    kp=200.0,
    ki=0.0,
    kd=0.5,
    effort_limit=20.0,
    velocity_limit=11.0,
    use_physx_damping=True,
    viscous_friction=0.44,
)

# Minimal robot config for canele
CANELE_MINIMAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_CANELE_USD_PATH,
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
        pos=(0.0, 0.0, 0.95),  # Initial pose
        joint_pos={
            'torso_yaw': 0.0,
            'left_hip_yaw': 0.0,
            'left_hip_pitch': -0.2094,
            'left_hip_roll': 0.0349,
            'left_knee_pitch': 0.4189,
            'left_ankle_pitch': -0.2443,
            'left_ankle_roll': -0.0349,
            'left_shoulder_yaw': 0.0,
            'left_shoulder_pitch': 0.0,
            'left_shoulder_roll': 0.0873,
            'left_elbow_yaw': 0.0,
            'left_elbow_pitch': -0.1745,
            'left_wrist_yaw': 0.0,
            'left_wrist_roll': 0.0,
            'left_wrist_pitch': 0.0,
            'right_hip_yaw': 0.0,
            'right_hip_pitch': -0.2094,
            'right_hip_roll': -0.0349,
            'right_knee_pitch': 0.4189,
            'right_ankle_pitch': -0.2443,
            'right_ankle_roll': 0.0349,
            'right_shoulder_yaw': 0.0,
            'right_shoulder_pitch': 0.0,
            'right_shoulder_roll': -0.0873,
            'right_elbow_yaw': 0.0,
            'right_elbow_pitch': -0.1745,
            'right_wrist_yaw': 0.0,
            'right_wrist_roll': 0.0,
            'right_wrist_pitch': 0.0,
        },
        joint_vel={'.*': 0.0},
    ),
    actuators={
        'x8_120':
            X8_120_PID_ACTUATOR_CFG.replace(joint_names_expr=[
                'left_hip_pitch',
                'right_hip_pitch',
            ],),
        'x6_60':
            X6_60_PID_ACTUATOR_CFG.replace(joint_names_expr=[
                'torso_yaw',
                'left_hip_yaw',
                'left_hip_roll',
                'left_ankle_roll',
                'right_hip_yaw',
                'right_hip_roll',
                'right_ankle_roll',
            ],),
        'x6_p36':
            X6_P36_PID_ACTUATOR_CFG.replace(joint_names_expr=[
                'left_shoulder_yaw',
                'left_shoulder_pitch',
                'left_shoulder_roll',
                'left_elbow_pitch',
                'right_shoulder_yaw',
                'right_shoulder_pitch',
                'right_shoulder_roll',
                'right_elbow_pitch',
            ],),
        'dm_j4340':
            DM_J4340_PID_ACTUATOR_CFG.replace(joint_names_expr=[
                'left_wrist_roll',
                'left_wrist_pitch',
                'right_wrist_roll',
                'right_wrist_pitch',
            ],),
        'knee_pitch':
            KNEE_PITCH_PID_ACTUATOR_CFG.replace(joint_names_expr=[
                'left_knee_pitch',
                'right_knee_pitch',
            ],),
        'ankle_pitch':
            ANKLE_PITCH_PID_ACTUATOR_CFG.replace(joint_names_expr=[
                'left_ankle_pitch',
                'right_ankle_pitch',
            ],),
        'elbow_yaw':
            ELBOW_YAW_PID_ACTUATOR_CFG.replace(joint_names_expr=[
                'left_elbow_yaw',
                'right_elbow_yaw',
            ],),
        'wrist_yaw':
            WRIST_YAW_PID_ACTUATOR_CFG.replace(joint_names_expr=[
                'left_wrist_yaw',
                'right_wrist_yaw',
            ],),
    },
)
