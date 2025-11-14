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
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),  # Initial Pose
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),

    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*"],   # Actuators
            effort_limit_sim=150.0,
            stiffness=60.0,
            damping=3.0,
        ),
    },
)
