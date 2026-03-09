from __future__ import annotations

from .canele_task_env import LOWER_BODY_JOINTS


def get_io_descriptors(mode: str = "walk") -> dict:
    obs_terms = [
        {
            "name": "imu_history_roll_pitch",
            "shape": [8],
            "dtype": "float32",
            "description": "Four-step history of roll and pitch.",
        },
        {
            "name": "imu_history_gyro",
            "shape": [12],
            "dtype": "float32",
            "description": "Four-step history of body angular velocity xyz.",
        },
        {
            "name": "action_history",
            "shape": [4 * len(LOWER_BODY_JOINTS)],
            "dtype": "float32",
            "description": "Four-step history of commanded lower-body joint actions.",
        },
    ]

    return {
        "policy": {
            "name": "canele_policy",
            "framework": "onnx",
        },
        "observations": {
            "policy": obs_terms,
        },
        "actions": [
            {
                "name": "joint_position_action",
                "shape": [len(LOWER_BODY_JOINTS)],
                "dtype": "float32",
                "description": "Normalized action for lower-body joints.",
                "joint_names": list(LOWER_BODY_JOINTS),
            }
        ],
        "meta": {
            "task": "Canele",
            "mode": mode,
            "observation_dim": 20 + 4 * len(LOWER_BODY_JOINTS),
            "action_dim": len(LOWER_BODY_JOINTS),
            "notes": [
                "Bimo-style observation layout for Canele.",
                "Observation = imu history + action history.",
            ],
        },
    }
