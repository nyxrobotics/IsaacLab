# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Canele",
    entry_point=f"{__name__}.canele_task_env:CaneleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.canele_task_env:CaneleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl:CanelePPORunnerCfg",
    },
)
