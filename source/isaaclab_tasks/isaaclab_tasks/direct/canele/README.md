# Canele package

Requested layout:

canele/
├ __init__.py
├ README.md
├ common/
│  ├ __init__.py
│  ├ canele_cfg.py
│  ├ canele.usd
│  ├ canele.urdf
│  ├ canele.sdf
│  └ meshes/
├ manager_based/
│  ├ __init__.py
│  ├ rough_env_cfg.py
│  └ flat_env_cfg.py
├ direct/
│  ├ __init__.py
│  ├ canele_task_env.py
│  ├ io_descriptors.py
│  ├ rewards/
│  │  ├ __init__.py
│  │  ├ canele_rewards_env.py
│  │  ├ canele_rewards_walk.py
│  │  └ canele_rewards_joint.py
│  └ terminations/
│     ├ __init__.py
│     └ canele_terminations.py
├ agents/
│  ├ __init__.py
│  ├ rsl_rl_ppo_cfg.py
│  ├ skrl_flat_ppo_cfg.yaml
│  └ skrl_rough_ppo_cfg.yaml
