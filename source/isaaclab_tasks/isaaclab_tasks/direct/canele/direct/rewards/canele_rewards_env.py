from __future__ import annotations

import torch

def is_alive(env) -> torch.Tensor:
    return (~env.termination_manager.terminated).float()

def is_terminated(env) -> torch.Tensor:
    return env.termination_manager.terminated.float()
