# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Local IO-descriptor helpers for Canele custom observation terms.

These helpers attach Isaac Lab IO descriptors to custom ObsTerm functions so that
`export_io_descriptors=True` exports non-empty `observations.policy` entries.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

try:
    from isaaclab.envs.utils.io_descriptors import (
        GenericObservationIODescriptor,
        generic_io_descriptor,
        record_dtype,
        record_joint_names,
        record_shape,
    )
except ImportError:
    # Compatibility fallback for installations that expose these symbols from a
    # different module path.
    from isaaclab.utils.io import (  # type: ignore
        GenericObservationIODescriptor,
        generic_io_descriptor,
        record_dtype,
        record_joint_names,
        record_shape,
    )


def _record_history_shape(output: torch.Tensor, descriptor: GenericObservationIODescriptor, **kwargs) -> None:
    """Record the flattened shape and keep it consistent with the exported tensor."""
    record_shape(output=output, descriptor=descriptor, **kwargs)


def _record_history_layout(
    output: torch.Tensor,
    descriptor: GenericObservationIODescriptor,
    *,
    history_length: int,
    terms_per_step: int,
    **kwargs,
) -> None:
    """Record extra metadata for flattened history observations."""
    descriptor.history_length = int(history_length)
    descriptor.terms_per_step = int(terms_per_step)
    descriptor.flatten_history_dim = True
    descriptor.history_shape = [int(history_length), int(terms_per_step)]
    descriptor.is_history_term = True



def history_observation_descriptor(
    *,
    observation_type: str,
    terms_per_step: int,
    history_length: int,
    units: str,
    source: str,
    normalization: str,
    axes: Sequence[str] | None = None,
    include_joint_names: bool = False,
    on_inspect_extra: Sequence[Callable[..., Any]] | None = None,
) -> Callable:
    """Build a generic IO descriptor for flattened history observations.

    The wrapped observation function still returns a flattened tensor of shape
    `[num_envs, history_length * terms_per_step]`, but the descriptor also stores
    the unflattened logical layout in extra metadata so downstream runners can
    reconstruct the history semantics.
    """
    hooks: list[Callable[..., Any]] = [
        _record_history_shape,
        record_dtype,
        lambda output, descriptor, **kwargs: _record_history_layout(
            output,
            descriptor,
            history_length=history_length,
            terms_per_step=terms_per_step,
            **kwargs,
        ),
    ]
    if include_joint_names:
        hooks.append(record_joint_names)
    if on_inspect_extra:
        hooks.extend(on_inspect_extra)

    descriptor_kwargs: dict[str, Any] = {
        "observation_type": observation_type,
        "units": units,
        "source": source,
        "normalization": normalization,
        "history_length": int(history_length),
        "terms_per_step": int(terms_per_step),
        "flatten_history_dim": True,
        "on_inspect": hooks,
    }
    if axes is not None:
        descriptor_kwargs["axes"] = list(axes)

    base_decorator = generic_io_descriptor(**descriptor_kwargs)

    def _decorate(func: Callable) -> Callable:
        if func.__doc__ is None:
            func.__doc__ = f"{observation_type} history observation for IO descriptor export."
        return base_decorator(func)

    return _decorate
