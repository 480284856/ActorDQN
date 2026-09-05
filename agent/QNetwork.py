"""Neural network used to approximate the action-value function."""

from math import prod
from numbers import Integral
from typing import Sequence

import torch
from torch import nn


class QNetwork(nn.Module):
    """Map one environment observation to one Q-value per discrete action.

    The observation may already be a vector or may have several dimensions (as
    the maze observation does). It is flattened before passing through two
    hidden layers. The output deliberately has no activation: Q-values are
    unbounded real numbers.
    """

    def __init__(
        self,
        input_dim: int | Sequence[int],
        output_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        self.input_shape = _validate_input_shape(input_dim)
        output_dim = _validate_positive_integer(output_dim, "output_dim")
        hidden_dims = tuple(
            _validate_positive_integer(width, "hidden dimension")
            for width in hidden_dims
        )

        layer_widths = (prod(self.input_shape), *hidden_dims, output_dim)
        self.model = nn.Sequential(*_build_mlp_layers(layer_widths))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return Q-values, preserving an optional leading batch dimension."""
        if not torch.is_tensor(observation):
            raise TypeError("observation must be a torch.Tensor")

        is_single_observation = observation.ndim == len(self.input_shape)
        if is_single_observation:
            observation = observation.unsqueeze(0)

        q_values = self.model(torch.flatten(observation, start_dim=1))
        return q_values.squeeze(0) if is_single_observation else q_values


def _build_mlp_layers(layer_widths: Sequence[int]) -> list[nn.Module]:
    """Build linear layers with ReLU activations between, but not after, them."""
    layers: list[nn.Module] = []
    for index, (input_width, output_width) in enumerate(
        zip(layer_widths, layer_widths[1:])
    ):
        layers.append(nn.Linear(input_width, output_width))
        if index < len(layer_widths) - 2:
            layers.append(nn.LayerNorm(output_width))
            layers.append(nn.ReLU())
    return layers


def _validate_input_shape(input_dim: int | Sequence[int]) -> tuple[int, ...]:
    """Normalize a flat input size or observation shape to a validated shape."""
    if isinstance(input_dim, int):
        return (_validate_positive_integer(input_dim, "input_dim"),)

    input_shape = tuple(input_dim)
    if not input_shape:
        raise ValueError("input_dim must contain at least one dimension")
    return tuple(
        _validate_positive_integer(size, "input dimension") for size in input_shape
    )


def _validate_positive_integer(value: int, name: str) -> int:
    """Reject invalid layer dimensions early with a useful error message."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)
