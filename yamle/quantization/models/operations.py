import torch
import torch.nn as nn


class QuantizableAdd(nn.Module):
    """A simple class implementing residual addition but with a FloatFunctional object."""

    def __init__(self) -> None:
        super(QuantizableAdd, self).__init__()
        self._add = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """The forward function of the residual addition."""
        return self._add.add(x, y)
