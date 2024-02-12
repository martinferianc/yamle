import torch
import torch.nn as nn

from yamle.defaults import TINY_EPSILON


class TemperatureScaler(nn.Module):
    """This is a simple temperature scaling layer that is applied to the logits of a model.

    Args:
        temperature (float): The initial temperature. Default: 1.0.
        mode (str): When to turn it on and off. Default: 'both'. Can select from 'train', 'eval', 'both'.
        train (bool): Whether the temperature is trainable or not. Default: False.
    """

    def __init__(
        self, temperature: float = 1.0, mode: str = "both", train: bool = False
    ) -> None:
        super().__init__()
        # At first the gradient is not computed for the temperature parameter.
        assert mode in [
            "train",
            "eval",
            "both",
        ], "The mode must be one of 'train', 'eval', 'both'."
        self._mode = mode
        self._T = nn.Parameter(torch.tensor(temperature), requires_grad=train)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the layer."""
        if self._mode == "train" and self.training:
            return x / torch.clamp(self._T, min=TINY_EPSILON, max=1e8)
        elif self._mode == "eval" and not self.training:
            return x / torch.clamp(self._T, min=TINY_EPSILON, max=1e8)
        elif self._mode == "both":
            return x / torch.clamp(self._T, min=TINY_EPSILON, max=1e8)
        return x

    def extra_repr(self) -> str:
        return super().extra_repr() + f"T={self._T.item():.2f}, mode={self._mode}"
