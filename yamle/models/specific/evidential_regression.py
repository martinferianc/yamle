from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalGammaLinear(nn.Linear):
    """This class defines the normal-gamma linear layer.

    It has 4 output features: mean, variance, alpha, and beta.

    Args:
        in_features (int): The number of input features.
        bias (bool): Whether to use bias or not.
    """

    def __init__(self, in_features: int, bias: bool = True) -> None:
        super().__init__(in_features, 4, bias)

    def _evidence(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the softplus function to calculate evidence."""
        return F.softplus(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super().forward(x)
        mu, logv, logalpha, logbeta = torch.split(output, 1, dim=-1)
        v = self._evidence(logv)
        alpha = self._evidence(logalpha) + 1.0
        beta = self._evidence(logbeta)
        return torch.cat([mu, v, alpha, beta], dim=-1)


class NormalGammaConv2d(nn.Conv2d):
    """This class defines the normal-gamma convolutional layer.

    It has 4 output features: mean, variance, alpha, and beta.

    Args:
        in_channels (int): Number of input channels.
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolutional kernel.
        stride (Union[int, Tuple[int, int]], optional): Stride for the convolution operation. Default: 1.
        padding (Union[int, Tuple[int, int]], optional): Zero-padding added to both sides of the input. Default: 0.
        dilation (Union[int, Tuple[int, int]], optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        bias (bool, optional): Whether to use bias or not. Default: True.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super(NormalGammaConv2d, self).__init__(
            in_channels,
            4,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def _evidence(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the softplus function to calculate evidence."""
        return F.softplus(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super(NormalGammaConv2d, self).forward(x)
        mu, logv, logalpha, logbeta = torch.split(output, self.out_channels, dim=1)
        v = self._evidence(logv)
        alpha = self._evidence(logalpha) + 1.0
        beta = self._evidence(logbeta)
        return torch.cat([mu, v, alpha, beta], dim=1)
