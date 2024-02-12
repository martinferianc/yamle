from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from yamle.models.specific.svi_utils import GaussianMeanField


class LinearRBNN(nn.Linear):
    """This is a wrapper around a linear layer that implements rank-1 Bayesian neural network.

    If the method is `additive` then the rank 1 weights are initialized with 0 mean.
    If the method is `multiplicative` then the rank 1 weights are initialized with 1 mean.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        bias (bool): If set to `True`, the layer will have a bias.
        num_members (int): The number of members in the ensemble.
        prior_mean (float): The mean of the prior distribution.
        log_variance (float): The initial value of the log of the standard deviation of the weights.
        prior_log_variance (float): The initial value of the log of the standard deviation of the prior distribution.
        method (str): The method whether `additive` or `multiplicative` to be used for the rank-1 approximation.
        num_members (int): The number of members in the ensemble.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        prior_mean: float = 0.0,
        log_variance: float = 0.0,
        prior_log_variance: float = 0.0,
        method: str = "additive",
        num_members: int = 1,
    ) -> None:
        super(LinearRBNN, self).__init__(in_features, out_features, bias)
        r_shape = (num_members, in_features)
        s_shape = (num_members, out_features)
        initial_mean = 0.0 if method == "additive" else 1.0
        self.r = GaussianMeanField(
            r_shape, initial_mean, log_variance, prior_mean, prior_log_variance
        )
        self.s = GaussianMeanField(
            s_shape, initial_mean, log_variance, prior_mean, prior_log_variance
        )
        self._num_members = num_members
        self._method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the layer."""
        R = self.r.sample().unsqueeze(0)
        S = self.s.sample().unsqueeze(0)
        N = x.shape[0]
        M = self._num_members
        x = x.reshape(N // M, M, -1)
        if self._method == "additive":
            output = F.linear(x + R, self.weight, None) + S
        elif self._method == "multiplicative":
            output = F.linear(x * R, self.weight, None) * S

        if self.bias is not None:
            output += self.bias
        return output.reshape(N, -1)


class Conv2dRBNN(nn.Conv2d):
    """This is a wrapper around a convolutional layer that implements rank-1 Bayesian neural network.

    If the method is `additive` then the rank 1 weights are initialized with 0 mean.
    If the method is `multiplicative` then the rank 1 weights are initialized with 1 mean.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (Union[int, Tuple[int, int]]): The size of the convolutional kernel.
        stride (Union[int, Tuple[int, int]]): The stride of the convolution.
        padding (Union[int, Tuple[int, int]]): The padding of the convolution.
        dilation (Union[int, Tuple[int, int]]): The dilation of the convolution.
        groups (int): The number of groups.
        bias (bool): If set to `True`, the layer will have a bias.
        prior_mean (float): The mean of the prior distribution.
        log_variance (float): The initial value of the log of the standard deviation of the weights.
        prior_log_variance (float): The initial value of the log of the standard deviation of the prior distribution.
        method (str): The method whether `additive` or `multiplicative` to be used for the rank-1 approximation.
        num_members (int): The number of members in the ensemble.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        prior_mean: float = 0.0,
        log_variance: float = 0.0,
        prior_log_variance: float = 0.0,
        method: str = "additive",
        num_members: int = 1,
    ) -> None:

        super(Conv2dRBNN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        r_shape = (num_members, in_channels)
        s_shape = (num_members, out_channels)
        initial_mean = 0.0 if method == "additive" else 1.0
        self.r = GaussianMeanField(
            r_shape, initial_mean, log_variance, prior_mean, prior_log_variance
        )
        self.s = GaussianMeanField(
            s_shape, initial_mean, log_variance, prior_mean, prior_log_variance
        )
        self._num_members = num_members
        self._method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the layer."""
        R = self.r.sample().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        S = self.s.sample().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        N = x.shape[0]
        M = self._num_members
        x = x.reshape(N // M, M, x.shape[1], x.shape[2], x.shape[3])
        if self._method == "additive":
            x = x + R
        elif self._method == "multiplicative":
            x = x * R
        x = x.reshape(N, x.shape[2], x.shape[3], x.shape[4])
        x = F.conv2d(
            x, self.weight, None, self.stride, self.padding, self.dilation, self.groups
        )
        x = x.reshape(N // M, M, x.shape[1], x.shape[2], x.shape[3])
        if self._method == "additive":
            x = x + S
        elif self._method == "multiplicative":
            x = x * S
        x = x.reshape(N, x.shape[2], x.shape[3], x.shape[4])

        if self.bias is not None:
            x += self.bias

        return x
