from typing import Tuple, Optional, Any, Type
import torch
import torch.nn as nn
import torch.nn.functional as F

from yamle.models.specific.svi_utils import (
    GaussianMeanField,
    VariationalDropoutMeanField,
    VariationalWeight,
)
from yamle.defaults import TINY_EPSILON, DISABLED_VI_KEY


def disable_svi_replacement(m: nn.Module) -> None:
    """This method is used to disable the SVI replacement for a module.
    It will do it recursively for all the submodules.

    Args:
        m (nn.Module): The module to disable the dropout replacement for.
    """
    for module in m.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            setattr(module, DISABLED_VI_KEY, True)


def enable_svi_replacement(m: nn.Module) -> None:
    """This method is used to enable the SVI replacement for a module.
    It will do it recursively for all the submodules.

    Args:
        m (nn.Module): The module to enable the dropout replacement for.
    """
    for module in m.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            setattr(module, DISABLED_VI_KEY, False)


def replace_with_svi_lrt(
    model: nn.Module, prior_mean: float, log_variance: float, prior_log_variance: float
) -> nn.Module:
    """This method is used to replace all the `nn.Linear` or `nn.Conv2d` layers
       with a `LinearSVILRT` or `Conv2dSVILRT` layer respectively.

    Args:
        model (nn.Module): The model to replace the layers in.
        prior_mean (float): The mean of the prior distribution.
        log_variance (float): The initial value of the log of the standard deviation of the weights.
        prior_log_variance (float): The initial value of the log of the standard deviation of the prior distribution.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            if hasattr(child, DISABLED_VI_KEY) and getattr(child, DISABLED_VI_KEY):
                continue
            setattr(
                model,
                name,
                LinearSVILRT(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    prior_mean=prior_mean,
                    log_variance=log_variance,
                    prior_log_variance=prior_log_variance,
                ),
            )
        elif isinstance(child, nn.Conv2d):
            if hasattr(child, DISABLED_VI_KEY) and getattr(child, DISABLED_VI_KEY):
                continue
            setattr(
                model,
                name,
                Conv2dSVILRT(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias is not None,
                    prior_mean=prior_mean,
                    log_variance=log_variance,
                    prior_log_variance=prior_log_variance,
                ),
            )
        else:
            replace_with_svi_lrt(child, prior_mean, log_variance, prior_log_variance)
    return model


def replace_with_svi_lrtvd(
    model: nn.Module, prior_mean: float, log_variance: float, prior_log_variance: float
) -> nn.Module:
    """This method is used to replace all the `nn.Linear` or `nn.Conv2d` layers
       with a `LinearSVILRTVD` or `Conv2dSVILRTVD` layer respectively.

    Args:
        model (nn.Module): The model to replace the layers in.
        prior_mean (float): The mean of the prior distribution.
        log_variance (float): The initial value of the log of the standard deviation of the weights.
        prior_log_variance (float): The initial value of the log of the standard deviation of the prior distribution.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            if hasattr(child, DISABLED_VI_KEY) and getattr(child, DISABLED_VI_KEY):
                continue
            setattr(
                model,
                name,
                LinearSVILRTVD(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    prior_mean=prior_mean,
                    log_variance=log_variance,
                    prior_log_variance=prior_log_variance,
                ),
            )
        elif isinstance(child, nn.Conv2d):
            if hasattr(child, DISABLED_VI_KEY) and getattr(child, DISABLED_VI_KEY):
                continue
            setattr(
                model,
                name,
                Conv2dSVILRTVD(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    bias=child.bias is not None,
                    prior_mean=prior_mean,
                    log_variance=log_variance,
                    prior_log_variance=prior_log_variance,
                ),
            )
        else:
            replace_with_svi_lrtvd(child, prior_mean, log_variance, prior_log_variance)
    return model


def replace_with_svi_rt(
    model: nn.Module, prior_mean: float, log_variance: float, prior_log_variance: float
) -> nn.Module:
    """This method is used to replace all the `nn.Linear` or `nn.Conv2d` layers
       with a `LinearSVI` or `Conv2dSVI` layer respectively.

    Args:
        model (nn.Module): The model to replace the layers in.
        prior_mean (float): The mean of the prior distribution.
        log_variance (float): The initial value of the log of the standard deviation of the weights.
        prior_log_variance (float): The initial value of the log of the standard deviation of the prior distribution.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            if hasattr(child, DISABLED_VI_KEY) and getattr(child, DISABLED_VI_KEY):
                continue
            setattr(
                model,
                name,
                LinearSVIRT(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    prior_mean=prior_mean,
                    log_variance=log_variance,
                    prior_log_variance=prior_log_variance,
                ),
            )
        elif isinstance(child, nn.Conv2d):
            if hasattr(child, DISABLED_VI_KEY) and getattr(child, DISABLED_VI_KEY):
                continue
            setattr(
                model,
                name,
                Conv2dSVIRT(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias is not None,
                    prior_mean=prior_mean,
                    log_variance=log_variance,
                    prior_log_variance=prior_log_variance,
                ),
            )
        else:
            replace_with_svi_rt(child, prior_mean, log_variance, prior_log_variance)
    return model


def replace_with_flipout_svi(
    model: nn.Module,
    prior_mean: float,
    log_variance: float,
    prior_log_variance: float,
    p: float = 0.5,
    method: str = "gaussian",
) -> nn.Module:
    """This method is used to replace all the `nn.Linear` or `nn.Conv2d` layers
       with a `LinearSVIFlipOut` or `Conv2dSVIFlipOut` layer respectively.

    Args:
        model (nn.Module): The model to replace the layers in.
        prior_mean (float): The mean of the prior distribution.
        log_variance (float): The initial value of the log of the standard deviation of the weights.
        prior_log_variance (float): The initial value of the log of the standard deviation of the prior distribution.
        p (float): The probability in the `DropConnect` layer.
        method (str): The method to use for the Monte Carlo approximation.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            if hasattr(child, DISABLED_VI_KEY) and getattr(child, DISABLED_VI_KEY):
                continue
            setattr(
                model,
                name,
                LinearSVIFlipOut(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    prior_mean=prior_mean,
                    log_variance=log_variance,
                    prior_log_variance=prior_log_variance,
                    p=p,
                    method=method,
                ),
            )
        elif isinstance(child, nn.Conv2d):
            if hasattr(child, DISABLED_VI_KEY) and getattr(child, DISABLED_VI_KEY):
                continue
            setattr(
                model,
                name,
                Conv2dSVIFlipOut(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    bias=child.bias is not None,
                    prior_mean=prior_mean,
                    log_variance=log_variance,
                    prior_log_variance=prior_log_variance,
                    p=p,
                    method=method,
                ),
            )
        else:
            replace_with_flipout_svi(
                child, prior_mean, log_variance, prior_log_variance, p, method
            )
    return model


def _gaussian_disurbance(
    weight: torch.Tensor,
    std: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    std_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """This method is used to compute the Gaussian disturbance to the weights."""
    sigma_weight = torch.log1p(std)
    delta_weight = sigma_weight * torch.randn_like(sigma_weight)

    delta_bias = None
    if bias is not None:
        sigma_bias = torch.log1p(torch.exp(std_bias))
        delta_bias = sigma_bias * torch.randn_like(sigma_bias)

    return delta_weight, delta_bias


def _dropconnect_disurbance(
    weight: torch.Tensor, p: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """This method is used to compute the dropconnect disturbance to the weights."""
    weight = weight / (1 - p)
    delta_weight = 2 * weight * torch.rand_like(weight) - weight

    delta_bias = None
    if bias is not None:
        bias = bias / (1 - p)
        delta_bias = 2 * bias * torch.rand_like(bias) - bias

    return delta_weight, delta_bias


def sample_gaussian_noise(
    x_shape: Tuple[int, ...], device: torch.device
) -> torch.Tensor:
    """This method samples from the distribution."""
    eta = torch.randn(x_shape, device=device)
    eta = eta / (torch.frobenius_norm(eta) + TINY_EPSILON)
    r = torch.randn(1, device=device)
    return eta * r


class LinearSVIRT(nn.Linear):
    """This is a wrapper around a linear layer that implements the basic reparameterization trick.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        bias (bool): If set to `True`, the layer will have a bias.
        distribution (Type[VariationalWeight]): The distribution to use for the weights.
        prior_mean (float): The prior mean of the weights.
        log_variance (float): The initial value of the log variance of the weights.
        prior_log_variance (float): The prior log variance of the weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        distribution: Type[VariationalWeight] = GaussianMeanField,
        prior_mean: float = 0.0,
        log_variance: float = 0.0,
        prior_log_variance: float = 0.0,
    ) -> None:
        super(LinearSVIRT, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )
        weight = self.weight.data.clone()
        weight_shape = weight.shape
        del self.weight
        self.weight = distribution(
            weight_shape, weight, log_variance, prior_mean, prior_log_variance
        )
        if self.bias is not None:
            bias = self.bias.data.clone()
            del self.bias
            self.bias = distribution(
                (out_features,), bias, log_variance, prior_mean, prior_log_variance
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the layer."""
        weight = self.weight.sample()
        bias = self.bias.sample() if self.bias is not None else None
        return F.linear(x, weight, bias)


class LinearSVILRT(LinearSVIRT):
    """This is a wrapper around a linear layer that implements the local reparameterization trick."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = F.linear(
            x, self.weight.mean, self.bias.mean if self.bias is not None else None
        )
        std = torch.sqrt(
            TINY_EPSILON
            + F.linear(
                x**2,
                self.weight.variance,
                self.bias.variance if self.bias is not None else None,
            )
        )
        std = torch.nan_to_num(std)
        mean = torch.nan_to_num(mean)
        return mean + std * sample_gaussian_noise(mean.shape, mean.device)


class LinearSVILRTVD(LinearSVIRT):
    """This is a wrapper around a linear layer that implements the local reparameterization trick with variational dropout."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        prior_mean: float = 0,
        log_variance: float = 0,
        prior_log_variance: float = 0,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            VariationalDropoutMeanField,
            prior_mean,
            log_variance,
            prior_log_variance,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_alpha = self.weight.log_alpha
        log_alpha_bias = self.bias.log_alpha if self.bias is not None else None
        mean = F.linear(
            x, self.weight.mean, self.bias.mean if self.bias is not None else None
        )
        mean = torch.nan_to_num(mean)
        if self.training:
            std = torch.sqrt(
                TINY_EPSILON
                + F.linear(
                    x**2,
                    log_alpha.exp() * self.weight.mean**2,
                    log_alpha_bias.exp() * self.bias.mean**2
                    if self.bias is not None
                    else None,
                )
            )
            std = torch.nan_to_num(std)
            return mean + std * sample_gaussian_noise(mean.shape, mean.device)
        else:
            return mean


class LinearSVIFlipOut(LinearSVIRT):
    """This is a wrapper around a linear layer that implements stochastic variational inference.

    It is based on Flipout trick. It can be implemented with respect to a Gaussian variational posterior or
    simply dropconnect.

    Args:
        method (str): The method to use for the Flipout trick. Can be either `gaussian` or `dropconnect`.
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        bias (bool): If set to `True`, the layer will have a bias.
        prior_mean (Optional[float]): The prior mean of the weights. Optional, only used if method is "gaussian".
        log_variance (Optional[float]): The initial value of the log variance of the weights. Optional, only used if method is "gaussian".
        prior_log_variance (Optional[float]): The prior log variance of the weights. Optional, only used if method is "gaussian".
        p (Optional[float]): The probability of dropping a weight. Optional, only used if method is "dropconnect".
    """

    def __init__(self, method: str, p: Optional[float] = None, **kwargs: Any) -> None:
        super(LinearSVIFlipOut, self).__init__(**kwargs)
        if method not in ["gaussian", "dropconnect"]:
            raise ValueError(f"Unknown method: {method}.")
        if method == "dropconnect":
            del self.weight
            self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features))
            if kwargs["bias"]:
                del self.bias
                self.bias = nn.Parameter(torch.randn(self.out_features))
            self._p = nn.Parameter(torch.tensor(p), requires_grad=False)
            self.reset_parameters()
        self._p_value = p if method != "dropconnect" else self._p
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the layer."""
        if self.method == "gaussian":
            delta_weight, delta_bias = _gaussian_disurbance(
                self.weight.mean,
                self.weight.std,
                self.bias.mean if self.bias is not None else None,
                self.bias.std if self.bias is not None else None,
            )
            weight = self.weight.mean
            bias = self.bias.mean if self.bias is not None else None
        elif self.method == "dropconnect":
            delta_weight, delta_bias = _dropconnect_disurbance(
                self.weight, self._p, self.bias
            )
            weight = self.weight / (1 - self._p)
            bias = self.bias / (1 - self._p) if self.bias is not None else None

        outputs = F.linear(x, weight, bias)
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        perturbed_outputs = (
            F.linear(x * sign_input, delta_weight, delta_bias) * sign_output
        )
        return outputs + perturbed_outputs

    def extra_repr(self) -> str:
        return super().extra_repr() + f", method={self.method}, p={self._p_value}"


class Conv2dSVIRT(nn.Conv2d):
    """This is a wrapper around a convolutional layer that implements the basic reparameterization trick.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the kernel.
        stride (int): The stride of the convolution.
        padding (int): The padding of the convolution.
        dilation (int): The dilation of the convolution.
        groups (int): The number of groups.
        bias (bool): If set to `True`, the layer will have a bias.
        padding_mode (str): The padding mode to use.
        distribution (Type[VariationalWeight]): The distribution to use for the weights.
        prior_mean (float): The prior mean of the weights.
        log_variance (float): The initial value of the log variance of the weights.
        prior_log_variance (float): The prior log variance of the weights.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        distribution: Type[VariationalWeight] = GaussianMeanField,
        prior_mean: float = 0.0,
        log_variance: float = 0.0,
        prior_log_variance: float = 0.0,
    ) -> None:
        super(Conv2dSVIRT, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=padding_mode,
            dilation=dilation,
            groups=groups,
        )
        weight = self.weight.data.clone()
        del self.weight
        weight_shape = weight.shape
        self.weight = distribution(
            weight_shape, weight, log_variance, prior_mean, prior_log_variance
        )
        if self.bias is not None:
            bias = self.bias.data.clone()
            del self.bias
            self.bias = distribution(
                (out_channels,), bias, log_variance, prior_mean, prior_log_variance
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the layer."""
        weight = self.weight.sample()
        bias = self.bias.sample() if self.bias is not None else None
        return F.conv2d(
            x, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )


class Conv2dSVILRT(Conv2dSVIRT):
    """This is a wrapper around a convolutional layer that implements the local reparameterization trick."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = F.conv2d(
            x,
            self.weight.mean,
            self.bias.mean if self.bias is not None else None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        std = torch.sqrt(
            TINY_EPSILON
            + F.conv2d(
                x**2,
                self.weight.variance,
                self.bias.variance if self.bias is not None else None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        )
        mean = torch.nan_to_num(mean)
        std = torch.nan_to_num(std)
        return mean + std * sample_gaussian_noise(mean.shape, mean.device)


class Conv2dSVILRTVD(Conv2dSVILRT):
    """This is a wrapper around a Conv2d layer that implements the local reparameterization trick with variational dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        prior_mean: float = 0.0,
        log_variance: float = 0.0,
        prior_log_variance: float = 0.0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=padding_mode,
            dilation=dilation,
            groups=groups,
            distribution=VariationalDropoutMeanField,
            prior_mean=prior_mean,
            log_variance=log_variance,
            prior_log_variance=prior_log_variance,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_alpha = self.weight.log_alpha
        log_alpha_bias = self.bias.log_alpha if self.bias is not None else None
        mean = F.conv2d(
            x,
            self.weight.mean,
            self.bias.mean if self.bias is not None else None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        mean = torch.nan_to_num(mean)
        if self.training:
            std = torch.sqrt(
                TINY_EPSILON
                + F.conv2d(
                    x**2,
                    log_alpha.exp() * self.weight.mean**2,
                    log_alpha_bias.exp() * self.bias.mean**2
                    if self.bias is not None
                    else None,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
            )
            std = torch.nan_to_num(std)
            return mean + std * sample_gaussian_noise(mean.shape, mean.device)
        else:
            return mean


class Conv2dSVIFlipOut(Conv2dSVIRT):
    """This is a wrapper around a convolutional layer that implements stochastic variational inference.

    It is based on Flipout trick. It can be implemented with respect to a Gaussian variational posterior or
    simply dropconnect.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the kernel.
        stride (int): The stride of the convolution.
        padding (int): The padding of the convolution.
        bias (bool): If set to `True`, the layer will have a bias.
        method (str): The method to use. Can be "gaussian" or "dropconnect".
        prior_mean (Optional[float]): The prior mean of the weights. Optional, only used if method is "gaussian".
        log_variance (Optional[float]): The initial value of the log variance of the weights. Optional, only used if method is "gaussian".
        prior_log_variance (Optional[float]): The prior log variance of the weights. Optional, only used if method is "gaussian".
        p (Optional[float]): The probability of dropping a weight. Optional, only used if method is "dropconnect".
    """

    def __init__(self, method: str, p: float, **kwargs: Any) -> None:
        super(Conv2dSVIFlipOut, self).__init__(**kwargs)
        if method not in ["gaussian", "dropconnect"]:
            raise ValueError(f"Unknown method: {method}.")
        if method == "dropconnect":
            del self.weight
            self.weight = nn.Parameter(
                torch.randn(
                    self.out_channels,
                    self.in_channels,
                    self.kernel_size,
                    self.kernel_size,
                )
            )
            if kwargs["bias"]:
                del self.bias
                self.bias = nn.Parameter(torch.randn(self.out_channels))
            self._p = nn.Parameter(torch.tensor(p), requires_grad=False)
            self.reset_parameters()

        self._p_value = p if method != "dropconnect" else self._p
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the layer."""
        if self.method == "gaussian":
            delta_weight, delta_bias = _gaussian_disurbance(
                self.weight.mean,
                self.weight.std,
                self.bias.mean if self.bias is not None else None,
                self.bias.std if self.bias is not None else None,
            )
            weight = self.weight.mean
            bias = self.bias.mean if self.bias is not None else None
        elif self.method == "dropconnect":
            delta_weight, delta_bias = _dropconnect_disurbance(
                self.weight, self._p, self.bias
            )
            weight = self.weight / (1 - self._p)
            bias = self.bias / (1 - self._p) if self.bias is not None else None

        outputs = F.conv2d(x, weight, bias, self.stride, self.padding)
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        perturbed_outputs = (
            F.conv2d(
                x * sign_input, delta_weight, delta_bias, self.stride, self.padding
            )
            * sign_output
        )
        return outputs + perturbed_outputs

    def extra_repr(self) -> str:
        return super().extra_repr() + f", method={self.method}, p={self._p_value}"
