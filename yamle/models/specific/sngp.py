from typing import Optional, Tuple
from torch import nn
from torch.nn.utils.spectral_norm import (
    SpectralNorm,
    SpectralNormLoadStateDictPreHook,
    SpectralNormStateDictHook,
)
import torch
import torch.nn as nn
from torch.nn.functional import normalize, conv_transpose2d, conv2d, batch_norm
from torch.nn.modules.batchnorm import _NormBase

from yamle.defaults import TINY_EPSILON, MODULE_INPUT_SHAPE_KEY

import math

"""
The code for this is largely inspired by: https://github.com/y0ast/DUE/blob/main/due/sngp.py
"""


def random_orthogonal_matrix(rows: int, cols: int) -> torch.Tensor:
    """Returns a random orthogonal matrix of shape (rows, cols)."""
    q, _ = torch.linalg.qr(torch.randn(rows, cols))
    return q


class RFF(nn.Module):
    """This class implements the Random Fourier Features layer as a replacement for Gaussian Process.

    The code is inspired by: https://github.com/y0ast/DUE/blob/main/due/sngp.py

    Args:
        in_features (int): The number of input features.
        random_features (int): The number of random features.
        out_features (int): The number of output features.
        mean_field_factor (float): The mean field factor.
        m (float): The gamma for exponential moving average for updating the precision matrix.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        random_features: int,
        mean_field_factor: float = 1.0,
        m: float = 0.99,
    ) -> None:
        super(RFF, self).__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._random_features = random_features
        self._m = m
        self._mean_field_factor = mean_field_factor
        self._final_epoch = False

        if random_features <= in_features:
            W = random_orthogonal_matrix(in_features, random_features)
        else:
            dim_left = random_features
            W = []
            while dim_left > in_features:
                W.append(random_orthogonal_matrix(in_features, in_features))
                dim_left -= in_features
            W.append(random_orthogonal_matrix(in_features, dim_left))
            W = torch.cat(W, dim=1)

        norm = torch.randn(W.shape) ** 2
        W *= torch.sqrt(norm.sum(0))
        self.register_buffer("W", W)

        B = torch.empty(random_features).uniform_(0, 2 * math.pi)
        self.register_buffer("B", B)

        self.beta = nn.Linear(random_features, out_features)

        precision = torch.zeros(random_features, random_features)
        self.register_buffer("precision", precision)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to compute the covariance matrix."""
        return math.sqrt(2.0 / self._out_features) * torch.cos(x @ self.W + self.B)

    def _mean(self, phi: torch.Tensor) -> torch.Tensor:
        """This method is used to compute the mean of the posterior."""
        return self.beta(phi)

    @torch.no_grad()
    def compute_covariance(self) -> None:
        """This method is used to compute the covariance matrix from the precision matrix."""
        self.precision = torch.inverse(self.precision)

    @torch.no_grad()
    def _update_precision(self, phi: torch.Tensor) -> None:
        """This method is used to update the precision matrix per each batch."""

        precision = torch.mm(phi.t(), phi)
        precision += torch.eye(precision.shape[0], device=precision.device)

        self.precision.data = self._m * self.precision + (1.0 - self._m) * precision

    @torch.no_grad()
    def _mean_field_logit_approximation(
        self, mean: torch.Tensor, covariance: torch.Tensor
    ) -> torch.Tensor:
        """This method is used to compute the mean field logit approximation."""
        scale = torch.clamp(
            torch.sqrt(1.0 + torch.diagonal(covariance) * self._mean_field_factor),
            min=TINY_EPSILON,
        ).unsqueeze(1)
        return mean / scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the layer."""
        phi = self._phi(x)
        mean = self._mean(phi)
        if self.training:
            if self._final_epoch:
                self._update_precision(phi)
            return mean
        else:
            predicted_covariance = torch.clamp(
                torch.mm(phi, self.precision @ phi.t()), min=TINY_EPSILON
            )
            return self._mean_field_logit_approximation(mean, predicted_covariance)


class SpectralNormLinear(SpectralNorm):
    """This class implements the spectral normalization for linear layers only."""

    def compute_weight(
        self, module: nn.Module, do_power_iteration: bool
    ) -> torch.Tensor:
        """This method is used to compute the spectral norm.

        Use the original method to compute the weight divided by `sigma`, then retrieve the `sigma`
        and compare it against the `coeff` to decide the scaling factor.
        """
        weight_divided_by_sigma = super().compute_weight(module, do_power_iteration)
        weight = getattr(module, self.name + "_orig")
        sigma = torch.mean(weight / weight_divided_by_sigma)
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor
        return weight

    @staticmethod
    def apply(
        module: nn.Module,
        name: str,
        n_power_iterations: int,
        dim: int,
        eps: float,
        coeff: float,
    ) -> "SpectralNormLinear":
        """This method is used to apply the spectral norm to the module."""
        fn = super(SpectralNormLinear, SpectralNormLinear).apply(
            module, name, n_power_iterations, dim, eps
        )
        fn.coeff = coeff
        return fn


class SpectralNormConv(SpectralNorm):
    """This class implements the spectral normalization for convolutional layers only."""

    def compute_weight(
        self, module: nn.Module, do_power_iteration: bool
    ) -> torch.Tensor:
        """This method is used to compute the spectral norm."""
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")

        # get settings from conv-module (for transposed convolution parameters)
        stride = module.stride
        padding = module.padding

        if do_power_iteration:
            with torch.no_grad():
                output_padding = 0
                if stride[0] > 1:
                    # Note: the below does not generalize to stride > 2
                    output_padding = 1 - self.input_shape[-1] % 2

                for _ in range(self.n_power_iterations):
                    v_s = conv_transpose2d(
                        u.view(self.outputs_dim),
                        weight,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                    )
                    v = normalize(v_s.view(-1), dim=0, eps=self.eps, out=v)

                    u_s = conv2d(
                        v.view(self.input_shape),
                        weight,
                        stride=stride,
                        padding=padding,
                        bias=None,
                    )
                    u = normalize(u_s.view(-1), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        weight_v = conv2d(
            v.view(self.input_shape), weight, stride=stride, padding=padding, bias=None
        )
        weight_v = weight_v.view(-1)
        sigma = torch.dot(u.view(-1), weight_v)
        factor = torch.max(torch.ones(1, device=weight.device), sigma / self.coeff)
        weight = weight / factor

        return weight

    @staticmethod
    def apply(
        module: nn.Module,
        name: str,
        n_power_iterations: int,
        eps: float,
        coeff: float,
        input_shape: Tuple[int, int, int, int],
    ) -> "SpectralNormConv":
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNormConv) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name)
                )

        fn = SpectralNormConv(name, n_power_iterations, eps=eps)
        fn.coeff = coeff
        fn.input_shape = input_shape
        weight = module._parameters[name]

        with torch.no_grad():
            num_input_shape = (
                input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
            )
            v = normalize(torch.randn(num_input_shape), dim=0, eps=fn.eps)

            # get settings from conv-module (for transposed convolution)
            stride = module.stride
            padding = module.padding
            # forward call to infer the shape
            u = conv2d(
                v.view(input_shape), weight, stride=stride, padding=padding, bias=None
            )
            fn.outputs_dim = u.shape
            num_outputs_dim = (
                fn.outputs_dim[0]
                * fn.outputs_dim[1]
                * fn.outputs_dim[2]
                * fn.outputs_dim[3]
            )
            # overwrite u with random init
            u = normalize(torch.randn(num_outputs_dim), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)

        module.register_forward_pre_hook(fn)

        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


class SpectralBatchNorm(_NormBase):
    """This class implements the spectral normalization for batch normalization layers only."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.01,
        affine: bool = True,
        coeff: float = 1.0,
    ):  # momentum is 0.01 by default instead of 0.1 of BN which alleviates noisy power iteration
        # Code is based on torch.nn.modules._NormBase
        super(SpectralBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats=True
        )
        self.coeff = coeff

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        # before the foward pass, estimate the lipschitz constant of the layer and
        # divide through by it so that the lipschitz constant of the batch norm operator is approximately
        # 1
        weight = (
            torch.ones_like(self.running_var) if self.weight is None else self.weight
        )
        # see https://arxiv.org/pdf/1804.04368.pdf, equation 28 for why this is correct.
        lipschitz = torch.max(torch.abs(weight * (self.running_var + self.eps) ** -0.5))

        # if lipschitz of the operation is greater than coeff, then we want to divide the input by a constant to
        # force the overall lipchitz factor of the batch norm to be exactly coeff
        lipschitz_factor = torch.max(lipschitz / self.coeff, torch.ones_like(lipschitz))

        weight = weight / lipschitz_factor

        return batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


def spectral_norm(
    module,
    coeff: float,
    n_power_iterations: int = 1,
    name: str = "weight",
    eps: float = 1e-12,
    dim: Optional[int] = None,
) -> nn.Module:
    """
    This method is used to apply spectral norm to the module.

    Args:
        module (nn.Module): The module to apply spectral norm.
        coeff (float): The coefficient for scaling.
        n_power_iterations (int): The number of power iterations.
        name (str): The name of the weight.
        eps (float): The epsilon for numerical stability.
        dim (int): The dimension for the spectral norm.
    """
    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0
    if isinstance(module, torch.nn.Linear):
        SpectralNormLinear.apply(module, name, n_power_iterations, dim, eps, coeff)
    elif isinstance(module, torch.nn.Conv2d):
        input_shape = getattr(module, MODULE_INPUT_SHAPE_KEY, None)
        SpectralNormConv.apply(
            module, name, n_power_iterations, eps, coeff, input_shape[0]
        )
    elif isinstance(module, torch.nn.BatchNorm2d):
        module = SpectralBatchNorm(module.num_features, coeff=coeff)

    return module
