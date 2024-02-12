import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.ao.quantization import fuse_modules

from yamle.defaults import (
    CLASSIFICATION_KEY,
    DEPTH_ESTIMATION_KEY,
    REGRESSION_KEY,
    SEGMENTATION_KEY,
    TEXT_CLASSIFICATION_KEY,
    RECONSTRUCTION_KEY,
    TINY_EPSILON,
)

logging = logging.getLogger("pytorch_lightning")


def output_activation(
    x: torch.Tensor, task: str, dim: Optional[int] = None
) -> torch.Tensor:
    """This function applies the output activation.

    Args:
        x (torch.Tensor): The input tensor.
        task (str): If the task is 'classification', the output is 'softmax'.
                    If the task is 'regression', the output is 'exp()' for the variance if the output is of shape (batch_size, 2).
        dim (Optional[int]): The dimension to apply the activation on. Defaults to 1.
    """
    dim = 1 if dim is None else dim
    if task in [CLASSIFICATION_KEY, SEGMENTATION_KEY, TEXT_CLASSIFICATION_KEY]:
        # Replace NaNs with TINY_EPSILON.
        x = torch.nan_to_num(
            x, nan=TINY_EPSILON, posinf=TINY_EPSILON, neginf=TINY_EPSILON
        )
        # Clamp the values between -50 and 50.
        x = torch.clamp(x, min=-50, max=50)
        # Cast to float64 to avoid numerical issues.
        x = torch.softmax(x, dim=dim, dtype=torch.float64)
        # Clamp the probabilities between TINY_EPSILON and 1+TINY_EPSILON.
        x = torch.nan_to_num(
            x, nan=TINY_EPSILON, posinf=TINY_EPSILON, neginf=TINY_EPSILON
        )
        x = torch.clamp(x, min=TINY_EPSILON, max=1 + TINY_EPSILON)
        return x
    elif task in [REGRESSION_KEY, DEPTH_ESTIMATION_KEY, RECONSTRUCTION_KEY]:
        if x.shape[dim] == 2:
            # Clamp the real part of the variance to be between -6 and 1.
            # This is done to avoid numerical issues especially for quantization.
            var = torch.index_select(x, dim, torch.tensor([1]).to(x.device))
            # Replace NaNs with 0.
            var = torch.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
            var = torch.clamp(var, min=-6, max=1)
            # Apply the exponential.
            var = torch.exp(var)
            # Clamp the variance to be at least TINY_EPSILON.
            var = torch.clamp(var, min=TINY_EPSILON)
            # Replace the variance with the exponential.
            mean = torch.index_select(x, dim, torch.tensor([0]).to(x.device))
            x = torch.cat([mean, var], dim=dim)
        return x
    else:
        raise ValueError(f"Task {task} is not supported.")


class OutputActivation(nn.Module):
    """This class is used to apply the output activation.

    If the task is `classification`, the output is `softmax`.
    If the task is `regression`, the output is `exp()` for the variance if the output is of shape `(batch_size, 2)`.

    Args:
        task (str): The task to perform.
        dim (Optional[int]): The dimension to apply the activation on. Defaults to 1.
    """

    def __init__(self, task: str, dim: Optional[int] = None) -> None:
        super().__init__()
        self._task = task
        self._dim = dim if dim is not None else 1
        # The enable parameter is used to enable or disable the activation.
        self._enable = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This function applies the output activation."""
        if self._enable:
            return output_activation(x, self._task, self._dim)
        else:
            return x

    def extra_repr(self) -> str:
        """This function returns the extra representation of the output activation."""
        return super().extra_repr() + f"task={self._task}, dim={self._dim}"

    def enable(self) -> None:
        """Enable the activation."""
        self._enable = True

    def disable(self) -> None:
        """Disable the activation."""
        self._enable = False


class ReshapeOutput(nn.Module):
    """This class is used to reshape the output of the model depending on the number of members.

    It does so with respect to the second dimension that is created by the `num_members` from the third dimension.
    """

    def __init__(self, num_members: int) -> None:
        super().__init__()
        self._num_members = num_members

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This function reshapes the output of the model."""
        return x.reshape(
            x.shape[0], self._num_members, x.shape[1] // self._num_members, *x.shape[2:]
        ).contiguous()

    def extra_repr(self) -> str:
        return super().extra_repr() + f"num_members={self._num_members}"


class ReshapeInput(nn.Module):
    """This class is used to reshape the input of the model depending on the number of members.

    It folds the `num_members` dimension into the second dimension.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This function reshapes the input of the model."""
        return x.reshape(x.shape[0], -1, *x.shape[3:]).contiguous()


class Unsqueeze(nn.Module):
    """This class is used to unsqueeze a tensor to a given shape length."""

    def __init__(self, shape_length: int) -> None:
        super(Unsqueeze, self).__init__()
        self._shape_length = shape_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This function is used to unsqueeze the tensor."""
        while x.dim() < self._shape_length:
            x = x.unsqueeze(-1)
        return x


class Add(nn.Module):
    """
    A simple class implementing residual addition.

    The forward function is simply the addition of the two inputs.

    Args:
        inplace (bool): If True, the addition is done in-place.
    """

    def __init__(self, inplace: bool = False) -> None:
        super(Add, self).__init__()
        self._inplace = inplace

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """The forward function of the residual addition."""
        return x.add_(y) if self._inplace else x.add(y)


class Multiply(nn.Module):
    """
    A simple class implementing residual multiplication.

    The forward function is simply the multiplication of the two inputs.

    Args:
        inplace (bool): If True, the multiplication is done in-place.
    """

    def __init__(self, inplace: bool = False) -> None:
        super(Multiply, self).__init__()
        self._inplace = inplace

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """The forward function of the residual multiplication."""
        return x.mul_(y) if self._inplace else x.mul(y)


class ResidualLayer(nn.Module):
    """This class implements a residual layer.

    It consists of a `layer` followed by a residual addition with the input.
    The `layer` should be a `nn.Module` of `nn.Sequential` type.

    Args:
        layer (nn.Module): The layer to be used.
        inplace (bool): If True, the addition is done in-place.
    """

    def __init__(self, layer: nn.Module, inplace: bool = False) -> None:
        super(ResidualLayer, self).__init__()
        # Do not change the name of this variable. `StochasticDepth` uses it.
        self._layer = layer
        self._add = Add(inplace)

    def forward(
        self, x: torch.Tensor, identity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """The forward function of the residual layer.

        By default, the identity is the input `x`. It can be changed by passing a tensor to the `identity` argument.
        """
        if identity is None:
            identity = x
        return self._add(identity, self._layer(x))


class ParallelModel(nn.Module):
    """This class implements a parallel model.

    It consists of a list of `models` that are applied in parallel to the input.
    The `models` should be a list of `nn.Module` of `nn.Sequential` type.
    The input is assumed to be of shape `(batch_size, len(models), *)` or `(batch_size, *)`
    if `single_source` is True.

    Args:
        models (List[nn.Module]): The models to be used.
        single_source (bool): If `True`, the input is assumed to be of shape `(batch_size, *)`. If
                                `False`, the input is assumed to be of shape `(batch_size, len(models), *)`.
                                Defaults to `False`.
        inputs_dim (int): The dimension to split the input on. Defaults to 1. Used only if `single_source` is `False`.
        outputs_dim (int): The dimension to stack the outputs on. Defaults to 1.
        initialise_members_same (bool): If `True`, the members of the models are initialised to the same values. Defaults to `False`.
    """

    def __init__(
        self,
        models: List[nn.Module],
        single_source: bool = False,
        inputs_dim: int = 1,
        outputs_dim: int = 1,
        initialise_members_same: bool = False,
    ) -> None:
        super(ParallelModel, self).__init__()
        self._models = nn.ModuleList(models)
        self._num_members = len(models)
        self._inputs_dim = inputs_dim
        self._outputs_dim = outputs_dim
        self._single_source = single_source
        if initialise_members_same:
            self.initialise_members_same()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the parallel model."""
        if self._single_source:
            return torch.stack(
                [model(x) for model in self._models], dim=self._outputs_dim
            )
        else:
            x = torch.split(x, 1, dim=self._inputs_dim)
            x = [x_.squeeze(self._inputs_dim) for x_ in x]
            # Check if by accident the input was squeezed too much i.e. len(x.shape) == 1.
            if len(x[0].shape) == 1:
                x = [x_.unsqueeze(1) for x_ in x]
            num_inputs = len(x)
            # If the number of inputs is the same as the number of models, apply each model to the corresponding input.
            if num_inputs == self._num_members:
                return torch.stack(
                    [model(x_) for x_, model in zip(x, self._models)],
                    dim=self._outputs_dim,
                )
            # If the number of inputs is larger then split them proportionally to the number of models.
            elif num_inputs > self._num_members:
                num_inputs_per_model = num_inputs // self._num_members
                outputs = []
                for i in range(self._num_members):
                    for j in range(num_inputs_per_model):
                        outputs.append(self._models[i](x[i * num_inputs_per_model + j]))
                assert (
                    len(outputs) == num_inputs
                ), f"The number of outputs ({len(outputs)}) is not equal to the number of inputs ({num_inputs})."
                return torch.stack(outputs, dim=self._outputs_dim)
            else:
                raise ValueError(
                    f"The number of inputs ({num_inputs}) is smaller than the number of models ({self._num_members})."
                )

    @torch.no_grad()
    def initialise_members_same(self) -> None:
        """This is a helper function to initialise the members of the parallel model with the same weights.

        The weights are copied from the first model in the list to all the other models.
        """
        weights = list(self._models[0].parameters())
        for model in self._models[1:]:
            for param, weight in zip(model.parameters(), weights):
                param.data.copy_(weight.data)

    def __getitem__(self, index: int) -> nn.Module:
        """This function returns the model at the given index."""
        assert (
            index < self._num_members
        ), f"The index ({index}) is larger than the number of models ({self._num_members})."
        return self._models[index]

    def __len__(self) -> int:
        """This function returns the number of models in the parallel model."""
        return self._num_members


class Normalization(nn.Module):
    """This class implements a normalization layer.

    Args:
        norm (Optional[str]): The type of normalization to use. Defaults to None. Choices are `batch`, `layer` and `instance`.
        dimension (int): The dimension to normalise on. Defaults to 1.
        norm_kwargs (Dict[str, Any]): The keyword arguments to be passed to the normalization layer. Defaults to `{}.`
    """

    def __init__(
        self,
        norm: Optional[str] = None,
        dimension: int = 1,
        norm_kwargs: Dict[str, Any] = {},
    ) -> None:
        super(Normalization, self).__init__()
        if norm is not None:
            if norm == "batch":
                if dimension == 1:
                    self._norm = nn.BatchNorm1d(**norm_kwargs)
                elif dimension == 2:
                    self._norm = nn.BatchNorm2d(**norm_kwargs)
                elif dimension == 3:
                    self._norm = nn.BatchNorm3d(**norm_kwargs)
                else:
                    raise ValueError(
                        f"Invalid dimension {dimension} for batch normalization."
                    )
            elif norm == "layer":
                if "affine" in norm_kwargs:
                    norm_kwargs["elementwise_affine"] = norm_kwargs["affine"]
                    del norm_kwargs["affine"]
                if "num_features" in norm_kwargs:
                    norm_kwargs["normalized_shape"] = norm_kwargs["num_features"]
                    del norm_kwargs["num_features"]
                self._norm = nn.LayerNorm(**norm_kwargs)
            elif norm == "instance":
                if dimension == 1:
                    self._norm = nn.InstanceNorm1d(**norm_kwargs)
                elif dimension == 2:
                    self._norm = nn.InstanceNorm2d(**norm_kwargs)
                elif dimension == 3:
                    self._norm = nn.InstanceNorm3d(**norm_kwargs)
                else:
                    raise ValueError(
                        f"Invalid dimension {dimension} for instance normalization."
                    )
            else:
                raise ValueError(f"Invalid normalization type {norm}.")
        else:
            self._norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the normalization layer."""
        if self._norm is not None:
            return self._norm(x)
        else:
            return x


class Pooling(nn.Module):
    """This class implements a pooling layer.

    Args:
        pooling (Optional[str]): The type of pooling to use. Defaults to None. Choices are `max`, `avg`, `adaptive_max` and `adaptive_avg`.
        dimension (int): The dimension to pool on. Defaults to 1.
        pool_kwargs (Dict[str, Any]): The keyword arguments to be passed to the pooling layer. Defaults to `{}.`
    """

    def __init__(
        self,
        pooling: Optional[str] = None,
        dimension: int = 1,
        pool_kwargs: Dict[str, Any] = {},
    ) -> None:
        super(Pooling, self).__init__()
        if pooling is not None:
            if pooling == "max":
                if dimension == 1:
                    self._pool = nn.MaxPool1d(**pool_kwargs)
                elif dimension == 2:
                    self._pool = nn.MaxPool2d(**pool_kwargs)
                elif dimension == 3:
                    self._pool = nn.MaxPool3d(**pool_kwargs)
                else:
                    raise ValueError(f"Invalid dimension {dimension} for max pooling.")
            elif pooling == "avg":
                if dimension == 1:
                    self._pool = nn.AvgPool1d(**pool_kwargs)
                elif dimension == 2:
                    self._pool = nn.AvgPool2d(**pool_kwargs)
                elif dimension == 3:
                    self._pool = nn.AvgPool3d(**pool_kwargs)
                else:
                    raise ValueError(
                        f"Invalid dimension {dimension} for average pooling."
                    )
            elif pooling == "adaptive_max":
                if dimension == 1:
                    self._pool = nn.AdaptiveMaxPool1d(**pool_kwargs)
                elif dimension == 2:
                    self._pool = nn.AdaptiveMaxPool2d(**pool_kwargs)
                elif dimension == 3:
                    self._pool = nn.AdaptiveMaxPool3d(**pool_kwargs)
                else:
                    raise ValueError(
                        f"Invalid dimension {dimension} for adaptive max pooling."
                    )
            elif pooling == "adaptive_avg":
                if dimension == 1:
                    self._pool = nn.AdaptiveAvgPool1d(**pool_kwargs)
                elif dimension == 2:
                    self._pool = nn.AdaptiveAvgPool2d(**pool_kwargs)
                elif dimension == 3:
                    self._pool = nn.AdaptiveAvgPool3d(**pool_kwargs)
                else:
                    raise ValueError(
                        f"Invalid dimension {dimension} for adaptive average pooling."
                    )
            else:
                raise ValueError(f"Invalid pooling type {pooling}.")
        else:
            self._pool = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the pooling layer."""
        if self._pool is not None:
            return self._pool(x)
        else:
            return x


class Activation(nn.Module):
    """This class implements an activation function.

    Args:
        activation (Optional[str]): The type of activation to use. Defaults to None. Choices are `relu`, `sigmoid`, `tanh`, `softmax` and `log_softmax`.
        dimension (int): The dimension to apply the activation on. Defaults to 1.
    """

    def __init__(self, activation: Optional[str] = None, dimension: int = 1) -> None:
        super(Activation, self).__init__()
        if activation is not None:
            activation = activation.lower()
            if activation == "relu":
                self._activation = nn.ReLU()
            elif activation == "sigmoid":
                self._activation = nn.Sigmoid()
            elif activation == "tanh":
                self._activation = nn.Tanh()
            elif activation == "elu":
                self._activation = nn.ELU()
            elif activation == "gelu":
                self._activation = nn.GELU()
            elif activation == "silu":
                self._activation = nn.SiLU()
            elif activation == "leaky_relu":
                self._activation = nn.LeakyReLU()
            elif activation == "softmax":
                self._activation = nn.Softmax(dim=dimension)
            elif activation == "log_softmax":
                self._activation = nn.LogSoftmax(dim=dimension)
            else:
                raise ValueError(f"Invalid activation type {activation}.")
        else:
            self._activation = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the activation layer."""
        if self._activation is not None:
            return self._activation(x)
        else:
            return x


class MatrixMultiplication(nn.Module):
    """This class implements a matrix multiplication layer."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """The forward function of the matrix multiplication layer."""
        return torch.matmul(x, y)


class LinearNormActivation(nn.Module):
    """This class is used to create a linear layer followed by a normalization layer and an activation layer.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        bias (bool): If True, the layer has a bias.
        normalization (Optional[str]): The type of normalization to use. Defaults to None. Choices are `batch`, `layer` and `instance`.
        activation (str): The type of activation to use. Defaults to `relu`. Choices are `relu`, `sigmoid`, `tanh`, `softmax` and `log_softmax`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        normalization: Optional[str] = None,
        activation: str = "relu",
    ) -> None:
        super(LinearNormActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.norm_kwargs = {
            "norm_kwargs": {"affine": True, "num_features": out_features},
            "norm": normalization,
            "dimension": 1,
        }
        layers = [nn.Linear(in_features, out_features, bias)]
        layers.append(Normalization(**self.norm_kwargs))
        layers.append(Activation(activation=activation))
        self._linear_norm_activation = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the linear layer followed by normalization and ReLU."""
        return self._linear_norm_activation(x)

    def replace_layers_for_quantization(self) -> None:
        """This method is used to replace the layers for quantization.

        It merges the linear layer with the normalization layer and activation layer
        if possible.
        This is currently only possible if the normalization layer is batch normalization
        and the activation layer is ReLU.
        """
        if isinstance(self._linear_norm_activation[1], Normalization) and isinstance(
            self._linear_norm_activation[1]._norm, nn.BatchNorm1d
        ):
            fuse_modules(self._linear_norm_activation, [["0", "1._norm"]], inplace=True)

        elif isinstance(self._linear_norm_activation[1], Activation) and isinstance(
            self._linear_norm_activation[1]._activation, nn.ReLU
        ):
            fuse_modules(
                self._linear_norm_activation, [["0", "1._activation"]], inplace=True
            )


class Conv2dNormActivation(nn.Module):
    """This class is used to create a convolutional layer followed by a normalization layer and an activation layer.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the convolutional kernel. Default: 3.
        stride (int): The stride of the convolution. Default: 1.
        padding (int): The padding of the convolution. Default: 1.
        dilation (int): The dilation of the convolution. Default: 1.
        groups (int): The number of groups in the convolution. Default: 1.
        bias (bool): If True, the convolution has a bias. Default: True.
        normalization (Optional[str]): The type of normalization to use. Defaults to None. Choices are `batch`, `layer` and `instance`.
        activation (str): The type of activation to use. Defaults to `relu`. Choices are `relu`, `sigmoid`, `tanh`, `softmax` and `log_softmax`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        normalization: Optional[str] = None,
        activation: str = "relu",
    ) -> None:
        super(Conv2dNormActivation, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
            )
        ]
        self.norm_kwargs = {
            "norm_kwargs": {"affine": True, "num_features": out_channels},
            "norm": normalization,
            "dimension": 2,
        }
        layers.append(Normalization(**self.norm_kwargs))
        layers.append(Activation(activation))
        self._conv_norm_activation = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the convolutional norm ReLU layer."""
        return self._conv_norm_activation(x)

    def replace_layers_for_quantization(self) -> None:
        """This method is used to replace the layers for quantization.

        It merges the convolutional layer with the normalization layer and activation layer
        if possible.
        This is currently only possible if the normalization layer is batch normalization
        and the activation layer is ReLU.
        """
        if (
            isinstance(self._conv_norm_activation[1], Normalization)
            and isinstance(self._conv_norm_activation[1]._norm, nn.BatchNorm2d)
            and isinstance(self._conv_norm_activation[2]._activation, nn.ReLU)
        ):
            fuse_modules(
                self._conv_norm_activation,
                [["0", "1._norm", "2._activation"]],
                inplace=True,
            )

        elif isinstance(self._conv_norm_activation[1], Normalization) and isinstance(
            self._conv_norm_activation[1]._norm, nn.BatchNorm2d
        ):
            fuse_modules(self._conv_norm_activation, [["0", "1._norm"]], inplace=True)

        elif isinstance(self._conv_norm_activation[1], Activation) and isinstance(
            self._conv_norm_activation[1]._activation, nn.ReLU
        ):
            fuse_modules(
                self._conv_norm_activation, [["0", "1._activation"]], inplace=True
            )


class DoubleConv2d(nn.Module):
    """This class is used to create a double convolutional layer.

    It is composed of two convolutional layers, followed by a normalization layer and ReLU.
    The first convolutional layer has output channels size `out_channels`. If
    `residual` is True, the output of the second convolutional layer is added to
    the input of the first convolutional layer.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the convolutional kernel. Default: 3.
        stride (int): The stride of the convolution. Default: 1.
        padding (int): The padding of the convolution. Default: 1.
        bias (bool): If True, the convolution has a bias. Default: True.
        normalization (Optional[str]): The type of normalization to use. Defaults to `batch`. Choices are `batch`, `layer` and `instance`.
        activation (str): The type of activation to use. Defaults to `relu`. Choices are `relu`, `sigmoid`, `tanh`, `softmax` and `log_softmax`.
        residual (bool): If True, the output of the second convolutional layer is
            added to the input of the first convolutional layer. Default: True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        normalization: Optional[str] = "batch",
        activation: str = "relu",
        residual: bool = True,
    ) -> None:
        super(DoubleConv2d, self).__init__()

        self._conv_norm_activation = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=bias,
            normalization=normalization,
            activation=activation,
        )

        self._final_conv_norm_activation = Conv2dNormActivation(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=bias,
            normalization=normalization,
            activation=activation,
        )
        if residual:
            self._final_conv_norm_activation = ResidualLayer(
                self._final_conv_norm_activation
            )

        self._residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the double convolutional layer."""
        x = self._conv_norm_activation(x)
        return self._final_conv_norm_activation(x)


class DepthwiseSeparableConv2d(nn.Module):
    """This class is used to create a depthwise separable convolutional layer.

    It consists of a depthwise convolutional layer followed by a pointwise convolutional layer.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the convolutional kernel. Default: 3.
        stride (int): The stride of the convolution. Default: 1.
        padding (int): The padding of the convolution. Default: 1.
        bias (bool): If True, the convolution has a bias. Default: True.
        normalization (Optional[str]): The type of normalization to use. Defaults to None. Choices are `batch`, `layer` and `instance`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        normalization: Optional[str] = None,
    ) -> None:
        super(DepthwiseSeparableConv2d, self).__init__()
        self._depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=False,
        )
        if normalization is not None:
            self._normalization = Normalization(
                normalization,
                dimension=2,
                norm_kwargs={"affine": True, "num_features": in_channels},
            )
        self._pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """This function is used to initialize the parameters of the layer."""
        nn.init.kaiming_normal_(self._depthwise_conv.weight)
        nn.init.kaiming_normal_(self._pointwise_conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the separable convolutional layer."""
        x = self._depthwise_conv(x)
        if hasattr(self, "_normalization"):
            x = self._normalization(x)
        return self._pointwise_conv(x)


class CompletelySeparableConv2d(nn.Module):
    """This class is used to create a completely separable convolutional layer.

    It consists of two depthwise convolutional layers followed by a pointwise convolutional layer.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the convolutional kernel. Default: 3.
        stride (int): The stride of the convolution. Default: 1.
        padding (int): The padding of the convolution. Default: 1.
        bias (bool): If True, the convolution has a bias. Default: True.
        normalization (Optional[str]): The type of normalization to use. Defaults to None. Choices are `batch`, `layer` and `instance`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        normalization: Optional[str] = None,
    ) -> None:
        super(CompletelySeparableConv2d, self).__init__()
        self._depthwise_conv_1 = nn.Conv2d(
            in_channels,
            in_channels,
            (1, kernel_size),
            stride=(1, stride),
            padding=(0, padding),
            groups=in_channels,
            bias=False,
        )
        self._depthwise_conv_2 = nn.Conv2d(
            in_channels,
            in_channels,
            (kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            groups=in_channels,
            bias=False,
        )
        if normalization is not None:
            self._normalization = Normalization(
                normalization,
                dimension=2,
                norm_kwargs={"affine": True, "num_features": in_channels},
            )
        self._pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """This function is used to initialize the parameters of the layer."""
        nn.init.kaiming_normal_(self._depthwise_conv_1.weight)
        nn.init.kaiming_normal_(self._depthwise_conv_2.weight)
        nn.init.kaiming_normal_(self._pointwise_conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the separable convolutional layer."""
        x = self._depthwise_conv_1(x)
        x = self._depthwise_conv_2(x)
        if hasattr(self, "_normalization"):
            x = self._normalization(x)
        return self._pointwise_conv(x)


class SqueezeAndExcitation(nn.Module):
    """This class is used to create a squeeze and excitation layer.

    It is implemented according to the paper `Squeeze-and-Excitation Networks
    <https://arxiv.org/abs/1709.01507>`_.

    Args:
        in_out_channels (int): The number of input channels and output channels.
        reduction (int): The reduction factor. Default: 16.
        activation (Optional[str]): The type of activation function to use. Defaults to relu. Choices are `relu`, `leaky_relu` and `elu`.
    """

    def __init__(
        self,
        in_out_channels: int,
        reduction: int = 16,
        activation: Optional[str] = "relu",
    ) -> None:
        super(SqueezeAndExcitation, self).__init__()
        assert (
            in_out_channels % reduction == 0
        ), f"The number of input channels {in_out_channels} must be divisible by the reduction factor {reduction}."
        assert (
            in_out_channels >= reduction
        ), f"The number of input channels {in_out_channels} must be greater than or equal to the reduction factor {reduction}."
        self._global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self._fc = nn.Sequential(
            nn.Linear(in_out_channels, in_out_channels // reduction),
            Activation(activation),
            nn.Linear(in_out_channels // reduction, in_out_channels),
            Activation("sigmoid"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the squeeze and excitation layer."""
        B, C, _, _ = x.size()
        y = self._global_avg_pool(x).view(B, C)
        y = self._fc(y).view(B, C, 1, 1)
        return x * y


class LSTM(nn.Module):
    """This class is used to create a simple LSTM cell.

    Args:
        input_size (int): The number of input features.
        hidden_size (int): The number of hidden features.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super(LSTM, self).__init__()

        self._input_size = input_size
        self._hidden_size = hidden_size

        self._Wi = nn.Linear(self._input_size, self._hidden_size, bias=False)
        self._Wf = nn.Linear(self._input_size, self._hidden_size, bias=False)
        self._Wo = nn.Linear(self._input_size, self._hidden_size, bias=False)
        self._Wg = nn.Linear(self._input_size, self._hidden_size, bias=False)

        self._Ui = nn.Linear(self._hidden_size, self._hidden_size, bias=True)
        self._Uf = nn.Linear(self._hidden_size, self._hidden_size, bias=True)
        self._Uo = nn.Linear(self._hidden_size, self._hidden_size, bias=True)
        self._Ug = nn.Linear(self._hidden_size, self._hidden_size, bias=True)

        self._input_gate_sigmoid = nn.Sigmoid()
        self._forget_gate_sigmoid = nn.Sigmoid()
        self._output_gate_sigmoid = nn.Sigmoid()
        self._g_gate_tanh = nn.Tanh()
        self._h_tanh = nn.Tanh()

        self._input_add = Add()
        self._forget_add = Add()
        self._output_add = Add()
        self._g_add = Add()
        self._cell_add = Add()
        self._cell_multiply1 = Multiply()
        self._cell_multiply2 = Multiply()
        self._h_multiply = Multiply()

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The forward function of the LSTM cell.

        Args:
            x (torch.Tensor): The input tensor of shape `(batch_size, T, input_size)`.
            h (torch.Tensor, optional): The hidden state of shape `(batch_size, hidden_size)`.
            c (torch.Tensor, optional): The cell state of shape `(batch_size, hidden_size)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The complete hidden state of shape `(batch_size, T, hidden_size)`
            and the last hidden and cell state of shape `(batch_size, hidden_size)`.
        """

        T = x.shape[1]
        if h is None:
            h = torch.zeros(x.size(0), self._hidden_size, device=x.device)

        if c is None:
            c = torch.zeros(x.size(0), self._hidden_size, device=x.device)

        h_t = []
        for t in range(T):
            i = self._input_gate_sigmoid(
                self._input_add(self._Wi(x[:, t, :]), self._Ui(h))
            )
            f = self._forget_gate_sigmoid(
                self._forget_add(self._Wf(x[:, t, :]), self._Uf(h))
            )
            o = self._output_gate_sigmoid(
                self._output_add(self._Wo(x[:, t, :]), self._Uo(h))
            )
            g = self._g_gate_tanh(self._g_add(self._Wg(x[:, t, :]), self._Ug(h)))
            c = self._cell_add(self._cell_multiply1(f, c), self._cell_multiply2(i, g))
            h = self._h_multiply(o, self._h_tanh(c))
            h_t.append(h)
        return torch.stack(h_t, dim=1), h, c


class Lambda(nn.Module):
    """This class is used to create a lambda layer.

    Args:
        fn (Callable): The function to apply.
    """

    def __init__(self, fn: Callable) -> None:
        super(Lambda, self).__init__()
        self._fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the lambda layer."""
        return self._fn(x)


class Reduction(nn.Module):
    """Given a dimension this module implements a reduction operation.

    Can choose between `sum`, `mean`, `max`, `min` and `cat`. If `cat` is chosen, the dimension size will
    be multiplied by the number of tensors.

    Args:
        dim (int): The dimension to reduce.
        reduction (str): The reduction operation. Choices are: `sum`, `mean`, `max`, `min` and `cat`.
        alignment (bool): If True, an alighment score will be computed for each tensor followed by a softmax.
    """

    def __init__(
        self, dim: int, reduction: str = "sum", alignment: bool = False
    ) -> None:
        super(Reduction, self).__init__()
        self._dim = dim
        assert reduction in [
            "sum",
            "mean",
            "max",
            "min",
            "cat",
        ], f"Reduction {reduction} is not supported."
        self._reduction = reduction
        self._alignment = alignment

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the reduction layer."""
        if self._alignment and x.shape[1] > 1:
            x_flattened = x.view(x.size(0), x.size(1), -1)
            scale = x_flattened.size(-1) ** -0.5
            softmax = torch.softmax(
                x_flattened.bmm(x_flattened.transpose(1, 2)) / scale, dim=-1
            )
            x = softmax.bmm(x)
        if self._reduction == "sum":
            return torch.sum(x, dim=self._dim)
        elif self._reduction == "mean":
            return torch.mean(x, dim=self._dim)
        elif self._reduction == "max":
            return torch.max(x, dim=self._dim)[0]
        elif self._reduction == "min":
            return torch.min(x, dim=self._dim)[0]
        elif self._reduction == "cat":
            x = torch.split(x, 1, dim=self._dim)
            x = [x_.squeeze(dim=self._dim) for x_ in x]
            return torch.cat(x, dim=self._dim)
        else:
            raise ValueError(f"Reduction {self._reduction} is not supported.")

    def extra_repr(self) -> str:
        return super().extra_repr() + f"reduction={self._reduction}"


class ScalarMultiplier(nn.Module):
    """This operation element-wise multiplies a tensor with learnable parameters.

    The parameters are initialized to be 1.
    The operation can be enabled or disabled.

    Args:
        shape (tuple): The shape of the learnable parameters.
    """

    def __init__(self, shape: Tuple[int, ...]):
        super(ScalarMultiplier, self).__init__()
        self._shape = shape
        self._p = nn.Parameter(torch.zeros(*shape), requires_grad=True)
        self._sigmoid = nn.Sigmoid()
        self._enable = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the scalar multiplier.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        if self._enable:
            return x * self._sigmoid(self._p)
        else:
            return x * self._sigmoid(self._p.detach())

    def enable(self):
        """Enable the scalar multiplier."""
        self._enable = True

    def disable(self):
        """Disable the scalar multiplier."""
        self._enable = False

    def extra_repr(self) -> str:
        return super().extra_repr() + f"shape={self._shape}"


class ScalarAdder(ScalarMultiplier):
    """This operation element-wise adds a tensor with learnable parameters.

    The parameters are initialized to be 0.
    The operation can be enabled or disabled.

    Args:
        shape (tuple): The shape of the learnable parameters.
    """

    def __init__(self, shape: Tuple[int, ...]):
        super(ScalarAdder, self).__init__(shape)
        self._p.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the scalar adder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        if self._enable:
            return x + self._p
        else:
            return x + self._p.detach()


class LinearExtractor(nn.Module):
    """This module implements the linear neural network model.

    It is used to be either input multiplexer or output demultiplexer. It takes in `inputs_dim`,
    `expansion_factor` and `outputs_dim` and produces a linear neural network model with `activation`
    and optionally `norm` layers in between. The number of layers is determined by `depth`.

    Args:
        inputs_dim (int): The input dimension for the input number of features.
        expansion_factor (Optional[float]): The expansion factor for the hidden representation.
        hidden_dim (Optional[int]): The hidden dimension for the hidden representation. If hidden_dim is None,
            it will be set to `expansion_factor * inputs_dim`.
        outputs_dim (int): The output dimension for the output features.
        depth (int): The number of layers in the sequence.
        activation (str): The activation function to be used in the model. Default is `ReLU`.
        norm (bool): Whether to use normalization layers in the model. Default is `False`.
        normalization (str): The normalization layer to be used in the model. Default is `BatchNorm1d`.
        end_activation (bool): Whether to use activation function at the end of the model. Default is `False`.
        end_normalization (bool): Whether to use normalization layer at the end of the model. Default is `False`.
        residual (bool): Whether to use residual connections in the model. Default is `False`.
    """

    def __init__(
        self,
        inputs_dim: int,
        expansion_factor: Optional[float] = None,
        hidden_dim: Optional[int] = None,
        outputs_dim: int = 1,
        depth: int = 1,
        activation: str = "ReLU",
        norm: bool = False,
        normalization: str = "batch",
        end_activation: bool = False,
        end_normalization: bool = False,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self._inputs_dim = inputs_dim
        self._expansion_factor = expansion_factor
        self._outputs_dim = outputs_dim
        self._depth = depth
        layers = []
        self._norm = norm
        assert (
            hidden_dim is not None or expansion_factor is not None
        ), "Either hidden_dim or expansion_factor must be specified."
        self._hidden_dim = (
            int(expansion_factor * inputs_dim) if hidden_dim is None else hidden_dim
        )
        activation_blueprint = Activation
        if depth == 1:
            layers.append(nn.Linear(inputs_dim, outputs_dim))
        elif depth > 1:
            for i in range(depth - 1):
                layers.append(nn.Linear(inputs_dim, self._hidden_dim))
                if norm:
                    layers.append(
                        Normalization(
                            normalization,
                            dimension=1,
                            norm_kwargs={
                                "num_features": self._hidden_dim,
                                "affine": True,
                            },
                        )
                    )
                layers.append(activation_blueprint(activation))
                inputs_dim = self._hidden_dim
                if residual and i > 0:
                    residual_layer = ResidualLayer(
                        nn.Sequential(*layers[-3:] if norm else layers[-2:])
                    )
                    layers = layers[:-3] if norm else layers[:-2]
                    layers.append(residual_layer)
            layers.append(nn.Linear(inputs_dim, outputs_dim))

        if end_normalization and norm:
            layers.append(
                Normalization(
                    normalization,
                    dimension=1,
                    norm_kwargs={"num_features": outputs_dim, "affine": True},
                )
            )

        if end_activation:
            layers.append(activation_blueprint(activation))

        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the linear neural network model."""
        return self._model(x)

    def extra_repr(self) -> str:
        """The extra representation of the linear neural network model."""
        return (
            super().extra_repr()
            + f"inputs_dim={self._inputs_dim}, expansion_factor={self._expansion_factor}, outputs_dim={self._outputs_dim}, depth={self._depth}, hidden_dim={self._hidden_dim}, norm={self._norm}"
        )


class Conv2dExtractor(nn.Module):
    """This module implements the convolutional neural network model.

    It is used to be either input multiplexer or output demultiplexer. It takes in `input_channels`,
    `expansion_factor` and `output_channels` and produces a convolutional neural network model with
    `activation` and optionally `norm` layers in between. The number of layers is determined by
    `depth`. Global average pooling is optionally applied to the output through `end_pooling`.

    Args:
        input_channels (int): The input dimension for the input number of features.
        expansion_factor (Optional[float]): The expansion factor for the hidden representation.
        hidden_dim (Optional[int]): The hidden dimension for the hidden representation. If hidden_dim is None,
            it will be set to `expansion_factor * inputs_dim`.
        output_channels (int): The output dimension for the number of output features.
        depth (int): The number of layers in the sequence.
        activation (str): The activation function to be used in the model. Default is `ReLU`.
        norm (bool): Whether to use normalization layers in the model. Default is `False`.
        normalization (str): The normalization layer to be used in the model. Default is `BatchNorm2d`.
        pool (bool): Whether to use pooling layers in the model. Default is `False`.
        pooling (str): The pooling layer to be used in the model. Default is `max`.
        end_activation (bool): Whether to use activation function at the end of the model. Default is `False`.
        end_normalization (bool): Whether to use normalization layer at the end of the model. Default is `False`.
        end_pooling (bool): Whether to apply global average pooling to the output. Default is `False`.
        convolution (str): The convolutional layer to be used. Default is `conv2d`. Choices are
                           `completely_separable`, `depthwise_separable`, `conv2d`.
        residual (bool): Whether to use residual connections in the model. Default is `False`.
        se (bool): Whether to use squeeze and excitation in the model. Default is `False`.
    """

    def __init__(
        self,
        input_channels: int,
        expansion_factor: Optional[float] = None,
        hidden_dim: Optional[int] = None,
        output_channels: int = 1,
        depth: int = 1,
        activation: str = "ReLU",
        norm: bool = False,
        normalization: str = "batch",
        pool: bool = False,
        pooling: str = "max",
        end_activation: bool = False,
        end_normalization: bool = False,
        end_pooling: bool = False,
        convolution: str = "conv2d",
        residual: bool = False,
        se: bool = False,
    ) -> None:
        super().__init__()
        self._input_channels = input_channels
        self._expansion_factor = expansion_factor
        self._output_channels = output_channels
        self._depth = depth
        layers = []
        assert (
            hidden_dim is not None or expansion_factor is not None
        ), "Either hidden_dim or expansion_factor must be specified."
        self._hidden_channels = (
            int(expansion_factor * input_channels) if hidden_dim is None else hidden_dim
        )
        self._norm = norm
        activation_blueprint = Activation
        self._convolution = convolution
        if convolution == "completely_separable":
            convolution = CompletelySeparableConv2d
        elif convolution == "depthwise_separable":
            convolution = DepthwiseSeparableConv2d
        elif convolution == "conv2d":
            convolution = nn.Conv2d
        if depth == 1:
            layers.append(
                convolution(input_channels, output_channels, kernel_size=3, padding=1)
            )
        elif depth > 1:
            for i in range(depth - 1):
                layers.append(
                    convolution(
                        input_channels, self._hidden_channels, kernel_size=3, padding=1
                    )
                )
                if norm:
                    layers.append(
                        Normalization(
                            normalization,
                            dimension=2,
                            norm_kwargs={
                                "num_features": self._hidden_channels,
                                "affine": True,
                            },
                        )
                    )
                layers.append(activation_blueprint(activation))
                input_channels = self._hidden_channels
                if residual and i > 0:
                    residual_layer = ResidualLayer(
                        nn.Sequential(*layers[-3:] if norm else layers[-2:])
                    )
                    layers = layers[:-3] if norm else layers[:-2]
                    layers.append(residual_layer)
                if se:
                    se_layer = SqueezeAndExcitation(self._hidden_channels, reduction=8)
                    layers.append(se_layer)
                if pool:
                    pool_layer = Pooling(
                        pooling,
                        dimension=2,
                        pool_kwargs={"kernel_size": 2, "stride": 2},
                    )
                    layers.append(pool_layer)
            layers.append(
                convolution(input_channels, output_channels, kernel_size=3, padding=1)
            )

        if end_normalization and norm:
            layers.append(
                Normalization(
                    normalization,
                    dimension=2,
                    norm_kwargs={"num_features": output_channels, "affine": True},
                )
            )

        if end_activation:
            layers.append(activation_blueprint(activation))

        if end_pooling:
            layers.append(nn.AdaptiveAvgPool2d(1))
            layers.append(nn.Flatten())

        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the convolutional neural network model."""
        return self._model(x)

    def extra_repr(self) -> str:
        """The extra representation of the convolutional neural network model."""
        return (
            super().extra_repr()
            + f"input_channels={self._input_channels}, expansion_factor={self._expansion_factor}, output_channels={self._output_channels}, depth={self._depth}, hidden_channels={self._hidden_channels}, norm={self._norm}, convolution={self._convolution}"
        )
