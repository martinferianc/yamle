from typing import List, Tuple, Dict, Any, Union
import torch.nn as nn
import torch
import argparse
import math

from yamle.models.operations import (
    OutputActivation,
    LinearNormActivation,
    Conv2dNormActivation,
    ResidualLayer,
    Normalization,
    ReshapeOutput,
)

from yamle.defaults import REGRESSION_KEY, CLASSIFICATION_KEY

from yamle.models.model import BaseModel


class ConvNetModel(BaseModel):
    """This class is used to create a convolutional model similar to LeNet.

    It combines convolutional layers, followed each time by pooling layers,
    ended with fully connected layers.

    The first input layer is a Convolution, followed by normalization and ReLU,
    otherwise we used the `Conv2dNormActivation` classes and `LinearNormActivation` classes.
    The output is only a linear layer, followed by the `OutputActivation` class.

    Args:
        inputs_dim (Tuple[int, int, int]): The input dimensions.
        conv_hidden_dims (List[int]): The dimensions of the convolutional hidden layers.
        linear_hidden_dims (List[int]): The dimensions of the linear hidden layers.
        outputs_dim (int): The dimension of the output.
        normalization (Optional[str]): The normalization to use. Either 'batch', 'layer', 'instance' or `None`.
        activation (Optional[str]): The activation to use. Either 'relu', 'sigmoid', 'tanh' or `None`.
        task (str): The task to perform. Either 'classification' or 'regression'.
                    The task determined is `softmax` is used for the output layer.
    """

    tasks = [CLASSIFICATION_KEY, REGRESSION_KEY]

    def __init__(
        self,
        conv_hidden_dims: List[int],
        linear_hidden_dims: List[int],
        normalization: str,
        activation: str = "relu",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(ConvNetModel, self).__init__(*args, **kwargs)
        self._conv_hidden_dims = conv_hidden_dims
        self._linear_hidden_dims = linear_hidden_dims

        assert (
            conv_hidden_dims[-1] == linear_hidden_dims[0]
        ), f"The last convolutional layer ({conv_hidden_dims[-1]}) must be equal to the first linear layer ({linear_hidden_dims[0]})."

        norm = Normalization(
            normalization,
            dimension=2,
            norm_kwargs={"affine": True, "num_features": conv_hidden_dims[0]},
        )
        self._layers = nn.ModuleList()
        self._input = nn.Conv2d(
            self._inputs_dim[0], conv_hidden_dims[0], kernel_size=3, stride=1, padding=1
        )
        self._norm_relu = nn.Sequential(norm, nn.ReLU())

        self._hidden_outputs_dims = [conv_hidden_dims[0]]
        self._hidden_types = ["conv"]

        for i in range(1, len(conv_hidden_dims)):
            self._layers.append(
                Conv2dNormActivation(
                    conv_hidden_dims[i - 1],
                    conv_hidden_dims[i],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    normalization=normalization,
                    activation=activation,
                )
            )
            self._hidden_outputs_dims.append(conv_hidden_dims[i])
            self._hidden_types.append("conv")
            self._layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self._layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self._layers.append(nn.Flatten())

        for i in range(len(linear_hidden_dims) - 1):
            self._layers.append(
                LinearNormActivation(
                    linear_hidden_dims[i],
                    linear_hidden_dims[i + 1],
                    normalization=normalization,
                    activation=activation,
                )
            )
            self._hidden_outputs_dims.append(linear_hidden_dims[i + 1])
            self._hidden_types.append("linear")

        self._output = nn.Linear(linear_hidden_dims[-1], self._outputs_dim)
        self._output_activation = OutputActivation(self._task, dim=1)
        self._depth = len(self._hidden_outputs_dims)

    def forward(
        self,
        x: torch.Tensor,
        staged_output: bool = False,
        input_kwargs: Dict[str, Any] = {},
        output_kwargs: Dict[str, Any] = {},
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """This method is used to perform a forward pass through the model.

        The input is expected to be of shape `(batch_size, inputs_dim)`.
        The output is of shape `(batch_size, outputs_dim)`.

        Args:
            x (torch.Tensor): The input to the model.
            staged_output (bool): If True, the output is a tuple of the last layer and the hidden layers.
            input_kwargs (Dict[str, Any]): The kwargs for the input layer.
            output_kwargs (Dict[str, Any]): The kwargs for the output layer.
        """
        layers_outputs = []
        x = self._input(x, **input_kwargs)
        x = self._norm_relu(x)
        if staged_output:
            layers_outputs.append(x)
        for layer in self._layers:
            x = layer(x)
            if staged_output and isinstance(
                layer, (Conv2dNormActivation, LinearNormActivation, ResidualLayer)
            ):
                layers_outputs.append(x)

        x = self.final_layer(x, **output_kwargs)
        if staged_output:
            return x, layers_outputs
        return x

    def final_layer(self, x: torch.Tensor, **output_kwargs: Any) -> torch.Tensor:
        """This function is used to get the final layer output."""
        x = self._output(x, **output_kwargs)
        return self._output_activation(x)

    def add_method_specific_layers(self, method: str, **kwargs: Any) -> None:
        """This method is used to add method specific layers to the model.

        Args:
            method (str): The method to use.
        """
        super().add_method_specific_layers(method, **kwargs)

        if method in ["dun", "mimmo"]:
            self._reshaping_layers = nn.ModuleList()
            available_heads = [True] * (self._depth-1)
            if "heads" in kwargs and kwargs["heads"]:
                self._heads = nn.ModuleList()
            if "available_heads" in kwargs:
                available_heads = kwargs["available_heads"]
            for i, available_head in enumerate(available_heads):
                if not available_head:
                    continue
                layers = []
                if self._hidden_types[i] == "conv":
                    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
                    layers.append(nn.Flatten())
                layers.append(
                    nn.Linear(
                        self._hidden_outputs_dims[i], self._linear_hidden_dims[-1]
                    )
                )
                layers.append(nn.BatchNorm1d(self._linear_hidden_dims[-1]))
                layers.append(nn.ReLU())
                self._reshaping_layers.append(nn.Sequential(*layers))

                if "heads" in kwargs and kwargs["heads"]:
                    head = []
                    head.append(
                        nn.Linear(
                            self._linear_hidden_dims[-1], self._output[0].out_features
                        )
                    )
                    head.append(ReshapeOutput(num_members=kwargs["num_members"]))
                    self._heads.append(nn.Sequential(*head))

        elif method in ["early_exit"]:
            gamma = kwargs["gamma"]
            self._reshaping_layers = nn.ModuleList()
            hidden_feature_size_output = (
                self._output.in_features
                if method == "early_exit"
                else self._output[0].in_features
            )
            size_output = (
                self._output.out_features
                if method == "early_exit"
                else self._output[0].out_features
            )
            heads = [1] * self._depth
            if "heads" in kwargs and kwargs["heads"] is not None:
                heads = kwargs["heads"]
                assert (
                    len(heads) == self._depth
                ), f"Number of heads should be {self._depth}, but got {len(heads)}"
            for i in range(1, self._depth):
                if not heads[i - 1]:
                    continue
                layers = []
                hidden_feature_size = int(
                    math.sqrt(1 + gamma) ** (self._depth - i)
                    * hidden_feature_size_output
                )
                if self._hidden_types[i - 1] == "conv":
                    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
                    layers.append(nn.Flatten())
                if gamma > 0:
                    layers.append(
                        nn.Linear(self._hidden_outputs_dims[i - 1], hidden_feature_size)
                    )
                    layers.append(nn.ReLU())
                    layers.append(nn.Linear(hidden_feature_size, size_output))
                else:
                    layers.append(
                        nn.Linear(self._hidden_outputs_dims[i - 1], size_output)
                    )
                self._reshaping_layers.append(nn.Sequential(*layers))
        else:
            raise ValueError(f"Method {method} is not supported.")

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the model specific arguments to the parent parser."""
        parser = super(ConvNetModel, ConvNetModel).add_specific_args(parent_parser)
        parser.add_argument(
            "--model_conv_hidden_dims",
            type=str,
            default="[128, 128]",
            help="The convolutional hidden dimensions.",
        )
        parser.add_argument(
            "--model_linear_hidden_dims",
            type=str,
            default="[128, 128, 128]",
            help="The linear hidden dimensions.",
        )
        parser.add_argument(
            "--model_normalization",
            type=str,
            default="batch",
            choices=["batch", "layer", "instance", None],
            help="The normalization to use.",
        )
        parser.add_argument(
            "--model_activation",
            type=str,
            default="relu",
            choices=["relu", "sigmoid", "tanh", None],
            help="The activation to use.",
        )
        return parser


class ResidualConvNetModel(ConvNetModel):
    """This class is used to create a residual convolutional network."""

    def __init__(
        self,
        inputs_dim: Tuple[int, ...],
        outputs_dim: int,
        task: str,
        conv_hidden_dims: List[int] = [32, 128],
        linear_hidden_dims: List[int] = [128, 128],
        normalization: str = "batch",
    ) -> None:
        """This method is used to initialize the model.

        Args:
            inputs_dim (Tuple[int, ...]): The input dimensions.
            outputs_dim (int): The output dimensions.
            task (str): The task to perform.
            conv_hidden_dims (List[int]): The convolutional hidden dimensions.
            linear_hidden_dims (List[int]): The linear hidden dimensions.
            normalization (str): The normalization to use.
        """
        super().__init__(
            inputs_dim=inputs_dim,
            outputs_dim=outputs_dim,
            task=task,
            conv_hidden_dims=conv_hidden_dims,
            linear_hidden_dims=linear_hidden_dims,
            normalization=normalization,
        )
        # Wrap the Conv2dNormReLU and LinearReLUNorm layers with ResidualLayers
        for i in range(len(self._layers)):
            if isinstance(
                self._layers[i], (Conv2dNormActivation, LinearNormActivation)
            ):
                self._layers[i] = ResidualLayer(self._layers[i])
