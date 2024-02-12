import argparse
import math
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.ao.quantization import DeQuantStub, QuantStub, fuse_modules


from yamle.models.model import BaseModel
from yamle.models.operations import (
    LinearNormActivation,
    Normalization,
    OutputActivation,
    ReshapeOutput,
    ResidualLayer,
)

from yamle.defaults import REGRESSION_KEY, CLASSIFICATION_KEY


class FCModel(BaseModel):
    """This class is used to create a FC model with the given parameters.

    Args:
        hidden_dim (int): The dimensions of the hidden layers.
        width_multiplier (int): The width multiplier for the hidden layers. Default: 1.
        depth (int): The number of hidden layers.
        normalization (Optional[str]): The normalization to use. Either 'batch', 'linear', 'instance' or `None`.
        activation (Optional[str]): The activation to use. Either 'relu', 'linear' or `None`.
    """

    tasks = [CLASSIFICATION_KEY, REGRESSION_KEY]

    def __init__(
        self,
        hidden_dim: int,
        width_multiplier: int,
        depth: int,
        normalization: str,
        activation: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(FCModel, self).__init__(*args, **kwargs)
        self._inputs_dim = np.prod(self._inputs_dim)
        self._hidden_dim = hidden_dim * width_multiplier

        self._flatten = nn.Flatten()
        self._layers = nn.ModuleList()
        self._input = nn.Linear(self._inputs_dim, self._hidden_dim)
        self._relu = nn.ReLU()
        for i in range(depth):
            self._layers.append(
                LinearNormActivation(
                    self._hidden_dim,
                    self._hidden_dim,
                    normalization=normalization,
                    activation=activation,
                )
            )
        self._normalization = normalization
        self._output = nn.Linear(self._hidden_dim, self._outputs_dim)
        self._output_activation = OutputActivation(self._task, dim=1)
        self._depth = depth

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
        x = self._flatten(x)
        x = self._input(x, **input_kwargs)
        x = self._relu(x)
        for layer in self._layers:
            x = layer(x)
            if staged_output:
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
        norm_kwargs = {"affine": True}
        if method in ["dun", "mimmo"]:
            self._reshaping_layers = nn.ModuleList()
            available_heads = [True] * (self._depth - 1)
            if "heads" in kwargs and kwargs["heads"]:
                self._heads = nn.ModuleList()
            if "available_heads" in kwargs:
                available_heads = kwargs["available_heads"]
            for i, available_head in enumerate(available_heads):
                if not available_head:
                    continue
                layers = []
                layers.append(nn.Linear(self._hidden_dim, self._hidden_dim))
                layers.append(
                    Normalization(
                        norm=self._normalization,
                        dimension=1,
                        norm_kwargs={**norm_kwargs, "num_features": self._hidden_dim},
                    )
                )
                layers.append(nn.ReLU())
                self._reshaping_layers.append(nn.Sequential(*layers))

                if "heads" in kwargs and kwargs["heads"]:
                    head = []
                    head.append(
                        nn.Linear(self._hidden_dim, self._output[0].out_features)
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
                ), "Number of heads should be equal to number of layers."
            for i in range(1, self._depth):
                if not heads[i - 1]:
                    continue
                sequence = []
                hidden_feature_size = int(
                    math.sqrt(1 + gamma) ** (self._depth - i)
                    * hidden_feature_size_output
                )
                if gamma > 0:
                    sequence.append(nn.Linear(self._hidden_dim, hidden_feature_size))
                    sequence.append(nn.ReLU())
                    sequence.append(nn.Linear(hidden_feature_size, size_output))
                else:
                    sequence.append(nn.Linear(self._hidden_dim, size_output))
                self._reshaping_layers.append(nn.Sequential(*sequence))
        else:
            raise ValueError(f"Method {method} is not supported.")

    def replace_layers_for_quantization(self) -> None:
        """Fuses all the operations in the network.

        In this function we only need to fuse layers that are not in the blocks.
        e.g. the reshaping layers added by the method.
        """
        if self._added_method_specific_layers:
            if self._method in ["dun", "mimmo"]:
                for i in range(len(self._reshaping_layers)):
                    self._reshaping_layers[i] = fuse_modules(
                        self._reshaping_layers[i], [["0", "1._norm"]]
                    )

                if "heads" in self._method_kwargs and self._method_kwargs["heads"]:
                    for i in range(len(self._heads)):
                        self._heads[i] = nn.Sequential(self._heads[i], DeQuantStub())

            elif self._method in ["early_exit"]:
                for i in range(1, self._depth):
                    if not self._method_kwargs["heads"][i - 1]:
                        continue
                    if self._method_kwargs["gamma"] > 0:
                        self._reshaping_layers[i - 1] = fuse_modules(
                            self._reshaping_layers[i - 1], [["2", "3"]]
                        )
                    self._reshaping_layers[i - 1] = nn.Sequential(
                        self._reshaping_layers[i - 1], DeQuantStub()
                    )

        # Add quantization stubs to the input and dequantization stubs to the output.
        self._input = nn.Sequential(QuantStub(), self._input)
        self._output = nn.Sequential(self._output, DeQuantStub())

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the model specific arguments to the parent parser."""
        parser = super(FCModel, FCModel).add_specific_args(parent_parser)
        parser.add_argument(
            "--model_hidden_dim",
            type=int,
            default=128,
            help="The dimensions of the hidden layers.",
        )
        parser.add_argument(
            "--model_width_multiplier",
            type=int,
            default=1,
            help="The width multiplier for the hidden layers.",
        )
        parser.add_argument(
            "--model_depth", type=int, default=3, help="The number of hidden layers."
        )
        parser.add_argument(
            "--model_normalization",
            type=str,
            default=None,
            choices=["batch", "instance", None],
            help="The normalization to use.",
        )
        parser.add_argument(
            "--model_activation",
            type=str,
            default="relu",
            choices=["relu", "linear", "sigmoid", "tanh", None],
            help="The activation to use.",
        )
        return parser


class ResidualFCModel(FCModel):
    """This class is used to create a residual FC model.

    Each hidden layer can be understood as a residual block. The input is added to the output of the hidden layer.
    All the hidden layers are followed by a ReLU activation and have the same dimension.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(ResidualFCModel, self).__init__(*args, **kwargs)
        # Replace the linear layers with residual layers except the first and last layer
        for i in range(len(self._layers)):
            if isinstance(self._layers[i], LinearNormActivation):
                self._layers[i] = ResidualLayer(self._layers[i])
