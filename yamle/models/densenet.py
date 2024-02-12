import argparse
import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.ao.quantization import DeQuantStub, QuantStub, fuse_modules

from yamle.models.operations import OutputActivation, ReshapeOutput
from yamle.models.model import BaseModel

from yamle.defaults import REGRESSION_KEY, CLASSIFICATION_KEY


class EmptyBlock(nn.Module):
    """This class defines an empty block that does nothing, but signals where to cache the hidden states."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """This function applies the forward pass.

        Args:
            inputs (torch.Tensor): Input tensor.
        """
        return inputs


class DenseLayer(nn.Module):
    """This class defines the primitive of the DenseNet architecture.

    Args:
        inplanes (int): Number of input features.
        growth_rate (int): Number of output features.
        normalization (Type[nn.Module]): The normalization to use.
        normalization_kwargs (Dict[str, Any]): The keyword arguments for the normalization.
        bn_size (int): Bottleneck size.
        dropout_rate (float): Dropout rate.
    """

    def __init__(
        self,
        inplanes: int,
        growth_rate: int,
        normalization: Type[nn.Module],
        normalization_kwargs: Dict[str, Any],
        bn_size: int,
        dropout_rate: float,
    ) -> None:
        super(DenseLayer, self).__init__()
        self.norm1 = normalization(inplanes, **normalization_kwargs)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            inplanes, bn_size * growth_rate, kernel_size=1, stride=1, bias=False
        )
        self.norm2 = normalization(bn_size * growth_rate, **normalization_kwargs)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.dropout = nn.Dropout(p=dropout_rate)

    def bn_function(self, inputs: torch.Tensor) -> torch.Tensor:
        """This function applies bottleneck function.

        Args:
            inputs (torch.Tensor): Input tensor.
        """
        concatenated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concatenated_features)))
        return bottleneck_output

    def forward(self, inputs: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """This function applies the forward pass.

        Args:
            inputs (torch.Tensor): Input tensor.
        """
        if isinstance(inputs, torch.Tensor):
            prev_features = [inputs]
        else:
            prev_features = inputs

        new_features = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        new_features = self.dropout(new_features)
        return new_features

    def replace_layers_for_quantization(self) -> None:
        """This function replaces the layers for quantization."""
        # Unfortunately the batch normalisation is the first layer of the block
        # which means that we cannot fuse it with the convolutional layer
        pass


class Transition(nn.Module):
    """This class defines the transition layer of the DenseNet architecture.

    Args:
        inplanes (int): Number of input features.
        outplanes (int): Number of output features.
        normalization (Type[nn.Module]): The normalization to use.
        normalization_kwargs (Dict[str, Any]): The keyword arguments for the normalization.
    """

    def __init__(
        self,
        inplanes: int,
        outplanes: int,
        normalization: Type[nn.Module],
        normalization_kwargs: Dict[str, Any],
    ) -> None:
        super(Transition, self).__init__()
        self.norm = normalization(inplanes, **normalization_kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """This function applies the forward pass.

        Args:
            inputs (torch.Tensor): Input tensor.
        """
        out = self.conv(self.relu(self.norm(inputs)))
        out = self.avgpool(out)
        return out


class DenseBlock(nn.Module):
    """This class defines the DenseBlock of the DenseNet architecture.

    Args:
        inplanes (int): Number of input features.
        growth_rate (int): Number of output features.
        normalization (Type[nn.Module]): The normalization to use.
        normalization_kwargs (Dict[str, Any]): The keyword arguments for the normalization.
        bn_size (int): Bottleneck size.
        dropout_rate (float): Dropout rate.
        n_layers (int): Number of layers.
    """

    def __init__(
        self,
        inplanes: int,
        growth_rate: int,
        normalization: Type[nn.Module],
        normalization_kwargs: Dict[str, Any],
        bn_size: int,
        dropout_rate: float,
        n_layers: int,
    ) -> None:
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = DenseLayer(
                inplanes + i * growth_rate,
                growth_rate,
                normalization,
                normalization_kwargs,
                bn_size,
                dropout_rate,
            )
            self.layers.append(layer)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """This function applies the forward pass.

        Args:
            inputs (torch.Tensor): Input tensor.
        """
        features = [inputs]
        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, dim=1)


class DenseNetModel(BaseModel):
    """This class defines the DenseNet architecture as described in the paper.
    Densely Connected Convolutional Networks: https://arxiv.org/abs/1608.06993

    The code is based on the implementation of torchvision: https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py

    Args:
        layers (List[int]): Number of layers in each block.
        depth (int): The depth of the network respective to the the length of the layers list.
        bn_size (int): Bottleneck size.
        growth_rate (int): The growth rate multiplier.
        initial_planes (int): Number of initial planes.
        width_multiplier (float): Width multiplier to multiply the initial number of features.
        normalization (Optional[str]): The normalization to use. Can be either 'batch', 'instance', `group`, `layer`, or `None`. Defaults to 'batch'.
        dropout_rate (float): Dropout rate.
    """

    tasks = [
        CLASSIFICATION_KEY,
        REGRESSION_KEY,
    ]

    def __init__(
        self,
        layers: List[int] = [6, 12, 24, 16],
        depth: int = 4,
        bn_size: int = 4,
        growth_rate: int = 32,
        initial_planes: int = 64,
        width_multiplier: float = 1.0,
        normalization: Optional[str] = "batch",
        dropout_rate: float = 0.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(DenseNetModel, self).__init__(*args, **kwargs)
        assert depth is None or (
            len(layers) >= depth and depth >= 1
        ), f"Depth must be between 1 and {len(layers)} but got {depth}"

        self._layers = layers
        self._initial_planes = initial_planes * width_multiplier
        self._width_multiplier = width_multiplier
        self._dropout_rate = dropout_rate
        if depth is None:
            self._depth = len(layers)
        else:
            self._depth = depth

        norm = nn.Identity
        assert normalization in [
            "batch",
            "instance",
            "group",
            "layer",
            None,
        ], f"Normalization {normalization} is not supported."
        norm_kwargs = {}
        if normalization == "batch":
            norm = nn.BatchNorm2d
        elif normalization == "instance":
            norm = nn.InstanceNorm2d
            norm_kwargs = {"affine": True}
        elif normalization == "group":
            norm = nn.GroupNorm
        elif normalization == "layer":
            norm = nn.LayerNorm
        self._normalization = norm
        self._norm_kwargs = norm_kwargs

        if (
            len(self._inputs_dim) == 3
            and self._inputs_dim[1] <= 64
            and self._inputs_dim[2] <= 64
        ):
            self._input = nn.Conv2d(
                self._inputs_dim[0],
                self._initial_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        else:
            self._input = nn.Conv2d(
                self._inputs_dim[0],
                self._initial_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        self._layers = nn.ModuleList()
        self._layers.append(norm(self._initial_planes, **norm_kwargs))
        self._layers.append(nn.ReLU(inplace=True))
        if self._inputs_dim[1] > 64 or self._inputs_dim[2] > 64:
            self._layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        num_features = self._initial_planes
        self._planes = []  # Will collet the number of output planes for each block
        for i in range(self._depth):
            self._layers.append(
                DenseBlock(
                    inplanes=num_features,
                    n_layers=layers[i],
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    dropout_rate=dropout_rate,
                    normalization=norm,
                    normalization_kwargs=norm_kwargs,
                )
            )
            num_features += layers[i] * growth_rate
            if i != self._depth-1:
                self._layers.append(
                    Transition(
                        inplanes=num_features,
                        outplanes=num_features // 2,
                        normalization=norm,
                        normalization_kwargs=norm_kwargs,
                    )
                )
                num_features = num_features // 2
            self._layers.append(EmptyBlock())
            self._planes.append(num_features)

        self._layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self._layers.append(nn.Flatten())
        self._output = nn.Linear(self._planes[self._depth-1], self._outputs_dim)
        self._output_activation = OutputActivation(self._task, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        staged_output: bool = False,
        input_kwargs: Dict[str, Any] = {},
        output_kwargs: Dict[str, Any] = {},
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """The forward function of the ResNet model.

        Args:
            x (torch.Tensor): The input tensor.
            staged_output (bool): Whether to return the output of each layer. Defaults to False.
            input_kwargs (Dict[str, Any]): The kwargs for the input layer.
            output_kwargs (Dict[str, Any]): The kwargs for the output layer.
        """
        layers_outputs = []
        x = self._input(x, **input_kwargs)
        for i in range(len(self._layers)):
            x = self._layers[i](x)
            # Make sure to cache the output of the transition layer
            # If transition layer
            if isinstance(self._layers[i], EmptyBlock):
                layers_outputs.append(x)

        if isinstance(self._layers[-1], nn.Flatten) and staged_output:
            layers_outputs[-1] = x

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
                layers.append(nn.Conv2d(self._planes[i], self._planes[self._depth-1], 1))
                layers.append(self._normalization(self._planes[self._depth-1]))
                layers.append(nn.ReLU())
                layers.append(nn.AdaptiveAvgPool2d(1))
                layers.append(nn.Flatten())
                self._reshaping_layers.append(nn.Sequential(*layers))

                if "heads" in kwargs and kwargs["heads"]:
                    head = []
                    head.append(
                        nn.Linear(
                            self._planes[self._depth-1], self._output[0].out_features
                        )
                    )
                    head.append(ReshapeOutput(num_members=kwargs["num_members"]))
                    self._heads.append(nn.Sequential(*head))

        elif method in ["early_exit"]:
            gamma = kwargs["gamma"]
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
            self._reshaping_layers = nn.ModuleList()
            heads = [1] * self._depth
            if "heads" in kwargs and kwargs["heads"] is not None:
                heads = kwargs["heads"]
                assert (
                    len(heads) == self._depth
                ), f"Number of heads should be {self._depth}, but got {len(heads)}"
            kwargs["heads"] = heads
            for i in range(1, self._depth):
                if not heads[i - 1]:
                    continue
                sequence = []
                hidden_feature_size = int(
                    math.sqrt(1 + gamma) ** (self._depth - i)
                    * hidden_feature_size_output
                )
                if gamma > 0:
                    sequence.append(nn.AdaptiveAvgPool2d(1))
                    sequence.append(nn.Flatten())
                    sequence.append(nn.Linear(self._planes[i - 1], hidden_feature_size))
                    sequence.append(nn.ReLU())
                    sequence.append(nn.Linear(hidden_feature_size, size_output))
                else:
                    sequence.append(nn.AdaptiveAvgPool2d(1))
                    sequence.append(nn.Flatten())
                    sequence.append(nn.Linear(self._planes[i - 1], size_output))
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
                        self._reshaping_layers[i], [["0", "1", "2"]], inplace=True
                    )

                if "heads" in self._method_kwargs and self._method_kwargs["heads"]:
                    for i in range(len(self._heads)):
                        self._heads[i] = nn.Sequential(self._heads[i], DeQuantStub())

                if self._method == "dun":
                    fuse_modules(
                        self, [["_input", "_layers.0", "_layers.1"]], inplace=True
                    )
                else:
                    fuse_modules(
                        self, [["_input.1", "_layers.0", "_layers.1"]], inplace=True
                    )

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
                fuse_modules(self, [["_input", "_layers.0", "_layers.1"]], inplace=True)
        else:
            # Find the input convolution, could be either _input or _input.X if the input is
            # in a sequential block.
            input_conv = "_input"
            if isinstance(self._input, (nn.Sequential, nn.ModuleList)):
                for i, layer in enumerate(self._input):
                    if isinstance(layer, nn.Conv2d):
                        input_conv = f"_input.{i}"
                        break
            fuse_modules(self, [[input_conv, "_layers.0", "_layers.1"]], inplace=True)

        # Add quantization stubs to the input and dequantization stubs to the output.
        self._input = nn.Sequential(QuantStub(), self._input)
        self._output = nn.Sequential(self._output, DeQuantStub())

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Add specific arguments to the parser."""
        parser = super(DenseNetModel, DenseNetModel).add_specific_args(parent_parser)
        parser.add_argument(
            "--model_layers",
            type=str,
            default="[6,12,24,16]",
            help="Number of layers in each block.",
        )
        parser.add_argument(
            "--model_depth",
            type=int,
            default=None,
            help="The depth of the network respective to the the length of the layers list.",
        )
        parser.add_argument(
            "--model_bn_size", type=int, default=4, help="Bottleneck size."
        )
        parser.add_argument(
            "--model_initial_planes",
            type=int,
            default=64,
            help="Number of initial planes.",
        )
        parser.add_argument(
            "--model_growth_rate",
            type=int,
            default=32,
            help="Number of output features.",
        )
        parser.add_argument(
            "--model_width_multiplier",
            type=float,
            default=1,
            help="Width multiplier that multiplies the initial number of features.",
        )
        parser.add_argument(
            "--model_normalization",
            type=str,
            default="batch",
            help="The normalization to use. Can be either 'batch', 'instance', `group`, `layer`, or `None`.",
        )
        parser.add_argument(
            "--model_dropout_rate", type=float, default=0.0, help="Dropout rate."
        )
        return parser
