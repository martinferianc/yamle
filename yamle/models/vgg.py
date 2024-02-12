import argparse
import copy
import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.ao.quantization import DeQuantStub, QuantStub, fuse_modules

from yamle.models.operations import OutputActivation, ReshapeOutput
from yamle.models.model import BaseModel

from yamle.defaults import REGRESSION_KEY, CLASSIFICATION_KEY


class EmptyBlock(nn.Module):
    """This is just an empty block that does nothing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This function does nothing."""
        return x


class VGGBlock(nn.Module):
    """This class implements a VGG block.

    Args:
        inplanes (int): The number of input planes.
        planes (int): The number of output planes.
        pooling (bool): Whether to use pooling after the block.
        normalization (Type[nn.Module]): The normalization to use.
        normalization_kwargs (Dict[str, Any]): The kwargs for the normalization.
    """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        pooling: bool,
        normalization: Optional[Type[nn.Module]] = None,
        normalization_kwargs: Optional[Dict[str, Any]] = {},
    ) -> None:
        super().__init__()
        self._pooling = pooling
        self._normalization = normalization
        self._normalization_kwargs = normalization_kwargs
        layers = []

        layers.append(
            nn.Conv2d(
                inplanes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False if self._normalization is not None else True,
            )
        )
        if self._normalization is not None:
            layers.append(self._normalization(planes, **self._normalization_kwargs))

        layers.append(nn.ReLU(inplace=True))

        if self._pooling:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.f = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the block."""
        return self.f(x)

    def replace_layers_for_quantization(self) -> None:
        """Fuses all the operations in the block."""
        # First fuse the operations in the residual layer
        # It might be the case that dropout (MCDropout) is inserted in front of the convolution
        # in that case the first layer is not a convolution but nn.Sequential where the first layer is dropout
        # and the second layer is convolution
        if isinstance(self.f[0], nn.Sequential) and isinstance(self.f[0][1], nn.Conv2d):
            if self._normalization is not None:
                fuse_modules(self.f, [["0.1", "1", "2"]], inplace=True)
            else:
                fuse_modules(self.f, [["0.1", "1"]], inplace=True)
        else:
            if self._normalization is not None:
                fuse_modules(self.f, [["0", "1", "2"]], inplace=True)
            else:
                fuse_modules(self.f, [["0", "1"]], inplace=True)


class Blocks(nn.Module):
    """A class implementing a stack of VGG blocks.

    Args:
        inplanes (int): The number of input planes.
        planes (int): The number of output planes.
        blocks (int): The number of blocks.
        pooling (bool): Whether to use pooling after the block.
        normalization (Type[nn.Module]): The normalization to use.
        normalization_kwargs (Dict[str, Any]): The kwargs for the normalization.
    """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        blocks: int,
        pooling: bool,
        normalization: Optional[Type[nn.Module]] = None,
        normalization_kwargs: Optional[Dict[str, Any]] = {},
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(blocks):
            self.layers.append(
                VGGBlock(
                    inplanes=inplanes if i == 0 else planes,
                    planes=planes,
                    pooling=pooling if i == blocks - 1 else False,
                    normalization=normalization,
                    normalization_kwargs=normalization_kwargs,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Iterate through the blocks."""
        for block in self.layers:
            x = block(x)
        return x


class VGGModel(BaseModel):
    """
    This class implements the VGG architecture as described in the paper:
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    and the paper can be found at: https://arxiv.org/abs/1409.1556.

    Args:
        layers (List[int]): The number of layers in the VGG architecture per block.
        depth (int): The depth of the network respective to the the length of the layers list.
        planes (List[int]): The number of planes in each block of layers.
        width_multiplier (int): The width multiplier for the planes list.
        pooling (List[bool]): Whether to use pooling after each block of layers.
        normalization (Optional[str]): The normalization to use. Can be either 'batch', 'instance', 'group', 'layer', or `None`. Defaults to 'batch'.
    """

    tasks = [REGRESSION_KEY, CLASSIFICATION_KEY]

    def __init__(
        self,
        layers: List[int] = [2, 2, 3, 3, 3],
        depth: Optional[int] = None,
        planes: List[int] = [64, 128, 256, 512, 512],
        width_multiplier: int = 1,
        pooling: List[bool] = [False, True, True, True, True],
        normalization: Optional[str] = "batch",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert len(planes) == len(
            layers
        ), f"Planes {planes} and layers {layers} must have the same length."
        assert len(pooling) == len(
            layers
        ), f"Pooling {pooling} and layers {layers} must have the same length."
        assert depth is None or (
            len(planes) >= depth and depth >= 1
        ), f"Depth {depth} must be between 1 and {len(planes)}."
        assert (
            width_multiplier > 0
        ), f"Width multiplier must be greater than 0. Got {width_multiplier}."

        self._num_layers = copy.deepcopy(layers)
        self._num_layers[0] -= 1  # The stem already counts as a layer
        self._planes = planes
        self._pooling = pooling
        if depth is None:
            self._depth = len(layers)
        else:
            self._depth = depth
        self._width_multiplier = width_multiplier

        norm_kwargs = {}
        norm = nn.Identity
        assert normalization in [
            "batch",
            "instance",
            "group",
            "layer",
            None,
        ], f"Normalization {normalization} is not supported."
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

        self._input = nn.Conv2d(
            self._inputs_dim[0],
            planes[0] * self._width_multiplier,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self._norm = norm(planes[0] * self._width_multiplier, **norm_kwargs)
        self._relu = nn.ReLU(inplace=True)
        self._max_pooling = (
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            if pooling[0] and self._num_layers[0] == 0
            else nn.Identity()
        )
        self._layers = nn.ModuleList()

        if self._num_layers[0] == 0:
            self._layers.append(EmptyBlock())

        for i in range(self._depth):
            if self._num_layers[i] > 0:
                self._layers.append(
                    Blocks(
                        inplanes=planes[i] * self._width_multiplier
                        if i == 0
                        else planes[i - 1] * self._width_multiplier,
                        planes=planes[i] * self._width_multiplier,
                        blocks=layers[i],
                        pooling=self._pooling[i],
                        normalization=norm,
                        normalization_kwargs=norm_kwargs,
                    )
                )

        self._global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._flatten = nn.Flatten()

        self._output = nn.Linear(
            planes[self._depth - 1] * self._width_multiplier, self._outputs_dim
        )
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
        x = self._norm(x)
        x = self._relu(x)
        x = self._max_pooling(x)
        for i in range(len(self._layers)):
            x = self._layers[i](x)
            if staged_output:
                layers_outputs.append(x)
        x = self._global_avgpool(x)
        x = self._flatten(x)
        if staged_output:
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
            available_heads = [True] * (self._depth - 1)
            if "heads" in kwargs and kwargs["heads"]:
                self._heads = nn.ModuleList()
            if "available_heads" in kwargs:
                available_heads = kwargs["available_heads"]
            for i, available_head in enumerate(available_heads):
                if not available_head:
                    continue
                layers = []
                layers.append(
                    nn.Conv2d(
                        self._planes[i] * self._width_multiplier,
                        self._planes[self._depth - 1] * self._width_multiplier,
                        1,
                    )
                )
                layers.append(
                    self._normalization(
                        self._planes[self._depth - 1] * self._width_multiplier,
                        **self._norm_kwargs,
                    )
                )
                layers.append(nn.ReLU())
                layers.append(nn.AdaptiveAvgPool2d(1))
                layers.append(nn.Flatten())
                self._reshaping_layers.append(nn.Sequential(*layers))

                if "heads" in kwargs and kwargs["heads"]:
                    head = []
                    head.append(
                        nn.Linear(
                            self._planes[self._depth - 1] * self._width_multiplier,
                            self._output[0].out_features,
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
            heads = [1] * (self._depth)
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
                    sequence.append(
                        nn.Linear(
                            self._planes[i - 1] * self._width_multiplier,
                            hidden_feature_size,
                        )
                    )
                    sequence.append(nn.ReLU())
                    sequence.append(nn.Linear(hidden_feature_size, size_output))
                else:
                    sequence.append(nn.AdaptiveAvgPool2d(1))
                    sequence.append(nn.Flatten())
                    sequence.append(
                        nn.Linear(
                            self._planes[i - 1] * self._width_multiplier, size_output
                        )
                    )
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
        parser = super(VGGModel, VGGModel).add_specific_args(parent_parser)
        parser.add_argument(
            "--model_layers",
            type=str,
            default="[1, 1, 2, 2, 2]",
            help="The number of layers in the model.",
        )
        parser.add_argument(
            "--model_depth",
            type=int,
            default=None,
            help="The depth of the model relative to the layers list.",
        )
        parser.add_argument(
            "--model_planes",
            type=str,
            default="[16, 32, 64, 128, 128]",
            help="The number of planes in the respective blocks.",
        )
        parser.add_argument(
            "--model_width_multiplier",
            type=int,
            default=1,
            help="The width multiplier for the model relative to the planes list.",
        )
        parser.add_argument(
            "--model_pooling",
            type=str,
            default="[1, 1, 1, 1, 0]",
            help="The pooling in the respective blocks.",
        )
        parser.add_argument(
            "--model_normalization",
            type=str,
            default="batch",
            choices=["batch", "group", "instance", "layer", "none"],
            help="The normalization to use in the model.",
        )
        return parser
