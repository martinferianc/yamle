import argparse
import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.ao.quantization import DeQuantStub, QuantStub, fuse_modules

from yamle.models.operations import OutputActivation, ReshapeOutput, ResidualLayer
from yamle.models.model import BaseModel

from yamle.defaults import REGRESSION_KEY, CLASSIFICATION_KEY


class BasicBlock(nn.Module):
    """
    A basic block for ResNet. It consists of two convolutional layers with batch normalization and ReLU activation.
    Args:
        inplanes (int): The number of input channels.
        planes (int): The number of output channels.
        stride (int): The stride of the first convolutional layer.
        normalization (Type[nn.Module]): The normalization to use.
        normalization_kwargs (Dict[str, Any]): The keyword arguments for the normalization.
        first (bool): Whether this is the first block in the sequence. Defaults to False.
    """

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        normalization: Type[nn.Module] = nn.BatchNorm2d,
        normalization_kwargs: Dict[str, Any] = {},
        first: bool = False,  # Here it is not used
    ) -> None:
        super().__init__()
        conv = nn.Conv2d
        self.downsample: Optional[nn.Module] = None
        if stride != 1:
            self.downsample = nn.Sequential(
                conv(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                normalization(planes, **normalization_kwargs),
            )
        else:
            self.downsample = None
        f = nn.Sequential(
            conv(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            normalization(planes, **normalization_kwargs),
            nn.ReLU(inplace=True),
            conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            normalization(planes, **normalization_kwargs),
        )
        self.f = ResidualLayer(f)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the basic block."""
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.f(x, identity=identity)
        out = self.relu(out)
        return out

    def replace_layers_for_quantization(self) -> None:
        """Fuses all the operations in the block."""
        # First fuse the operations in the residual layer
        # It might be the case that dropout (MCDropout) is inserted in front of the convolution
        # in that case the first layer is not a convolution but nn.Sequential where the first layer is dropout
        # and the second layer is convolution
        if isinstance(self.f._layer[0], nn.Sequential) and isinstance(
            self.f._layer[0][1], nn.Conv2d
        ):
            fuse_modules(self.f._layer, [["0.1", "1", "2"]], inplace=True)
        else:
            fuse_modules(self.f._layer, [["0", "1", "2"]], inplace=True)

        if isinstance(self.f._layer[3], nn.Sequential) and isinstance(
            self.f._layer[3][1], nn.Conv2d
        ):
            fuse_modules(self.f._layer, [["3.1", "4"]], inplace=True)
        else:
            fuse_modules(self.f._layer, [["3", "4"]], inplace=True)
        # If downsample is not None, fuse the operations in the downsample layer
        if self.downsample is not None:
            if isinstance(self.downsample[0], nn.Sequential) and isinstance(
                self.downsample[0][1], nn.Conv2d
            ):
                fuse_modules(self.downsample, [["0.1", "1"]], inplace=True)
            else:
                fuse_modules(self.downsample, [["0", "1"]], inplace=True)


class Bottleneck(nn.Module):
    """
    A bottleneck block for ResNet. It consists of 3 convolutional layers with batch normalization and ReLU activation.
    It is used to construct wide ResNet.

    Args:
        inplanes (int): The number of input channels.
        planes (int): The number of output channels.
        stride (int): The stride of the first convolutional layer.
        normalization (Type[nn.Module]): The normalization to use.
        normalization_kwargs (Dict[str, Any]): The keyword arguments for the normalization.
        first (bool): Whether this is the first block in the sequence. Defaults to False.
    """

    base_width_multiplier = 2
    groups = 1
    expansion = 4
    base_width = 32

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        normalization: Type[nn.Module] = nn.BatchNorm2d,
        normalization_kwargs: Dict[str, Any] = {},
        first: bool = False,
    ) -> None:
        super().__init__()
        conv = nn.Conv2d
        self.downsample: Optional[nn.Module] = None
        if stride != 1 or first:
            self.downsample = nn.Sequential(
                conv(
                    inplanes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                normalization(planes * self.expansion, **normalization_kwargs),
            )
        else:
            self.downsample = None
        width = (
            int(
                planes
                * ((self.base_width_multiplier * self.base_width) / self.base_width)
            )
            * self.groups
        )
        f = nn.Sequential(
            conv(inplanes, width, kernel_size=1, stride=1, bias=False, padding=0),
            normalization(width, **normalization_kwargs),
            nn.ReLU(inplace=True),
            conv(
                width,
                width,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                groups=self.groups,
            ),
            normalization(width, **normalization_kwargs),
            nn.ReLU(inplace=True),
            conv(
                width,
                planes * self.expansion,
                kernel_size=1,
                stride=1,
                bias=False,
                padding=0,
            ),
            normalization(planes * self.expansion, **normalization_kwargs),
        )
        self.f = ResidualLayer(f)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the bottleneck block."""
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.f(x, identity=identity)
        out = self.relu(out)
        return out

    def replace_layers_for_quantization(self) -> None:
        """Fuses all the operations in the block."""
        # First fuse the operations in the residual layer
        # It might be the case that dropout (MCDropout) is inserted in front of the convolution
        # in that case the first layer is not a convolution but nn.Sequential where the first layer is dropout
        # and the second layer is convolution
        if isinstance(self.f._layer[0], nn.Sequential) and isinstance(
            self.f._layer[0][1], nn.Conv2d
        ):
            fuse_modules(self.f._layer, [["0.1", "1", "2"]], inplace=True)

        else:
            fuse_modules(self.f._layer, [["0", "1", "2"]], inplace=True)

        if isinstance(self.f._layer[3], nn.Sequential) and isinstance(
            self.f._layer[3][1], nn.Conv2d
        ):
            fuse_modules(self.f._layer, [["3.1", "4", "5"]], inplace=True)
        else:
            fuse_modules(self.f._layer, [["3", "4", "5"]], inplace=True)

        if isinstance(self.f._layer[6], nn.Sequential) and isinstance(
            self.f._layer[6][1], nn.Conv2d
        ):
            fuse_modules(self.f._layer, [["6.1", "7"]], inplace=True)
        else:
            fuse_modules(self.f._layer, [["6", "7"]], inplace=True)

        # If downsample is not None, fuse the operations in the downsample layer
        if self.downsample is not None:
            if isinstance(self.downsample[0], nn.Sequential) and isinstance(
                self.downsample[0][1], nn.Conv2d
            ):
                fuse_modules(self.downsample, [["0.1", "1"]], inplace=True)
            else:
                fuse_modules(self.downsample, [["0", "1"]], inplace=True)


class Blocks(nn.Module):
    """
    A class implementing a stack of basic blocks.
    Args:
        inplanes (int): The number of input channels.
        planes (int): The number of output channels.
        blocks (int): The number of basic blocks.
        stride (int): The stride of the first convolutional layer.
        normalization (Type[nn.Module]): The normalization to use.
        normalization_kwargs (Dict[str, Any]): The keyword arguments for the normalization.
        block (Union[Type[BasicBlock], Type[Bottleneck]]): The block to use. Defaults to BasicBlock.
    """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int,
        normalization: Type[nn.Module],
        normalization_kwargs: Dict[str, Any] = {},
        block: Union[Type[BasicBlock], Type[Bottleneck]] = BasicBlock,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(blocks):
            self.layers.append(
                block(
                    inplanes=inplanes if i == 0 else planes * block.expansion,
                    planes=planes,
                    stride=stride if i == 0 else 1,
                    normalization=normalization,
                    normalization_kwargs=normalization_kwargs,
                    first=i == 0,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the stack of basic blocks."""
        for layer in self.layers:
            x = layer(x)
        return x


class ResNetModel(BaseModel):
    """
    This class implements the ResNet architecture as described in the paper:
    Deep Residual Learning for Image Recognition, published at the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2016.
    and the paper can be found at: https://arxiv.org/abs/1512.03385.

    The code is based on the implementation of the model is sourced from: https://pytorch.org/hub/pytorch_vision_resnet/

    Args:
        layers (List[int]): The number of basic blocks in each stack.
        depth (int): The depth of the network respective to the the length of the layers list.
        planes (List[int]): The number of output channels in each stack.
        width_multiplier (int): The width multiplier for the planes list.
        strides (List[int]): The stride of the first convolutional layer in each stack.
        normalization (Optional[str]): The normalization to use. Can be either 'batch', 'instance', `group`, `layer`, or `None`. Defaults to 'batch'.
        block (str): The block to use. Can be either 'basic' or 'bottleneck'. Defaults to 'basic'.
    """

    tasks = [REGRESSION_KEY, CLASSIFICATION_KEY]

    def __init__(
        self,
        layers: List[int] = [2, 2, 2, 2],
        depth: Optional[int] = None,
        planes: List[int] = [32, 64, 128, 256],
        width_multiplier: int = 1,
        strides: List[int] = [1, 2, 2, 2],
        normalization: Optional[str] = "batch",
        block: str = "basic",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert len(planes) == len(
            layers
        ), f"Planes {planes} and layers {layers} must have the same length."
        assert len(strides) == len(
            layers
        ), f"Strides {strides} and layers {layers} must have the same length."
        assert depth is None or (
            len(planes) >= depth and depth >= 1
        ), f"Depth {depth} must be between 1 and {len(planes)}."
        assert (
            width_multiplier > 0
        ), f"Width multiplier {width_multiplier} must be greater than 0."

        assert block in ["basic", "bottleneck"], f"Block {block} is not supported."
        if block == "basic":
            block = BasicBlock
        elif block == "bottleneck":
            block = Bottleneck
        else:
            raise NotImplementedError(f"Block {block} is not supported.")

        self._layers = layers
        self._planes = planes
        self._strides = strides
        if depth is None:
            self._depth = len(layers)
        else:
            self._depth = depth
        self._width_multiplier = width_multiplier
        self._block = block

        conv = nn.Conv2d
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

        if (
            len(self._inputs_dim) == 3
            and self._inputs_dim[1] <= 64
            and self._inputs_dim[2] <= 64
        ):
            self._input = conv(
                self._inputs_dim[0],
                planes[0] * self._width_multiplier,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        else:
            self._input = conv(
                self._inputs_dim[0],
                planes[0] * self._width_multiplier,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        self._norm = norm(planes[0] * self._width_multiplier, **norm_kwargs)
        self._relu = nn.ReLU(inplace=False)
        self._maxpool = (
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            if (self._inputs_dim[1] > 64 or self._inputs_dim[2] > 64)
            else nn.Identity()
        )
        self._layers = nn.ModuleList()

        for i in range(self._depth):
            self._layers.append(
                Blocks(
                    inplanes=planes[0] * self._width_multiplier
                    if i == 0
                    else planes[i - 1] * block.expansion * self._width_multiplier,
                    planes=planes[i] * self._width_multiplier,
                    blocks=layers[i],
                    stride=strides[i],
                    normalization=norm,
                    normalization_kwargs=norm_kwargs,
                    block=block,
                )
            )
        self._global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._flatten = nn.Flatten()

        self._output = nn.Linear(
            planes[self._depth - 1] * block.expansion * self._width_multiplier,
            self._outputs_dim,
        )
        self._output_activation = OutputActivation(self._task, dim=1)

        self._added_method_specific_layers = False

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
        x = self._maxpool(x)
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
                        self._planes[i]
                        * self._block.expansion
                        * self._width_multiplier,
                        self._planes[self._depth - 1]
                        * self._block.expansion
                        * self._width_multiplier,
                        1,
                    )
                )
                layers.append(
                    self._normalization(
                        self._planes[self._depth - 1]
                        * self._block.expansion
                        * self._width_multiplier
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
                            self._planes[self._depth - 1]
                            * self._block.expansion
                            * self._width_multiplier,
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
                            self._planes[i - 1]
                            * self._block.expansion
                            * self._width_multiplier,
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
                            self._planes[i - 1]
                            * self._block.expansion
                            * self._width_multiplier,
                            size_output,
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
        parser = super(ResNetModel, ResNetModel).add_specific_args(parent_parser)
        parser.add_argument(
            "--model_layers",
            type=str,
            default="[2, 2, 2, 2]",
            help="The number of basic blocks in each stack.",
        )
        parser.add_argument(
            "--model_depth", type=int, default=None, help="The number of stacks."
        )
        parser.add_argument(
            "--model_planes",
            type=str,
            default="[32, 64, 128, 256]",
            help="The number of output channels in each stack.",
        )
        parser.add_argument(
            "--model_width_multiplier",
            type=int,
            default=1,
            help="The width multiplier to use.",
        )
        parser.add_argument(
            "--model_strides",
            type=str,
            default="[1, 2, 2, 2]",
            help="The stride of the first convolutional layer in each stack.",
        )
        parser.add_argument(
            "--model_normalization",
            type=str,
            choices=["batch", "instance", "group", "layer", None],
            default="batch",
            help="The normalization to use.",
        )
        parser.add_argument(
            "--model_block",
            type=str,
            choices=["basic", "bottleneck"],
            default="basic",
            help="The block to use.",
        )
        return parser
