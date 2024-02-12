from typing import Union, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math

from yamle.models.operations import (
    DoubleConv2d,
    OutputActivation,
    Normalization,
    ReshapeOutput,
)
from yamle.models.model import BaseModel

from yamle.defaults import DEPTH_ESTIMATION_KEY, RECONSTRUCTION_KEY, SEGMENTATION_KEY


class DownBlock(nn.Module):
    """This class is used to create a down block of the UNet model.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        normalization (str): The normalization to use.
    """

    def __init__(self, in_channels: int, out_channels: int, normalization: str) -> None:
        super(DownBlock, self).__init__()
        self._conv = DoubleConv2d(
            in_channels, out_channels, normalization=normalization
        )
        self._pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function of the down block."""
        x = self._conv(x)
        return self._pool(x), x


class UpBlock(nn.Module):
    """This class is used to create an up block of the UNet model.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        normalization (str): The normalization to use.
    """

    def __init__(self, in_channels: int, out_channels: int, normalization: str) -> None:
        super(UpBlock, self).__init__()
        self._up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self._conv = DoubleConv2d(
            in_channels, out_channels, normalization=normalization
        )

    def _center_crop(self, x: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        """A helper function to center crop the input tensor, given a reference tensor."""
        diffY = x_ref.size()[2] - x.size()[2]
        diffX = x_ref.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """The forward function of the up block."""
        x = self._up(x)
        x = self._center_crop(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self._conv(x)


class UNetModel(BaseModel):
    """This class is used to create the UNet model.

    Args:
        init_features (int): The number of initial features.
        normalization (str): The type of normalization to use. Defaults to `batch`. Choices are `batch`, `layer`, `instance` or `None`.
    """

    tasks = [DEPTH_ESTIMATION_KEY, RECONSTRUCTION_KEY, SEGMENTATION_KEY]

    def __init__(
        self,
        init_features: int = 32,
        normalization: str = "batch",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(UNetModel, self).__init__(*args, **kwargs)
        assert normalization in ["batch", "layer", "instance", None]
        self._normalization = normalization
        self._features = [
            init_features,
            init_features * 2,
            init_features * 4,
            init_features * 8,
            init_features * 16,
        ]
        # Add one extra pointwise convolution to the UNet model such that
        # There is an explicit 1x1 convolution which can be manipulated by some method
        self._input = nn.Conv2d(self._inputs_dim[0], self._features[0], 1)
        self._down1 = DownBlock(self._features[0], self._features[0], normalization)
        self._down2 = DownBlock(self._features[0], self._features[1], normalization)
        self._down3 = DownBlock(self._features[1], self._features[2], normalization)
        self._down4 = DownBlock(self._features[2], self._features[3], normalization)
        self._center = DoubleConv2d(
            self._features[3], self._features[4], normalization=normalization
        )
        self._up4 = UpBlock(self._features[4], self._features[3], normalization)
        self._up3 = UpBlock(self._features[3], self._features[2], normalization)
        self._up2 = UpBlock(self._features[2], self._features[1], normalization)
        self._up1 = UpBlock(self._features[1], self._features[0], normalization)
        self._output = nn.Conv2d(self._features[0], self._outputs_dim, 1)
        self._output_activation = OutputActivation(self._task, dim=1)
        self._depth = 4

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
                layers.append(
                    nn.Upsample(
                        scale_factor=2**i, mode="bilinear", align_corners=True
                    )
                )
                layers.append(nn.Conv2d(self._features[i], self._features[0], 1))
                layers.append(
                    Normalization(
                        self._normalization,
                        dimension=2,
                        norm_kwargs={"num_features": self._features[0]},
                    )
                )
                layers.append(nn.ReLU())
                self._reshaping_layers.append(nn.Sequential(*layers))

                if "heads" in kwargs and kwargs["heads"]:
                    head = []
                    head.append(
                        nn.Conv2d(self._features[0], self._output[0].out_features, 1)
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
            for i in range(1, self._depth):
                if not heads[i - 1]:
                    continue
                layers = []
                hidden_feature_size = int(
                    math.sqrt(1 + gamma) ** (self._depth - i)
                    * hidden_feature_size_output
                )
                if gamma > 0:
                    layers.append(
                        nn.Upsample(
                            scale_factor=2 ** (i - 1),
                            mode="bilinear",
                            align_corners=True,
                        )
                    )
                    layers.append(
                        nn.Conv2d(self._features[i - 1], hidden_feature_size, 1)
                    )
                    layers.append(
                        Normalization(
                            self._normalization,
                            dimension=2,
                            norm_kwargs={"num_features": hidden_feature_size},
                        )
                    )
                    layers.append(nn.ReLU())
                    layers.append(nn.Conv2d(hidden_feature_size, size_output, 1))
                else:
                    layers.append(
                        nn.Upsample(
                            scale_factor=2 ** (i - 1),
                            mode="bilinear",
                            align_corners=True,
                        )
                    )
                    layers.append(nn.Conv2d(self._features[i - 1], size_output, 1))
                self._reshaping_layers.append(nn.Sequential(*layers))
        else:
            raise ValueError(f"Method {method} is not supported.")

    def forward(
        self,
        x: torch.Tensor,
        staged_output: bool = False,
        input_kwargs: Dict[str, Any] = {},
        output_kwargs: Dict[str, Any] = {},
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """The forward function of the UNet model.

        Args:
            x (torch.Tensor): The input tensor.
            staged_output (bool): Whether to return the output of each layer. Defaults to `False`.
            input_kwargs (Dict[str, Any]): The input arguments to pass to the input layer. Defaults to `{}`.
            output_kwargs (Dict[str, Any]): The input arguments to pass to the output layer. Defaults to `{}`.
        """
        layers_outputs = []
        x = self._input(x, **input_kwargs)
        x, x1 = self._down1(x)
        if staged_output:
            layers_outputs.append(x1)
        x, x2 = self._down2(x)
        if staged_output:
            layers_outputs.append(x2)
        x, x3 = self._down3(x)
        if staged_output:
            layers_outputs.append(x3)
        x, x4 = self._down4(x)
        if staged_output:
            layers_outputs.append(x4)
        x = self._center(x)
        x = self._up4(x, x4)
        x = self._up3(x, x3)
        x = self._up2(x, x2)
        x = self._up1(x, x1)
        x = self.final_layer(x, **output_kwargs)
        if staged_output:
            return x, layers_outputs
        return x

    def final_layer(self, x: torch.Tensor, **output_kwargs: Any) -> torch.Tensor:
        """This function is used to get the final layer output."""
        x = self._output(x, **output_kwargs)
        return self._output_activation(x)

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This function is used to add specific arguments to the parser."""
        parser = super(UNetModel, UNetModel).add_specific_args(parent_parser)
        parser.add_argument(
            "--model_init_features",
            type=int,
            default=32,
            help="The number of initial features.",
        )
        parser.add_argument(
            "--model_normalization",
            type=str,
            choices=["batch", "instance", "group", "layer", None],
            default="batch",
            help="The normalization to use.",
        )
        return parser
