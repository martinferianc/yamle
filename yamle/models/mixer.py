from typing import Tuple, Dict, Any, Union, List


import torch
from torch import nn

import argparse
import math
from functools import partial
from yamle.models.transformer import FeedForward, PreNorm
from yamle.models.visual_transformer import SpatialPositionalEmbedding
from yamle.models.operations import (
    ResidualLayer,
    Reduction,
    OutputActivation,
    ReshapeOutput,
)
from yamle.models.specific.mcdropout import disable_dropout_replacement
from yamle.models.model import BaseModel
from yamle.defaults import REGRESSION_KEY, CLASSIFICATION_KEY


class MixerLayer(nn.Sequential):
    """This class implements MLP-Mixer layer.

    It consists of a token-mixing MLP and a channel-mixing MLP.

    Args:
        tokens_dim (int): The dimension of the token mixing MLP. This is the embedding dimension.
        tokens_hidden_dim (int): The dimension of the hidden layer in the token mixing MLP.
        channels_dim (int): The dimension of the channel mixing MLP. This is the number of patches.
        channels_hidden_dim (int): The dimension of the hidden layer in the channel mixing MLP.
        dropout (float): The dropout rate.
    """

    def __init__(
        self,
        tokens_dim: int,
        tokens_hidden_dim: int,
        channels_dim: int,
        channels_hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        super().__init__(
            ResidualLayer(
                PreNorm(
                    tokens_dim,
                    FeedForward(channels_dim, channels_hidden_dim, dropout, chan_first),
                )
            ),
            ResidualLayer(
                PreNorm(
                    tokens_dim,
                    FeedForward(tokens_dim, tokens_hidden_dim, dropout, chan_last),
                )
            ),
        )
        self._tokens_dim = tokens_dim
        self._channels_dim = channels_dim
        self._dropout = dropout
        self._tokens_hidden_dim = tokens_hidden_dim
        self._channels_hidden_dim = channels_hidden_dim

    def extra_repr(self) -> str:
        return f"tokens_dim={self._tokens_dim}, tokens_hidden_dim={self._tokens_hidden_dim}, channels_dim={self._channels_dim}, channels_hidden_dim={self._channels_hidden_dim}, dropout={self._dropout}"


class MixerModel(BaseModel):
    """This class is used to create a MLP-Mixer model.

    Args:
        patch_size (int): The size of the patch to be used.
        tokens_dim (int): The dimension of the token mixing MLP. This is the embedding dimension.
        tokens_hidden_dim (int): The dimension of the hidden layer in the token mixing MLP.
        channels_hidden_dim (int): The dimension of the hidden layer in the channel mixing MLP.
        num_layers (int): The number of layers in the model.
        dropout (float): The dropout value.
        task (str): The task to be performed. It can be either `classification` or
            `regression`.
    """

    tasks = [CLASSIFICATION_KEY, REGRESSION_KEY]

    def __init__(
        self,
        patch_size: int = 4,
        tokens_dim: int = 128,
        tokens_hidden_dim: int = 512,
        channels_hidden_dim: int = 2048,
        num_layers: int = 8,
        dropout: float = 0.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._patch_size = patch_size
        self._tokens_dim = tokens_dim
        self._tokens_hidden_dim = tokens_hidden_dim
        self._channels_hidden_dim = channels_hidden_dim
        self._num_layers = num_layers
        self._dropout = dropout

        self._input = SpatialPositionalEmbedding(
            self._inputs_dim,
            patch_size,
            tokens_dim,
            dropout,
            num_cls_tokens=0,
            positional_embedding=False,
        )

        _, H, W = self._inputs_dim
        num_patches = (H // patch_size) * (W // patch_size)

        self._layers = nn.ModuleList()
        for _ in range(num_layers):
            self._layers.append(
                MixerLayer(
                    tokens_dim,
                    tokens_hidden_dim,
                    num_patches,
                    channels_hidden_dim,
                    dropout,
                )
            )
        self._layers.append(nn.LayerNorm(self._tokens_dim))
        self._layers.append(Reduction(dim=1, reduction="mean"))

        self._output = nn.Linear(self._tokens_dim, self._outputs_dim)
        self._output_activation = OutputActivation(self._task, dim=1)

        self._depth = num_layers
        # Disable dropout replacement for all the layers, transformer already has dropout
        disable_dropout_replacement(self)

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
                layers.append(Reduction(dim=1, reduction="mean"))
                layers.append(nn.LayerNorm(self._tokens_dim))
                layers.append(nn.Linear(self._tokens_dim, self._tokens_dim))
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(self._tokens_dim))
                self._reshaping_layers.append(nn.Sequential(*layers))

                if "heads" in kwargs and kwargs["heads"]:
                    head = []
                    head.append(
                        nn.Linear(self._tokens_dim, self._output[0].out_features)
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
                sequence = []
                hidden_feature_size = int(
                    math.sqrt(1 + gamma) ** (self._depth - i)
                    * hidden_feature_size_output
                )
                sequence.append(Reduction(dim=1, reduction="mean"))
                sequence.append(nn.LayerNorm(hidden_feature_size_output))
                if gamma > 0:
                    sequence.append(
                        nn.Linear(hidden_feature_size_output, hidden_feature_size)
                    )
                    sequence.append(nn.GELU())
                    sequence.append(nn.LayerNorm(hidden_feature_size))
                    sequence.append(nn.Linear(hidden_feature_size, size_output))
                else:
                    sequence.append(nn.Linear(hidden_feature_size_output, size_output))
                self._reshaping_layers.append(nn.Sequential(*sequence))
        else:
            raise ValueError(f"Method {method} is not supported.")

    def forward(
        self,
        x: torch.Tensor,
        staged_output: bool = False,
        input_kwargs: Dict[str, Any] = {},
        output_kwargs: Dict[str, Any] = {},
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.
            staged_output (bool, optional): Whether to return the output of each layer. Defaults to False.
            input_kwargs (Dict[str, Any], optional): The input kwargs. Defaults to {}.
            output_kwargs (Dict[str, Any], optional): The output kwargs. Defaults to {}.
        """

        layers_outputs = []
        x = self._input(x)
        for layer in self._layers:
            x = layer(x)
            if staged_output and isinstance(layer, MixerLayer):
                layers_outputs.append(x)

        if isinstance(self._layers[-1], Reduction) and staged_output:
            layers_outputs[-1] = x

        x = self.final_layer(x, **output_kwargs)
        if staged_output:
            return x, layers_outputs
        return x

    def final_layer(self, x: torch.Tensor, **output_kwargs: Any) -> torch.Tensor:
        """This function is used to get the final layer output."""
        x = self._output(x, **output_kwargs)
        return self._output_activation(x)

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """This function is used to add specific arguments to the parser."""
        parser = super(MixerModel, MixerModel).add_specific_args(parser)
        parser.add_argument(
            "--model_patch_size",
            type=int,
            default=4,
            help="The size of the patch to be used.",
        )
        parser.add_argument(
            "--model_tokens_dim",
            type=int,
            default=256,
            help="The dimension of the tokens. This is the embedding dimension.",
        )
        parser.add_argument(
            "--model_tokens_hidden_dim",
            type=int,
            default=512,
            help="The hidden dimension of the tokens.",
        )
        parser.add_argument(
            "--model_channels_hidden_dim",
            type=int,
            default=128,
            help="The hidden dimension of the channels.",
        )
        parser.add_argument(
            "--model_num_layers",
            type=int,
            default=4,
            help="The number of layers in the model.",
        )
        parser.add_argument(
            "--model_dropout",
            type=float,
            default=0.1,
            help="The dropout to be used in the model.",
        )
        return parser
