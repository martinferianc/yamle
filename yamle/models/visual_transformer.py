from typing import Tuple, Dict, Any, Union, List


import torch
from torch import nn

import argparse
from einops.layers.torch import Rearrange
from einops import repeat
import math
from yamle.models.operations import (
    OutputActivation,
    Reduction,
    Add,
    Lambda,
    ReshapeOutput,
)
from yamle.models.transformer import TransformerEncoderLayer
from yamle.models.model import BaseModel
from yamle.models.specific.mcdropout import disable_dropout_replacement

from yamle.defaults import REGRESSION_KEY, CLASSIFICATION_KEY


class SpatialPositionalEmbedding(nn.Module):
    """This class is used to create a spatial positional embedding to be used in the
    visual transformer for 2D images.

    Args:
        inputs_dim (Tuple[int, int, int]): The dimension of the input.
        patch_size (int): The size of the patch.
        embedding_dim (int): The dimension of the embedding.
        dropout (float): The dropout rate.
        num_cls_tokens (int): The number of class tokens. Defaults to 1.
        positional_embedding (bool): Whether to use positional embedding. Defaults to True.
    """

    def __init__(
        self,
        inputs_dim: Tuple[int, int, int],
        patch_size: int,
        embedding_dim: int,
        dropout: float = 0.0,
        num_cls_tokens: int = 1,
        positional_embedding: bool = True,
    ) -> None:
        super().__init__()
        C, H, W = inputs_dim
        assert (
            H % patch_size == 0
        ), f"Image dimensions must be divisible by the patch size. Got {H} and {patch_size}."
        assert (
            W % patch_size == 0
        ), f"Image dimensions must be divisible by the patch size. Got {W} and {patch_size}."
        self._inputs_dim = inputs_dim
        self._patch_size = patch_size
        self._embedding_dim = embedding_dim
        self._dropout = dropout
        self._num_cls_tokens = num_cls_tokens

        self._num_patches = (H // patch_size) ** 2
        self._patch_dim = C * patch_size**2

        self._to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            nn.LayerNorm(self._patch_dim),
            nn.Linear(self._patch_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        
        # Disable dropout replacement for positional embedding
        disable_dropout_replacement(self._to_patch_embedding._modules["2"])

        self._positional_embedding = (
            nn.Parameter(
                torch.randn(1, self._num_patches + self._num_cls_tokens, embedding_dim),
                requires_grad=True,
            )
            if positional_embedding
            else None
        )
        self._cls_token = (
            nn.Parameter(
                torch.randn(1, self._num_cls_tokens, embedding_dim), requires_grad=True
            )
            if num_cls_tokens > 0
            else None
        )
        self._drop = nn.Dropout(dropout)
        self._add = Add()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to get the forward pass of the model."""
        x = self._to_patch_embedding(x)
        b, _, _ = x.shape
        if self._cls_token is not None:
            cls_tokens = repeat(self._cls_token, "() n d -> b n d", b=b)
            x = torch.cat((cls_tokens, x), dim=1)
        if self._positional_embedding is not None:
            x = self._add(x, self._positional_embedding)
        return self._drop(x)

    def get_cls_token_indices(self) -> torch.Tensor:
        """This method is used to get the indices of the class tokens.

        They are added as the first tokens in the sequence.
        """
        return torch.arange(self._num_cls_tokens)


class VisualTransformerModel(BaseModel):
    """This class is used to create a visual transformer model.

    Args:
        patch_size (int): The size of the patch to be used.
        pooling (str): The pooling to be used. It can be either `mean` or `cls`.
        embedding_dim (int): The number of expected features in the input.
        num_heads (int): The number of heads in the multiheadattention models.
        depth (int): The number of sub-encoder-layers in the encoder.
        num_cls_tokens (int): The number of class tokens. Defaults to 1.
        hidden_dim (int): The dimension of the feedforward network model.
        width_multiplier (int): The width multiplier for the hidden dimension.
        dropout (float): The dropout value.
    """

    tasks = [
        CLASSIFICATION_KEY,
        REGRESSION_KEY,
    ]

    def __init__(
        self,
        patch_size: int = 4,
        pooling: str = "mean",
        embedding_dim: int = 128,
        num_heads: int = 6,
        depth: int = 4,
        num_cls_tokens: int = 1,
        hidden_dim: int = 512,
        width_multiplier: int = 1,
        dropout: float = 0.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert pooling in [
            "mean",
            "cls",
        ], f"Pooling must be either `mean` or `cls`. Got {pooling}."
        self._embedding_dim = embedding_dim
        self._num_heads = num_heads
        self._head_dim = embedding_dim // num_heads
        self._hidden_dim = hidden_dim * width_multiplier
        self._patch_size = patch_size
        self._pooling = pooling
        self._num_cls_tokens = num_cls_tokens

        self._input = SpatialPositionalEmbedding(
            self._inputs_dim, patch_size, embedding_dim, dropout, num_cls_tokens
        )

        self._layers = nn.ModuleList()
        for _ in range(depth):
            self._layers.append(
                TransformerEncoderLayer(
                    self._embedding_dim,
                    self._num_heads,
                    self._head_dim,
                    self._hidden_dim,
                    dropout,
                    causal=False,
                )
            )
        self._layers.append(nn.LayerNorm(self._embedding_dim))
        self._layers.append(
            Reduction(dim=1, reduction="mean") if pooling == "mean" else nn.Identity()
        )

        self._output = (
            nn.Linear(self._embedding_dim, self._outputs_dim)
            if pooling == "mean"
            else nn.Linear(
                self._embedding_dim * self._num_cls_tokens, self._outputs_dim
            )
        )
        self._output_activation = OutputActivation(self._task, dim=1)

        self._depth = depth

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
                layers.append(nn.Linear(self._embedding_dim, self._embedding_dim))
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(self._embedding_dim))
                layers.append(
                    Reduction(dim=1, reduction="mean")
                    if self._pooling == "mean"
                    else Lambda(lambda x: x[:, self._input.get_cls_token_indices()])
                )
                self._reshaping_layers.append(nn.Sequential(*layers))

                if "heads" in kwargs and kwargs["heads"]:
                    head = []
                    head.append(
                        nn.Linear(self._embedding_dim, self._output[0].out_features)
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
            heads = [1] * (self._depth)
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
                sequence.append(
                    Reduction(dim=1, reduction="mean")
                    if self._pooling == "mean"
                    else Lambda(
                        lambda x: x[:, self._input.get_cls_token_indices()].reshape(
                            x.shape[0], -1
                        )
                    )
                )
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
            if staged_output and isinstance(layer, TransformerEncoderLayer):
                layers_outputs.append(x)

        if isinstance(self._layers[-1], Reduction) and staged_output:
            layers_outputs[-1] = x

        x = self.final_layer(x, **output_kwargs)
        if staged_output:
            return x, layers_outputs
        return x

    def final_layer(self, x: torch.Tensor, **output_kwargs: Any) -> torch.Tensor:
        """This function is used to get the final layer output."""
        if self._pooling == "cls":
            # CLS token is the first token
            B = x.shape[0]
            x = x[:, self._input.get_cls_token_indices()].reshape(B, -1)

        x = self._output(x, **output_kwargs)
        return self._output_activation(x)

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """This function is used to add specific arguments to the parser."""
        parser = super(
            VisualTransformerModel, VisualTransformerModel
        ).add_specific_args(parser)
        parser.add_argument(
            "--model_patch_size",
            type=int,
            default=4,
            help="The size of the patch to be used.",
        )
        parser.add_argument(
            "--model_embedding_dim",
            type=int,
            default=128,
            help="The number of expected features in the input.",
        )
        parser.add_argument(
            "--model_pooling",
            type=str,
            default="cls",
            choices=["mean", "cls"],
            help="The pooling to be used.",
        )
        parser.add_argument(
            "--model_num_heads",
            type=int,
            default=4,
            help="The number of heads in the multiheadattention models.",
        )
        parser.add_argument(
            "--model_depth",
            type=int,
            default=2,
            help="The number of sub-encoder-layers in the encoder.",
        )
        parser.add_argument(
            "--model_num_cls_tokens",
            type=int,
            default=1,
            help="The number of cls tokens.",
        )
        parser.add_argument(
            "--model_hidden_dim",
            type=int,
            default=512,
            help="The dimension of the feedforward network model.",
        )
        parser.add_argument(
            "--model_width_multiplier",
            type=int,
            default=1,
            help="The width multiplier for the hidden dimension.",
        )
        parser.add_argument(
            "--model_dropout", type=float, default=0.1, help="The dropout value."
        )
        return parser
