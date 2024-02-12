from typing import Tuple, Dict, Any
import math

import torch
import torch.nn as nn
import argparse
from einops import rearrange
from yamle.models.operations import (
    OutputActivation,
    MatrixMultiplication,
    ResidualLayer,
)

from yamle.models.model import BaseModel
from yamle.models.specific.mcdropout import disable_dropout_replacement

from yamle.defaults import TEXT_CLASSIFICATION_KEY


class PreNorm(nn.Sequential):
    """This class implements the pre-normalization layer.

    Args:
        dim (int): The dimension of the input.
        module (nn.Module): The module to be applied after the normalization.
    """

    def __init__(self, dim: int, module: nn.Module) -> None:
        super().__init__(nn.LayerNorm(dim), module)


class FeedForward(nn.Sequential):
    """This class implements the feed-forward layer.

    It consists of two linear layers with GELU activation and dropout.

    Args:
        dim (int): The dimension of the input.
        hidden_dim (int): The dimension of the hidden layer.
        dropout (float): The dropout rate.
        dense (nn.Module): The dense layer to be used. Defaults to nn.Linear.
    """

    def __init__(
        self, dim: int, hidden_dim: int, dropout: float, dense: nn.Module = nn.Linear
    ) -> None:
        super().__init__(
            dense(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            dense(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self._dim = dim
        self._hidden_dim = hidden_dim
        self._dropout = dropout
        
        # Disable dropout replacement for the feed-forward layer
        disable_dropout_replacement(self._modules["0"])
        disable_dropout_replacement(self._modules["3"])
        

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f", dim={self._dim}, hidden_dim={self._hidden_dim}, dropout={self._dropout}"
        )


class Attention(nn.Module):
    """This class implements the attention layer.

    It computes multi-head attention.

    Args:
        dim (int): The dimension of the input.
        heads (int): The number of heads.
        dim_head (int): The dimension of each head.
        dropout (float): The dropout rate.
        causal (bool): Whether to use causal attention. Defaults to False.
    """

    def __init__(
        self, dim: int, heads: int, dim_head: int, dropout: float, causal: bool = False
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self._dim = dim
        self._inner_dim = inner_dim
        self._dim_head = dim_head
        self._dropout = dropout
        self._heads = heads
        self._scale = dim_head**-0.5
        self._causal = causal

        self._attend = nn.Softmax(dim=-1)
        self._drop = nn.Dropout(dropout)
        self._to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self._to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        self._matrix_multiplication1 = MatrixMultiplication()
        self._matrix_multiplication2 = MatrixMultiplication()
        
        # Disable dropout replacement for qkv layer and the first layer of the output layer
        disable_dropout_replacement(self._to_qkv)
        if isinstance(self._to_out, nn.Sequential):
            disable_dropout_replacement(self._to_out._modules["0"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        qkv = self._to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self._heads), qkv
        )

        dots = self._matrix_multiplication1(q, k.transpose(-1, -2)) * self._scale
        if self._causal:
            mask = torch.ones_like(dots).triu_(1).bool()
            dots.masked_fill_(mask, float("-inf"))
        attn = self._attend(dots)
        attn = self._drop(attn)

        out = self._matrix_multiplication2(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self._to_out(out)

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f", dim={self._dim}, inner_dim={self._inner_dim}, dim_head={self._dim_head}, dropout={self._dropout}, heads={self._heads}, scale={self._scale}, causal={self._causal}"
        )


class TransformerEncoderLayer(nn.Sequential):
    """This class implements the transformer encoder layer.

    It consists of a multi-head attention layer and a feed-forward layer.
    It also implements the residual connection and layer normalization.

    Args:
        dim (int): The dimension of the input.
        heads (int): The number of heads.
        dim_head (int): The dimension of each head.
        mlp_dim (int): The dimension of the hidden layer in the feed-forward layer.
        dropout (float): The dropout rate.
        causal (bool): Whether to use causal attention. Defaults to False.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
        causal: bool = False,
    ) -> None:
        super().__init__(
            ResidualLayer(
                PreNorm(
                    dim,
                    Attention(
                        dim,
                        heads=heads,
                        dim_head=dim_head,
                        dropout=dropout,
                        causal=causal,
                    ),
                )
            ),
            ResidualLayer(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
        )
        self._dim = dim
        self._heads = heads
        self._dim_head = dim_head
        self._mlp_dim = mlp_dim
        self._dropout = dropout
        self._causal = causal

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f", dim={self._dim}, heads={self._heads}, dim_head={self._dim_head}, mlp_dim={self._mlp_dim}, dropout={self._dropout}, causal={self._causal}"
        )


class PositionalEncoding(nn.Module):
    """This class is used to create a module to implement the positional encoding.

    Args:
        inputs_dim (int): The total size of token embeddings.
        embedding_dim (int): The number of expected features in the input.
        dropout (float): The dropout value.
        max_len (int): The max length of the expected input.
    """

    def __init__(
        self, inputs_dim: int, embedding_dim: int, dropout: float, max_len: int = 5000
    ) -> None:
        super().__init__()
        self._embedding = nn.Embedding(inputs_dim, embedding_dim)
        self._dropout = nn.Dropout(p=dropout)
        self._scale = math.sqrt(embedding_dim)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )
        pe = torch.zeros(max_len, 1, embedding_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("_pe", pe)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self._embedding(x) * self._scale
        x = x + self._pe[: x.size(0)]
        return self._dropout(x)

    def reset_parameters(self) -> None:
        """This function is used to initialize the parameters of the model."""
        self._embedding.weight.data.uniform_(-0.1, 0.1)


class TransformerModel(BaseModel):
    """This class is used to create a Transformer decoder model.

    It is based on the PyTorch implementation of the Transformer model.

    Args:
        embedding_dim (int): The embedding dimensions of the model.
        num_heads (int): The number of heads in the multiheadattention models.
        num_decoder_layers (int): The number of sub-decoder-layers in the decoder.
        hidden_dim (int): The dimension of the feedforward network model.
        dropout (float): The dropout value.
    """

    tasks = [TEXT_CLASSIFICATION_KEY]

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_decoder_layers: int,
        hidden_dim: int,
        dropout: float,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(TransformerModel, self).__init__(*args, **kwargs)

        self._positional_encoding = PositionalEncoding(
            self._outputs_dim, embedding_dim, dropout
        )

        # We use TransformerEncoderLayer as the decoder layer beacause it is easier to set
        # Causal mask just by setting the `is_causal` parameter to True.
        head_dim = embedding_dim // num_heads
        self._decoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim=embedding_dim,
                    heads=num_heads,
                    dim_head=head_dim,
                    mlp_dim=hidden_dim,
                    dropout=dropout,
                    causal=True,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self._output = nn.Linear(embedding_dim, self._outputs_dim)
        # Implement weight parameter sharing in the output layer and the positional encoding layer
        self._output.weight = self._positional_encoding._embedding.weight
        self._output_activation = OutputActivation(self._task, dim=2)

        self._depth = num_decoder_layers
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """This function is used to initialize the parameters of the model."""
        self._output.bias.data.zero_()
        self._output.weight.data.uniform_(-0.1, 0.1)

    def forward(
        self,
        x: torch.Tensor,
        staged_output: bool = False,
        input_kwargs: Dict[str, Any] = {},
        output_kwargs: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Forward pass of the model.

        Note that the input has a shape of `(batch_size, seq_len)`.

        Args:
            x (torch.Tensor): The input tensor.
            staged_output (bool): Whether to return the output of each layer.
            input_kwargs (Dict[str, Any]): The kwargs for the input layer.
            output_kwargs (Dict[str, Any]): The kwargs for the output layer.
        """
        layers_outputs = []
        x = self._positional_encoding(x)
        for e in self._decoder:
            x = e(x, is_causal=True)
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

    def generate(
        self, input: torch.Tensor, max_len: int, temperature: float = 1.0, **kwargs: Any
    ) -> torch.Tensor:
        """This function is used to generate output by passing the input through the model."""
        for _ in range(max_len):
            x = self._positional_encoding(input)
            for e in self._decoder:
                x = e(x, is_causal=True)
            # Get the last token
            x = x[:, [-1], :]
            x = self._output(x)
            x = torch.softmax(x / temperature, dim=-1)
            x = torch.multinomial(x, num_samples=1)
            # Add the new token to the input
            input = torch.cat([input, x], dim=1)
        return input[:, :-max_len, :]

    def add_method_specific_layers(self, method: str, **kwargs: Any) -> None:
        """This method is used to add method specific layers to the model.

        Args:
            method (str): The method to use.
        """
        super().add_method_specific_layers(method, **kwargs)

        if method in ["dun", "mimmo"]:
            self._reshaping_layers = nn.ModuleList(
                [nn.Identity() for _ in range(self._depth - 1)]
            )
        else:
            raise ValueError(f"Method {method} is not supported.")

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """This function is used to add specific arguments to the parser."""
        parser = super(TransformerModel, TransformerModel).add_specific_args(parser)
        parser.add_argument(
            "--model_embedding_dim",
            type=int,
            default=200,
            help="The number of expected features in the input.",
        )
        parser.add_argument(
            "--model_num_heads",
            type=int,
            default=2,
            help="The number of heads in the multiheadattention models.",
        )
        parser.add_argument(
            "--model_num_decoder_layers",
            type=int,
            default=2,
            help="The number of decoder layers.",
        )
        parser.add_argument(
            "--model_hidden_dim",
            type=int,
            default=200,
            help="The dimension of the feedforward network model.",
        )
        parser.add_argument(
            "--model_dropout", type=float, default=0.2, help="The dropout value."
        )
        return parser
