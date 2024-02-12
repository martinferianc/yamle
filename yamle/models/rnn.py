from typing import List, Tuple, Union, Dict, Any
import torch.nn as nn
import torch
import argparse

from yamle.models.operations import LSTM, OutputActivation
from yamle.models.model import BaseModel

from yamle.defaults import REGRESSION_KEY, CLASSIFICATION_KEY, RECONSTRUCTION_KEY


class RNNModel(BaseModel):
    """This class is used to create a LSTM model with the given parameters.

    Args:
        hidden_dim (int): The dimension of the hidden layers.
        width_multiplier (int): The width multiplier for the hidden layers.
        depth (int): The number of hidden layers.
    """

    tasks = [REGRESSION_KEY, CLASSIFICATION_KEY]

    def __init__(
        self,
        hidden_dim: int,
        width_multiplier: int,
        depth: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(RNNModel, self).__init__(*args, **kwargs)
        self._inputs_dim = self._inputs_dim[-1]
        self._hidden_dim = hidden_dim
        self._width_multiplier = width_multiplier
        self._depth = depth

        self._hidden_dim = self._hidden_dim * self._width_multiplier

        self._input = nn.Linear(self._inputs_dim, hidden_dim)

        self._layers = nn.ModuleList()
        for i in range(depth):
            self._layers.append(LSTM(hidden_dim, hidden_dim))

        self._output = nn.Linear(hidden_dim, self._outputs_dim)
        self._output_activation = OutputActivation(self._task, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        staged_output: bool = False,
        input_kwargs: Dict[str, Any] = {},
        output_kwargs: Dict[str, Any] = {},
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """The forward function of the model.

        Args:
            x (torch.Tensor): The input tensor.
            staged_output (bool): Whether to return the output of each layer. Defaults to False.
            input_kwargs (Dict[str, Any]): The kwargs for the input layer.
            output_kwargs (Dict[str, Any]): The kwargs for the output layer.
        """
        layers_outputs = []
        h = None
        assert (
            len(x.shape) == 3
        ), f"The input shape should be `(batch_size, seq_len, inputs_dim)`, but got {x.shape}."
        x = self._input(x, **input_kwargs)
        for i in range(len(self._layers)):
            x, h, _ = self._layers[i](x)
            if staged_output:
                layers_outputs.append(h)
        x = h
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

        if method == "dun":
            self._reshaping_layers = nn.ModuleList(
                [
                    nn.Linear(self._hidden_dims[i], self._hidden_dims[-1])
                    for i in range(self._depth - 1)
                ]
            )
        else:
            raise ValueError(f"Method {method} is not supported.")

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the model specific arguments to the parent parser."""
        parser = super(RNNModel, RNNModel).add_specific_args(parent_parser)

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
        return parser


class RNNAutoEncoderModel(BaseModel):
    """This class is used to create a LSTM model with the given parameters.

    This is an autoencoder model, so the input and output dimensions are the same.
    The encoder consists of LSTM layers, while the decoder consists of LSTM layers.
    The encoder's last hidden state is repeated and used as the input to the decoder.
    Then all the hidden states of the decoder are processed through a linear layer to get the
    two times the input dimension, one for mean and one for variance.

    Args:
        hidden_dim (int): The dimension of the hidden layers.
        width_multiplier (int): The width multiplier for the hidden layers.
        encoder_depth (int): The number of hidden layers for the encoder.
        decoder_depth (int): The number of hidden layers for the decoder.
    """

    tasks = [RECONSTRUCTION_KEY]

    def __init__(
        self,
        hidden_dim: int,
        width_multiplier: int,
        encoder_depth: int,
        decoder_depth: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(RNNAutoEncoderModel, self).__init__(*args, **kwargs)
        self._inputs_dim = self._inputs_dim[-1]
        self._hidden_dim = hidden_dim * width_multiplier
        self._width_multiplier = width_multiplier
        self._encoder_depth = encoder_depth
        self._decoder_depth = decoder_depth
        self._depth = self._encoder_depth + self._decoder_depth

        self._input = nn.Linear(self._inputs_dim, hidden_dim)

        self._layers = nn.ModuleList()
        for i in range(encoder_depth):
            self._layers.append(LSTM(hidden_dim, hidden_dim))

        self._layers.append(InputRepeater())

        for i in range(decoder_depth):
            self._layers.append(LSTM(hidden_dim, hidden_dim))

        self._output = nn.Linear(hidden_dim, self._outputs_dim)
        self._output_activation = OutputActivation(self._task, dim=2)

    def forward(
        self,
        x: torch.Tensor,
        staged_output: bool = False,
        input_kwargs: Dict[str, Any] = {},
        output_kwargs: Dict[str, Any] = {},
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """The forward function of the model.

        Args:
            x (torch.Tensor): The input tensor.
            staged_output (bool): Whether to return the output of each layer. Defaults to
                False.
            input_kwargs (Dict[str, Any]): The kwargs for the input layer.
            output_kwargs (Dict[str, Any]): The kwargs for the output layer.
        """
        layers_outputs = []
        h = None
        assert (
            len(x.shape) == 3
        ), f"The input shape should be `(batch_size, seq_len, inputs_dim)`, but got {x.shape}."
        T = x.shape[1]
        x = self._input(x, **input_kwargs)

        for i in range(len(self._layers)):
            if isinstance(self._layers[i], InputRepeater):
                x = self._layers[i](h, T)
                continue
            else:
                x, h, _ = self._layers[i](x)
                if staged_output:
                    layers_outputs.append(x)

        x = self.final_layer(x, **output_kwargs)
        # Permute the output to be of shape (batch_size, outputs_dim, seq_len)
        x = x.permute(0, 2, 1)
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
        """This method is used to add the model specific arguments to the parent parser."""
        parser = super(RNNAutoEncoderModel, RNNAutoEncoderModel).add_specific_args(
            parent_parser
        )

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
            "--model_encoder_depth",
            type=int,
            default=3,
            help="The number of hidden layers for the encoder.",
        )

        parser.add_argument(
            "--model_decoder_depth",
            type=int,
            default=3,
            help="The number of hidden layers for the decoder.",
        )

        return parser


class InputRepeater(nn.Module):
    """This class repeats the input for the given number of times."""

    def forward(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """This method repeats the input for the given number of times.

        Args:
            x (torch.Tensor): The input tensor.
            T (int): The number of times to repeat the input.
        """
        x = x.unsqueeze(1).repeat(1, T, 1)
        return x
