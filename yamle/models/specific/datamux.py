from typing import List, Optional
import torch
import torch.nn as nn

from yamle.models.operations import (
    ParallelModel,
    Reduction,
    DepthwiseSeparableConv2d,
    CompletelySeparableConv2d,
)
from yamle.regularizers.regularizer import BaseRegularizer


class Multiplexer(nn.Module):
    """This module implements the input multiplexer.

    The function is to take the input of shape `(batch_size, num_members, num_features)` and
    output a tensor of shape `(batch_size, num_features)`.

    The output is produced through a  learnable `precoder` that is applied through a `ParallelModel`
    to each member of the input.

    Then the outputs is processed through the `coder`, which is applied to each member of the input
    and the member number of the coder is desired by the desired number of members.

    The output of the `coder` is then processed through the `postcoder`, which is applied to each
    member of the input.

    The output is then finally reduced through the `reduction` method. It can be optionally
    normalised through the `normalization` layer.

    Args:
        inputs_dim (int): The input dimension for the input tuple.
        outputs_dim (int): The output dimension for the output tuple.
        reduction (str): The reduction method to use. Defaults to "mean".
        coder (Optional[List[nn.Module]]): The coder layer to be applied to each member of the input.
        precoder (Optional[List[nn.Module]]): The precoder layer to be applied to each member of the input.
        postcoder (Optional[List[nn.Module]]): The postcoder layer to be applied to each member of the input.
        reduction_normalization (Optional[nn.Module]): The normalization layer to be applied to the output.
        feature_regularizer (BaseRegularizer): The feature_regularizer to be applied to the encoding of each member.
    """

    def __init__(
        self,
        inputs_dim: int,
        outputs_dim: int,
        reduction: str = "mean",
        coder: List[nn.Module] = None,
        precoder: Optional[List[nn.Module]] = None,
        postcoder: Optional[List[nn.Module]] = None,
        reduction_normalization: Optional[nn.Module] = None,
        feature_regularizer: Optional[BaseRegularizer] = None,
    ) -> None:

        super().__init__()
        self._inputs_dim = inputs_dim
        self._outputs_dim = outputs_dim

        self._coder = (
            ParallelModel(coder, inputs_dim=inputs_dim, outputs_dim=outputs_dim)
            if coder is not None
            else nn.Identity()
        )

        self._precoder = (
            ParallelModel(precoder, inputs_dim=inputs_dim, outputs_dim=outputs_dim)
            if precoder is not None
            else nn.Identity()
        )

        self._postcoder = (
            ParallelModel(postcoder, inputs_dim=inputs_dim, outputs_dim=outputs_dim)
            if postcoder is not None
            else nn.Identity()
        )

        self._reduction = Reduction(dim=outputs_dim, reduction=reduction)

        self._feature_regularizer = feature_regularizer
        self._regularization_value: torch.Tensor = 0.0
        self._regularizer_active: bool = True

        self._reduction_normalization = reduction_normalization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._precoder(x)
        x = self._coder(x)
        x = self._postcoder(x)

        if (
            self._feature_regularizer is not None
            and self.training
            and self._regularizer_active
        ):
            self._regularization_value = self._feature_regularizer(x)

        x = self._reduction(x)

        if self._reduction_normalization is not None:
            x = self._reduction_normalization(x)

        return x

    def initialise_members_same(self):
        """This is a helper function to initialise the members of the parallel model with the same weights."""
        self._coder.initialise_members_same() if self._coder is not None else None
        self._precoder.initialise_members_same() if self._precoder is not None else None
        self._postcoder.initialise_members_same() if self._postcoder is not None else None

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f"inputs_dim={self._inputs_dim}, outputs_dim={self._outputs_dim}"
        )

    def get_feature_regularization_value(self):
        """This function returns the regularization value of the model."""
        return (
            self._regularization_value
            if isinstance(self._regularization_value, torch.Tensor)
            else torch.tensor(
                0.0, dtype=torch.float32, device=next(self.parameters()).device
            )
        )

    def reset_feature_regularization_value(self):
        """This function resets the regularization value of the model."""
        self._regularization_value = 0.0

    def enable_feature_regularizer(self):
        """This function enables the feature regularizer."""
        self._regularizer_active = True

    def disable_feature_regularizer(self):
        """This function disables the feature regularizer."""
        self._regularizer_active = False


class Demultiplexer(nn.Module):
    """This module implements the demultiplexer part.

    The function is to take the input of shape `(batch_size, num_features)` and output a tensor of
    shape `(batch_size, num_members, num_output_features)`. The output is produced through a
    learnable encoding of parallel layers where each layer, or a sequence of layers, is applied
    to the same input.

    Args:
        parallel_layers (List[nn.Module]): The layers that will be applied to each member.
        outputs_dim (int): The output dimension for the output tuple.
    """

    def __init__(self, parallel_layers: List[nn.Module], outputs_dim: int) -> None:
        super().__init__()
        self._outputs_dim = outputs_dim
        self._parallel_layers = ParallelModel(
            parallel_layers, outputs_dim=outputs_dim, single_source=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._parallel_layers(x)

    def initialise_members_same(self):
        """This is a helper function to initialise the members of the parallel model with the same weights."""
        self._parallel_layers.initialise_members_same()

    def extra_repr(self) -> str:
        return super().extra_repr() + f"outputs_dim={self._outputs_dim}"
