from typing import Any, Callable

import torch
import torch.nn as nn

from yamle.models.model import BaseModel

import logging

logging = logging.getLogger("pytorch_lightning")


class MIMMMOWrapper(nn.Module):
    """This is a wrapper for a MIMMO module which makes the predictions from any ``BaseModel``.

    This is to wrap the forward method which should return all the predictions from the model.

    Args:
        model (BaseModel): The model to wrap.
        evaluation_depth_weights_function (Callable): The function to use to compute the depth weights.
    """

    def __init__(
        self, model: BaseModel, evaluation_depth_weights_function: Callable
    ) -> None:
        super().__init__()
        self.model = model
        self._evaluation_depth_weights_function = evaluation_depth_weights_function

    def forward(self, x: torch.Tensor, **forward_kwargs: Any) -> torch.Tensor:
        """This method is used to perform a forward pass of the model."""
        last_layer, stages = self.model(x, staged_output=True, **forward_kwargs)

        # Since the last layer uses the last hidden layer
        # we can remove it
        stages = stages[:-1]
        outputs = []
        offset = 0
        for i, h in enumerate(stages):
            if not self.model._available_heads[i]:
                offset += 1
                continue
            h = self.model._reshaping_layers[i - offset](h)
            h = (
                self.model.final_layer(h)
                if not self.model._additional_heads
                else self.model._output_activation(self.model._heads[i - offset](h))
            )
            # A single output has shape `(batch_size, num_members, predictions)`
            outputs.append(h)

        outputs.append(last_layer)
        # Note that the output shape is `(batch_size, depth, num_members, predictions)`
        return torch.stack(outputs, dim=1)

    @property
    def _input(self) -> nn.Module:
        """This property is used to get the input layer of the model."""
        return self.model._input

    @property
    def _output(self) -> nn.Module:
        """This property is used to get the output layer of the model."""
        return self.model._output

    @property
    def _heads(self) -> nn.ModuleList:
        """This property is used to get the heads of the model."""
        return self.model._heads

    @property
    def _reshaping_layers(self) -> nn.ModuleList:
        """This property is used to get the reshaping layers of the model."""
        return self.model._reshaping_layers

    @property
    def _prior_depth_weights(self) -> torch.Tensor:
        """This property is used to get the prior depth weights of the model."""
        return self.model._prior_depth_weights

    @property
    def _depth_weights(self) -> torch.Tensor:
        """This property is used to get the depth weights of the model."""
        return self.model._depth_weights

    @property
    def _available_heads(self) -> torch.Tensor:
        """This property is used to get the available heads of the model."""
        return self.model._available_heads

    @property
    def _additional_heads(self) -> bool:
        """This property is used to get whether the model has additional heads."""
        return self.model._additional_heads

    @property
    def _output_activation(self) -> nn.Module:
        """This property is used to get the output activation of the model."""
        return self.model._output_activation

    @property
    def _depth(self) -> int:
        """This property is used to get the depth of the model."""
        return self.model._depth

    def final_layer(self, x: torch.Tensor, **output_kwargs: Any) -> torch.Tensor:
        """This function is used to get the final layer output."""
        return self.model.final_layer(x, **output_kwargs)

    def reset(self) -> None:
        """This method is used to reset the model e.g. at the start of a new epoch."""
        self.model.reset()

    def replace_layers_for_quantization(self) -> None:
        """Fuses all the operations in the network.

        In this function we only need to fuse layers that are not in the blocks.
        e.g. the reshaping layers added by the method.
        """
        self.model.replace_layers_for_quantization()