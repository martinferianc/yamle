from typing import Any
import torch
import torch.nn as nn
import argparse

from yamle.methods.mimo import MIMOMethod
from yamle.models.operations import (
    Unsqueeze,
    OutputActivation,
    ReshapeOutput,
    ReshapeInput,
)


def replace_layers_with_grouped_convs(
    model: nn.Module, M: int, alpha: int, gamma: int
) -> nn.Module:
    """This method replaces all `nn.Linear` and `nn.Conv2d` layers with grouped versions.

    Each layer is replaced with a layer where the input and output dimensions are multiplied by the
    `alpha`.

    Args:
        model (nn.Module): The model to replace the layers of.
        M (int): The number of members in the ensemble.
        alpha (int): The width multiplier.
        gamma (int): The subgroups multiplier.

    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and not hasattr(child, "do_not_replace"):
            setattr(
                model,
                name,
                nn.Conv2d(
                    child.in_features * alpha,
                    child.out_features * alpha,
                    kernel_size=1,
                    groups=gamma * M,
                    bias=child.bias is not None,
                ),
            )
        elif isinstance(child, nn.Conv2d) and not hasattr(child, "do_not_replace"):
            setattr(
                model,
                name,
                nn.Conv2d(
                    child.in_channels * alpha,
                    child.out_channels * alpha,
                    child.kernel_size,
                    child.stride,
                    child.padding,
                    groups=gamma * M,
                    bias=child.bias is not None,
                ),
            )
        elif isinstance(child, nn.BatchNorm2d) and not hasattr(child, "do_not_replace"):
            setattr(model, name, nn.BatchNorm2d(child.num_features * alpha))
        else:
            replace_layers_with_grouped_convs(child, M, alpha, gamma)
    return model


class PEMethod(MIMOMethod):
    """This class is the extension of the base method for packed-ensemble methods.

    Args:
        alpha (int): The expansion multiplier for the width of the model. It is used to multiply the
            number of input and output channels of each layer.
        gamma (int): The subgroups multiplier. It is used to multiply the number of groups in each
            convolutional layer together with the `num_members` parameter, such as `(num_members * gamma)`.
    """

    def __init__(self, alpha: int, gamma: int, *args: Any, **kwargs: Any) -> None:
        assert (
            alpha >= 1
        ), f"The alpha parameter should be larger or equal to 1, but got {alpha}."
        assert (
            gamma >= 1
        ), f"The gamma parameter should be larger or equal to 1, but got {gamma}."
        assert (
            int(alpha) == alpha
        ), f"The alpha parameter should be an integer, but got {alpha}."
        assert (
            int(gamma) == gamma
        ), f"The gamma parameter should be an integer, but got {gamma}."
        self._alpha = alpha
        self._gamma = gamma
        super(PEMethod, self).__init__(*args, **kwargs)
        replace_layers_with_grouped_convs(
            self.model, self._num_members, self._alpha, self._gamma
        )

    def _post_init(self) -> None:
        """This method is called after the initialization of the method."""
        self._replace_input_and_output_layers()

    def analyse(self, save_path: str) -> None:
        """This method analyses the model and saves the results to a file."""
        pass

    def _replace_input_and_output_layers(self) -> None:
        # Replace the first layer with one where the input dimension is multiplied by the number of
        # members.
        if isinstance(self.model._input, nn.Linear):
            self.model._input = nn.Sequential(
                Unsqueeze(shape_length=4),
                ReshapeInput(),
                nn.Conv2d(
                    in_channels=self.model._input.in_features * self._num_members,
                    out_channels=self.model._input.out_features * self._alpha,
                    kernel_size=1,
                    groups=self._gamma * self._num_members,
                    bias=self.model._input.bias is not None,
                ),
            )
            self.model._input[1].do_not_replace = True
        elif isinstance(self.model._input, torch.nn.Conv2d):
            self.model._input = nn.Sequential(
                ReshapeInput(),
                torch.nn.Conv2d(
                    in_channels=self.model._input.in_channels * self._num_members,
                    out_channels=self.model._input.out_channels * self._alpha,
                    kernel_size=self.model._input.kernel_size,
                    stride=self.model._input.stride,
                    padding=self.model._input.padding,
                    groups=self._gamma * self._num_members,
                    bias=self.model._input.bias is not None,
                ),
            )
            self.model._input[1].do_not_replace = True
        else:
            raise ValueError(
                "The first layer of the model should be either a `torch.nn.Linear` or a "
                "`torch.nn.Conv2d`."
            )
        if isinstance(self.model._output, torch.nn.Linear):
            self.model._output = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(
                    self.model._output.in_features * self._alpha,
                    self.model._output.out_features * self._num_members,
                    kernel_size=1,
                    groups=self._num_members * self._gamma,
                    bias=self.model._output.bias is not None,
                ),
                ReshapeOutput(self._num_members),
            )
            self.model._output[1].do_not_replace = True
            self.model._output_activation = OutputActivation(task=self._task, dim=2)
        else:
            raise ValueError(
                "The last layer of the model should be a `torch.nn.Linear`."
            )

        # Check if model has adaptive avg pooling layer and flatten in the end, if yes remove them
        if isinstance(self.model._layers[-2], nn.AdaptiveAvgPool2d) and isinstance(
            self.model._layers[-1], nn.Flatten
        ):
            self.model._layers = self.model._layers[:-2]

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method adds the specific arguments for the MIMO method."""
        parser = super(PEMethod, PEMethod).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_alpha", type=int, default=1, help="The width multiplier."
        )
        parser.add_argument(
            "--method_gamma", type=int, default=1, help="The subgroups multiplier."
        )
        return parser
