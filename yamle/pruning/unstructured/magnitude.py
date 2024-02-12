from typing import Any
import torch
import torch.nn as nn
import argparse

from yamle.pruning.pruner import BasePruner
from yamle.utils.pruning_utils import (
    get_all_prunable_weights,
    is_layer_prunable,
    is_parameter_prunable,
)


class UnstructuredMagnitudePruner(BasePruner):
    """This is the base class for unstructured magnitude-based pruning.

    It will prune the weights with the lowest absolute magnitude. The threshold is determined
    by the pruning percentage. The pruning percentage is the percentageage of weights to prune.

    Args:
        pruning_percentage (float): The percentageage of weights to prune.
    """

    def __init__(self, percentage: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert 0.0 <= percentage <= 1.0, "Pruning percentage must be between 0 and 1."
        self._percentage = percentage

    def __call__(self, m: nn.Module) -> float:
        """This method is used to prune the model."""
        # Get all the weights in the model
        weights = get_all_prunable_weights(m)

        # Find the magnitude of the weight at a given percentile
        threshold = torch.abs(weights).kthvalue(int(self._percentage * len(weights)))[0]

        # Prune the weights
        for module in m.modules():
            if is_layer_prunable(module):
                for p in module.parameters():
                    if is_parameter_prunable(p):
                        # Create a mask to prune the weights, `True` means prune
                        mask = torch.abs(p.data) < threshold
                        self.prune_parameter(p, mask)

        return threshold.item()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(percentage={self._percentage})"

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the pruner specific arguments to the parent parser."""
        parser = super(
            UnstructuredMagnitudePruner, UnstructuredMagnitudePruner
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--pruner_percentage",
            type=float,
            default=0.5,
            help="The percentageage of weights to prune.",
        )
        return parser
