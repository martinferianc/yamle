from typing import Any
from abc import ABC, abstractmethod
import argparse
import torch.nn as nn
import torch
from tabulate import tabulate

from yamle.defaults import MASK_PRUNING_KEY, FORMER_DATA_PRUNING_KEY
from yamle.utils.pruning_utils import is_layer_prunable, is_parameter_prunable

import logging

logging = logging.getLogger("pytorch_lightning")


class BasePruner(ABC):
    """This is the base class for all prune methods.

    The pruner's call method will be used to prune the model.
    Additionally there is a function to analyse the model and print a summary of the pruning
    per each named parameter.
    """

    @abstractmethod
    def __call__(self, m: nn.Module) -> Any:
        """This method is used to prune the model."""
        pass

    @staticmethod
    def prune_parameter(p: nn.Parameter, mask: torch.Tensor) -> None:
        """This function is used to prune a parameter based on a mask.

        The mask's `True` values will be pruned. `False` values will be kept.

        This function and this function only should be used to prune parameters.
        """
        assert (
            p.shape == mask.shape
        ), f"Parameter shape {p.shape} and mask shape {mask.shape} do not match."
        if hasattr(p, MASK_PRUNING_KEY):
            logging.warning(
                "The parameter already has a mask. The new mask will be used."
            )

        setattr(p, MASK_PRUNING_KEY, mask.detach().cpu().clone())
        setattr(p, FORMER_DATA_PRUNING_KEY, p.data.detach().cpu().clone())
        p.data[mask] = 0.0

    @staticmethod
    def recover_parameter(p: nn.Parameter) -> None:
        """This function is used to recover a parameter from a mask."""
        if hasattr(p, FORMER_DATA_PRUNING_KEY):
            p.data = getattr(p, FORMER_DATA_PRUNING_KEY).to(p.device)
            # Delete the data but keep the mask such that there is a record of the pruning
            delattr(p, FORMER_DATA_PRUNING_KEY)
        else:
            logging.warning("No former data was found for this parameter.")

    def recover(self, m: nn.Module) -> None:
        """This method is used to recover the model from a mask."""
        for module in m.modules():
            if is_layer_prunable(module):
                for p in module.parameters():
                    if is_parameter_prunable(p):
                        self.recover_parameter(p)

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the pruner specific arguments to the parent parser."""
        return argparse.ArgumentParser(parents=[parent_parser], add_help=False)

    def summary(self, module: nn.Module) -> int:
        """This method is used to print a summary of the pruning per parameter.

        The pruned weights are recognised as the weights with a value of 0.

        The format of the is: the first column is the name of the parameter,
        the second column is the total number of weights in the parameter, the third column
        is the number of pruned weights in the parameter, the fourth column is the percentage
        of pruned weights in the parameter. The last row is the total number of pruned weights
        in the model, the total number of weights in the model and the percentage of pruned weights.
        Returns the total number of pruned weights.
        """
        table = []
        total_pruned = 0
        total_weights = 0
        non_pruned_parameters = []
        for name, m in module.named_modules():
            if is_layer_prunable(m):
                for parameter_name, p in m.named_parameters():
                    if is_parameter_prunable(p):
                        total_weights += p.numel()
                        pruned = torch.sum(p.data == 0).item()
                        total_pruned += pruned
                        table.append(
                            [
                                f"{name}.{parameter_name}",
                                p.numel(),
                                pruned,
                                pruned / p.numel(),
                            ]
                        )
                    else:
                        non_pruned_parameters.append(f"{name}.{parameter_name}")
            else:
                # If the module is a leaf module add the name to the non-pruned parameters.
                if len(list(m.children())) == 0:
                    non_pruned_parameters.append(name)
        table.append(
            ["Total", total_weights, total_pruned, total_pruned / total_weights]
        )
        logging.info(
            tabulate(
                table,
                headers=["Parameter", "Total", "Pruned", "Pruned [%]"],
                tablefmt="github",
            )
        )
        logging.info(
            f"Non-pruned parameters/layers: {', '.join(non_pruned_parameters)}"
        )
        return total_pruned

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class DummyPruner(BasePruner):
    """This is a dummy pruner class which does not do any pruning."""

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def summary(self, m: nn.Module) -> int:
        logging.info("No pruning was performed.")
        return 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
