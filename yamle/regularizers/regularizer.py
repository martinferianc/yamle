from typing import List, Union, Any, Tuple
from abc import ABC
import torch
import torch.nn as nn
import argparse

from yamle.defaults import DISABLED_REGULARIZER_KEY


class BaseRegularizer(ABC):
    """This is a general class for regularizers applied to the model (L1, L2, etc.)."""

    def __call__(self, model: Union[nn.Module, torch.Tensor]) -> torch.Tensor:
        """This method is used to calculate the regularization loss."""
        return torch.tensor(0.0, device=next(model.parameters()).device)

    def get_parameters(self, model: nn.Module) -> List[nn.Parameter]:
        """This method is used to get the parameters of the model that should be regularized."""
        params = []
        for param in model.parameters():
            if param.requires_grad:
                if hasattr(param, DISABLED_REGULARIZER_KEY) and getattr(
                    param, DISABLED_REGULARIZER_KEY
                ):
                    continue
                params.append(param)
        return params

    def get_names(self, model: nn.Module) -> Tuple[List[str], List[str]]:
        """This method is used to get the names of the parameters of the model that should and should not be regularized."""
        regularized = []
        not_regularized = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if hasattr(param, DISABLED_REGULARIZER_KEY) and getattr(
                    param, DISABLED_REGULARIZER_KEY
                ):
                    not_regularized.append(name)
                else:
                    regularized.append(name)
            else:
                not_regularized.append(name)
        return regularized, not_regularized

    def on_after_training_step(
        self, model: nn.Module, *args: Any, **kwargs: Any
    ) -> None:
        """This method is used to update the model after a given training step.

        It can be used to implement a weight decay strategy, e.g. update the weights after each training
        batch by decaying them with a given factor multiplied by the learning rate.
        """
        pass

    def on_after_backward(self, model: nn.Module, *args: Any, **kwargs: Any) -> None:
        """This method is used to update the model after the backward pass.

        It can be used to update the model after the backward pass, e.g. add noise to the gradients.
        """
        pass

    def on_after_train_epoch(self, model: nn.Module, *args: Any, **kwargs: Any) -> None:
        """This method is used to update the model after a given training epoch.

        It can be used to add noise to the model after each training epoch.
        """
        pass

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add specific arguments to the parser."""
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        return parser

    def __repr__(self) -> str:
        return f"Regularizer()"


class DummyRegularizer(BaseRegularizer):
    """This is a class for a dummy regularizer that does nothing."""

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """This method is used to calculate the regularization loss."""
        for arg in args:
            if isinstance(arg, torch.Tensor):
                device = arg.device
                break
            elif isinstance(arg, nn.Module):
                device = next(arg.parameters()).device
                break
        return torch.tensor(0.0, device=device)

    def __repr__(self) -> str:
        return "DummyRegularizer()"
