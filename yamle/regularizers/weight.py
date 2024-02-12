from typing import Any

from yamle.regularizers.regularizer import BaseRegularizer

import torch
import torch.nn as nn


class L1Regularizer(BaseRegularizer):
    """This is a class for L1 regularization."""

    def __call__(self, model: nn.Module) -> torch.Tensor:
        """This method is used to calculate the regularization loss."""
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for param in self.get_parameters(model):
            loss += torch.sum(torch.abs(param))
        return loss

    def __repr__(self) -> str:
        return f"L1()"


class L2Regularizer(BaseRegularizer):
    """This is a class for L2 regularization."""

    def __call__(self, model: nn.Module) -> torch.Tensor:
        """This method is used to calculate the regularization loss."""
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for param in self.get_parameters(model):
            loss += torch.sum(param**2)
        return loss * 0.5

    def __repr__(self) -> str:
        return f"L2()"


class L1L2Regularizer(BaseRegularizer):
    """This is a class for combined L1 and L2 regularization."""

    def __call__(self, model: nn.Module) -> torch.Tensor:
        """This method is used to calculate the regularization loss."""
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for param in self.get_parameters(model):
            loss += 0.5 * torch.sum(param**2)
            loss += torch.sum(torch.abs(param))
        return loss

    def __repr__(self) -> str:
        return f"L1L2()"


class WeightDecayRegularizer(BaseRegularizer):
    """This is a class for weight decay regularization.

    It is implemented in an inefficient manner to be compatible with any optimizer.

    During the ``__call__`` method, the weights at time ``t`` are cached.
    Then, during the `update_on_step` method, the weights, which were already updated by the optimizer, are further updated by weight decay.

    The weight decay is applied as follows: 

        w_{t+1} = (1 - weight) * w_{t} - \eta * ∇L(w_{t+1})

    Hence, after the optimization step, assuming that only ``w_{t+1} = w_{t} - \eta * ∇L(w_{t+1})`` was applied, 
    we need to apply the ``-weight * w_{t}`` term. The weight is scaled by the learning rate.
    """

    def __call__(self, model: nn.Module) -> torch.Tensor:
        """This method is used to cache all the weight values *before* the optimization step.

        This is done such that the weights can then be updated at the very end of the training batch.
        """
        for param in self.get_parameters(model):
            param._cached_weight = param.data.clone().detach()
        return torch.tensor(0.0, device=next(model.parameters()).device)

    def on_after_training_step(
        self, model: nn.Module, weight: float, lr: float, *args: Any, **kwargs: Any
    ) -> None:
        """This method is used to update the model on a given step."""
        for param in self.get_parameters(model):
            param.data.add_(param._cached_weight, alpha=-weight * lr)
            # Reset the cached weight
            del param._cached_weight

    def __repr__(self) -> str:
        return f"WeightDecay()"
