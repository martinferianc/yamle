from typing import Any

import torch.nn as nn
from yamle.defaults import TINY_EPSILON
from yamle.regularizers.regularizer import BaseRegularizer

import torch
import argparse


class GradientNoiseRegularizer(BaseRegularizer):
    """This is a class for a gradient noise regularization.

    It adds a noise sampled from a normal distribution with mean 0 and standard deviation `std` to the gradient.

    It follows the paper: https://arxiv.org/pdf/1511.06807.pdf

    Args:
        eta (float): The standard deviation of the normal distribution from which the noise is sampled.
        gamma (float): The factor by which the noise is multiplied.
    """

    def __init__(self, eta: float, gamma: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert (
            eta >= 0
        ), "The standard deviation of the normal distribution must be non-negative."
        assert (
            eta > 0
        ), "The standard deviation of the normal distribution must be positive."
        assert 0 <= gamma <= 1, f"The factor must be between 0 and 1, but got {gamma}."
        self._eta = eta
        self._gamma = gamma

    def _var(self, epoch: int) -> float:
        """Return the variance of the noise at a given epoch."""
        return self._eta / ((1 + epoch) ** self._gamma + TINY_EPSILON)

    def on_after_backward(
        self, model: nn.Module, epoch: int, *args: Any, **kwargs: Any
    ) -> None:
        """Add noise to the gradients after the backward pass."""
        var = self._var(epoch)
        for param in model.parameters():
            if param.grad is not None:
                param.grad += torch.randn_like(param.grad) * var

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """This method is used to add specific arguments to the parser."""
        parser = super(
            GradientNoiseRegularizer, GradientNoiseRegularizer
        ).add_specific_args(parser)
        parser.add_argument(
            "--regularizer_eta",
            type=float,
            default=0.1,
            help="The standard deviation of the normal distribution from which the noise is sampled.",
        )
        parser.add_argument(
            "--regularizer_gamma",
            type=float,
            default=0.55,
            help="The factor by which the noise is multiplied.",
        )
        return parser

    def __repr__(self) -> str:
        return f"GradientNoiseRegularizer(eta={self._eta}, gamma={self._gamma})"
