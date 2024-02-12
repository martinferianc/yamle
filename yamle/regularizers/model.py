from typing import Any

import torch.nn as nn
from yamle.regularizers.regularizer import BaseRegularizer

import torch
import argparse


class ShrinkAndPerturbRegularizer(BaseRegularizer):
    """This is a class for a shrink and perturb regularization.

    It shrinks the weights by a factor of `l` and adds a noise sampled from a
    normal distribution with mean 0 and standard deviation `std` to the weights at
    a certain epoch frequency.

    There is also a second argument which limits the starting epoch and the ending epoch
    within which the shrink and perturb regularization is applied.

    It follows the paper: https://arxiv.org/pdf/1910.08475.pdf

    Args:
        l (float): The factor by which the weights are shrunk.
        std (float): The standard deviation of the normal distribution from which the noise is sampled.
        start_epoch (int): The epoch at which the shrink and perturb regularization starts. Default is 0, which means that the regularization is applied from the beginning of the training.
        end_epoch (int): The epoch at which the shrink and perturb regularization ends. Default is -1, which means that the regularization is applied until the end of the training.
        epoch_frequency (int): The frequency at which the shrink and perturb regularization is applied.
    """

    def __init__(
        self,
        l: float,
        std: float,
        start_epoch: int,
        end_epoch: int,
        epoch_frequency: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert 0 <= l <= 1, f"The shrink factor must be between 0 and 1, but got {l}."

        assert (
            std >= 0
        ), f"The standard deviation of the normal distribution must be non-negative, but got {std}."

        assert (
            start_epoch >= 0
        ), f"The start epoch must be non-negative, but got {start_epoch}."

        assert (
            end_epoch == -1 or end_epoch >= start_epoch
        ), f"The end epoch must be greater than or equal to the start epoch, but got {end_epoch} and {start_epoch}."

        assert (
            epoch_frequency > 0
        ), f"The epoch frequency must be greater than 0, but got {epoch_frequency}."

        self._l = l
        self._std = std
        self._start_epoch = start_epoch
        self._end_epoch = end_epoch
        self._epoch_frequency = epoch_frequency

    def on_after_train_epoch(
        self, model: nn.Module, epoch: int, *args: Any, **kwargs: Any
    ) -> None:
        """Add noise to the weights after a given training epoch.

        For all parameters that require gradients, the weights are shrunk by a factor of `l` and a noise sampled from a
        normal distribution with mean 0 and standard deviation `std` is added to the weights.
        """
        if (
            epoch >= self._start_epoch
            and (epoch <= self._end_epoch or self._end_epoch == -1)
            and epoch % self._epoch_frequency == 0
            and epoch != 0
        ):
            for param in model.parameters():
                if param.requires_grad:
                    param.data = (
                        param.data * self._l + torch.randn_like(param.data) * self._std
                    )

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """This method is used to add specific arguments to the parser."""
        parser = super(
            ShrinkAndPerturbRegularizer, ShrinkAndPerturbRegularizer
        ).add_specific_args(parser)
        parser.add_argument(
            "--regularizer_l",
            type=float,
            default=0.1,
            help="The factor by which the weights are shrunk.",
        )
        parser.add_argument(
            "--regularizer_std",
            type=float,
            default=0.1,
            help="The standard deviation of the normal distribution from which the noise is sampled.",
        )
        parser.add_argument(
            "--regularizer_start_epoch",
            type=int,
            default=0,
            help="The epoch at which the shrink and perturb regularization starts.",
        )
        parser.add_argument(
            "--regularizer_end_epoch",
            type=int,
            default=-1,
            help="The epoch at which the shrink and perturb regularization ends.",
        )
        parser.add_argument(
            "--regularizer_epoch_frequency",
            type=int,
            default=1,
            help="The frequency at which the shrink and perturb regularization is applied.",
        )
        return parser
