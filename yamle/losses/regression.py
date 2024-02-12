from typing import Any, List, Optional

import torch
import torch.nn as nn
import argparse

from yamle.losses.loss import BaseLoss
from yamle.defaults import TINY_EPSILON, REGRESSION_KEY, RECONSTRUCTION_KEY, DEPTH_ESTIMATION_KEY


class GaussianNegativeLogLikelihoodLoss(BaseLoss):
    """This defines the base negative log-likelihood loss.

    It assumes that the input shape is `(batch_size, num_members, 1)` of only mean is predicted
    or `(batch_size, num_members, 2)` if mean and variance are predicted.
    The first feature is the mean and the second is the variance.
    No matter what the reduction it is always averaged over the `num_members`.

    The loss can also be weighted by a weight tensor of shape `(batch_size)`.

    Args:
        flatten (bool): Whether to flatten the input. Defaults to `False`.
    """
    
    tasks = [REGRESSION_KEY, DEPTH_ESTIMATION_KEY, RECONSTRUCTION_KEY]

    def __init__(self, flatten: bool = False, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._loss = nn.GaussianNLLLoss(reduction="none")
        self._flatten = flatten

    def __call__(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """This method is used to compute the loss."""
        if self._flatten:
            B = y_hat.shape[0]
            M = y_hat.shape[1]
            y_hat = y_hat.reshape(B, M, 2, -1)
            y = y.reshape(B, -1)
        num_members = y_hat.shape[1]
        loss = 0.0
        mean = y_hat[:, :, 0]
        variance = y_hat[:, :, 1] if y_hat.shape[2] == 2 else torch.ones_like(mean)
        assert torch.all(
            variance > 0.0
        ), f"The variance is not positive. Got {variance[variance <= 0.0]}."

        for i in range(num_members):
            sample_loss = self._loss(
                mean[:, i].squeeze(),
                y.squeeze(),
                variance[:, i].squeeze() + TINY_EPSILON,
            )
            loss += self._process_sample_loss(sample_loss, i, weights)
        loss = self._process_member_loss(loss, num_members)
        return self._process_feature_loss(loss, dim=list(range(0, loss.ndim)))

    def __repr__(self) -> str:
        return f"GaussianNegativeLogLikelihoodLoss(reduction_per_member={self._reduction_per_member}, reduction_per_sample={self._reduction_per_sample}, reduction_per_feature={self._reduction_per_feature})"

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the specific arguments for this loss."""
        parser = super(
            GaussianNegativeLogLikelihoodLoss, GaussianNegativeLogLikelihoodLoss
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--loss_flatten",
            type=int,
            help="Whether to flatten the input.",
            default=0,
            choices=[0, 1],
        )
        return parser


class MeanSquaredError(BaseLoss):
    """This defines the mean squared error loss.

    It assumes that the input shape is `(batch_size, num_members, 1)`.
    If there are more features, only the first feature is used.
    No matter what the reduction it is always averaged over the `num_members`.

    The loss can also be weighted by a weight tensor of shape `(batch_size)`.

    Args:
        flatten (bool): Whether to flatten the input. Defaults to False.
    """
    
    tasks = [REGRESSION_KEY, DEPTH_ESTIMATION_KEY, RECONSTRUCTION_KEY]

    def __init__(self, flatten: bool = False, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._loss = nn.MSELoss(reduction="none")
        self._flatten = flatten

    def __call__(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """This method is used to compute the loss."""
        if self._flatten:
            B = y_hat.shape[0]
            M = y_hat.shape[1]
            mean_and_variance = y_hat.shape[2]
            y_hat = y_hat.reshape(B, M, mean_and_variance, -1)
            y = y.reshape(B, -1)

        num_members = y_hat.shape[1]
        loss = 0.0
        for i in range(num_members):
            sample_loss = self._loss(y_hat[:, i, 0], y)
            loss += self._process_sample_loss(sample_loss, i, weights)
        loss = self._process_member_loss(loss, num_members)
        return self._process_feature_loss(loss, dim=list(range(0, loss.ndim)))

    def __repr__(self) -> str:
        return f"MeanSquaredError(reduction_per_sample={self._reduction_per_sample}, reduction_per_member={self._reduction_per_member}, reduction_per_feature={self._reduction_per_feature})"

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the specific arguments for this loss."""
        parser = super(MeanSquaredError, MeanSquaredError).add_specific_args(
            parent_parser
        )
        parser.add_argument(
            "--loss_flatten",
            type=int,
            help="Whether to flatten the input.",
            default=0,
            choices=[0, 1],
        )
        return parser


class QuantileRegressionLoss(BaseLoss):
    """This defines the quantile regression loss.

    It assumes that the input shape is `(batch_size, num_members, quantiles)`
    The losses for different `quantiles` are averaged.

    The loss can also be weighted by a weight tensor of shape `(batch_size)`.

    Args:
        quantiles (List[float]): The quantiles to be used for computing the loss.
    """
    
    tasks = [REGRESSION_KEY, DEPTH_ESTIMATION_KEY, RECONSTRUCTION_KEY]

    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        for quantile in quantiles:
            assert (
                0.0 < quantile < 1.0
            ), f"The quantile must be between 0 and 1. Got {quantile}."
        self._quantiles = quantiles

    def _loss(
        self, y_hat: torch.Tensor, y: torch.Tensor, quantile: float
    ) -> torch.Tensor:
        """This method computes the loss for a single quantile."""
        return torch.max(quantile * (y - y_hat), (quantile - 1) * (y - y_hat))

    def __call__(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """This method is used to compute the loss."""
        num_members = y_hat.shape[1]
        loss = 0.0
        for i in range(num_members):
            for j, quantile in enumerate(self._quantiles):
                quantile_loss = self._loss(y_hat[:, i, j], y, quantile)
                loss += self._process_sample_loss(quantile_loss, i, weights)
            loss /= len(self._quantiles)
        return self._process_member_loss(loss, num_members)

    def __repr__(self) -> str:
        return f"QuantileRegressionLoss(quantiles={self._quantiles}, reduction_per_sample={self._reduction_per_sample}, reduction_per_member={self._reduction_per_member}, reduction_per_feature={self._reduction_per_feature})"

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the specific arguments for this loss."""
        parser = super(
            QuantileRegressionLoss, QuantileRegressionLoss
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--loss_quantiles",
            type=str,
            help="The quantiles to be used for computing the loss.",
            default="[0.1,0.5,0.9]",
        )
        return parser
