from typing import Any, Optional, Union, List

import torch
import argparse

from yamle.losses.loss import BaseLoss
from yamle.defaults import TINY_EPSILON, CLASSIFICATION_KEY, TEXT_CLASSIFICATION_KEY, SEGMENTATION_KEY


def one_hot(
    y: torch.Tensor, num_classes: int, label_smoothing: float = 0.0
) -> torch.Tensor:
    """One-hot encodes the target and directly applies label smoothing.

    Taken from: https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
    """
    return (
        torch.zeros(size=(y.size(0), num_classes, *y.size()[1:]), device=y.device)
        .fill_(label_smoothing / (num_classes - 1))
        .scatter_(1, y.data.unsqueeze(1), 1.0 - label_smoothing)
    )


class CrossEntropyLoss(BaseLoss):
    """This defines the base cross-entropy loss.

    It assumes that the input shape is `(batch_size, num_members, num_classes)`.
    No matter what the reduction it is always averaged over the `num_members`.
    The target is assumed to be of shape `(batch_size)`, it needs to be one-hot encoded.

    The input is assumed to be probabilities.

    The loss can also be weighted by a weight tensor of shape `(batch_size)`.

    Args:
        label_smoothing (float): The amount of label smoothing to apply.
        one_hot_target (bool): Whether the target is already one-hot encoded. Defaults to `False`.
        class_weights (Optional[Union[torch.Tensor, List[float]]]): The weights to apply to each class. Defaults to `None`.
        flatten (bool): Whether to flatten the predictions and the targets. Defaults to `False`.
    """
    
    tasks = [CLASSIFICATION_KEY, SEGMENTATION_KEY]

    def __init__(
        self,
        label_smoothing: float = 0.0,
        one_hot_target: bool = False,
        class_weights: Optional[Union[torch.Tensor, List[float]]] = None,
        flatten: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert 0.0 <= label_smoothing <= 1.0, "Label smoothing must be between 0 and 1."
        self._label_smoothing = label_smoothing
        self._one_hot_target = one_hot_target
        self._class_weights = class_weights
        self._flatten = flatten

    def _loss(self, y_hat: torch.Tensor, y_one_hot: torch.Tensor) -> torch.Tensor:
        """Computes the negative log likelihood loss."""
        # Check if `y_hat` is already probabilities.
        assert (
            y_hat.shape == y_one_hot.shape
        ), f"The shapes of the predictions and the targets do not match. Got {y_hat.shape} and {y_one_hot.shape}."
        assert torch.all(y_hat >= 0.0) and torch.all(
            y_hat <= 1.1
        ), f"The predictions are not probabilities. Got {y_hat[y_hat < 0.0]} or {y_hat[y_hat > 1.1]}."
        # Check that y_hat sums approximately to 1.
        assert torch.allclose(
            y_hat.sum(dim=1), torch.ones_like(y_hat.sum(dim=1)), atol=1e-2
        ), f"The predictions do not sum to 1. Got {y_hat.sum(dim=1)}."
        class_weights = None
        if self._class_weights is not None:
            if isinstance(self._class_weights, list):
                self._class_weights = torch.tensor(
                    self._class_weights, device=y_hat.device
                )
            assert (
                self._class_weights.shape[0] == y_hat.shape[-1]
            ), f"The number of class weights ({self._class_weights.shape[0]}) does not match the number of classes ({y_hat.shape[-1]})."
            # Check the device.
            self._class_weights = self._class_weights.to(y_hat.device)
            class_weights = self._class_weights
        else:
            class_weights = torch.ones(y_hat.shape[-1], device=y_hat.device)

        return -torch.sum(
            y_one_hot * class_weights * torch.log(y_hat + TINY_EPSILON), dim=1
        )

    def __call__(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """This method is used to compute the loss."""
        num_members = y_hat.shape[1]
        if self._flatten:
            B = y_hat.shape[0]
            C = y_hat.shape[2]
            y_hat = y_hat.reshape(B, num_members, C, -1)
            y = y.reshape(B, -1)
        loss = 0.0
        one_hot_y = (
            one_hot(y, y_hat.shape[-1], self._label_smoothing)
            if not self._one_hot_target
            else y
        )
        for i in range(num_members):
            sample_loss = self._loss(y_hat[:, i], one_hot_y)
            loss += self._process_sample_loss(sample_loss, i, weights)

        return self._process_member_loss(loss, num_members)

    def __repr__(self) -> str:
        return f"CrossEntropyLoss(reduction_per_sample={self._reduction_per_sample}, reduction_per_member={self._reduction_per_member}, reduction_per_feature={self._reduction_per_feature}, label_smoothing={self._label_smoothing}, one_hot_target={self._one_hot_target}, class_weights={self._class_weights}"

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = super(CrossEntropyLoss, CrossEntropyLoss).add_specific_args(
            parent_parser
        )
        parser.add_argument(
            "--loss_label_smoothing",
            type=float,
            default=0.0,
            help="The amount of label smoothing to apply.",
        )
        parser.add_argument(
            "--loss_one_hot_target",
            type=int,
            choices=[0, 1],
            default=0,
            help="Whether the target is already one-hot encoded.",
        )
        parser.add_argument(
            "--loss_class_weights",
            type=str,
            default=None,
            help="The weights to apply to each class.",
        )
        parser.add_argument(
            "--loss_flatten",
            type=int,
            choices=[0, 1],
            default=0,
            help="Whether to flatten the predictions and the targets.",
        )
        return parser


class TextCrossEntropyLoss(CrossEntropyLoss):
    """This defines the base cross-entropy loss.

    It assumes that the input shape is `(batch_size, num_members, sequence_length, num_classes)`.
    No matter what the reduction it is always averaged over the `num_members`.
    The target is assumed to be of shape `(batch_size, sequence_length)`.

    The input is assumed to be probabilities.

    The loss can also be weighted by a weight tensor of shape `(batch_size)`.

    Args:
        label_smoothing (float): The amount of label smoothing to apply.
    """
    
    tasks = [TEXT_CLASSIFICATION_KEY]

    def _loss(self, y_hat: torch.Tensor, y_one_hot: torch.Tensor) -> torch.Tensor:
        """Computes the negative log likelihood loss."""
        # Check if `y_hat` is already probabilities.
        assert (
            y_hat.shape == y_one_hot.shape
        ), f"The shapes of the predictions and the targets do not match. Got {y_hat.shape} and {y_one_hot.shape}."
        assert torch.all(y_hat >= 0.0) and torch.all(
            y_hat <= 1.1
        ), f"The predictions are not probabilities. Got {y_hat[y_hat < 0.0]} or {y_hat[y_hat > 1.1]}."
        return -torch.sum(y_one_hot * torch.log(y_hat + TINY_EPSILON), dim=2)

    def __call__(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """This method is used to compute the loss."""
        num_members = y_hat.shape[1]
        loss = 0.0
        y_original_shape = y.shape
        one_hot_y = one_hot(y.reshape(-1), y_hat.shape[-1], self._label_smoothing)
        one_hot_y = one_hot_y.reshape(y_original_shape[0], y_original_shape[1], -1)

        for i in range(num_members):
            sample_loss = self._loss(y_hat[:, i], one_hot_y).mean(dim=1)
            loss += self._process_sample_loss(sample_loss, i, weights)

        return self._process_member_loss(loss, num_members)

    def __repr__(self) -> str:
        return f"TextCrossEntropyLoss(reduction_per_sample={self._reduction_per_sample}, reduction_per_member={self._reduction_per_member}, reduction_per_feature={self._reduction_per_feature}, label_smoothing={self._label_smoothing})"
