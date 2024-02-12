from typing import Any, List, Optional

import torch
import argparse

from yamle.losses.loss import BaseLoss
from yamle.defaults import TINY_EPSILON


class SoftIntersectionOverUnionLoss(BaseLoss):
    """This defines the soft intersection over union loss for semantic segmentation.

    It assumes that the input shape is `(batch_size, num_members, num_classes, height, width)`.
    No matter what the reduction it is always averaged over the `num_members`.

    The input is assumed to be probabilities.

    The loss can also be weighted by a weight tensor of shape `(batch_size)`.

    Args:
        factor (float): The softness factor. Defaults to 1.0.
        ignore_indices (List[int]): The indices to ignore. Defaults to [].
    """

    def __init__(
        self, factor: float = 1.0, ignore_indices: List[int] = [], **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._factor = factor
        self._ignore_indices = ignore_indices

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
            sample_loss = self._soft_iou(y_hat[:, i], y)
            loss += self._process_sample_loss(sample_loss, i, weights)
        return self._process_member_loss(loss, num_members)

    def _soft_iou(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """This method is used to compute the soft IoU loss."""
        if y_hat.shape != y.shape:
            y_one_hot = torch.zeros(
                y_hat.shape, device=y_hat.device, requires_grad=False
            ).long()
            y = y_one_hot.scatter_(1, y.unsqueeze(1), 1)
        intersection = torch.sum(y_hat * y, dim=[2, 3])
        union = torch.sum(y_hat, dim=[2, 3]) + torch.sum(y, dim=[2, 3]) - intersection
        if self._ignore_indices is not None:
            for i in self._ignore_indices:
                intersection[:, i] = 0
                union[:, i] = 0
        total_classes = y_hat.shape[1] - len(self._ignore_indices)
        return 1 - torch.sum(
            (intersection + self._factor) / (union + self._factor) / total_classes,
            dim=1,
        )

    def __repr__(self) -> str:
        return f"SoftIntersectionOverUnionLoss(reduction_per_sample={self._reduction_per_sample}, reduction_per_member={self._reduction_per_member}, reduction_per_feature={self._reduction_per_feature}, factor={self._factor}, ignore_indices={self._ignore_indices})"

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the loss specific arguments to the parent parser."""
        parser = super(
            SoftIntersectionOverUnionLoss, SoftIntersectionOverUnionLoss
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--loss_factor", type=float, default=1.0, help="The softness factor."
        )
        parser.add_argument(
            "--loss_ignore_indices",
            type=str,
            default="[]",
            help="The indices to ignore.",
        )
        return parser


class FocalLoss(BaseLoss):
    """This defines the focal loss for semantic segmentation.

    It assumes that the input shape is `(batch_size, num_members, num_classes, height, width)`.
    No matter what the reduction it is always averaged over the `num_members`.

    The input is assumed to be probabilities.

    The loss can also be weighted by a weight tensor of shape `(batch_size)`.

    Args:
        alpha (float): The alpha factor. Defaults to 0.25.
        gamma (float): The gamma factor. Defaults to 2.0.
        ignore_indices (List[int]): The indices to ignore. Defaults to [].
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_indices: List[int] = [],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_indices = ignore_indices

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
            sample_loss = self._focal_loss(y_hat[:, i], y)
            loss += self._process_sample_loss(sample_loss, i, weights)
        return self._process_member_loss(loss, num_members)

    def _focal_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """This method is used to compute the focal loss."""
        if y_hat.shape != y.shape:
            y_one_hot = torch.zeros(y_hat.shape, device=y_hat.device).long()
            y = y_one_hot.scatter_(1, y.unsqueeze(1), 1)

        # Compute the focal loss
        focal = (
            -self._alpha
            * y
            * torch.pow(1 - y_hat + TINY_EPSILON, self._gamma)
            * torch.log(y_hat + TINY_EPSILON)
        )
        focal = torch.sum(focal, dim=(2, 3))

        loss = 0.0
        for i in range(y_hat.shape[1]):
            if i not in self._ignore_indices:
                loss += focal[:, i]
        return loss / (y_hat.shape[1] - len(self._ignore_indices))

    def __repr__(self) -> str:
        return f"FocalLoss(reduction_per_sample={self._reduction_per_sample}, reduction_per_member={self._reduction_per_member}, reduction_per_feature={self._reduction_per_feature}, alpha={self._alpha}, gamma={self._gamma}, ignore_indices={self._ignore_indices})"

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the loss specific arguments to the parent parser."""
        parser = super(FocalLoss, FocalLoss).add_specific_args(parent_parser)
        parser.add_argument(
            "--loss_alpha", type=float, default=0.25, help="The alpha factor."
        )
        parser.add_argument(
            "--loss_gamma", type=float, default=2.0, help="The gamma factor."
        )
        parser.add_argument(
            "--loss_ignore_indices",
            type=str,
            default="[]",
            help="The indices to ignore.",
        )
        return parser
