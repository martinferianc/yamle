from typing import Any, Optional

import torch
import argparse
import torch.nn.functional as F
from yamle.losses.loss import BaseLoss
from yamle.defaults import TINY_EPSILON


class NoiseContrastiveEstimatorLoss(BaseLoss):
    """This defines the noise contrastive estimation (NCE) loss.

    It assumes that the input shape is `(batch_size, num_members, num_classes)`.
    No matter what the reduction it is always averaged over the `num_members`.

    Args:
        temperature (float): The temperature to use for the softmax. Defaults to 1.0.
        similarity (str): The similarity function to use. Defaults to `cosine`. Choices are `cosine` and `dot`.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        similarity: str = "cosine",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert similarity in [
            "cosine",
            "dot",
        ], f"Similarity function must be either `cosine` or `dot`. Got {similarity}."
        assert (
            temperature > 0
        ), f"Temperature must be greater than 0. Got {temperature}."
        self._similarity = similarity
        self._temperature = temperature

    def _cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the cosine similarity between two tensors."""
        return F.cosine_similarity(x, y, dim=-1, eps=TINY_EPSILON)

    def _dot_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the dot similarity between two tensors."""
        return torch.matmul(x, y.transpose(-1, -2))

    def _loss(
        self,
        y_hat: torch.Tensor,
        y_hat_positive: torch.Tensor,
        y_hat_negative: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the NCE loss.

        The `y_hat` tensor contains the default predictions for `x` samples. The shape is `(batch_size, num_classes)`.
        The `y_hat_positive` tensor contains the predictions for the positive samples. The shape is `(batch_size, num_classes)`.
        The `y_hat_negative` tensor contains the predictions for the negative samples. The shape is `(batch_size, K, num_classes)`.
        """
        assert (
            y_hat.shape == y_hat_positive.shape
        ), f"The shapes of the predictions do not match. Got {y_hat.shape}, {y_hat_positive.shape}."
        assert (
            y_hat.shape[0] == y_hat_negative.shape[0]
        ), f"The batch sizes of the predictions do not match. Got {y_hat.shape[0]}, {y_hat_negative.shape[0]}."

        if self._similarity == "cosine":
            similarity_fn = self._cosine_similarity
        elif self._similarity == "dot":
            similarity_fn = self._dot_similarity
        else:
            raise NotImplementedError(
                f"Similarity function {self._similarity} is not implemented."
            )

        similarity_positive = (
            similarity_fn(y_hat, y_hat_positive) / self._temperature
        ).exp()
        similarity_negative = (
            similarity_fn(
                y_hat.unsqueeze(1).repeat(1, y_hat_negative.shape[1], 1), y_hat_negative
            )
            / self._temperature
        ).exp()

        loss = -torch.log(
            similarity_positive
            / (
                similarity_positive
                + torch.sum(similarity_negative, dim=1)
                + TINY_EPSILON
            )
            + TINY_EPSILON
        )
        return loss

    def __call__(
        self,
        y_hat: torch.Tensor,
        y_hat_positive: torch.Tensor,
        y_hat_negative: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """This method is used to compute the NCE loss."""
        num_members = y_hat_positive.shape[1]
        loss = 0.0
        for i in range(num_members):
            sample_loss = self._loss(
                y_hat[:, i], y_hat_positive[:, i], y_hat_negative[:, i]
            )
            loss += self._process_sample_loss(sample_loss, i, weights)

        return self._process_member_loss(loss, num_members)

    def __repr__(self) -> str:
        return f"NoiseContrastiveEstimatorLoss(reduction_per_sample={self._reduction_per_sample}, reduction_per_member={self._reduction_per_member}, reduction_per_feature={self._reduction_per_feature}, similarity={self._similarity})"

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = super(
            NoiseContrastiveEstimatorLoss, NoiseContrastiveEstimatorLoss
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--loss_temperature",
            type=float,
            default=1.0,
            help="The temperature to use for the softmax.",
        )
        parser.add_argument(
            "--loss_similarity",
            type=str,
            choices=["cosine", "dot"],
            default="cosine",
            help="The similarity function to use.",
        )
        return parser
