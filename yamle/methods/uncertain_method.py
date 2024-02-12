from typing import Any, List, Dict, Optional
import torch
import argparse
import torchmetrics

from yamle.methods.method import BaseMethod
from yamle.models.specific.svi_utils import KulbackLeiblerParameterLoss
from yamle.defaults import (
    LOSS_KEY,
    LOSS_KL_KEY,
    TRAIN_KEY,
    VALIDATION_KEY,
    TEST_KEY,
    MEMBERS_DIM,
    MIN_TENDENCY,
)
from yamle.evaluation.metrics.algorithmic import metrics_factory


class MemberMethod(BaseMethod):
    """This class is the extension of the base method for which the prediciton is performed using multiple members.

    Args:
        num_members (int): The number of members to be used for the prediction.
    """

    training_num_members = 1

    def __init__(self, num_members: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._num_members = num_members

    def _create_metrics(self, metrics_kwargs: Dict[str, Any]) -> None:
        """This method is used to create the metrics to be used for training, validation and testing."""
        self.metrics = {
            TRAIN_KEY: metrics_factory(**metrics_kwargs, per_member=False),
            VALIDATION_KEY: metrics_factory(**metrics_kwargs, per_member=True),
            TEST_KEY: metrics_factory(**metrics_kwargs, per_member=True),
        }

    def _loss_per_member(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        weights_per_sample: Optional[torch.Tensor] = None,
        weights_per_member: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """A helper method to calculate the loss per member and averaged per member.

        The `y_hat` is assumed to have the shape `(batch_size, num_members, target_dim)`.
        The `y` is assumed to have the shape `(batch_size, num_members)`.
        The `weights_per_sample` is assumed to have the shape `(batch_size, num_members)`.
        The `weights_per_member` is assumed to have the shape `(num_members)`.
        """
        num_members = y_hat.shape[1]
        assert (
            y.shape[1] == num_members
        ), f"The number of members in the `y_hat` and `y` should be the same. Got {num_members} and {y.shape[1]}."
        if weights_per_member is None:
            weights_per_member = torch.ones(num_members, device=y.device, dtype=y.dtype)
        assert (
            len(weights_per_member) == num_members
        ), f"The number of members in the `y_hat` and `weights_per_member` should be the same. Got {num_members} and {len(weights_per_member)}."
        loss = 0.0
        for i in range(num_members):
            weights_per_sample_per_member = (
                None if weights_per_sample is None else weights_per_sample[:, i]
            )
            loss += weights_per_member[i] * self._loss(
                y_hat[:, i].unsqueeze(1), y[:, i], weights_per_sample_per_member
            )
        return loss

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the specific arguments for the class."""
        parser = super(MemberMethod, MemberMethod).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_num_members",
            type=int,
            default=1,
            help="The number of members to be used for the prediction. Default: 1.",
        )
        return parser


class MCSamplingMethod(MemberMethod):
    """This class is the extension of the base method for which the prediciton is performed using Monte Carlo sampling."""

    def _create_metrics(self, metrics_kwargs: Dict[str, Any]) -> None:
        """This method is used to create the metrics to be used for training, validation and testing.

        For the Monte Carlo sampling, we do not care about the individual members.
        """
        self.metrics = {
            TRAIN_KEY: metrics_factory(**metrics_kwargs, per_member=False),
            VALIDATION_KEY: metrics_factory(**metrics_kwargs, per_member=False),
            TEST_KEY: metrics_factory(**metrics_kwargs, per_member=False),
        }

    def _predict(self, x: torch.Tensor, **forward_kwargs: Any) -> torch.Tensor:
        """This method is used to perform a forward pass of the model.

        It is done with respect to the number of samples specified in the constructor.
        """
        outputs = []
        num_members = self.training_num_members if self.training else self._num_members
        for _ in range(num_members):
            outputs.append(super()._predict(x, **forward_kwargs))
        return torch.cat(outputs, dim=MEMBERS_DIM)


class SVIMethod(MCSamplingMethod):
    """This class is the extension of the base method for stochastic variational inference methods.
    That need to minimize the KL divergence between the prior and the posterior for their parameters.

    Args:
        alpha (float): The alpha to be used for the trade-off between the likelihood and the KL divergence.
    """

    def __init__(self, alpha: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert alpha >= 0, "The alpha must be non-negative."
        self._alpha = alpha
        self._kl_loss = KulbackLeiblerParameterLoss(self.model)

    def _create_metrics(self, metrics_kwargs: Dict[str, Any]) -> None:
        """This method is used to create the metrics to be used for training, validation and testing."""
        super()._create_metrics(metrics_kwargs)
        self._add_additional_metrics({LOSS_KL_KEY: torchmetrics.MeanMetric()}, tendencies=[MIN_TENDENCY])

    def _training_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """This method is used to perform a single training step.

        It assumes that the batch has a shape `(batch_size, num_features)`.
        It assumes that the output of the model has a shape `(batch_size, n_samples, num_classes)`.
        """
        output = super()._training_step(batch, batch_idx)
        kl_divergence = self._kl_loss() * self._alpha
        output[LOSS_KEY] += kl_divergence
        output[LOSS_KL_KEY] = kl_divergence
        return output

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = super(SVIMethod, SVIMethod).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_alpha",
            type=float,
            default=0.001,
            help="The alpha to be used for the trade-off between the likelihood and the prior. Default: 0.001.",
        )
        return parser
