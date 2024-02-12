from yamle.utils.operation_utils import average_predictions
from yamle.defaults import (
    LOSS_KEY,
    TARGET_KEY,
    TINY_EPSILON,
    PREDICTION_KEY,
    MEAN_PREDICTION_KEY,
    PREDICTION_PER_MEMBER_KEY,
    TARGET_PER_MEMBER_KEY,
    CLASSIFICATION_KEY,
    SEGMENTATION_KEY,
    TRAIN_KEY,
    VALIDATION_KEY,
    TEST_KEY,
    MEMBERS_DIM,
    INPUT_KEY,
    MAX_TENDENCY,
)
from yamle.evaluation.metrics.algorithmic import metrics_factory
from yamle.methods.uncertain_method import MemberMethod
from typing import Any, List, Dict, Optional
import torch
import argparse
import logging
import torchmetrics

logging = logging.getLogger("pytorch_lightning")


class EarlyExitMethod(MemberMethod):
    """This class is the extension of the base method for which the prediciton is performed through the method of:
    early exits from the architecture which are auxiliary heads that are added to the model at different depths.

    Args:
        alpha (float): The term to scale the loss of the auxiliary heads. Default: 1.0.
        beta (float): The term to scale the diversity loss which is a cross entropy between all exit pairs. Default: 0.0.
        gamma (float): The term to scale the hidden feature size when creating the auxiliary heads. Default: 0.3.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 0.3,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not hasattr(kwargs["model"], "_depth"):
            raise ValueError(
                "The model should have a `_depth` attribute which is the number of hidden layers."
            )

        kwargs["num_members"] = kwargs["model"]._depth
        kwargs["metrics_kwargs"]["num_members"] = kwargs["model"]._depth
        self._depth = kwargs["model"]._depth
        super().__init__(*args, **kwargs)

        assert 0 <= alpha <= 1, f"alpha should be between 0 and 1, got {alpha}"
        assert beta >= 0, f"beta should be non-negative, got {beta}"
        assert gamma >= 0, f"gamma should be non-negative, got {gamma}"

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

        if beta > 0:
            assert self._task in [
                CLASSIFICATION_KEY,
                SEGMENTATION_KEY,
            ], f"Task {self._task} is not supported."

        logging.warning(
            f"The number of members is set to {self._num_members} because of the depth of the model."
        )

        self.model.add_method_specific_layers(method="early_exit", gamma=gamma)

    def _create_metrics(self, metrics_kwargs: Dict[str, Any]) -> None:
        """This method is used to create the metrics to be used for training, validation and testing."""
        self.metrics = {
            TRAIN_KEY: metrics_factory(**metrics_kwargs, per_member=True),
            VALIDATION_KEY: metrics_factory(**metrics_kwargs, per_member=True),
            TEST_KEY: metrics_factory(**metrics_kwargs, per_member=True),
        }
        self._add_additional_metrics(
            {f"{LOSS_KEY}_diversity": torchmetrics.MeanMetric()},
            tendencies=[MAX_TENDENCY],
        )

    def _diversity_loss(self, y_hat: torch.Tensor) -> torch.Tensor:
        """This method is used to compute the diversity loss between all pairs of the predictions.

        Args:
            y_hat (torch.Tensor): The predictions of the model with a shape `(batch_size, num_members, num_classes)`.
        """
        if self._beta == 0 and self._task not in [CLASSIFICATION_KEY, SEGMENTATION_KEY]:
            return torch.zeros(1, device=y_hat.device)
        loss = 0.0
        for i in range(self._num_members):
            for j in range(i + 1, self._num_members):
                loss += -torch.sum(
                    y_hat[:, i] * torch.log(y_hat[:, j] + TINY_EPSILON), dim=1
                ).mean()
        return loss / (self._num_members * (self._num_members - 1) / 2)

    def _predict(self, x: torch.Tensor, **forward_kwargs: Any) -> torch.Tensor:
        """This method is used to perform a forward pass of the model."""
        last_layer, stages = self.model(x, staged_output=True, **forward_kwargs)
        # Since the last layer uses the last hidden layer
        # we can remove it
        stages = stages[:-1]
        outputs = []
        for i, h in enumerate(stages):
            h = self.model._reshaping_layers[i](h)
            h = self.model._output_activation(h)
            outputs.append(h)
        outputs.append(last_layer)
        return torch.stack(outputs, dim=MEMBERS_DIM)

    def _step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
        phase: str = TRAIN_KEY,
    ) -> Dict[str, torch.Tensor]:
        """This method is used to perform a single training step.

        It assumes that the batch has a shape `(batch_size, num_features)`.
        It assumes that the output of the model has a shape `(batch_size, n_samples, num_classes)`.
        """
        x, y = batch
        y = torch.stack([y for _ in range(self._num_members)], dim=MEMBERS_DIM)

        y_hat = self._predict(x)

        outputs = {}
        if self.training:
            loss = self._loss_per_member(
                y_hat, y, weights_per_member=[self._alpha] * (self._depth - 1) + [1.0]
            )
            loss_diversity = self._diversity_loss(y_hat)
            loss = loss + self._beta * loss_diversity
        else:
            loss = torch.zeros(1, device=y_hat.device)
            loss_diversity = torch.zeros(1, device=y_hat.device)

        y_hat_permember = y_hat.detach()
        y_permember = y.detach()
        y = y[:, 0]
        y_hat_mean = average_predictions(y_hat, self._task)

        outputs[LOSS_KEY] = loss
        outputs[f"{LOSS_KEY}_diversity"] = loss_diversity.detach()
        outputs[TARGET_KEY] = y.detach()
        outputs[INPUT_KEY] = x.detach()
        outputs[PREDICTION_KEY] = y_hat.detach()
        outputs[MEAN_PREDICTION_KEY] = y_hat_mean.detach()
        outputs[PREDICTION_PER_MEMBER_KEY] = y_hat_permember.detach()
        outputs[TARGET_PER_MEMBER_KEY] = y_permember.detach()
        return outputs

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the specific arguments of the EarlyExitMethod."""
        parser = super(EarlyExitMethod, EarlyExitMethod).add_specific_args(
            parent_parser
        )
        parser.add_argument(
            "--method_alpha",
            type=float,
            default=1.0,
            help="The term to scale the loss of the auxiliary heads.",
        )
        parser.add_argument(
            "--method_beta",
            type=float,
            default=0.0,
            help="The term to scale the diversity loss which is a cross entropy between all axit pairs.",
        )
        parser.add_argument(
            "--method_gamma",
            type=float,
            default=0.3,
            help="The term to scale the hidden feature size when creating the auxiliary heads.",
        )
        return parser
