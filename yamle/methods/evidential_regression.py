from typing import Any, List, Dict, Optional
import torch.nn as nn
import torch
import argparse
import torch.distributions as distributions
import torchmetrics

from yamle.defaults import (
    TINY_EPSILON,
    LOSS_KEY,
    LOSS_KL_KEY,
    TARGET_KEY,
    PREDICTION_KEY,
    MEAN_PREDICTION_KEY,
    TRAIN_KEY,
    MEMBERS_DIM,
    REGRESSION_KEY,
    INPUT_KEY,
    MIN_TENDENCY,
)
from yamle.methods.uncertain_method import (
    MCSamplingMethod,
    MemberMethod,
)
from yamle.models.specific.evidential_regression import (
    NormalGammaLinear,
    NormalGammaConv2d,
)
from yamle.losses.evidential_regression import EvidentialRegressionLoss
from yamle.utils.operation_utils import average_predictions

import logging

logging = logging.getLogger("pytorch_lightning")


class EvidentialRegressionMethod(MCSamplingMethod):
    """This class is the extension of the base method for which the prediciton is performed using Evidence Regression.

    Args:
        alpha (float): The alpha parameter regularising between the negative log likelihood and the KL divergence.
    """

    tasks = [REGRESSION_KEY]

    def __init__(self, alpha: float = 1, *args: Any, **kwargs: Any) -> None:
        super(EvidentialRegressionMethod, self).__init__(*args, **kwargs)
        assert isinstance(
            self.model._output, (nn.Linear, nn.Conv2d)
        ), f"The last layer must be a linear or convolutional layer, got {type(self.model._output)}"
        assert isinstance(
            self._loss, EvidentialRegressionLoss
        ), f"The loss must be an instance of EvidentialRegressionLoss, got {type(self._loss)}"
        assert alpha >= 0, f"alpha must be non-negative, got {alpha}"
        self._alpha = alpha
        self._replace_output_layer()

    def _create_metrics(self, metrics_kwargs: Dict[str, Any]) -> None:
        """This method is used to create the metrics to be used for training, validation and testing."""
        super(MemberMethod, self)._create_metrics(metrics_kwargs)
        self._add_additional_metrics(
            {
                f"{LOSS_KEY}_individual": torchmetrics.MeanMetric(),
                LOSS_KL_KEY: torchmetrics.MeanMetric(),
            },
            tendencies=[MIN_TENDENCY, MIN_TENDENCY],
        )

    def _replace_output_layer(self) -> None:
        """This method is used to replace the output layer of the model with the evidential regression layer."""
        if isinstance(self.model._output, nn.Linear):
            self.model._output = NormalGammaLinear(
                in_features=self.model._output.in_features,
                bias=self.model._output.bias is not None,
            )
        elif isinstance(self.model._output, nn.Conv2d):
            self.model._output = NormalGammaConv2d(
                in_channels=self.model._output.in_channels,
                kernel_size=self.model._output.kernel_size,
                stride=self.model._output.stride,
                padding=self.model._output.padding,
                dilation=self.model._output.dilation,
                groups=self.model._output.groups,
                bias=self.model._output.bias is not None,
            )
        else:
            raise NotImplementedError(
                f"Unsupported output layer type {type(self.model._output)}"
            )

        # Replace output_activation with identity
        self.model._output_activation = nn.Identity()

    def _sample(self, y_hat: torch.Tensor) -> torch.Tensor:
        """This method is used to sample from the output of the model.

        The last layer output of the model gives the `mu, v, alpha, beta` parameters of the Normal-Inverse-Gamma distribution.
        Given the parameters, we can sample from the distributions for mean and variance.
        For variance we sample the inverse gamma distribution and for mean we sample from the normal distribution.
        """
        mu, v, alpha, beta = torch.split(y_hat, 1, dim=-1)
        var = distributions.transformed_distribution.TransformedDistribution(
            distributions.gamma.Gamma(alpha, beta),
            distributions.transforms.PowerTransform(
                torch.tensor([-1.0]).to(self.device)
            ),
        ).sample()
        mean = distributions.normal.Normal(
            mu, torch.sqrt(var) / (v + TINY_EPSILON)
        ).sample()
        return torch.cat([mean, var], dim=-1)

    def _step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
        phase: str = TRAIN_KEY,
    ) -> Dict[str, torch.Tensor]:
        """This method is used to perform a single step."""
        x, y = batch
        params = super(MCSamplingMethod, self)._predict(x, unsqueeze=False)
        loss_nll, loss_reg = self._loss(params, y)

        if self.training:
            # During training, do only a single sample
            y_hat = self._sample(params).unsqueeze(dim=MEMBERS_DIM)
        else:
            # During validation and testing, sample `num_members` times and average the predictions
            outputs = []
            for _ in range(self._num_members):
                y_hat = self._sample(params).unsqueeze(dim=MEMBERS_DIM)
                outputs.append(y_hat)
            y_hat = torch.cat(outputs, dim=MEMBERS_DIM)

        y_hat_mean = average_predictions(y_hat, self._task)

        outputs = {}
        outputs[LOSS_KEY] = loss_nll + self._alpha * loss_reg
        outputs[LOSS_KL_KEY] = loss_reg.detach()
        outputs[f"{LOSS_KEY}_individual"] = loss_nll.detach()
        outputs[TARGET_KEY] = y.detach()
        outputs[INPUT_KEY] = x.detach()
        outputs[PREDICTION_KEY] = y_hat.detach()
        outputs[MEAN_PREDICTION_KEY] = y_hat_mean.detach()
        return outputs

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the specific arguments for the class."""
        parser = super(
            EvidentialRegressionMethod, EvidentialRegressionMethod
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_alpha",
            type=float,
            default=1,
            help="The alpha parameter regularising between the negative log likelihood and the KL divergence.",
        )
        return parser
