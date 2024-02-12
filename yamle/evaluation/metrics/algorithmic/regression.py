from typing import Any, Optional

import torchmetrics
import torch
import torch.nn.functional as F

from yamle.defaults import TINY_EPSILON
from yamle.utils.operation_utils import weighted_regression_variance


class NegativeLogLikelihood(torchmetrics.Metric):
    """Calculate the negative log-likelihood for regression.

    The negative log-likelihood is a proper metric for evaluating the uncertainty
    in regression. It measures the mean log-likelihood of the true outcome.

    Args:
        reduction (str): Reduction method for the metric.

        flatten (bool): If `True` the input and predictions will be flattened.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(
        self, reduction: str = "mean", flatten: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reduction = reduction
        self._flatten = flatten

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model, mean and or variance.
                                    Mean is assumed to be the first element. Variance
                                    is assumed to be the second element.
            target (torch.Tensor): Ground truth values.
        """
        mean = preds[:, 0]
        variance = preds[:, 1] if preds.shape[1] > 1 else torch.ones_like(mean)
        if self._flatten:
            batch_size = mean.shape[0]
            mean = mean.reshape(batch_size, -1)
            variance = variance.reshape(batch_size, -1)
            target = target.reshape(batch_size, -1)
        self.sum += F.gaussian_nll_loss(
            mean.squeeze(), target.squeeze(), variance.squeeze(), reduction="sum"
        )
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the negative log-likelihood."""
        if self._reduction == "mean":
            return self.sum / self.count
        elif self._reduction == "sum":
            return self.sum
        else:
            raise ValueError(f"Reduction {self._reduction} not supported.")


class MeanAbsoluteError(torchmetrics.Metric):
    """Calculate the mean absolute error for regression.

    The mean absolute error is a proper metric for evaluating the uncertainty
    in regression. It measures the mean absolute error of the true outcome.

    Args:
        reduction (str): Reduction method for the metric.
        flatten (bool): If `True` the input and predictions will be flattened.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(
        self, reduction: str = "mean", flatten: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reduction = reduction
        self._flatten = flatten

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model, mean and or variance.
                                    Mean is assumed to be the first element. Variance
                                    is assumed to be the second element.
            target (torch.Tensor): Ground truth values.
        """
        mean = preds[:, 0]
        if self._flatten:
            batch_size = mean.shape[0]
            mean = mean.reshape(batch_size, -1)
            target = target.reshape(batch_size, -1)
        self.sum += F.l1_loss(mean.squeeze(), target.squeeze(), reduction="sum")
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the mean absolute error."""
        if self._reduction == "mean":
            return self.sum / self.count
        elif self._reduction == "sum":
            return self.sum
        else:
            raise ValueError(f"Reduction {self._reduction} not supported.")


class MeanSquaredError(torchmetrics.Metric):
    """Calculate the mean squared error for regression.

    The mean squared error is a proper metric for evaluating the uncertainty
    in regression. It measures the mean squared error of the true outcome.

    Args:
        reduction (str): Reduction method for the metric.
        flatten (bool): If `True` the input and predictions will be flattened.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(
        self, reduction: str = "mean", flatten: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reduction = reduction
        self._flatten = flatten

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model, mean and or variance.
                                    Mean is assumed to be the first element. Variance
                                    is assumed to be the second element.
            target (torch.Tensor): Ground truth values.
        """
        mean = preds[:, 0]
        if self._flatten:
            batch_size = mean.shape[0]
            mean = mean.reshape(batch_size, -1)
            target = target.reshape(batch_size, -1)
        self.sum += F.mse_loss(mean.squeeze(), target.squeeze(), reduction="sum")
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the mean squared error."""
        if self._reduction == "mean":
            return self.sum / self.count
        elif self._reduction == "sum":
            return self.sum
        else:
            raise ValueError(f"Reduction {self._reduction} not supported.")


class RootMeanSquaredError(MeanSquaredError):
    """Calculate the root mean squared error for regression."""

    def compute(self) -> torch.Tensor:
        """Compute the root mean squared error."""
        assert (
            self._reduction == "mean"
        ), f"Reduction {self._reduction} not supported. Use mean."
        return torch.sqrt(super().compute())


class PredictiveUncertainty(torchmetrics.Metric):
    """Calculate the predictive uncertainty for regression.

    The predictive uncertainty is a proper metric for evaluating the uncertainty
    in regression. It measures the mean predictive uncertainty of the true outcome.
    It is stored as a natural logarithm.

    Args:
        reduction (str): Reduction method for the metric.
        flatten (bool): If `True` the input and predictions will be flattened.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(
        self,
        reduction: str = "mean",
        flatten: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reduction = reduction
        self._flatten = flatten

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model, mean and or variance.
                                    Mean is assumed to be the first element. Variance
                                    is assumed to be the second element.
            target (torch.Tensor): Ground truth values.
        """
        variance = preds[:, 1] if preds.shape[1] > 1 else torch.ones_like(preds[:, 0])
        if self._flatten:
            batch_size = variance.shape[0]
            variance = variance.reshape(batch_size, -1)

        self.sum += torch.sum(torch.log(variance + TINY_EPSILON))
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the predictive uncertainty."""
        if self._reduction == "mean":
            return self.sum / self.count
        elif self._reduction == "sum":
            return self.sum
        else:
            raise ValueError(f"Reduction {self._reduction} not supported.")


class AleatoricUncertainty(torchmetrics.Metric):
    """Calculate the aleatoric uncertainty for regression.

    The aleatoric uncertainty is a proper metric for evaluating the uncertainty
    in regression. It measures the mean aleatoric uncertainty of the true outcome.
    It is stored as a natural logarithm.

    Args:
        reduction (str): Reduction method for the metric.
        flatten (bool): If `True` the input and predictions will be flattened.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(
        self, reduction: str = "mean", flatten: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reduction = reduction
        self._flatten = flatten

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        preds_individual: torch.Tensor,
        averaging_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model, mean and or variance.
                                    Mean is assumed to be the first element. Variance
                                    is assumed to be the second element.
            target (torch.Tensor): Ground truth values.
            preds_individual (torch.Tensor): Predictions from model, mean and or variance.
                                    Mean is assumed to be the first element. Variance
                                    is assumed to be the second element.
            averaging_weights (torch.Tensor): Weights to use for averaging the individual predictions. If `None` uniform averaging is used.
        """
        individual_variance = (
            preds_individual[:, :, 1]
            if preds_individual.shape[2] > 1
            else torch.ones_like(preds_individual)
        )
        if self._flatten:
            batch_size = individual_variance.shape[0]
            num_members = individual_variance.shape[1]
            individual_variance = individual_variance.reshape(
                batch_size, num_members, -1
            )
        if averaging_weights is None:
            num_members = individual_variance.shape[1]
            averaging_weights = (
                torch.ones(
                    individual_variance.shape[1], device=individual_variance.device
                )
                / num_members
            )

        aleatoric = torch.sum(individual_variance * averaging_weights, dim=1)

        self.sum += torch.sum(torch.log(aleatoric + TINY_EPSILON))
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the predictive uncertainty."""
        if self._reduction == "mean":
            return self.sum / self.count
        elif self._reduction == "sum":
            return self.sum
        else:
            raise ValueError(f"Reduction {self._reduction} not supported.")


class EpistemicUncertainty(torchmetrics.Metric):
    """Calculate the epistemic uncertainty for regression.

    The epistemic uncertainty is a proper metric for evaluating the uncertainty
    in regression. It measures the mean epistemic uncertainty of the true outcome.
    It is stored as a natural logarithm.

    Args:
        reduction (str): Reduction method for the metric.
        flatten (bool): If `True` the input and predictions will be flattened.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(
        self, reduction: str = "mean", flatten: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reduction = reduction
        self._flatten = flatten

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        preds_individual: torch.Tensor,
        averaging_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model, mean and or variance.
                                    Mean is assumed to be the first element. Variance
                                    is assumed to be the second element.
            target (torch.Tensor): Ground truth values.
            preds_individual (torch.Tensor): Predictions from model, mean and or variance.
                                    Mean is assumed to be the first element. Variance
                                    is assumed to be the second element.
            averaging_weights (torch.Tensor): Weights to use for averaging the individual predictions. If `None` uniform averaging is used.
        """
        individual_mean = preds_individual[:, :, 0]
        if self._flatten:
            batch_size = individual_mean.shape[0]
            num_members = individual_mean.shape[1]
            individual_mean = individual_mean.reshape(batch_size, num_members, -1)

        if averaging_weights is None:
            num_members = individual_mean.shape[1]
            averaging_weights = (
                torch.ones(individual_mean.shape[1], device=individual_mean.device)
                / num_members
            )

        epistemic = weighted_regression_variance(individual_mean, averaging_weights)

        # If the uncertainty is all zeros, then the log is undefined, we set the sum to zero
        if torch.all(epistemic == 0):
            self.sum += 0
        else:
            self.sum += torch.sum(torch.log(epistemic + TINY_EPSILON))
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the predictive uncertainty."""
        if self._reduction == "mean":
            return self.sum / self.count
        elif self._reduction == "sum":
            return self.sum
        else:
            raise ValueError(f"Reduction {self._reduction} not supported.")
