from typing import Any, Optional

import torchmetrics
import torch
import torch.nn.functional as F

from yamle.defaults import TINY_EPSILON


class BrierScore(torchmetrics.Metric):
    """Calculate the Brier Score for multi-class classification.

    The Brier Score is a metric for evaluation uncertainty in probabilistic
    classification. It measures the mean squared difference between the predicted
    probability and the true outcome.

    The input is assumed to be probabilities, so the output of a softmax layer.

    Args:
        reduction (str): Reduction method for the metric.
        flatten (bool): If `True` the input and predictions will be flattened.
        permute (bool): If `True` the input and predictions will be permuted from
        `(batch_size, num_classes, N, ...)` to `(batch_size, N, ..., num_classes)` before flattening.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(
        self,
        reduction: str = "mean",
        flatten: bool = False,
        permute: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reduction = reduction
        self._flatten = flatten
        self._permute = permute

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model (probabilities).
            target (torch.Tensor): Ground truth values.
        """
        if self._flatten:
            if self._permute:
                preds = preds.permute(0, *(tuple(range(2, preds.ndim)) + (1,)))
            preds = preds.reshape(-1, preds.shape[-1])
            target = target.flatten()
        one_hot = F.one_hot(target, num_classes=preds.shape[1]).float()
        self.sum += torch.sum((preds - one_hot) ** 2).item()
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the Brier Score."""
        if self._reduction == "mean":
            return self.sum / self.count
        elif self._reduction == "sum":
            return self.sum
        else:
            raise ValueError(f"Reduction {self._reduction} not supported.")


class Accuracy(torchmetrics.Accuracy):
    """Calculate the accuracy of a classification model.

    Args:
        flatten (bool): If `True` the input and predictions will be flattened.
        permute (bool): If `True` the input and predictions will be permuted from
        `(batch_size, num_classes, N, ...)` to `(batch_size, N, ..., num_classes)` before flattening.
    """

    def __init__(
        self, flatten: bool = False, permute: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._flatten = flatten
        self._permute = permute

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model (probabilities).
            target (torch.Tensor): Ground truth values.
        """
        if self._flatten:
            if self._permute:
                preds = preds.permute(0, *(tuple(range(2, preds.ndim)) + (1,)))
            preds = preds.reshape(-1, preds.shape[-1])
            target = target.flatten()

        super().update(preds, target)


class Precision(torchmetrics.Precision):
    """Calculate the precision of a classification model.

    Args:
        flatten (bool): If `True` the input and predictions will be flattened.
        permute (bool): If `True` the input and predictions will be permuted from
        `(batch_size, num_classes, N, ...)` to `(batch_size, N, ..., num_classes)` before flattening.
    """

    def __init__(
        self, flatten: bool = False, permute: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._flatten = flatten
        self._permute = permute

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model (probabilities).
            target (torch.Tensor): Ground truth values.
        """
        if self._flatten:
            if self._permute:
                preds = preds.permute(0, *(tuple(range(2, preds.ndim)) + (1,)))
            preds = preds.reshape(-1, preds.shape[-1])
            target = target.flatten()
        super().update(preds, target)


class Recall(torchmetrics.Recall):
    """Calculate the recall of a classification model.

    Args:
        flatten (bool): If `True` the input and predictions will be flattened.
        permute (bool): If `True` the input and predictions will be permuted from
        `(batch_size, num_classes, N, ...)` to `(batch_size, N, ..., num_classes)` before flattening.
    """

    def __init__(
        self, flatten: bool = False, permute: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._flatten = flatten
        self._permute = permute

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model (probabilities).
            target (torch.Tensor): Ground truth values.
        """
        if self._flatten:
            if self._permute:
                preds = preds.permute(0, *(tuple(range(2, preds.ndim)) + (1,)))
            preds = preds.reshape(-1, preds.shape[-1])
            target = target.flatten()
        super().update(preds, target)


class F1Score(torchmetrics.F1Score):
    """Calculate the F1 Score of a classification model.

    Args:
        flatten (bool): If `True` the input and predictions will be flattened.
        permute (bool): If `True` the input and predictions will be permuted from
        `(batch_size, num_classes, N, ...)` to `(batch_size, N, ..., num_classes)` before flattening.
    """

    def __init__(
        self, flatten: bool = False, permute: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._flatten = flatten
        self._permute = permute

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model (probabilities).
            target (torch.Tensor): Ground truth values.
        """
        if self._flatten:
            if self._permute:
                preds = preds.permute(0, *(tuple(range(2, preds.ndim)) + (1,)))
            preds = preds.reshape(-1, preds.shape[-1])
            target = target.flatten()
        super().update(preds, target)


class AUROC(torchmetrics.AUROC):
    """Calculate the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    The ROC AUC score is equivalent to the probability that a randomly chosen
    positive example ranks higher than a randomly chosen negative example.

    Args:
        flatten (bool): If `True` the input and predictions will be flattened.
        permute (bool): If `True` the input and predictions will be permuted from
        `(batch_size, num_classes, N, ...)` to `(batch_size, N, ..., num_classes)` before flattening.
    """

    def __init__(
        self, flatten: bool = False, permute: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._flatten = flatten
        self._permute = permute

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model (probabilities).
            target (torch.Tensor): Ground truth values.
        """
        if self._flatten:
            if self._permute:
                preds = preds.permute(0, *(tuple(range(2, preds.ndim)) + (1,)))
            preds = preds.reshape(-1, preds.shape[-1])
            target = target.flatten()
        super().update(preds, target)


class NegativeLogLikelihood(torchmetrics.Metric):
    """Calculate the negative log-likelihood for multi-class classification.

    The negative log-likelihood is a metric for evaluation uncertainty in
    probabilistic classification. It measures the mean log-likelihood of the
    true outcome.

    The input is assumed to be probabilities, so the output of a softmax layer.

    Args:
        reduction (str): Reduction method for the metric.
        flatten (bool): If `True` the input and predictions will be flattened.
        permute (bool): If `True` the input and predictions will be permuted from
        `(batch_size, num_classes, N, ...)` to `(batch_size, N, ..., num_classes)` before flattening.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(
        self,
        reduction: str = "mean",
        flatten: bool = False,
        permute: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reduction = reduction
        self._flatten = flatten
        self._permute = permute

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model (probabilities).
            target (torch.Tensor): Ground truth values.
        """
        if self._flatten:
            if self._permute:
                preds = preds.permute(0, *(tuple(range(2, preds.ndim)) + (1,)))
            preds = preds.reshape(-1, preds.shape[-1])
            target = target.flatten()

        one_hot = F.one_hot(target, num_classes=preds.shape[-1])
        self.sum += torch.sum(-torch.log(preds + TINY_EPSILON) * one_hot).item()
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the negative log-likelihood."""
        if self._reduction == "mean":
            return self.sum / self.count
        elif self._reduction == "sum":
            return self.sum
        else:
            raise ValueError(f"Reduction {self._reduction} not supported.")


class Perplexity(torchmetrics.Metric):
    """Calculate the perplexity for multi-class classification.

    The perplexity is a metric for evaluation uncertainty in probabilistic
    classification. It measures the average uncertainty of the predicted
    probability distribution.

    The input is assumed to be probabilities, so the output of a softmax layer.

    Args:
        reduction (str): Reduction method for the metric.
        flatten (bool): If `True` the input and predictions will be flattened.
        permute (bool): If `True` the input and predictions will be permuted from
        `(batch_size, num_classes, N, ...)` to `(batch_size, N, ..., num_classes)` before flattening.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(
        self,
        reduction: str = "mean",
        flatten: bool = False,
        permute: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reduction = reduction
        self._flatten = flatten
        self._permute = permute

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model (probabilities).
            target (torch.Tensor): Ground truth values.
        """
        if self._flatten:
            if self._permute:
                preds = preds.permute(0, *(tuple(range(2, preds.ndim)) + (1,)))
            preds = preds.reshape(-1, preds.shape[-1])
            target = target.flatten()

        self.sum += F.cross_entropy(
            torch.log(preds + TINY_EPSILON), target, reduction="sum"
        )
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the perplexity."""
        if self._reduction == "mean":
            return torch.exp(self.sum / self.count)
        elif self._reduction == "sum":
            return torch.exp(self.sum)
        else:
            raise ValueError(f"Reduction {self._reduction} not supported.")


class PredictiveUncertainty(torchmetrics.Metric):
    """Calculate the complete predictive uncertainty for multi-class classification.

    This metric computes the combined uncertainty of the model. It is the sum of the
    aleatoric and epistemic uncertainty.

    The input is assumed to be probabilities, so the output of a softmax layer.

    Args:
        reduction (str): Reduction method for the metric.
        flatten (bool): If `True` the input and predictions will be flattened.
        permute (bool): If `True` the input and predictions will be permuted from
        `(batch_size, num_classes, N, ...)` to `(batch_size, N, ..., num_classes)` before flattening.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(
        self,
        reduction: str = "mean",
        flatten: bool = False,
        permute: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reduction = reduction
        self._flatten = flatten
        self._permute = permute

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model (probabilities).
            target (torch.Tensor): Ground truth values.
        """
        if self._flatten is not None:
            if self._permute:
                preds = preds.permute(0, *(tuple(range(2, preds.ndim)) + (1,)))
            preds = preds.reshape(-1, preds.shape[-1])

        self.sum += torch.sum(-preds * torch.log(preds + TINY_EPSILON)).item()
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the entropy."""
        if self._reduction == "mean":
            return self.sum / self.count
        elif self._reduction == "sum":
            return self.sum
        else:
            raise ValueError(f"Reduction {self._reduction} not supported.")


class AleatoricUncertainty(torchmetrics.Metric):
    """Calculate the aleatoric uncertainty for multi-class classification.

    The input is assumed to be probabilities, so the output of a softmax layer.

    Args:
        reduction (str): Reduction method for the metric.
        flatten (bool): If `True` the input and predictions will be flattened.
        permute (bool): If `True` the input and predictions will be permuted from
        `(batch_size, num_classes, N, ...)` to `(batch_size, N, ..., num_classes)` before flattening.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(
        self,
        reduction: str = "mean",
        flatten: bool = False,
        permute: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reduction = reduction
        self._flatten = flatten
        self._permute = permute

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, preds_individual: torch.Tensor, averaging_weights: Optional[torch.Tensor] = None
    ) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model (probabilities). These are the mean predictions across number of samples.
            target (torch.Tensor): Ground truth values.
            preds_individual (torch.Tensor): Predictions from model (probabilities). These are the individual predictions across number of samples.
            averaging_weights (torch.Tensor): Weights to use for averaging the individual predictions. If `None` uniform averaging is used.
        """
        if self._flatten:
            if self._permute:
                preds = preds.permute(0, *(tuple(range(2, preds.ndim)) + (1,)))
                preds_individual = preds_individual.permute(
                    0, *(tuple(range(3, preds_individual.ndim)) + (1, 2))
                )
            preds = preds.reshape(-1, preds.shape[-1])
            preds_individual = preds_individual.reshape(
                -1, preds_individual.shape[1], preds_individual.shape[-1]
            )
        num_members = preds_individual.shape[1]
        if averaging_weights is None:
            averaging_weights = torch.ones(num_members, device=preds_individual.device)/num_members
        entropy = -torch.sum(preds_individual * torch.log(preds_individual + TINY_EPSILON), dim=2)
        aleatoric = torch.sum(averaging_weights * entropy, dim=1)
        self.sum += torch.sum(aleatoric).item()
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the aleatoric uncertainty."""
        if self._reduction == "mean":
            return self.sum / self.count
        elif self._reduction == "sum":
            return self.sum
        else:
            raise ValueError(f"Reduction {self._reduction} not supported.")


class EpistemicUncertainty(torchmetrics.Metric):
    """Calculate the epistemic uncertainty for multi-class classification.

    The input is assumed to be probabilities, so the output of a softmax layer.

    Args:
        reduction (str): Reduction method for the metric.
        flatten (bool): If `True` the input and predictions will be flattened.
        permute (bool): If `True` the input and predictions will be permuted from
        `(batch_size, num_classes, N, ...)` to `(batch_size, N, ..., num_classes)` before flattening.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(
        self,
        reduction: str = "mean",
        flatten: bool = False,
        permute: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reduction = reduction
        self._flatten = flatten
        self._permute = permute

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, preds_individual: torch.Tensor, averaging_weights: Optional[torch.Tensor] = None
    ) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model (probabilities). These are the mean predictions across number of samples.
            target (torch.Tensor): Ground truth values.
            preds_individual (torch.Tensor): Predictions from model (probabilities). These are the individual predictions across number of samples.
            averaging_weights (torch.Tensor): Weights to use for averaging the individual predictions. If `None` uniform averaging is used.
        """
        if self._flatten:
            if self._permute:
                preds = preds.permute(0, *(tuple(range(2, preds.ndim)) + (1,)))
                preds_individual = preds_individual.permute(
                    0, *(tuple(range(3, preds_individual.ndim)) + (1, 2))
                )
            preds = preds.reshape(-1, preds.shape[-1])
            preds_individual = preds_individual.reshape(
                -1, preds_individual.shape[1], preds_individual.shape[-1]
            )
        num_members = preds_individual.shape[1]
        if averaging_weights is None:
            averaging_weights = torch.ones(num_members, device=preds_individual.device)/num_members
        
        predictive = -torch.sum(preds * torch.log(preds + TINY_EPSILON), dim=1)
        aleatoric_entropy = -torch.sum(preds_individual * torch.log(preds_individual + TINY_EPSILON), dim=2)
        aleatoric = torch.sum(averaging_weights * aleatoric_entropy, dim=1)
        
        epistemic = predictive - aleatoric
        self.sum += torch.sum(epistemic).item()
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the epistemic uncertainty."""
        if self._reduction == "mean":
            return self.sum / self.count
        elif self._reduction == "sum":
            return self.sum
        else:
            raise ValueError(f"Reduction {self._reduction} not supported.")


class ClassificationDiversity(torchmetrics.Metric):
    """Calculate the classification diversity for multi-class classification.

    It is computed based on the ratio of the errors between the individual ensemble member predictions.

    The input is assumed to be probabilities, so the output of a softmax layer.

    Args:
        flatten (bool): If `True` the input and predictions will be flattened.
        permute (bool): If `True` the input and predictions will be permuted from
        `(batch_size, num_classes, N, ...)` to `(batch_size, N, ..., num_classes)` before flattening.
        num_members (int): Number of members in the ensemble.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = True

    def __init__(
        self,
        num_members: int = 10,
        flatten: bool = False,
        permute: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._flatten = flatten
        self._permute = permute
        assert num_members >= 1, "Number of members must be greater than 1."
        self._num_members = num_members

        self.add_state(
            "confusion_matrix",
            default=torch.zeros(
                (num_members, num_members, 2, 2), dtype=torch.long, requires_grad=False
            ),
            dist_reduce_fx="sum",
        )

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, preds_individual: torch.Tensor, averaging_weights: Optional[torch.Tensor] = None
    ) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model (probabilities). These are the mean predictions across number of samples. Not used.
            target (torch.Tensor): Ground truth values.
            preds_individual (torch.Tensor): Predictions from model (probabilities). These are the predictions from each member of the ensemble.
            averaging_weights (torch.Tensor): Weights to use for averaging the individual predictions. If `None` uniform averaging is used.
        """
        if self._flatten:
            if self._permute:
                preds_individual = preds_individual.permute(
                    0, *(tuple(range(3, preds_individual.ndim)) + (1, 2))
                )
            preds_individual = preds_individual.reshape(
                -1, preds_individual.shape[1], preds_individual.shape[-1]
            )
            target = target.flatten()

        preds_individual = torch.argmax(preds_individual, dim=2)
        num_members = min(preds_individual.shape[1], self._num_members)
        for i in range(num_members - 1):
            for k in range(i + 1, num_members):
                preds_i = preds_individual[:, i]
                preds_k = preds_individual[:, k]
                preds_i_eq_target = preds_i == target
                preds_i_neq_target = preds_i != target
                preds_k_eq_target = preds_k == target
                preds_k_neq_target = preds_k != target
                self.confusion_matrix[i, k, 0, 0] += torch.sum(
                    torch.logical_and(preds_i_eq_target, preds_k_eq_target)
                )
                self.confusion_matrix[i, k, 0, 1] += torch.sum(
                    torch.logical_and(preds_i_eq_target, preds_k_neq_target)
                )
                self.confusion_matrix[i, k, 1, 0] += torch.sum(
                    torch.logical_and(preds_i_neq_target, preds_k_eq_target)
                )
                self.confusion_matrix[i, k, 1, 1] += torch.sum(
                    torch.logical_and(preds_i_neq_target, preds_k_neq_target)
                )

    def compute(self) -> torch.Tensor:
        """Compute the classification diversity.

        Returns:
            torch.Tensor: The classification diversity.
        """
        diversity = torch.tensor(0.0, device=self.confusion_matrix.device)
        for i in range(self._num_members - 1):
            for k in range(i + 1, self._num_members):
                diversity += (
                    self.confusion_matrix[i, k, 0, 1]
                    + self.confusion_matrix[i, k, 1, 0]
                ) / torch.sum(self.confusion_matrix[i, k])
        # The TINY_EPSILON is to avoid division by zero if there is only one member in the ensemble.
        return (2 * diversity) / (
            self._num_members * (self._num_members - 1) + TINY_EPSILON
        )

    def reset(self) -> None:
        """Reset all of the metric states."""
        self.confusion_matrix.data.zero_()


class CalibrationError(torchmetrics.CalibrationError):
    """Calculate the expected calibration error for multi-class classification.

    The input is assumed to be probabilities, so the output of a softmax layer.

    Args:
        flatten (bool): If `True` the input and predictions will be flattened.
        permute (bool): If `True` the input and predictions will be permuted from
        `(batch_size, num_classes, N, ...)` to `(batch_size, N, ..., num_classes)` before flattening.
    """

    def __init__(
        self,
        flatten: bool = False,
        permute: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._flatten = flatten
        self._permute = permute

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model (probabilities).
            target (torch.Tensor): Ground truth values.
        """
        if self._flatten:
            if self._permute:
                preds = preds.permute(0, *(tuple(range(2, preds.ndim)) + (1,)))
            preds = preds.reshape(-1, preds.shape[-1])
            target = target.flatten()
        super().update(preds, target)
        
class ClassConditionalCalibrationError(torchmetrics.Metric):
    """Calculate the expected calibration error for multi-class classification.

    Compute the calibration error for each class separately.
    Then average the calibration error across classes.
    This implements the "macro" averaging method for "micro" averaging, this
    is equivalent to the `CalibrationError` metric.
    
    Args:
        task (str): The task of the model. Either "multiclass" or "binary".
        n_bins (int): Number of bins to use for calibration error.
        norm (str): Norm to use for the calibration error. Either "l1" or "l2".
        flatten (bool): If `True` the input and predictions will be flattened.
        permute (bool): If `True` the input will be permuted from
        `(batch_size, num_classes, N, ...)` to `(batch_size, N, ..., num_classes)` before flattening.
    """
    def __init__(
        self,
        task: str = "multiclass",
        n_bins: int = 15,
        norm: str = "l1",
        num_classes: int = 10,
        flatten: bool = False,
        permute: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        
        assert task in ["multiclass", "binary"], f"Unknown task: {task}"
        assert norm in ["l1", "l2"], f"Unknown norm: {norm}"
        assert n_bins >= 1, f"Number of bins must be greater than 1."
        assert num_classes >= 2, f"Number of classes must be greater than 1."
        
        self._task = task
        self._n_bins = n_bins
        self._norm = norm
        self._num_classes = num_classes
        self._flatten = flatten
        self._permute = permute
        
        self.add_state("confidences", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("labels", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("bins", default=torch.linspace(0, 1, n_bins + 1), dist_reduce_fx="sum")
 
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets."""
        if self._flatten:
            if self._permute:
                preds = preds.permute(0, *(tuple(range(2, preds.ndim)) + (1,)))
            preds = preds.reshape(-1, preds.shape[-1])
            target = target.flatten()
            
        if self._task == "binary":
            # Add an extra dimension to the predictions which will be the 1 - p.
            preds = torch.cat((preds, 1 - preds), dim=1)
            assert preds.shape[1] == 2, f"Binary classification must have 2 classes."
        else:
            assert preds.shape[1] == self._num_classes, f"Number of classes must be {self._num_classes}."
            
        # Convert the target to one-hot encoding.
        target_one_hot = F.one_hot(target, num_classes=preds.shape[1]).float()
        
        assert self._num_classes == target_one_hot.shape[1], f"Number of classes must be {self._num_classes}."

        self.confidences = torch.cat((self.confidences, preds.detach())) if self.confidences.numel() != 0 else preds.detach()
        self.labels = torch.cat((self.labels, target_one_hot.detach())) if self.labels.numel() != 0 else target_one_hot.detach()
        
    def reset(self) -> None:
        """Reset all of the metric states."""
        self.confidences = torch.tensor([]).to(self.confidences.device)
        self.labels = torch.tensor([]).to(self.labels.device)
        
    def compute(self) -> torch.Tensor:
        """Compute the calibration error."""
        error = []
        num_classes = self.confidences.shape[1]
        for i in range(self.confidences.shape[1]):
            # Select the confidences and labels for the current class.
            labels = self.labels[torch.argmax(self.confidences, dim=1) == i][:, i].contiguous()
            confidences = self.confidences[torch.argmax(self.confidences, dim=1) == i][:, i].contiguous()
            error.append(self._compute_calibration_error(confidences, labels)/num_classes)
        error = sum(error)
        if self._norm == "l2":
            error = torch.sqrt(error)
        return error
        
    def _compute_calibration_error(self, probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """This function computes the sum weighted calibration error."""         
        if probabilities.numel() == 0:
            return torch.tensor(0.0, device=probabilities.device)
        
        bin_indices = torch.bucketize(probabilities, self.bins)
        n_bins = self.bins.numel() - 1
        sums = torch.bincount(bin_indices.cpu(), weights=probabilities.cpu(), minlength=n_bins).to(probabilities.device)
        sums = sums.type(torch.float32)
        counts = torch.bincount(bin_indices.cpu(), minlength=n_bins).to(probabilities.device)
        counts = counts + TINY_EPSILON
        
        confidences = sums / counts
        accuracies = torch.bincount(bin_indices.cpu(), weights=labels.cpu(), minlength=n_bins).to(probabilities.device) / counts
        
        calibration_errors = accuracies - confidences
        
        if self._norm == "l1":
            calibration_errors_normed = calibration_errors
        elif self._norm == "l2":
            calibration_errors_normed = torch.square(calibration_errors)
        else:
            raise ValueError(f"Unknown norm: {self._norm}")
        
        weighting = counts / float(probabilities.numel())
        weighted_calibration_error = calibration_errors_normed * weighting
        
        return torch.sum(torch.abs(weighted_calibration_error))