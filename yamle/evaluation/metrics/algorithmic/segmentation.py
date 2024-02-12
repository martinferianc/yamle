from typing import List, Optional, Any

import torchmetrics
import torch
from yamle.defaults import TINY_EPSILON


class IntersectionOverUnion(torchmetrics.Metric):
    """Calculate the intersection over union (IoU) of two tensors.

    The input is assumed to be probabilities, so the output of a softmax layer.

    Args:
        num_classes (int): Number of classes in the dataset. Defaults to 2.
        flatten (bool): Whether to flatten the input. Defaults to False.
        ignore_indices (Optional[List[int]]): List of indices to ignore. Defaults to None.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(
        self,
        num_classes: int = 10,
        flatten: bool = False,
        ignore_indices: Optional[List[int]] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._num_classes = num_classes
        self._flatten = flatten
        self._ignore_indices = ignore_indices

        self.add_state(
            "confusion_matrix",
            default=torch.zeros((self._num_classes, self._num_classes)),
            dist_reduce_fx="sum",
        )

    def _confusion_matrix(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """A helper function to compute the confusion matrix."""
        mask = (y >= 0) & (y < self._num_classes)
        cm = torch.bincount(
            self._num_classes * y[mask].long() + y_hat[mask],
            minlength=self._num_classes**2,
        ).reshape(self._num_classes, self._num_classes)
        if self._ignore_indices is not None:
            cm[:, self._ignore_indices] = 0
            cm[self._ignore_indices, :] = 0
        return cm

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model (probabilities).
            target (torch.Tensor): Ground truth values.
        """
        if self._flatten:
            preds = preds.permute(0, *(tuple(range(2, preds.ndim)) + (1,))).reshape(
                -1, preds.shape[1]
            )
            target = target.flatten()
        preds = preds.argmax(dim=1)

        self.confusion_matrix += self._confusion_matrix(preds, target)

    def compute(self) -> torch.Tensor:
        """Compute the intersection over union."""
        # The tiny epsilon is added to avoid division by zero.
        iou = torch.diag(self.confusion_matrix) / (
            self.confusion_matrix.sum(dim=1)
            + self.confusion_matrix.sum(dim=0)
            - torch.diag(self.confusion_matrix)
            + TINY_EPSILON
        )
        ignored_indices = (
            0 if self._ignore_indices is None else len(self._ignore_indices)
        )
        return torch.sum(iou) / (self._num_classes - ignored_indices)
