from typing import Type, Callable, Optional

from yamle.losses.loss import DummyLoss
from yamle.losses.classification import CrossEntropyLoss, TextCrossEntropyLoss
from yamle.losses.contrastive import NoiseContrastiveEstimatorLoss
from yamle.losses.regression import (
    GaussianNegativeLogLikelihoodLoss,
    MeanSquaredError,
    QuantileRegressionLoss,
)
from yamle.losses.segmentation import FocalLoss, SoftIntersectionOverUnionLoss
from yamle.losses.evidential_regression import EvidentialRegressionLoss

AVAILABLE_LOSSES = {
    "crossentropy": CrossEntropyLoss,
    "nce": NoiseContrastiveEstimatorLoss,
    "textcrossentropy": TextCrossEntropyLoss,
    "gaussiannll": GaussianNegativeLogLikelihoodLoss,
    "mse": MeanSquaredError,
    "quantile": QuantileRegressionLoss,
    "focal": FocalLoss,
    "softiou": SoftIntersectionOverUnionLoss,
    "evidentialregression": EvidentialRegressionLoss,
    None: DummyLoss,
    "dummy": DummyLoss,
}


def loss_factory(loss_type: Optional[str] = None) -> Type[Callable]:
    """This function is used to create a loss instance based on the loss type.

    Args:
        loss_type (str): The type of loss to create.
    """
    if loss_type not in AVAILABLE_LOSSES:
        raise ValueError(f"Unknown loss type {loss_type}.")
    return AVAILABLE_LOSSES[loss_type]
