from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from yamle.defaults import (
    TINY_EPSILON,
    CLASSIFICATION_KEY,
    REGRESSION_KEY,
    SEGMENTATION_KEY,
    MEMBERS_DIM,
    TEXT_CLASSIFICATION_KEY,
    DEPTH_ESTIMATION_KEY,
    RECONSTRUCTION_KEY,
)


@torch.no_grad()
def average_predictions(
    predictions: torch.Tensor, task: str, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """This function is used to average the predictions.

    Note that it is assumed that no gradients are passed through the predictions
    when this function is called.

    Args:
        predictions (torch.Tensor): The predictions to average.
        task (str): The task to perform the averaging for.
        weights (Optional[torch.Tensor]): The weights to use for the averaging. If the weights are not provided, the mean is used.
    """
    if weights is None:
        weights = (
            torch.ones(
                predictions.shape[0],
                predictions.shape[MEMBERS_DIM],
                device=predictions.device,
            )
            / predictions.shape[MEMBERS_DIM]
        )
    assert torch.all(
        torch.sum(weights, dim=MEMBERS_DIM)
    ), f"The weights should sum up to 1.0, but they sum up to {torch.sum(weights, dim=MEMBERS_DIM)}."
    # In case the shape is smaller, increase the dimension
    # This weights all predictions equally for the same sample.
    while len(weights.shape) < len(predictions.shape):
        weights = weights.unsqueeze(-1)
    if task in [CLASSIFICATION_KEY, SEGMENTATION_KEY, TEXT_CLASSIFICATION_KEY]:
        predictions_mean = torch.sum(predictions * weights, dim=MEMBERS_DIM)
    elif task in [REGRESSION_KEY, DEPTH_ESTIMATION_KEY, RECONSTRUCTION_KEY]:
        predictions_mean = torch.sum(
            predictions[:, :, 0] * weights.squeeze(-1), dim=MEMBERS_DIM
        )
        predictions_mean = predictions_mean.unsqueeze(1)
        variance = (
            weighted_regression_variance(
                predictions[:, :, 0], weights.squeeze(-1)
            ).unsqueeze(1)
            if predictions.shape[MEMBERS_DIM] > 1
            else torch.zeros_like(predictions_mean)
        )
        variance += torch.sum(
            predictions[:, :, 1] * weights.squeeze(-1), dim=MEMBERS_DIM
        ).unsqueeze(1)
        predictions_mean = torch.cat([predictions_mean, variance], dim=1)
    else:
        raise ValueError(f"Task {task} is not supported.")
    return predictions_mean


def weighted_regression_variance(
    predictions: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """This function is used to calculate the regression variance.

    Args:
        predictions (torch.Tensor): The predictions to calculate the variance for. The shape should be `(batch_size, num_members, *prediction_dims)`.
        weights (Optional[torch.Tensor]): The weights to use for the variance calculation. If the weights are not provided, the mean is used.
    """
    if predictions.shape[MEMBERS_DIM] == 1: # This is for numerical stability
        return torch.zeros_like(predictions).squeeze(MEMBERS_DIM)
    
    
    if weights is None:
        weights = torch.ones_like(predictions) / predictions.shape[1]

    predictions_mean = torch.sum(predictions * weights, dim=MEMBERS_DIM).unsqueeze(
        MEMBERS_DIM
    )
    variance = torch.sum(
        weights * (predictions - predictions_mean) ** 2, dim=MEMBERS_DIM
    )
    return variance

def repeat_inputs(
    inputs: torch.Tensor, num: int) -> torch.Tensor:
    """This method is used to concatenate the `num` inputs into the first dimension."""
    new_inputs = []
    for i in range(len(inputs)):
        for _ in range(num):
            new_inputs.append(inputs[i].unsqueeze(0))
    new_inputs = torch.cat(new_inputs, dim=0)
    return new_inputs


def repeat_inputs_in_batch(batch: List[torch.Tensor], num: int) -> List[torch.Tensor]:
    """This method is used to concatenate the `num` dimension into the batch dimension."""
    x, y = batch
    new_x = repeat_inputs(x, num)
    return [new_x, y]


def entropy(probabilities: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Compute the entropy of a probability distribution.
    Args:
        probabilities (torch.Tensor): The probability distribution.
        dim (int, optional): The dimension to compute the entropy. Defaults to 1.
    Returns:
        torch.Tensor: The entropy of the distribution.
    """
    return -torch.sum(probabilities * torch.log(probabilities + TINY_EPSILON), dim=dim)


@torch.no_grad()
def classification_uncertainty_decomposition(
    samples: torch.Tensor,
    probabilities: bool = False,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose the output with repsect to predictive, aleatoric and epistemic uncertainties.

    Args:
        samples (torch.Tensor): The samples to decompose. The shape should be `(batch_size, num_members, num_classes)`.
        probabilities (bool, optional): Whether the samples are probabilities. Defaults to False.
        weights (Optional[torch.Tensor], optional): The weights to use for the decomposition. Defaults to None.
    """
    if len(samples.shape) == 2:
        samples = samples.unsqueeze(1)
    if weights is None:
        weights = (
            torch.ones(
                samples.shape[0],
                samples.shape[MEMBERS_DIM],
                device=samples.device,
            )
            / samples.shape[MEMBERS_DIM]
        )
    if not probabilities:
        # Check if the samples are truly probabilities
        assert torch.all(
            torch.sum(samples, dim=2) == 1
        ), f"The samples should be logits, but they sum up to {torch.sum(samples, dim=2)}."
        softmax = F.softmax(samples, dim=2)
    else:
        softmax = samples
    mean_softmax = torch.sum(weights.unsqueeze(-1) * softmax, dim=MEMBERS_DIM)

    predictive = entropy(mean_softmax, dim=1)
    aleatoric = torch.sum(weights * entropy(softmax, dim=2), dim=MEMBERS_DIM)
    epistemic = predictive - aleatoric
    return predictive, aleatoric, epistemic


@torch.no_grad()
def regression_uncertainty_decomposition(
    samples: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose the output with repsect to mean, predictive, aleatoric and epistemic uncertainties.

    Args:
        samples (torch.Tensor): The samples to decompose. The shape should be `(batch_size, num_members, 2, output_dim)`.
        weights (Optional[torch.Tensor], optional): The weights to use for the decomposition. Defaults to None.
    """
    if len(samples.shape) == 2:
        samples = samples.unsqueeze(1)

    if weights is None:
        weights = (
            torch.ones(
                samples.shape[0],
                samples.shape[MEMBERS_DIM],
                device=samples.device,
            )
            / samples.shape[MEMBERS_DIM]
        )

    mean = samples[:, :, 0]
    aleatoric_variance = (
        samples[:, :, 1] if samples.shape[2] > 1 else torch.zeros_like(mean)
    )

    mean = torch.sum(mean * weights, dim=MEMBERS_DIM)
    epistemic_variance = weighted_regression_variance(samples[:, :, 0], weights)
    aleatoric_variance = torch.sum(aleatoric_variance * weights, dim=MEMBERS_DIM)
    predictive_variance = epistemic_variance + aleatoric_variance

    return mean, predictive_variance, aleatoric_variance, epistemic_variance
