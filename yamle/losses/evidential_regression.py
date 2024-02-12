"""
Adopted from: https://github.com/aamini/evidential-deep-learning/
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from yamle.losses.loss import BaseLoss
from yamle.defaults import TINY_EPSILON, REGRESSION_KEY, DEPTH_ESTIMATION_KEY, RECONSTRUCTION_KEY


class NIG_NLL(nn.Module):
    """Negative log-likelihood loss for Normal Inverse Gamma (NIG) distribution."""

    def forward(
        self,
        y: torch.Tensor,
        gamma: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss function."""
        twoBlambda = 2 * beta * (1 + v)

        nll = (
            0.5 * torch.log(torch.tensor(np.pi) / (v + TINY_EPSILON) + TINY_EPSILON)
            - alpha * torch.log(twoBlambda + TINY_EPSILON)
            + (alpha + 0.5)
            * torch.log(v * (y - gamma) ** 2 + twoBlambda + TINY_EPSILON)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        return torch.mean(nll)


class NIG_Reg(nn.Module):
    """Regularization loss for Normal Inverse Gamma (NIG) distribution."""

    def forward(
        self,
        y: torch.Tensor,
        gamma: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss function."""
        error = torch.abs(y - gamma)
        evi = 2 * v + (alpha)
        reg = error * evi
        return torch.mean(reg)


class EvidentialRegressionLoss(BaseLoss):
    """Evidential regression loss for probabilistic regression."""
    
    tasks = [REGRESSION_KEY, DEPTH_ESTIMATION_KEY, RECONSTRUCTION_KEY]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._nll = NIG_NLL()
        self._reg = NIG_Reg()

    def __call__(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma, v, alpha, beta = torch.split(y_hat, 1, dim=-1)
        loss_nll = self._nll(y, gamma, v, alpha, beta)
        loss_reg = self._reg(y, gamma, v, alpha, beta)
        return loss_nll, loss_reg

    def __repr__(self) -> str:
        return "EvidentialRegression()"
