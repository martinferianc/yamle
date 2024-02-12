from typing import Any, List
from yamle.regularizers.regularizer import BaseRegularizer

import torch
import argparse


class L1FeatureRegularizer(BaseRegularizer):
    """This is a class for L1 regularization for the output features."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to calculate the regularization loss."""
        batch_size = x.shape[0]
        return torch.abs(x).sum() / batch_size

    def __repr__(self) -> str:
        return f"L1FeatureRegularizer()"


class L2FeatureRegularizer(BaseRegularizer):
    """This is a class for L2 regularization for the output features."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to calculate the regularization loss."""
        batch_size = x.shape[0]
        return (torch.sum(x**2) * 0.5) / batch_size

    def __repr__(self) -> str:
        return f"L2Feature()"


class InnerProductFeatureRegularizer(BaseRegularizer):
    """This is a class for inner product regularization.

    Given a tensor `x` which can be split in dimension `dim` into `n` tensors `x_1, ..., x_n`, the regularization loss is calculated as:

    `loss = sum_{i=1}^{n} sum_{j=i+1}^{n} x_i * x_j`
    `loss = loss / (n*(n-1)/2)`

    Args:
        dim (int): The dimension over which split the tensor to then calculate the inner product as a cartesian product.
    """

    def __init__(self, dim: int = 1, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._dim = dim

    def _split_and_reshape_tensor_on_dim(self, x: torch.Tensor) -> List[torch.Tensor]:
        """This method is used to split the tensor on the given dimension and then reshape it."""
        batch_size = x.shape[0]
        x = torch.split(x, 1, dim=self._dim)
        x = [x_.squeeze(dim=self._dim).view(batch_size, -1) for x_ in x]
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to calculate the regularization loss."""
        loss = 0.0
        x = self._split_and_reshape_tensor_on_dim(x)
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                loss += torch.sum(x[i] * x[j], dim=1).mean()
        return loss / (len(x) * (len(x) - 1) / 2)

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """This method is used to add specific arguments to the parser."""
        parser = super(
            InnerProductFeatureRegularizer, InnerProductFeatureRegularizer
        ).add_specific_args(parser)
        parser.add_argument(
            "--regularizer_dim",
            type=int,
            default=1,
            help="The dimension over which split the tensor to then calculate the inner product as a cartesian product.",
        )
        return parser

    def __repr__(self) -> str:
        return f"InnerProductFeatureRegularizer(dim={self._dim})"


class CosineSimilarityFeatureRegularizer(InnerProductFeatureRegularizer):
    """This is a class for cosine similarity regularization.

    Given a tensor `x` which can be split in dimension `dim` into `n` tensors `x_1, ..., x_n`, the regularization loss is calculated as:

    `loss = sum_{i=1}^{n} sum_{j=i+1}^{n} cos(x_i, x_j)`
    `loss = loss / (n*(n-1)/2)`

    The `cos` function is the cosine similarity between `x_i` and `x_j`.
    `cos(x_i, x_j) = x_i * x_j / (||x_i|| * ||x_j||)`
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to calculate the regularization loss."""
        loss = 0.0
        x = self._split_and_reshape_tensor_on_dim(x)
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                loss += torch.cosine_similarity(x[i], x[j], dim=1).mean()
        return loss / (len(x) * (len(x) - 1) / 2)

    def __repr__(self) -> str:
        return f"CosineSimilarityFeatureRegularizer(dim={self._dim})"


class CorrelationFeatureRegularizer(CosineSimilarityFeatureRegularizer):
    """This is a class for correlation regularization.

    Correlation is the cosine similarity between centered versions of x and y.
    Unlike the cosine, the correlation is invariant to both scale and location changes of x and y.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to calculate the regularization loss."""
        x = self._split_and_reshape_tensor_on_dim(x)
        for i in range(len(x)):
            x[i] = x[i] - x[i].mean(dim=1, keepdim=True)
        x = torch.stack(x, dim=self._dim)
        return super().__call__(x)

    def __repr__(self) -> str:
        return f"CorrelationFeatureRegularizer(dim={self._dim})"
