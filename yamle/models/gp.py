from typing import Tuple, List, Union, Dict, Any

import argparse
import torch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
    LMCVariationalStrategy,
)
from gpytorch.means import ZeroMean, ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal

from yamle.defaults import CLASSIFICATION_KEY


class GPModel(ApproximateGP):
    """This class is used to create a Gaussian Process model with the given parameters.

    Args:
        prior_mean (str): The prior mean function. Can be 'zero' or 'constant'.
        prior_covariance (str): The prior covariance function. Can be 'rbf', 'matern32', 'matern52'.
        inducing_points (torch.Tensor): The inducing points.
        num_latent (int): The latent dimension.
        num_outputs (int): The number of outputs.
        task (str): The task to perform. Either 'classification' or 'regression'.
                    The task determined is `softmax` is used for the output layer.
    """

    def __init__(
        self,
        prior_mean: str,
        prior_covariance: str,
        inducing_points: torch.Tensor,
        num_latent: int,
        num_outputs: int,
        task: str,
    ) -> None:
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0),
            batch_shape=torch.Size([num_latent]),
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        if task == CLASSIFICATION_KEY:
            variational_strategy = LMCVariationalStrategy(
                variational_strategy,
                num_tasks=num_outputs,
                num_latents=num_latent,
                latent_dim=-1,
            )

        super().__init__(variational_strategy)
        assert prior_mean in ["zero", "constant"]
        assert prior_covariance in ["matern32", "matern52", "rbf"]

        if prior_mean == "zero":
            self._prior_mean = ZeroMean(batch_shape=torch.Size([num_latent]))
        elif prior_mean == "constant":
            self._prior_mean = ConstantMean(batch_shape=torch.Size([num_latent]))
        else:
            raise ValueError(f"The prior mean function {prior_mean} is not supported.")

        if prior_covariance == "rbf":
            self._prior_covariance = ScaleKernel(
                RBFKernel(batch_shape=torch.Size([num_latent])),
                batch_shape=torch.Size([num_latent]),
            )
        elif prior_covariance == "matern32":
            self._prior_covariance = ScaleKernel(
                MaternKernel(nu=1.5, batch_shape=torch.Size([num_latent])),
                batch_shape=torch.Size([num_latent]),
            )
        elif prior_covariance == "matern52":
            self._prior_covariance = ScaleKernel(
                MaternKernel(nu=2.5, batch_shape=torch.Size([num_latent])),
                batch_shape=torch.Size([num_latent]),
            )
        else:
            raise ValueError(
                f"The prior covariance function {prior_covariance} is not supported."
            )

        self._task = task

    def forward(
        self,
        x: torch.Tensor,
        staged_output: bool = False,
        input_kwargs: Dict[str, Any] = {},
        output_kwargs: Dict[str, Any] = {},
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """This function is used to perform the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.
            staged_output (bool): Whether to return the intermediate outputs. Not used in this model.
            input_kwargs (Dict[str, Any]): The kwargs for the input layer.
            output_kwargs (Dict[str, Any]): The kwargs for the output layer.
        """
        assert not staged_output, "The staged output is not supported for this model."
        mean = self._prior_mean(x)
        covariance = self._prior_covariance(x)
        return MultivariateNormal(mean, covariance)

    def final_layer(self, x: torch.Tensor) -> torch.Tensor:
        """This function is used to get the final layer output."""
        pass

    def add_method_specific_layers(self, method: str) -> None:
        """This method is used to add method specific layers to the model.

        Args:
            method (str): The method to use.
        """
        pass

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the model specific arguments to the parent parser."""
        pass

    def reset(self) -> None:
        """This function is used to reset the model after each epoch."""
        pass
