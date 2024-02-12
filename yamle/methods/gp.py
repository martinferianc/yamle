from yamle.defaults import (
    LOSS_KEY,
    TARGET_KEY,
    PREDICTION_KEY,
    MEAN_PREDICTION_KEY,
    REGRESSION_KEY,
    CLASSIFICATION_KEY,
    TRAIN_KEY,
    INPUT_KEY,
)
from yamle.data.datamodule import SurrogateDataset
from yamle.models.gp import GPModel
from yamle.methods.method import BaseMethod
from gpytorch.likelihoods import GaussianLikelihood, SoftmaxLikelihood
import gpytorch
import torch
from typing import Any, Dict, List, Optional
import argparse
import logging

logging = logging.getLogger("pytorch_lightning")


class GPMethod(BaseMethod):
    """This class implements the Gaussian Process (GP) method.

    Args:
        prior_mean (str): The prior mean function.
        prior_covariance (str): The prior covariance function.
        num_inducing_points (int): The inducing points.
        num_latent (int): The latent dimension.
    """

    tasks = [CLASSIFICATION_KEY, REGRESSION_KEY]

    def __init__(
        self,
        prior_mean: str = "constant",
        prior_covariance: str = "rbf",
        num_inducing_points: int = 100,
        num_latent: int = 3,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        logging.warn(
            "This method defines it's own model. The model defined in the configuration will be ignored."
        )

        assert num_inducing_points > 0
        self._num_inducing_points = num_inducing_points

        assert num_latent > 0
        self._num_latent = num_latent

        if self._task == CLASSIFICATION_KEY:
            self._likelihood = SoftmaxLikelihood(
                num_classes=self._outputs_dim, mixing_weights=False
            )
            num_outputs = self._outputs_dim
        elif self._task == REGRESSION_KEY:
            self._likelihood = GaussianLikelihood()
            num_outputs = 1
            if self._num_latent != 1:
                raise ValueError(
                    f"Number of latent dimensions must be 1 for regression task. Got {self._num_latent}."
                )
        else:
            raise ValueError(
                f"Task {self._task} not supported by the Gaussian Process method."
            )

        train_dataset = self._datamodule.train_dataset()
        inducing_points = self._get_inducing_points(train_dataset)

        del self.model
        self.model = GPModel(
            prior_mean=prior_mean,
            prior_covariance=prior_covariance,
            inducing_points=inducing_points,
            num_latent=self._num_latent,
            num_outputs=num_outputs,
            task=self._task,
        )

        logging.warning(
            "This method defines it's own loss. The loss defined in the configuration will be ignored."
        )
        del self._loss

        self._loss_mll = gpytorch.mlls.VariationalELBO(
            self._likelihood, self.model, num_data=len(self._datamodule.train_dataset())
        )

    def state_dict(self) -> Dict[str, Any]:
        """Get the state dictionary of the model."""
        state_dict = super().state_dict()
        state_dict["likelihood"] = self._likelihood.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state dictionary of the model."""
        super().load_state_dict(state_dict)
        self._likelihood.load_state_dict(state_dict["likelihood"])

    def get_parameters(self, recurse: bool = True) -> List[torch.nn.Parameter]:
        """This method is used to get the parameters of the model."""
        return list(self.model.parameters(recurse=recurse)) + list(
            self._likelihood.parameters(recurse=recurse)
        )

    def _get_inducing_points(self, dataset: SurrogateDataset) -> torch.Tensor:
        """This method is used to get the inducing points.

        Args:
            dataset (SurrogateDataset): The dataset to be used to get the inducing points.

        Returns:
            torch.Tensor: The inducing points.
        """
        indices = torch.randperm(len(dataset))[: self._num_inducing_points]
        return torch.stack([dataset[i][0] for i in indices], dim=0)

    def _loss_f(self, outputs: Any, targets: torch.Tensor) -> torch.Tensor:
        """A function to compute the loss. It adds the required `-`."""
        return -self._loss_mll(outputs, targets).mean()

    def _predict(self, x: torch.Tensor, **forward_kwargs: Any) -> torch.Tensor:
        """This method is used to perform a forward pass of the model.

        Args:
            x (torch.Tensor): The input to the model.
            **forward_kwargs (Any): The keyword arguments to be passed to the forward pass of the model.
        """
        return self.model(x, **forward_kwargs)

    def _step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
        phase: str = TRAIN_KEY,
    ) -> Dict[str, torch.Tensor]:
        """This method is used to perform a single step.

        Args:
            batch (List[torch.Tensor]): The batch of data.
            **forward_kwargs (Any): The keyword arguments to be passed to the forward pass of the model.
        """
        x, y = batch

        output = self._predict(x.squeeze())
        loss = self._loss_f(output, y.squeeze())
        y_hat = self._likelihood(output)

        if self._task == REGRESSION_KEY:
            mean = y_hat.mean.t().squeeze(-1)
            variance = y_hat.variance.t().squeeze(-1)
            y_hat_mean = torch.stack([mean, variance], dim=1)
            y_hat = y_hat_mean.unsqueeze(1)
        elif self._task == CLASSIFICATION_KEY:
            y_hat_mean = y_hat.probs.mean(dim=0)
            y_hat = y_hat.probs.permute(1, 0, 2)

        return {
            LOSS_KEY: loss,
            PREDICTION_KEY: y_hat.detach(),
            MEAN_PREDICTION_KEY: y_hat_mean.detach(),
            TARGET_KEY: y.detach(),
            INPUT_KEY: x.detach(),
        }

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = super(GPMethod, GPMethod).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_prior_mean",
            type=str,
            default="constant",
            help="The prior mean function.",
        )
        parser.add_argument(
            "--method_prior_covariance",
            type=str,
            default="matern32",
            help="The prior covariance function.",
        )
        parser.add_argument(
            "--method_num_inducing_points",
            type=int,
            default=100,
            help="The inducing points.",
        )
        parser.add_argument(
            "--method_num_latent", type=int, default=1, help="The latent dimension."
        )
        return parser
