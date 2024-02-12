from typing import List, Dict, Any
import torch
import argparse
import torch.nn as nn

from yamle.methods.uncertain_method import SVIMethod
from yamle.models.specific.rbnn import LinearRBNN, Conv2dRBNN
from yamle.utils.operation_utils import repeat_inputs_in_batch, average_predictions
from yamle.defaults import LOSS_KEY, TARGET_KEY, PREDICTION_KEY, MEAN_PREDICTION_KEY


def replace_with_rbnn(
    model: nn.Module,
    num_members: int,
    prior_mean: float,
    log_variance,
    prior_log_variance: float,
    method: str,
) -> nn.Module:
    """This method is used to replace all the `nn.Linear`, `nn.Conv2d` layers
       with a `LinearRBNN` and `Conv2dRBNN` respectively.

    Args:
        model (nn.Module): The model to replace the layers in.
        num_members (int): The number of members in the ensemble.
        prior_mean (float): The mean of the prior distribution.
        log_variance (float): The initial value of the log of the standard deviation of the weights.
        prior_log_variance (float): The initial value of the log of the standard deviation of the prior distribution.
        method (str): The method whether `additive` or `multiplicative` to be used for the rank-1 approximation.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(
                model,
                name,
                LinearRBNN(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    num_members=num_members,
                    prior_mean=prior_mean,
                    log_variance=log_variance,
                    prior_log_variance=prior_log_variance,
                    method=method,
                ),
            )
        elif isinstance(child, nn.Conv2d):
            setattr(
                model,
                name,
                Conv2dRBNN(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias is not None,
                    num_members=num_members,
                    prior_mean=prior_mean,
                    log_variance=log_variance,
                    prior_log_variance=prior_log_variance,
                    method=method,
                ),
            )
        else:
            replace_with_rbnn(
                child, num_members, prior_mean, log_variance, prior_log_variance, method
            )


class RBNNMethod(SVIMethod):
    """This class is the extension of the base method for Rank-1 Bayesian Neural Networks.

    Args:
        num_members (int): The number of members in the ensemble.
        prior_mean (float): The mean of the prior distribution.
        log_variance (float): The initial value of the log of the standard deviation of the weights.
        prior_log_variance (float): The initial value of the log of the standard deviation of the prior distribution.
        method (str): The method whether `additive` or `multiplicative` to be used for the rank-1 approximation.
    """

    def __init__(
        self,
        prior_mean: float = 1.0,
        log_variance: float = -3.0,
        prior_log_variance: float = -3.0,
        method: str = "additive",
        **kwargs
    ):
        super().__init__(**kwargs)
        assert method in ["additive", "multiplicative"]
        replace_with_rbnn(
            self.model,
            self._num_members,
            prior_mean,
            log_variance,
            prior_log_variance,
            method,
        )

    def _validation_test_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """This method is used to perform a single validation/test step.

        It assumes that the batch has a shape `(batch_size, num_features)`.
        It assumes that the output of the model has a shape `(batch_size, n_samples, num_classes)`.
        """
        x, y = repeat_inputs_in_batch(batch, self._num_members)
        y_hat = self._predict(x)
        y_hat = y_hat.reshape(
            -1, self._num_members * self._num_members, *y_hat.shape[2:]
        )
        loss = self._loss(y_hat, y)
        y_hat_mean = average_predictions(y_hat, self._task)
        output = {
            LOSS_KEY: loss,
            TARGET_KEY: y.detach(),
            PREDICTION_KEY: y_hat.detach(),
            MEAN_PREDICTION_KEY: y_hat_mean.detach(),
        }
        return output

    def _validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """This method is used to perform a single validation step."""
        return self._validation_test_step(batch, batch_idx)

    def _test_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """This method is used to perform a single test step."""
        return self._validation_test_step(batch, batch_idx)

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method adds the specific arguments for the MIMO method."""
        parser = super(RBNNMethod, RBNNMethod).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_prior_mean",
            type=float,
            default=1.0,
            help="The mean of the prior distribution.",
        )
        parser.add_argument(
            "--method_log_variance",
            type=float,
            default=-3.0,
            help="The initial value of the log of the standard deviation of the weights.",
        )
        parser.add_argument(
            "--method_prior_log_variance",
            type=float,
            default=-3.0,
            help="The initial value of the log of the standard deviation of the prior distribution.",
        )
        parser.add_argument(
            "--method_method",
            type=str,
            default="additive",
            help="The method whether `additive` or `multiplicative` to be used for the rank-1 approximation.",
        )
        return parser
