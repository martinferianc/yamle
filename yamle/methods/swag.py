from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
import argparse
import copy

from yamle.methods.method import BaseMethod
from yamle.defaults import (
    MEMBERS_DIM,
    TRAIN_KEY,
    VALIDATION_KEY,
    TEST_KEY,
    TINY_EPSILON,
)
from yamle.evaluation.metrics.algorithmic import metrics_factory

import logging

logging = logging.getLogger("pytorch_lightning")


class SWAGMethod(BaseMethod):
    """This class is the extension of the base method for stochastic weight averaging.

    This method was described in the paper "A Simple Baseline for Bayesian Uncertainty in Deep Learning":
    https://arxiv.org/pdf/1902.02476.pdf.

    Args:
        covariance (bool): Whether to estimate the full covariance matrix.
        fullrank (bool): Whether to use the full rank covariance matrix.
        apply_to_normalisation (bool): Whether to apply the method to the normalisation layers.
        scale (float): The scale of the sampling.
        num_members (int): The number of samples to take when sampling the weights during testing.
        epochs_to_collect (List[int]): The epochs to collect the weights from.
    """

    def __init__(
        self,
        covariance: bool,
        fullrank: bool,
        apply_to_normalisation: bool,
        scale: float,
        num_members: int,
        epochs_to_collect: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._covariance = covariance
        self._fullrank = fullrank
        self._apply_to_normalisation = apply_to_normalisation
        self._scale = scale
        self._num_members = num_members
        assert (
            len(epochs_to_collect) > 1
        ), f"The number of epochs to collect must be greater than 1. Got {len(epochs_to_collect)}."
        assert (
            min(epochs_to_collect) >= 0
        ), f"The epochs to collect must be greater than or equal to 0. Got {min(epochs_to_collect)}."
        self._epochs_to_collect = epochs_to_collect

        self._max_num_of_collected_models = len(self._epochs_to_collect)
        self._num_of_collected_models = 0

        # This list stores the referene to the parameters in the model
        # We add attributes to the parameters to store the mean, square mean and covariance matrix
        self._swag_parameters: List[nn.Parameter] = []

        self._initialise_swag_model()

    def state_dict(self) -> Dict[str, Any]:
        """This method is used to get the state dictionary of the method."""
        state_dict = super().state_dict()
        # Store the actual values of the parameters
        state_dict["swag_parameters"] = [
            {
                "_mean": getattr(p, "_mean", None),
                "_sq_mean": getattr(p, "_sq_mean", None),
                "_cov_mat_sqrt": getattr(p, "_cov_mat_sqrt", None),
                "_training": getattr(p, "_training", None),
            }
            for p in self._swag_parameters
        ]

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """This method is used to load the state dictionary of the method."""
        super().load_state_dict(state_dict)
        # Load the actual values of the parameters
        for p, state in zip(self._swag_parameters, state_dict.pop("swag_parameters")):
            assert (
                p.data.shape == state["_mean"].shape
            ), f"The shape of the mean is not the same as the parameter. Got {p.data.shape} and {state['_mean'].shape}."
            setattr(p, "_training", state["_training"])
            setattr(p, "_mean", state["_mean"])
            setattr(p, "_sq_mean", state["_sq_mean"])
            setattr(p, "_cov_mat_sqrt", state["_cov_mat_sqrt"])

    def _create_metrics(self, metrics_kwargs: Dict[str, Any]) -> None:
        """This method is used to create the metrics to be used for training, validation and testing.

        For the Monte Carlo sampling, we do not care about the individual members.
        """
        self.metrics = {
            TRAIN_KEY: metrics_factory(**metrics_kwargs, per_member=False),
            VALIDATION_KEY: metrics_factory(**metrics_kwargs, per_member=False),
            TEST_KEY: metrics_factory(**metrics_kwargs, per_member=False),
        }

    def _predict(self, x: torch.Tensor, **forward_kwargs: Any) -> torch.Tensor:
        """This method is used to perform a forward pass through the current model."""
        # During training simple give the current model
        if self.training:
            return super()._predict(x, **forward_kwargs)
        else:
            # During testing, we need to sample from the distribution
            outputs = []
            for i in range(self._num_members):
                self._sample()
                self._set_weights_to_sample()
                outputs.append(super()._predict(x, **forward_kwargs))

            return torch.cat(outputs, dim=MEMBERS_DIM)

    def _sample(self) -> None:
        """This method is used to sample from the distribution of the weights."""
        if self._fullrank:
            self._sample_fullrank()
        else:
            self._sample_blockwise()

    def _set_weights_to_sample(self) -> None:
        """This method is used to set the weights to the sampled weights."""
        for p in self._swag_parameters:
            p.data = (
                getattr(p, "_sample").clone().to(next(self.model.parameters()).device)
            )

    def _set_weights_to_training(self) -> None:
        """This method is used to set the weights to the training weights."""
        for p in self._swag_parameters:
            p.data = (
                getattr(p, "_training").clone().to(next(self.model.parameters()).device)
            )

    def _update_training_weights(self) -> None:
        """This method is used to update the training weights."""
        for p in self._swag_parameters:
            setattr(p, "_training", p.data.clone().cpu())

    def _sample_blockwise(self) -> None:
        """This method is used to sample blockwise from the distribution of the weights."""
        for p in self._swag_parameters:
            mean = getattr(p, "_mean", None)
            sq_mean = getattr(p, "_sq_mean", None)

            eps = torch.randn_like(mean)

            var = torch.clamp(sq_mean - mean**2, min=TINY_EPSILON)

            scaled_diag_sample = self._scale * torch.sqrt(var) * eps

            if self._covariance:
                cov_mat_sqrt = getattr(p, "_cov_mat_sqrt", None)
                if cov_mat_sqrt is None:
                    continue
                eps = cov_mat_sqrt.new_empty((cov_mat_sqrt.size(0), 1)).normal_()
                cov_sample = (
                    self._scale / ((self._max_num_of_collected_models - 1) ** 0.5)
                ) * cov_mat_sqrt.t().matmul(eps).view_as(mean)

                if self._fullrank:
                    w = mean + scaled_diag_sample + cov_sample
                else:
                    w = mean + scaled_diag_sample

            else:
                w = mean + scaled_diag_sample

            setattr(p, "_sample", w)

    @staticmethod
    def _flatten_params(
        parameters: List[Union[nn.Parameter, torch.Tensor]]
    ) -> torch.Tensor:
        """This method is used to flatten the parameters."""
        temp = (
            [p.data.view(-1, 1) for p in parameters]
            if isinstance(parameters[0], nn.Parameter)
            else [p.view(-1, 1) for p in parameters]
        )
        return torch.cat(temp, dim=0)

    @staticmethod
    def _unflatten_params_like(
        parameters: torch.Tensor, like: List[Union[nn.Parameter, torch.Tensor]]
    ) -> List[torch.Tensor]:
        """This method is used to unflatten the parameters."""
        temp = []
        index = 0
        for p in like:
            temp.append(parameters[index : index + p.numel()].view(p.size()))
            index += p.numel()
        return temp

    def _sample_fullrank(self) -> None:
        """This method is used to sample from the full rank distribution of the weights."""
        scale_sqrt = self._scale**0.5

        mean_list: List[torch.Tensor] = []
        sq_mean_list: List[torch.Tensor] = []

        if self._covariance:
            cov_mat_sqrt_list = []

        for p in self._swag_parameters:
            mean = getattr(p, "_mean", None)
            sq_mean = getattr(p, "_sq_mean", None)

            if self._covariance:
                cov_mat_sqrt = getattr(p, "_cov_mat_sqrt", None)
                if cov_mat_sqrt is None:
                    continue
                cov_mat_sqrt_list.append(cov_mat_sqrt)

            mean_list.append(mean)
            sq_mean_list.append(sq_mean)

        mean = self._flatten_params(mean_list)
        sq_mean = self._flatten_params(sq_mean_list)

        # Draw diagonal variance sample
        var = torch.clamp(sq_mean - mean**2, min=TINY_EPSILON)
        var_sample = var.sqrt() * torch.randn_like(var)

        # If covariance draw low rank sample
        if self._covariance:
            cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1)

            cov_sample = cov_mat_sqrt.t().matmul(
                cov_mat_sqrt.new_empty((cov_mat_sqrt.size(0),)).normal_()
            )
            cov_sample /= (self._max_num_of_collected_models - 1) ** 0.5

            rand_sample = var_sample + cov_sample
        else:
            rand_sample = var_sample

        # Update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample
        sample = sample.unsqueeze(0)

        # Unflatten new sample like the mean sample
        samples_list = self._unflatten_params_like(sample, mean_list)

        for p, sample in zip(self._swag_parameters, samples_list):
            setattr(p, "_sample", sample)

    def _collect_model(self) -> None:
        """This method is used to collect the model.

        It is done by updating the mean and covariance parameters.
        """
        for p in self._swag_parameters:
            mean = getattr(p, "_mean", None)
            sq_mean = getattr(p, "_sq_mean", None)

            # First moment
            mean = mean * self._num_of_collected_models / (
                self._num_of_collected_models + 1.0
            ) + p.data.detach().clone().cpu() / (self._num_of_collected_models + 1.0)

            # Second moment
            sq_mean = sq_mean * self._num_of_collected_models / (
                self._num_of_collected_models + 1.0
            ) + p.data.detach().clone().cpu() ** 2 / (
                self._num_of_collected_models + 1.0
            )

            # Square root of covariance matrix
            if self._covariance:
                cov_mat_sqrt = getattr(p, "_cov_mat_sqrt", None)

                # Block covariance matrices, store deviation from current mean
                dev = (p.data.detach().clone().cpu() - mean).view(-1, 1)
                cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(-1, 1).t()), dim=0)

                # remove first column if we have stored too many models
                if (
                    self._num_of_collected_models + 1
                ) > self._max_num_of_collected_models:
                    cov_mat_sqrt = cov_mat_sqrt[1:, :]

                # Update the parameters
                setattr(p, "_cov_mat_sqrt", cov_mat_sqrt)

            # Update the parameters
            setattr(p, "_mean", mean)
            setattr(p, "_sq_mean", sq_mean)

        self._num_of_collected_models += 1

    def _initialise_swag_model(self) -> None:
        """This method is used to initialise the SWAG model.

        It iterates through all the parameters in the model and appends references to the parameters
        to a list and initialises the attributes of the parameters.
        """
        for module in self.model.modules():
            # If there are no children it is a leaf module
            if not len(list(module.children())) == 0:
                continue

            if not self._apply_to_normalisation and isinstance(
                module,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.LayerNorm,
                    nn.GroupNorm,
                ),
            ):
                continue
            for p in module.parameters():
                data = p.data
                setattr(p, "_mean", data.new(data.size()).zero_().cpu())
                setattr(p, "_sq_mean", data.new(data.size()).zero_().cpu())
                if self._covariance:
                    setattr(
                        p,
                        "_cov_mat_sqrt",
                        data.new_empty((0, data.numel())).zero_().cpu(),
                    )
                setattr(p, "_training", copy.deepcopy(p.data).cpu())
                self._swag_parameters.append(p)

    def on_train_epoch_end(self) -> None:
        """This method is used to collect the model at the end of each epoch."""
        super().on_train_epoch_end()
        if self._datamodule.validation_dataset() is None:
            if self.current_epoch in self._epochs_to_collect:
                logging.info(f"Collecting the model at epoch {self.current_epoch}.")
                self._collect_model()

            # The validation epoch start is not called if there is no validation dataset
            self._update_training_weights()

    def on_validation_epoch_start(self) -> None:
        """This method is used to cache the training weights."""
        super().on_validation_epoch_start()
        if not self.trainer.sanity_checking:
            if self.current_epoch in self._epochs_to_collect:
                logging.info(f"Collecting the model at epoch {self.current_epoch}.")
                self._collect_model()
            self._update_training_weights()

    def on_train_epoch_start(self) -> None:
        """This method is used to set the model to training mode."""
        super().on_train_epoch_start()
        self._set_weights_to_training()

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the specific arguments for the class."""
        parser = super(SWAGMethod, SWAGMethod).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_covariance",
            type=int,
            choices=[0, 1],
            default=0,
            help="Whether to estimate the full covariance matrix.",
        )
        parser.add_argument(
            "--method_fullrank",
            type=int,
            choices=[0, 1],
            default=0,
            help="Whether to use the full rank covariance matrix.",
        )
        parser.add_argument(
            "--method_apply_to_normalisation",
            type=int,
            choices=[0, 1],
            default=0,
            help="Whether to apply the method to the normalisation layers.",
        )
        parser.add_argument(
            "--method_scale",
            type=float,
            default=1.0,
            help="The scale of the sampling.",
        )
        parser.add_argument(
            "--method_num_members",
            type=int,
            default=1,
            help="The number of members to be used for the prediction. Default: 1.",
        )
        parser.add_argument(
            "--method_epochs_to_collect",
            type=str,
            nargs="+",
            help="The epochs to collect the weights from.",
        )
        return parser
