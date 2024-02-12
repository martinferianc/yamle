from typing import Any
import torch
import argparse

from yamle.methods.method import BaseMethod
from yamle.models.specific.sngp import RFF, spectral_norm
from yamle.defaults import CLASSIFICATION_KEY, SEGMENTATION_KEY


def enable_spectral_normalization(model: torch.nn.Module, coeff: float) -> None:
    """Replace all the layers in the model with spectral normalized layers.

    Args:
        model (torch.nn.Module): The model to enable spectral normalization for.
    """
    for name, child in model.named_children():
        if len(list(child.children())) > 0:
            enable_spectral_normalization(child, coeff)
        else:
            setattr(model, name, spectral_norm(child, coeff))


class SNGPMethod(BaseMethod):
    """This class is the extension of the base method for which the prediciton is performed through the method of:
    Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness


    The core of the method is to 1. enable spectral normalization for all `._residual` layers in the model
    and replace the `._output` layer with a `._output` layer with a Gaussian process

    Args:
        m (float): The gamma for exponential moving average for updating the precision matrix.
        random_features (int): The number of random features to use in the RFF layer.
        mean_field_factor (float): The factor to use for the mean field approximation.
        coeff (float): The coefficient for the spectral normalization.
    """

    tasks = [CLASSIFICATION_KEY, SEGMENTATION_KEY]

    def __init__(
        self,
        m: float = 0.99,
        random_features: int = 512,
        mean_field_factor: float = 1.0,
        coeff: float = 1.0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        enable_spectral_normalization(self.model, coeff=coeff)
        assert isinstance(
            self.model._output, torch.nn.Linear
        ), "The output layer must be a linear layer"
        self.model._output = RFF(
            self.model._output.in_features,
            self.model._output.out_features,
            random_features,
            mean_field_factor,
            m,
        )

    def on_train_epoch_start(self) -> None:
        """In the final epoch we need to update the precision matrix. The update is triggered by the `_final_epoch` flag
        set to `True`."""
        if self.current_epoch == self.trainer.max_epochs - 1:
            self.model._output._final_epoch = True
        return super().on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        if self.model._output._final_epoch:
            self.model._output._final_epoch = False
            self.model._output.compute_covariance()
        return super().on_train_epoch_end()

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the specific arguments for the DUN method."""
        parser = super(SNGPMethod, SNGPMethod).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_m",
            type=float,
            default=0.99,
            help="The gamma for exponential moving average for updating the precision matrix.",
        )
        parser.add_argument(
            "--method_random_features",
            type=int,
            default=512,
            help="The number of random features to use in the RFF layer.",
        )
        parser.add_argument(
            "--method_mean_field_factor",
            type=float,
            default=1.0,
            help="The factor to use for the mean field approximation.",
        )
        parser.add_argument(
            "--method_coeff",
            type=float,
            default=1.0,
            help="The coefficient for the spectral normalization.",
        )
        return parser
