from typing import Any, Tuple, List

import torch
import torch.nn as nn
import argparse

from yamle.methods.method import BaseMethod
from yamle.models.specific.temperature_scaling import TemperatureScaler
from yamle.defaults import CLASSIFICATION_KEY, SEGMENTATION_KEY


class TemperatureMethod(BaseMethod):
    """This class is the extension of the base method for temperature scaling.

    Args:
        calibration_learning_rate (float): The learning rate for the calibration.
    """

    tasks = [CLASSIFICATION_KEY, SEGMENTATION_KEY]

    def __init__(self, calibration_learning_rate: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model._output = nn.Sequential(self.model._output, TemperatureScaler())

        self.hparams.calibration_learning_rate = calibration_learning_rate

        self.calibration = False

    def calibrate(self) -> None:
        """This method is used to trigger the calibration."""
        self.calibration = True
        self.model._output[1]._T.requires_grad = True

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Any]]:
        """This method is used to configure the optimizers for the model.
        if the model is not calibrated, then the temperature parameter is not updated.

        if `self.calibration` is True, then only the temperature parameter is updated.
        """
        if not self.calibration:
            return super().configure_optimizers()
        else:
            return [
                torch.optim.LBFGS(
                    [self.model._output[1]._T],
                    lr=self.hparams.calibration_learning_rate,
                    line_search_fn="strong_wolfe",
                )
            ], []

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the specific arguments for the class."""
        parser = super(TemperatureMethod, TemperatureMethod).add_specific_args(
            parent_parser
        )
        parser.add_argument(
            "--method_calibration_learning_rate",
            help="The learning rate for the calibration.",
            type=float,
            default=0.001,
        )
        return parser
