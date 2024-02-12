import argparse
from typing import Any

from torch.utils.data import DataLoader

from yamle.trainers.trainer import BaseTrainer


class CalibrationTrainer(BaseTrainer):
    """This class defines a temperature trainer which first trains the model and then calibrates it.
    
    The training is on the training set and the calibration is on the calibration set.

    Args:
        calibration_epochs (int): The number of epochs to calibrate the model.
    """

    def __init__(self, calibration_epochs: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._calibration_epochs = calibration_epochs

    def fit(self, train_dataloader: DataLoader, validation_dataloader: DataLoader) -> float:
        """This method trains the method and then does the calibration.

        Args:
            train_dataloader (DataLoader): The dataloader to be used for training.
            validation_dataloader (DataLoader): The dataloader to be used for validation.
        """
        training_time = super().fit(train_dataloader, validation_dataloader)
        calibration_dataloader = self._datamodule.calibration_dataloader()
        if not hasattr(self._method, "calibrate"):
            raise ValueError("Make sure that the method has a calibrate method.")
        self._method.calibrate()
        self._initialize_trainer(epochs=self._calibration_epochs)
        calibration_time = super().fit(calibration_dataloader, validation_dataloader)
        return training_time + calibration_time

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method adds trainer arguments to the given parser.

        Args:
            parent_parser (ArgumentParser): The parser to which the arguments should be added.
        """
        parser = super(CalibrationTrainer, CalibrationTrainer).add_specific_args(
            parent_parser
        )
        parser.add_argument(
            "--trainer_calibration_epochs",
            type=int,
            default=10,
            help="The number of epochs to be used for training.",
        )
        return parser
