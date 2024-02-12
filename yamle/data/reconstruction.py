from argparse import ArgumentParser
from typing import Tuple, Union, Any

import torch
from torch.utils.data import TensorDataset
from pytorch_lightning import LightningModule

import scienceplots
import matplotlib.pyplot as plt

plt.style.use("science")

from yamle.data.custom import ECG5000DataModule
from yamle.defaults import (
    RECONSTRUCTION_KEY,
    TRAIN_KEY,
    VALIDATION_KEY,
    TEST_KEY,
    INPUT_KEY,
    TARGET_KEY,
    PREDICTION_KEY,
    AVERAGE_WEIGHTS_KEY,
)

from yamle.utils.operation_utils import regression_uncertainty_decomposition
from yamle.utils.file_utils import plots_file


class ECG5000ReconstructionDataModule(ECG5000DataModule):
    """Reconstruction data module for the ECG5000 dataset.

    If `anomaly` is set to `True`, then all the anomalous cases from the training set are appended to
    the test set. This can be used to create an autoencoder trained only on normal cases, being unable
    to reconstruct the anomalous cases. Thus training an autoencoder for anomaly detection.

    Args:
        anomaly (bool): If `True`, then all the anomalous cases from the training set are appended to
            the test set. Default: `False`.
    """

    outputs_dim = 2
    outputs_dtype = torch.float32
    inputs_dtype = torch.float32
    targets_dim = 140
    task = RECONSTRUCTION_KEY

    def __init__(self, anomaly: bool = False, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if self._train_target_transform is None:
            self._train_target_transform = ["datanormalize"]

        if self._test_target_transform is None:
            self._test_target_transform = ["datanormalize"]

        self._anomaly = anomaly

        self._train_classes: torch.Tensor = None  # The classes of the train data
        self._test_classes: torch.Tensor = None  # The classes of the test data

    def prepare_data(self) -> None:
        """Download and prepare the data"""
        # The train and test data are pandas dataframes
        train_data, test_data = super().prepare_data()

        # Separate the inputs and classes for the train and test data
        train_inputs = train_data.iloc[:, 1:]
        train_classes = train_data.iloc[:, 0] - 1  # The classes start from 1

        test_inputs = test_data.iloc[:, 1:]
        test_classes = test_data.iloc[:, 0] - 1  # The classes start from 1
        
        # Convert the inputs and classes to tensors
        train_inputs = torch.from_numpy(train_inputs.to_numpy()).float()
        train_classes = torch.from_numpy(train_classes.to_numpy()).long()
        
        test_inputs = torch.from_numpy(test_inputs.to_numpy()).float()
        test_classes = torch.from_numpy(test_classes.to_numpy()).long()

        if self._anomaly:
            # Append the anomalous cases to the test set
            anomalous_cases = train_classes != 0
            test_inputs = torch.cat([test_inputs, train_inputs[anomalous_cases]])
            test_classes = torch.cat([test_classes, train_classes[anomalous_cases]])
            # Remove the anomalous cases from the training set
            train_inputs = train_inputs[~anomalous_cases]
            train_classes = train_classes[~anomalous_cases]

        train_targets = train_inputs
        test_targets = test_inputs

        self._train_inputs = train_inputs
        self._test_classes = test_classes

        # Convert them into TensorDatasets
        self._train_dataset = TensorDataset(
            train_inputs.unsqueeze(1)
            .float()
            .permute(0, 2, 1),
            train_targets.unsqueeze(1)
            .float()
            .permute(0, 2, 1),
        )

        self._test_dataset = TensorDataset(
            test_inputs.unsqueeze(1)
            .float()
            .permute(0, 2, 1),
            test_targets.unsqueeze(1)
            .float()
            .permute(0, 2, 1),
        )

        # Calculate the mean and standard deviation of the training data
        self._data_mean, self._data_std = self._mean_std(self._train_dataset, index=0)

        # Calculate the maximum and minimum of the training data
        self._data_max, self._data_min = self._max_min(self._train_dataset, index=0)

    def _get_prediction(
        self,
        tester: LightningModule,
        x: torch.Tensor,
        y: Union[torch.Tensor, int],
        phase: str = TRAIN_KEY,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the prediction, input and target for the given input and target."""
        super()._get_prediction(tester, x, y, phase)
        x = x.to(tester.device)
        y = (
            torch.tensor(y).long().to(tester.device)
            if isinstance(y, int)
            else y.long().to(tester.device)
        )
        if phase == TRAIN_KEY:
            output = tester.training_step([x, y], batch_idx=0)
        elif phase == VALIDATION_KEY:
            output = tester.validation_step([x, y], batch_idx=0)
        elif phase == TEST_KEY:
            output = tester.test_step([x, y], batch_idx=0)
        y_hat = output[PREDICTION_KEY].cpu()
        x = output[INPUT_KEY].cpu()
        y = output[TARGET_KEY].cpu()
        average_weights = (
            output[AVERAGE_WEIGHTS_KEY].cpu() if AVERAGE_WEIGHTS_KEY in output else None
        )
        return y_hat, x, y, average_weights

    def plot(
        self, tester: LightningModule, save_path: str, specific_name: str = ""
    ) -> None:
        """Plots the dataset."""
        super().plot(tester, save_path, specific_name)

        fig, axs = plt.subplots(nrows=3, ncols=10, figsize=(60, 15))
        for i, dataloader in enumerate(
            [
                self.train_dataloader(),
                self.validation_dataloader(),
                self.test_dataloader(),
            ]
        ):
            inputs, targets = next(iter(dataloader))
            outputs = self._get_prediction(tester, inputs, targets, TEST_KEY)
            y_hat, _, _, average_weights = outputs

            (
                mean,
                predictive_variance,
                aleatoric_variance,
                epistemic_variance,
            ) = regression_uncertainty_decomposition(y_hat, weights=average_weights)

            # Convert the variance to standard deviation
            predictive_variance = torch.sqrt(predictive_variance)
            aleatoric_variance = torch.sqrt(aleatoric_variance)
            epistemic_variance = torch.sqrt(epistemic_variance)

            # Renormalize the data
            inputs = inputs * self._data_std + self._data_mean
            targets = targets * self._data_std + self._data_mean
            mean = mean * self._data_std.squeeze() + self._data_mean.squeeze()
            predictive_variance = (
                predictive_variance * self._data_std.squeeze()
                + self._data_mean.squeeze()
            )
            aleatoric_variance = (
                aleatoric_variance * self._data_std.squeeze()
                + self._data_mean.squeeze()
            )
            epistemic_variance = (
                epistemic_variance * self._data_std.squeeze()
                + self._data_mean.squeeze()
            )

            # Plot the input and the target
            # Also plot the variance as a shaded region
            # Plot the aleatoric variance as a shaded region

            # Plot the data
            for j in range(10):
                axs[i, j].plot(inputs[j, :], color="black", label="Input")
                axs[i, j].plot(targets[j, :], color="red", label="Target")
                axs[i, j].plot(mean[j, :], color="blue", linestyle="--", label="Mean")
                axs[i, j].fill_between(
                    torch.arange(140),
                    mean[j, :] - predictive_variance[j, :],
                    mean[j, :] + predictive_variance[j, :],
                    color="blue",
                    alpha=0.3,
                    label="Predictive Uncertainty",
                )
                axs[i, j].fill_between(
                    torch.arange(140),
                    mean[j, :] - aleatoric_variance[j, :],
                    mean[j, :] + aleatoric_variance[j, :],
                    color="orange",
                    alpha=0.3,
                    label="Aleatoric Uncertainty",
                )
                axs[i, j].grid()
                if i == 0 and j == 0:
                    axs[i, j].legend()

        plt.savefig(plots_file(save_path, specific_name), bbox_inches="tight")
        plt.close(fig)
        plt.clf()

    @staticmethod
    def add_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = super(
            ECG5000ReconstructionDataModule, ECG5000ReconstructionDataModule
        ).add_specific_args(parent_parser)

        parser.add_argument(
            "--datamodule_anomaly",
            type=int,
            choices=[0, 1],
            default=0,
            help="If `True`, then all the anomalous cases from the training set are appended to the test set. Default: `False`.",
        )
        return parser
