import argparse
import copy
import datetime
import glob
import os
import re
import shutil
from typing import Any, Callable, Optional, Tuple, Union
from urllib import request

import matplotlib.pyplot as plt
import medmnist
import pandas as pd
import scienceplots
import torch
import torchvision
from PIL import Image
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset, Subset, TensorDataset
from torchvision.datasets.utils import download_and_extract_archive

from yamle.data.datamodule import (
    BaseDataModule,
    RealWorldDataModule,
    RealWorldRegressionDataModule,
    VisionRegressionDataModule,
)
from yamle.data.dataset_wrappers import ImageRotationDataset, InputImagePaddingDataset
from yamle.third_party.imagenet_c import VisionCorruption
from yamle.defaults import (
    INPUT_KEY,
    PREDICTION_KEY,
    REGRESSION_KEY,
    TARGET_KEY,
    TEST_KEY,
    TRAIN_KEY,
    VALIDATION_KEY,
    AVERAGE_WEIGHTS_KEY
)
from yamle.third_party.medmnist import MedMNISTDatasetWrapper
from yamle.third_party.tinyimagenet import TinyImageNet
from yamle.utils.file_utils import plots_file
from yamle.utils.operation_utils import regression_uncertainty_decomposition

plt.style.use("science")


class ToyRegressionDataModule(BaseDataModule):
    """Data module for a toy regression 1D dataset.

    Implements a toy regression dataset with 1D inputs and 1D outputs.

    \$ y = sin(3x) + x^2 - 0.3 + \mathcal{N}(0, 0.2) + \mathcal{n}(0, `gaussian_noise_sigma`) \$ on the interval \$ [-1, 1] \$.
    but with a gap in the middle from \$ [-0.2, 0.2] \$.

    The validation and test data is on intervals \$ [-2, 2] \$ and witout $\mathcal{N}(0, 1)$.

    Args:
        num_samples (int): Number of samples to generate.
        test_samples (int): Number of samples to generate for the test set.
    """

    outputs_dim = 2
    targets_dim = 1
    outputs_dtype = torch.float
    inputs_dim = (1,)
    task = REGRESSION_KEY

    def __init__(
        self, num_samples: int, test_samples: int, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._num_samples = num_samples
        self._test_samples = test_samples
        self._generator = torch.Generator(device="cpu").manual_seed(self._seed)
        self._original_test_dataset: torch.utils.data.Dataset = None

    def _ground_truth(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the ground truth function."""
        return torch.sin(3 * x) + x**2 - 0.3

    def prepare_data(self) -> None:
        """Generate the toy regression dataset."""
        super().prepare_data()
        train_size = int((1 - self._validation_portion) * self._num_samples)
        validation_size = int(self._validation_portion * self._num_samples)

        train_x = torch.ones(train_size // 2, 1).uniform_(
            -1, -0.2, generator=self._generator
        )
        train_x = torch.cat(
            (
                train_x,
                torch.ones(train_size // 2, 1).uniform_(
                    0.2, 1, generator=self._generator
                ),
            ),
            dim=0,
        ).squeeze()
        train_y = (
            self._ground_truth(train_x)
            + torch.randn(train_size, generator=self._generator) * 0.2
        )

        train_x.unsqueeze_(1)
        train_y.unsqueeze_(1)
        train_dataset = torch.utils.data.TensorDataset(train_x, train_y)

        validation_x = torch.linspace(-2, 2, validation_size)
        validation_y = self._ground_truth(validation_x)
        validation_x.unsqueeze_(1)
        validation_y.unsqueeze_(1)
        validation_dataset = torch.utils.data.TensorDataset(validation_x, validation_y)

        test_x = torch.linspace(-2, 2, self._test_samples)
        test_y = self._ground_truth(test_x)
        test_x.unsqueeze_(1)
        test_y.unsqueeze_(1)
        test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

        self._train_dataset, self._validation_dataset, self._test_dataset = (
            train_dataset,
            validation_dataset,
            test_dataset,
        )

    @torch.no_grad()
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
        y = y.to(tester.device)
        if phase == TRAIN_KEY:
            output = tester.training_step([x, y], batch_idx=0)
        elif phase == VALIDATION_KEY:
            output = tester.validation_step([x, y], batch_idx=0)
        elif phase == TEST_KEY:
            output = tester.test_step([x, y], batch_idx=0)
        y_hat = output[PREDICTION_KEY].cpu()
        x = output[INPUT_KEY].cpu()
        y = output[TARGET_KEY].cpu()
        average_weights = output[AVERAGE_WEIGHTS_KEY].cpu() if AVERAGE_WEIGHTS_KEY in output else None
        return y_hat, x, y, average_weights

    @torch.no_grad()
    def plot(
        self, tester: LightningModule, save_path: str, specific_name: str = ""
    ) -> None:
        """Plot the data and the uncertainty of the model."""
        train_x, train_y = self._train_dataset._dataset.tensors
        test_x, test_y = self._test_dataset._dataset.tensors

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

        # Generate points from the ground truth function for plotting.
        x_min, x_max = test_x.min() - 1.0, test_x.max() + 1.0
        y_min, y_max = test_y.min() - 1.0, test_y.max() + 1.0
        x = torch.linspace(x_min, x_max, 1000).unsqueeze(1)
        y = self._ground_truth(x)
        output = self._get_prediction(tester, x, y, TEST_KEY)
        predictions = output[0]
        average_weights = output[3] if output[3] is not None else None
        (
            mean,
            total_uncertainty,
            aleatoric_uncertainty,
            epistemic_uncertainty,
        ) = regression_uncertainty_decomposition(predictions, average_weights)

        for i in range(3):
            axs[i].plot(
                x.squeeze().cpu(),
                y.squeeze().cpu(),
                label="Ground Truth",
                color="black",
                linestyle="dashed",
            )
            axs[i].plot(x.squeeze().cpu(), mean.squeeze(), label="Mean", color="blue")
            axs[i].scatter(
                train_x.squeeze(),
                train_y.squeeze(),
                color="red",
                label="Train",
                marker="o",
            )
            axs[i].scatter(
                test_x.squeeze(),
                test_y.squeeze(),
                color="green",
                label="Test",
                marker="x",
            )
            if i == 0:
                axs[i].set_title(f"Total Uncertainty")
                axs[i].fill_between(
                    x.squeeze().cpu(),
                    (mean - total_uncertainty).squeeze(),
                    (mean + total_uncertainty).squeeze(),
                    alpha=0.5,
                )
            elif i == 1:
                axs[i].set_title(f"Epistemic Uncertainty")
                axs[i].fill_between(
                    x.squeeze().cpu(),
                    (mean - epistemic_uncertainty).squeeze(),
                    (mean + epistemic_uncertainty).squeeze(),
                    alpha=0.5,
                )
            else:
                axs[i].set_title(f"Aleatoric Uncertainty")
                axs[i].fill_between(
                    x.squeeze().cpu(),
                    (mean - aleatoric_uncertainty).squeeze(),
                    (mean + aleatoric_uncertainty).squeeze(),
                    alpha=0.5,
                )
            axs[i].set_xlim(x_min, x_max)
            axs[i].set_ylim(y_min, y_max)
            axs[i].set_xlabel("x")
            axs[i].set_ylabel("y")
            axs[i].legend()

        plt.tight_layout()
        plt.savefig(plots_file(save_path, specific_name), bbox_inches="tight")
        plt.close(fig)
        plt.clf()

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Add dataset specific arguments to the parser."""
        parser = super(
            ToyRegressionDataModule, ToyRegressionDataModule
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--datamodule_num_samples",
            type=int,
            default=300,
            help="Number of samples to generate.",
        )
        parser.add_argument(
            "--datamodule_test_samples",
            type=int,
            default=200,
            help="Number of samples to generate for testing.",
        )
        return parser


class UCIRegressionDataModule(RealWorldRegressionDataModule):
    """Data module for the UCI regression datasets.

    Currently supports the following datasets:
        - Concrete
        - Energy
        - Boston
        - Wine
        - Yacht
        - Abalone
        - Telemonitoring

    Args:
        dataset (str): Name of the dataset to use.
    """

    task = REGRESSION_KEY
    outputs_dim = 2
    outputs_dtype = torch.float
    inputs_dtype = torch.float

    def __init__(self, dataset: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if dataset not in [
            "concrete",
            "energy",
            "boston",
            "wine",
            "yacht",
            "abalone",
            "telemonitoring",
        ]:
            raise ValueError(
                "Dataset must be one of the following: concrete, energy, boston, wine, yacht, abalone, telemonitoring."
            )
        self._dataset = dataset

    def plot(
        self, tester: LightningModule, save_path: str, specific_name: str = ""
    ) -> None:
        """Plots the dataset."""
        for dataloader in [
            self.train_dataloader(),
            self.validation_dataloader(),
            self.test_dataloader(),
        ]:
            inputs, targets = next(iter(dataloader))
            outputs = self._get_prediction(tester, inputs, targets, TEST_KEY)[0]
            # The first feature is the mean and the second is the standard deviation.
            for i in range(inputs.shape[0]):
                with open(
                    os.path.join(save_path, f"predictions_{specific_name}.txt"), "a"
                ) as f:
                    f.write(
                        f"Input: {inputs[i, :] * self._data_std + self._data_mean}\n"
                    )
                    f.write(
                        f"Output: {outputs[i, 0] * self._target_std + self._target_mean} +- {outputs[i, 1] * self._target_std}\n"
                    )
                    f.write(
                        f"Target: {targets[i] * self._target_std + self._target_mean}\n"
                    )


class MedMNISTRegressionDataModule(VisionRegressionDataModule):
    """Data module for the MedMNIST dataset.


    Args:
        dataset (str): Name of the dataset to use.
        pad_to_32 (bool): Whether to pad the images to 32x32. Defaults to True.
    """

    outputs_dtype = torch.float32

    def __init__(
        self, dataset: str, pad_to_32: bool = True, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        if dataset not in ["retinamnist"]:
            raise ValueError("Dataset not supported.")
        self._dataset = dataset
        self._pad_to_32 = pad_to_32
        if self._pad_to_32:
            self.inputs_dim = (self.inputs_dim[0], 32, 32)

    def prepare_data(self) -> None:
        """Download and prepare the data, the data is stored in `self._train_dataset`, `self._validation_dataset` and `self._test_dataset`."""
        super().prepare_data()
        if self._dataset == "retinamnist":
            self._train_dataset = MedMNISTDatasetWrapper(
                medmnist.RetinaMNIST(root=self._data_dir, split="train", download=True),
                pad_to_32=self._pad_to_32,
                target_normalization=True,
            )
            self._validation_dataset = MedMNISTDatasetWrapper(
                medmnist.RetinaMNIST(root=self._data_dir, split="val", download=True),
                pad_to_32=self._pad_to_32,
                target_normalization=True,
            )
            self._test_dataset = MedMNISTDatasetWrapper(
                medmnist.RetinaMNIST(root=self._data_dir, split="test", download=True),
                pad_to_32=self._pad_to_32,
                target_normalization=True,
            )
        else:
            raise ValueError("Dataset not supported.")

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Add dataset specific arguments to the parser."""
        parser = super(
            MedMNISTRegressionDataModule, MedMNISTRegressionDataModule
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--datamodule_pad_to_32",
            type=int,
            choices=[0, 1],
            default=1,
            help="Whether to pad the images to 32x32.",
        )
        return parser


class RetinaMNISTDataModule(MedMNISTRegressionDataModule):
    """Data module for the RetinaMNIST dataset."""

    inputs_dim = (3, 28, 28)
    outputs_dim = 2
    targets_dim = 1
    mean = (0.3974, 0.2446, 0.1554)
    std = (0.2977, 0.2001, 0.1503)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("retinamnist", *args, **kwargs)


class ConcreteUCIRegressionDataModule(UCIRegressionDataModule):
    """Data module for the Concrete UCI regression dataset."""

    inputs_dim = (8,)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset="concrete", *args, **kwargs)

    def _download_data(self) -> TensorDataset:
        """Downloads the dataset."""
        # Test if the data is already downloaded.
        if not os.path.exists(
            os.path.join(os.path.expanduser(self._data_dir), "Concrete_Data.xls")
        ):
            request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
                "Concrete_Data.xls",
            )
            shutil.move(
                "Concrete_Data.xls",
                os.path.join(os.path.expanduser(self._data_dir), "Concrete_Data.xls"),
            )
        data = pd.read_excel(os.path.join(self._data_dir, "Concrete_Data.xls"))
        data = data.dropna().to_numpy()
        data = torch.from_numpy(data).float()
        inputs, targets = data[:, :-1], data[:, -1]
        return TensorDataset(inputs, targets)


class EnergyUCIRegressionDataModule(UCIRegressionDataModule):
    """Data module for the Energy UCI regression dataset."""

    inputs_dim = (9,)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset="energy", *args, **kwargs)

    def _download_data(self) -> TensorDataset:
        """Downloads the dataset."""
        # Test if the data is already downloaded.
        if not os.path.exists(
            os.path.join(os.path.expanduser(self._data_dir), "ENB2012_data.xlsx")
        ):
            request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
                "ENB2012_data.xlsx",
            )
            shutil.move(
                "ENB2012_data.xlsx",
                os.path.join(os.path.expanduser(self._data_dir), "ENB2012_data.xlsx"),
            )
        data = pd.read_excel(os.path.join(self._data_dir, "ENB2012_data.xlsx"))
        data = data.dropna().to_numpy()
        data = torch.from_numpy(data).float()
        inputs, targets = data[:, :-1], data[:, -1]
        return TensorDataset(inputs, targets)


class BostonUCIRegressionDataModule(UCIRegressionDataModule):
    """Data module for the Boston UCI regression dataset."""

    inputs_dim = (13,)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset="boston", *args, **kwargs)

    def _download_data(self) -> TensorDataset:
        """Downloads the dataset."""
        # Test if the data is already downloaded.
        if not os.path.exists(
            os.path.join(os.path.expanduser(self._data_dir), "housing.data")
        ):
            request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                "housing.data",
            )
            shutil.move(
                "housing.data",
                os.path.join(os.path.expanduser(self._data_dir), "housing.data"),
            )
        data = pd.read_csv(
            os.path.join(self._data_dir, "housing.data"),
            delim_whitespace=True,
            header=None,
        )
        data = data.dropna().to_numpy()
        data = torch.from_numpy(data).float()
        inputs, targets = data[:, :-1], data[:, -1]
        return TensorDataset(inputs, targets)


class WineQualityUCIRegressionDataModule(UCIRegressionDataModule):
    """Data module for the Wine Quality UCI regression dataset."""

    inputs_dim = (11,)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset="wine", *args, **kwargs)

    def _download_data(self) -> TensorDataset:
        """Downloads the dataset."""
        # Test if the data is already downloaded.
        if not os.path.exists(
            os.path.join(os.path.expanduser(self._data_dir), "winequality-red.csv")
        ):
            request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                "winequality-red.csv",
            )
            shutil.move(
                "winequality-red.csv",
                os.path.join(os.path.expanduser(self._data_dir), "winequality-red.csv"),
            )
        data = pd.read_csv(os.path.join(self._data_dir, "winequality-red.csv"), sep=";")
        data = data.dropna().to_numpy()
        data = torch.from_numpy(data).float()
        inputs, targets = data[:, :-1], data[:, -1]
        return TensorDataset(inputs, targets)


class YachtUCIRegressionDataModule(UCIRegressionDataModule):
    """Data module for the Yacht Hydrodynamics UCI regression dataset."""

    inputs_dim = (6,)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset="yacht", *args, **kwargs)

    def _download_data(self) -> TensorDataset:
        """Downloads the dataset."""
        # Test if the data is already downloaded.
        if not os.path.exists(
            os.path.join(os.path.expanduser(self._data_dir), "yacht_hydrodynamics.data")
        ):
            request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
                "yacht_hydrodynamics.data",
            )
            shutil.move(
                "yacht_hydrodynamics.data",
                os.path.join(
                    os.path.expanduser(self._data_dir), "yacht_hydrodynamics.data"
                ),
            )
        data = pd.read_csv(
            os.path.join(self._data_dir, "yacht_hydrodynamics.data"),
            delim_whitespace=True,
            header=None,
        )
        data = data.dropna().to_numpy()
        data = torch.from_numpy(data).float()
        inputs, targets = data[:, :-1], data[:, -1]
        return TensorDataset(inputs, targets)


class AbaloneUCIRegressionDataModule(UCIRegressionDataModule):
    """Data module for the Abalone UCI regression dataset."""

    inputs_dim = (10,)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset="abalone", *args, **kwargs)

    def _download_data(self) -> TensorDataset:
        """Downloads the dataset."""
        # Test if the data is already downloaded.
        if not os.path.exists(
            os.path.join(os.path.expanduser(self._data_dir), "abalone.data")
        ):
            request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
                "abalone.data",
            )
            shutil.move(
                "abalone.data",
                os.path.join(os.path.expanduser(self._data_dir), "abalone.data"),
            )
        data = pd.read_csv(
            os.path.join(self._data_dir, "abalone.data"),
            delim_whitespace=True,
            header=None,
        )

        # Replace the sex column with one hot encoding
        data = pd.get_dummies(data, columns=[0])
        data = data.dropna().to_numpy()
        data = torch.from_numpy(data).float()
        inputs, targets = data[:, :-1], data[:, -1]
        return TensorDataset(inputs, targets)


class TelemonitoringUCIRegressionDataModule(UCIRegressionDataModule):
    """Data module for the Telemonitoring UCI regression dataset.

    We use the `subject#` column to split the data manually
    between train, validation, calibration and test sets to ensure that a patient
    does not appear in more than one set.

    We predict the `total_UPDRS` column.

    """

    inputs_dim = (20,)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset="telemonitoring", *args, **kwargs)

    def prepare_data(self) -> None:
        """Prepares the data for training, validation, calibration and testing."""
        super(RealWorldDataModule, self).prepare_data()
        data = self._download_data()
        (
            self._train_dataset,
            self._validation_dataset,
            self._calibration_dataset,
            self._test_dataset,
        ) = self._split_data(data)

        self._data_mean, self._data_std = self._mean_std(self._train_dataset, 0)
        self._target_mean, self._target_std = self._mean_std(self._train_dataset, 1)

    def _download_data(self) -> pd.DataFrame:
        """Downloads the dataset."""
        # Test if the data is already downloaded.
        if not os.path.exists(
            os.path.join(os.path.expanduser(self._data_dir), "parkinsons_updrs.data")
        ):
            request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data",
                "parkinsons_updrs.data",
            )
            shutil.move(
                "parkinsons_updrs.data",
                os.path.join(
                    os.path.expanduser(self._data_dir), "parkinsons_updrs.data"
                ),
            )
        data = pd.read_csv(
            os.path.join(self._data_dir, "parkinsons_updrs.data"),
            delim_whitespace=True,
            header=0,
        )
        data = data.dropna()
        return data

    def _split_data(
        self, data: pd.DataFrame
    ) -> Tuple[
        TensorDataset, Optional[TensorDataset], Optional[TensorDataset], TensorDataset
    ]:
        """This function splits the data into train, validation, calibration and test sets."""
        num_patients = len(data["subject#"].unique())
        train_size = int(
            (
                1
                - self._validation_portion
                - self._test_portion
                - self._calibration_portion
            )
            * num_patients
        )
        validation_size = int(self._validation_portion * num_patients)
        calibration_size = int(self._calibration_portion * num_patients)

        # Get the unique patient ids.
        patient_ids = data["subject#"].unique()

        # Split the patient ids into train, validation, calibration and test sets based on the ids.
        # Shuffle the ids.
        patient_ids = patient_ids[
            torch.randperm(
                len(patient_ids),
                generator=torch.Generator(device="cpu").manual_seed(self._seed),
            )
        ]
        train_patient_ids = patient_ids[:train_size]
        validation_patient_ids = patient_ids[train_size : train_size + validation_size]
        calibration_patient_ids = patient_ids[
            train_size
            + validation_size : train_size
            + validation_size
            + calibration_size
        ]
        test_patient_ids = patient_ids[
            train_size + validation_size + calibration_size :
        ]

        assert len(train_patient_ids) > 0, f"Train size needs to be greater than 0."
        assert len(test_patient_ids) > 0, f"Test size needs to be greater than 0."

        # Get the data for each patient.
        train_data = data[data["subject#"].isin(train_patient_ids)]
        validation_data = data[data["subject#"].isin(validation_patient_ids)] if (
            validation_size > 0
        ) else None
        calibration_data = data[data["subject#"].isin(calibration_patient_ids)] if (
            calibration_size > 0
        ) else None
        test_data = data[data["subject#"].isin(test_patient_ids)]

        # Remove the patient ids from the data.
        train_data = train_data.drop(columns=["subject#"])
        validation_data = (
            validation_data.drop(columns=["subject#"])
            if validation_data is not None
            else None
        )
        calibration_data = (
            calibration_data.drop(columns=["subject#"])
            if calibration_data is not None
            else None
        )
        test_data = test_data.drop(columns=["subject#"])

        # Select the features and targets.
        train_targets, train_inputs = train_data["total_UPDRS"], train_data.drop(
            columns=["total_UPDRS"]
        )
        if validation_data is not None:
            validation_targets, validation_inputs = validation_data[
                "total_UPDRS"
            ], validation_data.drop(columns=["total_UPDRS"])
        if calibration_data is not None:
            calibration_targets, calibration_inputs = calibration_data[
                "total_UPDRS"
            ], calibration_data.drop(columns=["total_UPDRS"])
        test_targets, test_inputs = test_data["total_UPDRS"], test_data.drop(
            columns=["total_UPDRS"]
        )

        # Convert the data to tensors.
        train_inputs, train_targets = (
            torch.from_numpy(train_inputs.to_numpy()).float(),
            torch.from_numpy(train_targets.to_numpy()).float(),
        )
        if validation_data is not None:
            validation_inputs, validation_targets = (
                torch.from_numpy(validation_inputs.to_numpy()).float(),
                torch.from_numpy(validation_targets.to_numpy()).float(),
            )
        if calibration_data is not None:
            calibration_inputs, calibration_targets = (
                torch.from_numpy(calibration_inputs.to_numpy()).float(),
                torch.from_numpy(calibration_targets.to_numpy()).float(),
            )
        test_inputs, test_targets = (
            torch.from_numpy(test_inputs.to_numpy()).float(),
            torch.from_numpy(test_targets.to_numpy()).float(),
        )

        # Create the TensorDatasets.
        train_dataset = TensorDataset(train_inputs, train_targets)
        validation_dataset = (
            TensorDataset(validation_inputs, validation_targets)
            if validation_data is not None
            else None
        )
        calibration_dataset = (
            TensorDataset(calibration_inputs, calibration_targets)
            if calibration_data is not None
            else None
        )
        test_dataset = TensorDataset(test_inputs, test_targets)

        return train_dataset, validation_dataset, calibration_dataset, test_dataset


class TimeSeriesRegressionDataModule(RealWorldRegressionDataModule):
    """This is a wrapper class for time series regression datasets.

    Args:
        time_window (int): The number of time steps to use for creating the time series.
    """

    def __init__(self, time_window: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert time_window > 0, "The time window must be greater than 0."
        self._time_window = time_window
        self.inputs_dim = (time_window, 1)

    def _create_time_series(self, data: torch.Tensor) -> TensorDataset:
        """Creates a time series dataset from the given data without overlapping
        between the time steps.

        Args:
            data (torch.Tensor): The data to create the time series from.
        """
        inputs, targets = [], []
        for i in range(0, len(data) - self._time_window, self._time_window):
            inputs.append(data[i : i + self._time_window])
            targets.append(data[i + self._time_window])
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        return TensorDataset(inputs, targets)

    def _mean_std(
        self, dataset: TensorDataset, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A helper function to compute the mean and standard deviation for `index`th column of the dataset."""
        mean = torch.mean(
            torch.stack([dataset[i][index] for i in range(len(dataset))]), dim=[0, 1]
        )
        std = torch.std(
            torch.stack([dataset[i][index] for i in range(len(dataset))]), dim=[0, 1]
        )
        return mean, std

    @torch.no_grad()
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
        y = y.to(tester.device)
        if phase == TRAIN_KEY:
            output = tester.training_step([x, y], batch_idx=0)
        elif phase == VALIDATION_KEY:
            output = tester.validation_step([x, y], batch_idx=0)
        elif phase == TEST_KEY:
            output = tester.test_step([x, y], batch_idx=0)
        y_hat = output[PREDICTION_KEY].cpu()
        x = output[INPUT_KEY].cpu()
        y = output[TARGET_KEY].cpu()
        return y_hat, x, y

    @torch.no_grad()
    def plot(
        self, tester: LightningModule, save_path: str, specific_name: str = ""
    ) -> None:
        """Plots the predictions of the model on the test set.

        Args:
            tester (LightningModule): The model to test.
            save_path (str): The path to save the plot.
            specific_name (str): A specific name to add to the plot name.
        """
        # Create a single figure and iterate through, training, validation, and test sets
        # and plot all of them on the same figure.
        fig, axs = plt.subplots(1, 3, figsize=(30, 5))
        predictions = []
        y = []
        average_weights = []
        for dataloader in [
            self.train_dataloader(shuffle=False),
            self.validation_dataloader(shuffle=False),
            self.test_dataloader(shuffle=False),
        ]:
            for batch in dataloader:
                input, target = batch
                y.append(target)
                output = self._get_prediction(tester, input, target, TEST_KEY)                
                predictions.append(output[0])
                if output[3] is not None:
                    average_weights.append(output[3])
        predictions = torch.cat(predictions, dim=0)
        average_weights = torch.cat(average_weights, dim=0) if len(average_weights) > 0 else None
        y = torch.cat(y, dim=0)
        x = torch.arange(0, len(y))

        (
            mean,
            total_uncertainty,
            aleatoric_uncertainty,
            epistemic_uncertainty,
        ) = regression_uncertainty_decomposition(predictions, weights=average_weights)

        # De-normalize the data
        y = y * self._target_std + self._target_mean
        mean = mean * self._target_std + self._target_mean
        total_uncertainty = torch.sqrt(total_uncertainty) * self._target_std
        aleatoric_uncertainty = torch.sqrt(aleatoric_uncertainty) * self._target_std
        epistemic_uncertainty = torch.sqrt(epistemic_uncertainty) * self._target_std

        y = y.cpu().numpy()
        x = x.cpu().numpy()
        mean = mean.cpu().numpy()
        total_uncertainty = total_uncertainty.cpu().numpy()
        aleatoric_uncertainty = aleatoric_uncertainty.cpu().numpy()
        epistemic_uncertainty = epistemic_uncertainty.cpu().numpy()

        for i in range(3):
            axs[i].plot(x, y, label="Ground Truth", color="black", linestyle="dashed")
            axs[i].plot(x, mean, label="Mean", color="blue")
            if i == 0:
                axs[i].set_title(f"Total Uncertainty")
                axs[i].fill_between(
                    x, (mean - total_uncertainty), (mean + total_uncertainty), alpha=0.5
                )
            elif i == 1:
                axs[i].set_title(f"Epistemic Uncertainty")
                axs[i].fill_between(
                    x,
                    (mean - epistemic_uncertainty),
                    (mean + epistemic_uncertainty),
                    alpha=0.5,
                )
            else:
                axs[i].set_title(f"Aleatoric Uncertainty")
                axs[i].fill_between(
                    x,
                    (mean - aleatoric_uncertainty),
                    (mean + aleatoric_uncertainty),
                    alpha=0.5,
                )
            axs[i].set_xlabel("x")
            axs[i].set_ylabel("y")
            axs[i].legend()

        plt.tight_layout()
        plt.savefig(plots_file(save_path, specific_name), bbox_inches="tight")
        plt.close(fig)
        plt.clf()

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Add dataset specific arguments to the parser."""
        parser = super(
            TemperatureTimeSeriesDataModule, TemperatureTimeSeriesDataModule
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--datamodule_time_window",
            type=int,
            default=10,
            help="The time window of the data.",
        )
        return parser

    def prepare_data(self) -> None:
        """Prepares the data for training, validation, and testing."""
        super().prepare_data()
        dataset = self._download_data()
        train_size = int(
            (1.0 - self._test_portion - self._validation_portion) * len(dataset)
        )
        validation_size = int(self._validation_portion * len(dataset))

        # Do not use random sampling beacuse we want to keep the time series
        # in order.
        train_indices = list(range(0, train_size))
        validation_indices = list(range(train_size, train_size + validation_size))
        test_indices = list(range(train_size + validation_size, len(dataset)))

        self._train_dataset = Subset(dataset, train_indices)
        self._validation_dataset = Subset(dataset, validation_indices)
        self._test_dataset = Subset(dataset, test_indices)

        self._data_mean, self._data_std = self._mean_std(self._train_dataset, 0)
        self._target_mean, self._target_std = self._mean_std(self._train_dataset, 1)


class TemperatureTimeSeriesDataModule(TimeSeriesRegressionDataModule):
    """Data module for temperature time series regression."""

    outputs_dim = 2
    inputs_dim = (1,)
    task = REGRESSION_KEY

    def _download_data(self) -> TensorDataset:
        request.urlretrieve(
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
            "daily-min-temperatures.csv",
        )
        shutil.move(
            "daily-min-temperatures.csv",
            os.path.join(
                os.path.expanduser(self._data_dir), "daily-min-temperatures.csv"
            ),
        )
        data = pd.read_csv(
            os.path.join(self._data_dir, "daily-min-temperatures.csv"),
            header=0,
            index_col=0,
        )
        data = data.dropna().to_numpy()
        data = torch.from_numpy(data).float()
        return self._create_time_series(data)


class WikiFaceDataset(Dataset):
    """Dataset for the WikiFace age prediction task.

    The dataset is the wikipedia portion of the dataset:
    https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

    Args:
        root_dir (str): The root directory to store the dataset.
        seed (int): The seed for the random number generator.
        image_size (Tuple[int, int]): The size of the image.
        test_portion (float): The portion of the dataset to use for testing.
        transform (Optional[Callable]): The transform to apply to the image.
        split (str): The split of the dataset to use. Either 'train' or 'test'.
    """

    md5sum = "f536eb7f5eae229ae8f286184364b42b"

    def __init__(
        self,
        root_dir: str,
        seed: int = 42,
        image_size: Tuple[int, int] = (64, 64),
        test_portion: float = 0.1,
        transform: Optional[Callable] = None,
        split: str = "train",
    ) -> None:
        assert split in ["train", "test"]

        self.root_dir = os.path.join(root_dir, "wiki_face")
        self.image_size = image_size

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
            download_and_extract_archive(
                url="https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar",
                download_root=self.root_dir,
                filename="wiki_crop.tar",
                md5=self.md5sum,
            )

            paths = glob.glob(os.path.join(self.root_dir, "wiki_crop", "*/*.jpg"))

            # Resize the images to the given size
            for path in paths:
                with open(path, "rb") as f:
                    image = Image.open(f)
                    image = image.convert("RGB")
                    image = image.resize(self.image_size)
                    image.save(path)

        paths = glob.glob(os.path.join(self.root_dir, "wiki_crop", "*/*.jpg"))
        self.paths = []
        # Filter out the paths where dob or year of photo taken cannot be parsed
        for path in paths:
            try:
                self._convert_path_to_age(path)
                self.paths.append(path)
            except Exception as e:
                pass

        self.transform = transform
        self.generator = torch.Generator().manual_seed(seed)

        # Split the dataset into train or test
        train_size = int(len(self.paths) * (1 - test_portion))

        indices = torch.randperm(len(self.paths), generator=self.generator).tolist()
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        self.indices = train_indices if split == "train" else test_indices

    def __len__(self) -> int:
        """Return the number of images"""
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the image and the age"""
        path = self.paths[self.indices[index]]
        with open(path, "rb") as f:
            image = Image.open(f)
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        age = self._convert_path_to_age(path)

        return image, age

    def _convert_path_to_age(self, path: str) -> float:
        """Convert the path to age"""
        # The file name is 2786_1921-05-21_1989.jpg
        # The age is 1921-05-21 - 1989-07-01
        # Load in the date of birth
        dob = path.split("/")[-1].split("_")[1]
        # Filter the dob with regex to be YYYY-MM-DD to prevent e.g. 2015-02-16UTC08:04
        dob = re.findall(r"\d{4}-\d{2}-\d{2}", dob)[0]
        # If month or day is 00, set it to 01
        dob = dob.replace("-00", "-01")
        dob = datetime.datetime.strptime(dob, "%Y-%m-%d")
        # Load in the date of photo taken
        photo_year_taken = int(path.split("/")[-1].split("_")[2].split(".")[0])
        # Set the date of photo taken to July 1st
        photo_taken = datetime.datetime(year=photo_year_taken, month=7, day=1)
        # Calculate the age and convert it to years
        age = photo_taken - dob
        assert age.days >= 0, "Age cannot be negative"
        age = age.days / 365.25

        # Normalize the age divide by 100 to prevent the age from being too large
        age /= 100

        return age


class TorchvisionRegressionDataModule(VisionRegressionDataModule):
    outputs_dtype = torch.float32

    def __init__(self, dataset: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if dataset not in ["wiki_face"]:
            raise ValueError("Dataset not supported.")
        self._dataset = dataset

    def prepare_data(self) -> None:
        """Download and prepare the data, the data is stored in `self._train_dataset` and `self._test_dataset`."""
        super().prepare_data()
        if self._dataset == "wiki_face":
            self._train_dataset = WikiFaceDataset(
                root_dir=self._data_dir,
                seed=self._seed,
                split="train",
                test_portion=self._test_portion,
            )
            self._test_dataset = WikiFaceDataset(
                root_dir=self._data_dir,
                seed=self._seed,
                split="test",
                test_portion=self._test_portion,
            )


class WikiFaceRegressionDataModule(TorchvisionRegressionDataModule):
    """Data module for the WikiFace dataset."""

    inputs_dim = (3, 64, 64)
    outputs_dim = 2
    targets_dim = 1
    outputs_dtype = torch.float32
    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2302, 0.2265, 0.2262)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("wiki_face", *args, **kwargs)


class TorchvisionRotationRegressionDataModule(VisionRegressionDataModule):
    """Data module for the rotation regression task.

    It can load the most common torchvision datasets and apply a rotation to the images
    to create a rotation regression task. The task is to predict the rotation of the image.

    Args:
        dataset (str): The dataset to load. Either 'mnist', 'fashionmnist', 'cifar10', 'cifar100', 'svhn', 'tinyimagenet'.
        pad_to_32 (bool): Whether to pad the images to 32x32.
        min_angle (float): The minimum angle to rotate the image. Defaults to 0.
        max_angle (float): The maximum angle to rotate the image. Defaults to 90.
    """

    outputs_dim = 2
    targets_dim = 1
    outputs_dtype = torch.float32

    def __init__(
        self,
        dataset: str,
        pad_to_32: bool = False,
        min_angle: float = 0,
        max_angle: float = 90,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if dataset not in [
            "mnist",
            "fashionmnist",
            "cifar10",
            "cifar100",
            "tinyimagenet",
            "svhn",
        ]:
            raise ValueError("Dataset not supported.")
        self._dataset = dataset
        self._pad_to_32 = pad_to_32
        if pad_to_32:
            assert dataset in [
                "mnist",
                "fashionmnist",
            ], "Padding only supported for 28x28 images."
            self.inputs_dim = (self.inputs_dim[0], 32, 32)
        self._min_angle = min_angle
        self._max_angle = max_angle

    def prepare_data(self) -> None:
        """Download and prepare the data, the data is stored in `self._train_dataset`, `self._validation_dataset` and `self._test_dataset`."""
        super().prepare_data()
        if self._dataset == "mnist":
            self._train_dataset = torchvision.datasets.MNIST(
                self._data_dir, train=True, download=True
            )
            self._test_dataset = torchvision.datasets.MNIST(
                self._data_dir, train=False, download=True
            )
        elif self._dataset == "fashionmnist":
            self._train_dataset = torchvision.datasets.FashionMNIST(
                self._data_dir, train=True, download=True
            )
            self._test_dataset = torchvision.datasets.FashionMNIST(
                self._data_dir, train=False, download=True
            )

        elif self._dataset == "svhn":
            self._train_dataset = torchvision.datasets.SVHN(
                self._data_dir, split="train", download=True
            )
            self._test_dataset = torchvision.datasets.SVHN(
                self._data_dir, split="test", download=True
            )
        elif self._dataset == "cifar10":
            self._train_dataset = torchvision.datasets.CIFAR10(
                self._data_dir, train=True, download=True
            )
            self._test_dataset = torchvision.datasets.CIFAR10(
                self._data_dir, train=False, download=True
            )

        elif self._dataset == "cifar100":
            self._train_dataset = torchvision.datasets.CIFAR100(
                self._data_dir, train=True, download=True
            )
            self._test_dataset = torchvision.datasets.CIFAR100(
                self._data_dir, train=False, download=True
            )
        elif self._dataset == "tinyimagenet":
            self._train_dataset = TinyImageNet(
                self._data_dir, split="train", download=True
            )
            self._test_dataset = TinyImageNet(
                self._data_dir, split="val", download=True
            )
        else:
            raise ValueError("Dataset not supported.")

        if self._pad_to_32:
            self._train_dataset = InputImagePaddingDataset(self._train_dataset, 2)
            self._test_dataset = InputImagePaddingDataset(self._test_dataset, 2)

        self._train_dataset = ImageRotationDataset(
            self._train_dataset,
            max_angle=self._max_angle,
            min_angle=self._min_angle,
            seed=self._seed,
        )
        self._test_dataset = ImageRotationDataset(
            self._test_dataset,
            max_angle=self._max_angle,
            min_angle=self._min_angle,
            seed=self._seed,
        )

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = super(
            TorchvisionRotationRegressionDataModule,
            TorchvisionRotationRegressionDataModule,
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--datamodule_pad_to_32",
            type=int,
            choices=[0, 1],
            default=0,
            help="Whether to pad the images to 32x32.",
        )
        parser.add_argument(
            "--datamodule_min_angle",
            type=float,
            default=0,
            help="The minimum angle to rotate the image. It is in degrees.",
        )
        parser.add_argument(
            "--datamodule_max_angle",
            type=float,
            default=90,
            help="The maximum angle to rotate the image. It is in degrees.",
        )
        return parser


class TinyImageNetRotationRegressionDataModule(TorchvisionRotationRegressionDataModule):
    """Data module for the rotation regression task on TinyImageNet."""

    inputs_dim = (3, 64, 64)
    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2302, 0.2265, 0.2262)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("tinyimagenet", *args, **kwargs)


class TorchvisionRotationRegressionDataModuleMNIST(
    TorchvisionRotationRegressionDataModule
):
    """Data module for the rotation regression task on MNIST."""

    inputs_dim = (1, 28, 28)
    mean = (0.1307,)
    std = (0.3081,)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("mnist", *args, **kwargs)

    def prepare_data(self) -> None:
        super().prepare_data()
        if self.test_augmentations is not None:
            for augmentation in self.test_augmentations:
                if augmentation in VisionCorruption.available_augmentations:
                    assert self._pad_to_32, (
                        f"Padding to 32 is required for MNIST with augmentation {augmentation}."
                    )


class TorchvisionRotationRegressionDataModuleFashionMNIST(
    TorchvisionRotationRegressionDataModule
):
    """Data module for the rotation regression task on FashionMNIST."""

    inputs_dim = (1, 28, 28)
    mean = (0.2860,)
    std = (0.3530,)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("fashionmnist", *args, **kwargs)

    def prepare_data(self) -> None:
        super().prepare_data()
        if self.test_augmentations is not None:
            for augmentation in self.test_augmentations:
                if augmentation in VisionCorruption.available_augmentations:
                    assert self._pad_to_32, (
                        f"Padding to 32 is required for MNIST with augmentation {augmentation}."
                    )


class TorchvisionRotationRegressionDataModuleSVHN(
    TorchvisionRotationRegressionDataModule
):
    """Data module for the rotation regression task on SVHN."""

    inputs_dim = (3, 32, 32)
    mean = (0.4377, 0.4438, 0.4728)
    std = (0.1980, 0.2010, 0.1970)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("svhn", *args, **kwargs)


class TorchvisionRotationRegressionDataModuleCIFAR10(
    TorchvisionRotationRegressionDataModule
):
    """Data module for the rotation regression task on CIFAR10."""

    inputs_dim = (3, 32, 32)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("cifar10", *args, **kwargs)


class TorchvisionRotationRegressionDataModuleCIFAR100(
    TorchvisionRotationRegressionDataModule
):
    """Data module for the rotation regression task on CIFAR100."""

    inputs_dim = (3, 32, 32)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("cifar100", *args, **kwargs)
