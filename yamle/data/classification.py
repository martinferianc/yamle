import argparse
import os
import shutil
from typing import Any, List, Tuple, Union
from urllib import request

import matplotlib.pyplot as plt
import medmnist
import pandas as pd
import scienceplots
import torch
import torchvision
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import LightningModule
from sklearn import datasets
from torch.utils.data import TensorDataset

from yamle.data.datamodule import (
    BaseDataModule,
    RealWorldClassificationDataModule,
    VisionClassificationDataModule,
)
from yamle.data.custom import ECG5000DataModule
from yamle.data.dataset_wrappers import InputImagePaddingDataset
from yamle.data.transforms import ClassificationDatasetSubset
from yamle.defaults import (
    CLASSIFICATION_KEY,
    INPUT_KEY,
    PREDICTION_KEY,
    MEAN_PREDICTION_KEY, 
    TARGET_KEY,
    TEST_KEY,
    TRAIN_KEY,
    VALIDATION_KEY,
    AVERAGE_WEIGHTS_KEY,
)
from yamle.third_party.medmnist import MedMNISTDatasetWrapper
from yamle.third_party.tinyimagenet import TinyImageNet
from yamle.third_party.imagenet_c import VisionCorruption
from yamle.utils.file_utils import plots_file
from yamle.utils.operation_utils import classification_uncertainty_decomposition

plt.style.use("science")

import logging

logging = logging.getLogger("pytorch_lightning")


class TorchvisionClassificationDataModule(VisionClassificationDataModule):
    """Data module for the torchvision datasets.

    Args:
        dataset (str): Name of the torchvision dataset. Currently supported are `mnist`, `fashion_mnist`, `cifar10`, `cifar100` and `tinyimagenet`.
        pad_to_32 (bool): Whether to pad the images to 32x32. Defaults to False.
    """

    outputs_dtype = torch.long

    def __init__(
        self, dataset: str, pad_to_32: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        if dataset not in [
            "mnist",
            "fashionmnist",
            "cifar10",
            "cifar3",
            "cifar5",
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
        elif self._dataset == "cifar3":
            self._train_dataset = ClassificationDatasetSubset(
                torchvision.datasets.CIFAR10(self._data_dir, train=True, download=True),
                indices=self._indices,
            )
            self._test_dataset = ClassificationDatasetSubset(
                torchvision.datasets.CIFAR10(
                    self._data_dir, train=False, download=True
                ),
                indices=self._indices,
            )
        # This is a version of the cifar10 dataset only with 5 classes
        elif self._dataset == "cifar5":
            self._train_dataset = ClassificationDatasetSubset(
                torchvision.datasets.CIFAR10(self._data_dir, train=True, download=True),
                indices=self._indices,
            )
            self._test_dataset = ClassificationDatasetSubset(
                torchvision.datasets.CIFAR10(
                    self._data_dir, train=False, download=True
                ),
                indices=self._indices,
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

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = super(
            TorchvisionClassificationDataModule, TorchvisionClassificationDataModule
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--datamodule_pad_to_32",
            type=int,
            choices=[0, 1],
            default=0,
            help="Whether to pad the images to 32x32.",
        )
        return parser


class TinyImageNetClassificationDataModule(TorchvisionClassificationDataModule):
    """Data module for the TinyImageNet dataset."""

    inputs_dim = (3, 64, 64)
    outputs_dim = 200
    targets_dim = 1
    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2302, 0.2265, 0.2262)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("tinyimagenet", *args, **kwargs)


class TorchvisionClassificationDataModuleMNIST(TorchvisionClassificationDataModule):
    """Data module for the MNIST dataset."""

    inputs_dim = (1, 28, 28)
    outputs_dim = 10
    targets_dim = 1
    mean = (0.1307,)
    std = (0.3081,)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("mnist", *args, **kwargs)

    def prepare_data(self) -> None:
        super().prepare_data()
        if self.test_augmentations is not None:
            for augmentation in self.test_augmentations:
                if augmentation in VisionCorruption.available_augmentations:
                    assert (
                        self._pad_to_32
                    ), f"Padding to 32 is required for MNIST with augmentation {augmentation}."


class TorchvisionClassificationDataModuleFashionMNIST(
    TorchvisionClassificationDataModule
):
    """Data module for the FashionMNIST dataset."""

    inputs_dim = (1, 28, 28)
    outputs_dim = 10
    targets_dim = 1
    mean = (0.2860,)
    std = (0.3530,)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("fashionmnist", *args, **kwargs)

    def prepare_data(self) -> None:
        super().prepare_data()
        if self.test_augmentations is not None:
            for augmentation in self.test_augmentations:
                if augmentation in VisionCorruption.available_augmentations:
                    assert (
                        self._pad_to_32
                    ), f"Padding to 32 is required for MNIST with augmentation {augmentation}."


class TorchvisionClassificationDataModuleSVHN(TorchvisionClassificationDataModule):
    """Data module for the SVHN dataset."""

    inputs_dim = (3, 32, 32)
    outputs_dim = 10
    targets_dim = 1
    mean = (0.4377, 0.4438, 0.4728)
    std = (0.1980, 0.2010, 0.1970)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("svhn", *args, **kwargs)


class TorchvisionClassificationDataModuleCIFAR10(TorchvisionClassificationDataModule):
    """Data module for the CIFAR10 dataset."""

    inputs_dim = (3, 32, 32)
    outputs_dim = 10
    targets_dim = 1
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("cifar10", *args, **kwargs)


class TorchvisionClassificationDataModuleCIFAR3(TorchvisionClassificationDataModule):
    """Data module for the CIFAR3 dataset."""

    inputs_dim = (3, 32, 32)
    outputs_dim = 3
    targets_dim = 1
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    def __init__(self, indices: List[int], *args: Any, **kwargs: Any) -> None:
        super().__init__("cifar3", *args, **kwargs)
        assert (
            len(indices) == 3
        ), f"Indices must be a list of length 3, got {len(indices)}."
        self._indices = indices

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Add dataset specific arguments to the parser."""
        parser = super(
            TorchvisionClassificationDataModuleCIFAR3,
            TorchvisionClassificationDataModuleCIFAR3,
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--datamodule_indices",
            type=str,
            default="[0, 1, 2]",
            help="Indices of the CIFAR10 classes to use.",
        )
        return parser


class TorchvisionClassificationDataModuleCIFAR5(TorchvisionClassificationDataModule):
    """Data module for the CIFAR5 dataset."""

    inputs_dim = (3, 32, 32)
    outputs_dim = 5
    targets_dim = 1
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    def __init__(self, indices: List[int], *args: Any, **kwargs: Any) -> None:
        super().__init__("cifar5", **kwargs)
        assert (
            len(indices) == 5
        ), f"Indices must be a list of length 5, but got {len(indices)}."
        self._indices = indices

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Add dataset specific arguments to the parser."""
        parser = super(
            TorchvisionClassificationDataModuleCIFAR5,
            TorchvisionClassificationDataModuleCIFAR5,
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--datamodule_indices",
            type=str,
            default="[0, 1, 2, 3, 4]",
            help="Indices of the CIFAR10 classes to use.",
        )
        return parser


class TorchvisionClassificationDataModuleCIFAR100(TorchvisionClassificationDataModule):
    """Data module for the CIFAR100 dataset."""

    inputs_dim = (3, 32, 32)
    outputs_dim = 100
    targets_dim = 1
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("cifar100", *args, **kwargs)


class MedMNISTClassificationDataModule(VisionClassificationDataModule):
    """Data module for the MedMNIST dataset.


    Args:
        dataset (str): Name of the dataset to use.
        pad_to_32 (bool): Whether to pad the images to 32x32. Defaults to True.
    """

    outputs_dtype = torch.long

    def __init__(
        self, dataset: str, pad_to_32: bool = True, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        if dataset not in ["pneumoniamnist", "dermamnist", "breastmnist", "bloodmnist"]:
            raise ValueError("Dataset not supported.")
        self._dataset = dataset
        self._pad_to_32 = pad_to_32
        if pad_to_32:
            # Since the images are 28x28, we might want to pad them to 32x32
            self.inputs_dim = (self.inputs_dim[0], 32, 32)

    def prepare_data(self) -> None:
        """Download and prepare the data, the data is stored in `self._train_dataset`, `self._validation_dataset` and `self._test_dataset`."""
        super().prepare_data()
        if self._dataset == "pneumoniamnist":
            self._train_dataset = MedMNISTDatasetWrapper(
                medmnist.PneumoniaMNIST(
                    root=self._data_dir, split="train", download=True
                ),
                pad_to_32=self._pad_to_32,
            )
            self._validation_dataset = MedMNISTDatasetWrapper(
                medmnist.PneumoniaMNIST(
                    root=self._data_dir, split="val", download=True
                ),
                pad_to_32=self._pad_to_32,
            )
            self._test_dataset = MedMNISTDatasetWrapper(
                medmnist.PneumoniaMNIST(
                    root=self._data_dir, split="test", download=True
                ),
                pad_to_32=self._pad_to_32,
            )
        elif self._dataset == "dermamnist":
            self._train_dataset = MedMNISTDatasetWrapper(
                medmnist.DermaMNIST(root=self._data_dir, split="train", download=True),
                pad_to_32=self._pad_to_32,
            )
            self._validation_dataset = MedMNISTDatasetWrapper(
                medmnist.DermaMNIST(root=self._data_dir, split="val", download=True),
                pad_to_32=self._pad_to_32,
            )
            self._test_dataset = MedMNISTDatasetWrapper(
                medmnist.DermaMNIST(root=self._data_dir, split="test", download=True),
                pad_to_32=self._pad_to_32,
            )
        elif self._dataset == "breastmnist":
            self._train_dataset = MedMNISTDatasetWrapper(
                medmnist.BreastMNIST(root=self._data_dir, split="train", download=True),
                pad_to_32=self._pad_to_32,
            )
            self._validation_dataset = MedMNISTDatasetWrapper(
                medmnist.BreastMNIST(root=self._data_dir, split="val", download=True),
                pad_to_32=self._pad_to_32,
            )
            self._test_dataset = MedMNISTDatasetWrapper(
                medmnist.BreastMNIST(root=self._data_dir, split="test", download=True),
                pad_to_32=self._pad_to_32,
            )
        elif self._dataset == "bloodmnist":
            self._train_dataset = MedMNISTDatasetWrapper(
                medmnist.BloodMNIST(root=self._data_dir, split="train", download=True),
                pad_to_32=self._pad_to_32,
            )
            self._validation_dataset = MedMNISTDatasetWrapper(
                medmnist.BloodMNIST(root=self._data_dir, split="val", download=True),
                pad_to_32=self._pad_to_32,
            )
            self._test_dataset = MedMNISTDatasetWrapper(
                medmnist.BloodMNIST(root=self._data_dir, split="test", download=True),
                pad_to_32=self._pad_to_32,
            )
        else:
            raise ValueError("Dataset not supported.")

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Add dataset specific arguments to the parser."""
        parser = super(
            MedMNISTClassificationDataModule, MedMNISTClassificationDataModule
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--datamodule_pad_to_32",
            type=int,
            choices=[0, 1],
            default=1,
            help="Whether to pad the images to 32x32.",
        )
        return parser


class PneumoniaMNISTClassificationDataModule(MedMNISTClassificationDataModule):
    """Data module for the PneumoniaMNIST dataset."""

    inputs_dim = (1, 28, 28)
    outputs_dim = 2
    targets_dim = 1
    mean = (0.5404,)
    std = (0.2824,)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("pneumoniamnist", *args, **kwargs)


class BreastMNISTClassificationDataModule(MedMNISTClassificationDataModule):
    """Data module for the BreastMNIST dataset."""

    inputs_dim = (1, 28, 28)
    outputs_dim = 2
    targets_dim = 1
    mean = (0.3304,)
    std = (0.2057,)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("breastmnist", *args, **kwargs)


class DermaMNISTClassificationDataModule(MedMNISTClassificationDataModule):
    """Data module for the DermaMNIST dataset."""

    inputs_dim = (3, 28, 28)
    outputs_dim = 7
    targets_dim = 1
    mean = (0.7637, 0.5383, 0.5615)
    std = (0.1371, 0.1540, 0.1690)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("dermamnist", *args, **kwargs)


class BloodMNISTClassificationDataModule(MedMNISTClassificationDataModule):
    """Data module for the BloodMNIST dataset."""

    inputs_dim = (3, 28, 28)
    outputs_dim = 8
    targets_dim = 1
    mean = (0.7943, 0.6596, 0.6962)
    std = (0.2156, 0.2415, 0.1179)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("bloodmnist", *args, **kwargs)


class ToyTwoMoonsClassificationDataModule(BaseDataModule):
    """Data module for a toy classification problem between data coming from 2 classes with 2 features.

    Args:
        noise (float): Noise to add to the data.
        num_samples (int): Number of samples to generate.
    """

    inputs_dim = (2,)
    outputs_dim = 2
    targets_dim = 1
    task = CLASSIFICATION_KEY
    inputs_dtype = torch.float32
    outputs_dtype = torch.long

    def __init__(
        self,
        noise: float,
        num_samples: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert 0 <= noise <= 1, "Noise must be between 0 and 1."
        self._noise = noise
        self._num_samples = num_samples
        self._generator = torch.Generator(device="cpu").manual_seed(self._seed)

    def prepare_data(self) -> None:
        """Generate the toy classification dataset."""
        super().prepare_data()
        train_size = int(
            (1 - self._validation_portion - self._test_portion) * self._num_samples
        )
        validation_size = int(self._validation_portion * self._num_samples)

        X, Y = datasets.make_moons(
            n_samples=self._num_samples, noise=self._noise, random_state=self._seed
        )
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).long()

        self._train_dataset = torch.utils.data.TensorDataset(
            X[:train_size], Y[:train_size]
        )
        self._validation_dataset = torch.utils.data.TensorDataset(
            X[train_size : train_size + validation_size],
            Y[train_size : train_size + validation_size],
        )
        self._test_dataset = torch.utils.data.TensorDataset(
            X[train_size + validation_size :], Y[train_size + validation_size :]
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

    @torch.no_grad()
    def plot(
        self, tester: LightningModule, save_path: str, specific_name: str = ""
    ) -> None:
        """Plot the data and the decision boundary of the model."""
        train_x, train_y = torch.stack(
            [x for x, _ in self._train_dataset]
        ), torch.stack([y for _, y in self._train_dataset])
        test_x, test_y = torch.stack([x for x, _ in self._test_dataset]), torch.stack(
            [y for _, y in self._test_dataset]
        )
        train_colors = ["red" if y == 0 else "blue" for y in train_y]
        test_colors = ["red" if y == 0 else "blue" for y in test_y]

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

        # Generate a grid of points to plot the uncertainty of the model
        x_min, x_max = train_x[:, 0].min() - 0.5, train_x[:, 0].max() + 0.5
        y_min, y_max = train_x[:, 1].min() - 0.5, train_x[:, 1].max() + 0.5
        xx, yy = torch.meshgrid(
            torch.linspace(x_min, x_max, 100),
            torch.linspace(y_min, y_max, 100),
            indexing="ij",
        )
        grid = torch.stack((xx.flatten(), yy.flatten()), dim=1).to(tester.device)
        labels = torch.zeros(grid.shape[0], dtype=torch.long).to(tester.device)
        predictions = []
        average_weights = []
        batch_size = 128
        for i in range(0, grid.shape[0], batch_size):
            output = self._get_prediction(
                tester,
                grid[i : i + batch_size],
                labels[i : i + batch_size],
                TEST_KEY,
            )
            predictions.append(output[0])
            if output[3] is not None:
                average_weights.append(output[3])

        predictions = torch.cat(predictions, dim=0)
        average_weights = (
            torch.cat(average_weights, dim=0) if len(average_weights) > 0 else None
        )
        (
            total_uncertainty,
            aleatoric_uncertainty,
            epistemic_uncertainty,
        ) = classification_uncertainty_decomposition(
            predictions, probabilities=True, weights=average_weights
        )

        for i in range(3):
            if i == 0:
                axs[i].set_title(f"Total Uncertainty")
                axs[i].contourf(xx, yy, total_uncertainty.reshape(xx.shape), alpha=0.5)
            elif i == 1:
                axs[i].set_title(f"Epistemic Uncertainty")
                axs[i].contourf(
                    xx, yy, epistemic_uncertainty.reshape(xx.shape), alpha=0.5
                )
            else:
                axs[i].set_title(f"Aleatoric Uncertainty")
                axs[i].contourf(
                    xx, yy, aleatoric_uncertainty.reshape(xx.shape), alpha=0.5
                )
            # Add a colorbar
            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(axs[i].collections[0], cax=cax)

            axs[i].scatter(
                train_x[:, 0], train_x[:, 1], c=train_colors, marker="o", label="Train"
            )
            axs[i].scatter(
                test_x[:, 0], test_x[:, 1], c=test_colors, marker="x", label="Test"
            )
            axs[i].set_xlim(xx.min(), xx.max())
            axs[i].set_ylim(yy.min(), yy.max())
            axs[i].set_xlabel("Feature 1")
            axs[i].set_ylabel("Feature 2")
            axs[i].legend()

        plt.savefig(plots_file(save_path, specific_name), bbox_inches="tight")
        plt.close(fig)
        plt.clf()

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Add dataset specific arguments to the parser."""
        parser = super(
            ToyTwoMoonsClassificationDataModule, ToyTwoMoonsClassificationDataModule
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--datamodule_num_samples",
            type=int,
            default=1000,
            help="Number of samples to generate.",
        )
        parser.add_argument(
            "--datamodule_noise",
            type=float,
            default=0.1,
            help="Noise to add to the data.",
        )

        return parser


class ToyTwoCirclesClassificationDataModule(ToyTwoMoonsClassificationDataModule):
    """Toy two ovals classification dataset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(ToyTwoCirclesClassificationDataModule, self).__init__(*args, **kwargs)

    def prepare_data(self) -> None:
        """Generate the toy classification dataset."""
        train_size = int(
            (1 - self._validation_portion - self._test_portion) * self._num_samples
        )
        validation_size = int(self._validation_portion * self._num_samples)

        X, Y = datasets.make_circles(
            n_samples=self._num_samples, noise=self._noise, random_state=self._seed
        )
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).long()

        self._train_dataset = torch.utils.data.TensorDataset(
            X[:train_size], Y[:train_size]
        )
        self._validation_dataset = torch.utils.data.TensorDataset(
            X[train_size : train_size + validation_size],
            Y[train_size : train_size + validation_size],
        )
        self._test_dataset = torch.utils.data.TensorDataset(
            X[train_size + validation_size :], Y[train_size + validation_size :]
        )


class UCIClassificationDataModule(RealWorldClassificationDataModule):
    """Data module for the UCI classification datasets.

    Currently supports the following datasets:
        - Breast Cancer
        - Adult income
        - Car evaluation
        - Credit default
        - Dermatology

    Args:
        dataset (str): Name of the dataset to use.
    """

    def __init__(self, dataset: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if dataset not in [
            "breast_cancer",
            "adult_income",
            "car_evaluation",
            "credit_default",
            "dermatology",
        ]:
            raise ValueError(
                "Dataset must be one of the following: breast_cancer, adult_income, car_evaluation, credit_default, dermatology."
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
                    f.write(f"Output: {outputs[i]}\n")
                    f.write(f"Target: {targets[i]}\n")


class BreastCancerUCIClassificationDataModule(UCIClassificationDataModule):
    """Data module for the Breast Cancer dataset."""

    # Number of classes in the dataset.
    outputs_dim = 2
    outputs_dtype = torch.long
    inputs_dim = (9,)
    inputs_dtype = torch.float32
    task = CLASSIFICATION_KEY

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset="breast_cancer", *args, **kwargs)

    def _download_data(self) -> TensorDataset:
        """Downloads the dataset."""
        # Test if the data is already downloaded.
        if not os.path.exists(
            os.path.join(
                os.path.expanduser(self._data_dir), "breast-cancer-wisconsin.data"
            )
        ):
            request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
                "breast-cancer-wisconsin.data",
            )
            shutil.move(
                "breast-cancer-wisconsin.data",
                os.path.join(
                    os.path.expanduser(self._data_dir), "breast-cancer-wisconsin.data"
                ),
            )
        data = pd.read_csv(
            os.path.join(self._data_dir, "breast-cancer-wisconsin.data"),
            header=None,
            na_values="?",
        )
        data = data.dropna().to_numpy()
        data = torch.from_numpy(data).float()
        inputs, targets = data[:, 1:-1], data[:, -1].long()
        targets[targets == 2] = 0
        targets[targets == 4] = 1
        return TensorDataset(inputs, targets)


class AdultIncomeUCIClassificationDataModule(UCIClassificationDataModule):
    """Data module for the Adult Income dataset."""

    # Number of classes in the dataset.
    outputs_dim = 2
    outputs_dtype = torch.long
    inputs_dim = (108,)
    inputs_dtype = torch.float32
    task = CLASSIFICATION_KEY

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset="adult_income", *args, **kwargs)

    def _download_data(self) -> TensorDataset:
        """Downloads the dataset."""
        # Test if the data is already downloaded.
        if not os.path.exists(
            os.path.join(os.path.expanduser(self._data_dir), "adult.data")
        ):
            request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                "adult.data",
            )
            shutil.move(
                "adult.data",
                os.path.join(os.path.expanduser(self._data_dir), "adult.data"),
            )
        data = pd.read_csv(
            os.path.join(self._data_dir, "adult.data"), header=None, na_values="?"
        )
        # Convert categorical variables to one-hot encoding.
        data = pd.get_dummies(data)
        data = data.dropna().to_numpy()
        for col in range(data.shape[1]):
            if isinstance(data[0, col], bool):
                data[:, col] = data[:, col].astype(np.float32)
        data = data.astype(np.float32)

        data = torch.from_numpy(data).float()
        inputs, targets = data[:, :-2], torch.argmax(data[:, -2:], dim=1)
        return TensorDataset(inputs, targets)


class CarEvaluationUCIClassificationDataModule(UCIClassificationDataModule):
    """Data module for the Car Evaluation dataset."""

    # Number of classes in the dataset.
    outputs_dim = 4
    outputs_dtype = torch.long
    inputs_dim = (21,)
    inputs_dtype = torch.float32
    task = CLASSIFICATION_KEY

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset="car_evaluation", *args, **kwargs)

    def _download_data(self) -> TensorDataset:
        """Downloads the dataset."""
        # Test if the data is already downloaded.
        if not os.path.exists(
            os.path.join(os.path.expanduser(self._data_dir), "car.data")
        ):
            request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
                "car.data",
            )
            shutil.move(
                "car.data", os.path.join(os.path.expanduser(self._data_dir), "car.data")
            )
        data = pd.read_csv(
            os.path.join(self._data_dir, "car.data"), header=None, na_values="?"
        )
        # Convert categorical variables to one-hot encoding.
        data = pd.get_dummies(data)
        data = data.dropna().to_numpy()
        data = torch.from_numpy(data).float()
        inputs, targets = data[:, :-4], data[:, -4:].argmax(dim=1)
        return TensorDataset(inputs, targets)


class CreditUCIClassificationDataModule(UCIClassificationDataModule):
    """Data module for the Credit Default dataset."""

    # Number of classes in the dataset.
    outputs_dim = 2
    outputs_dtype = torch.long
    inputs_dim = (23,)
    inputs_dtype = torch.float32
    task = CLASSIFICATION_KEY

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset="credit_default", *args, **kwargs)

    def _download_data(self) -> TensorDataset:
        """Downloads the dataset."""
        # Test if the data is already downloaded.
        if not os.path.exists(
            os.path.join(
                os.path.expanduser(self._data_dir), "default of credit card clients.xls"
            )
        ):
            request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
                "default of credit card clients.xls",
            )
            shutil.move(
                "default of credit card clients.xls",
                os.path.join(
                    os.path.expanduser(self._data_dir),
                    "default of credit card clients.xls",
                ),
            )
        data = pd.read_excel(
            os.path.join(self._data_dir, "default of credit card clients.xls"),
            header=1,
            na_values=".",
        )
        # Convert categorical variables to one-hot encoding.
        data = pd.get_dummies(data)
        data = data.dropna().to_numpy()
        data = torch.from_numpy(data).float()
        inputs, targets = data[:, 1:-1], data[:, -1].long()
        return TensorDataset(inputs, targets)


class DermatologyUCIClassificationDataModule(UCIClassificationDataModule):
    """Data module for the Dermatology dataset."""

    # Number of classes in the dataset.
    outputs_dim = 6
    outputs_dtype = torch.long
    inputs_dim = (34,)
    inputs_dtype = torch.float32
    task = CLASSIFICATION_KEY

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset="dermatology", *args, **kwargs)

    def _download_data(self) -> TensorDataset:
        """Downloads the dataset."""
        # Test if the data is already downloaded.
        if not os.path.exists(
            os.path.join(os.path.expanduser(self._data_dir), "dermatology.data")
        ):
            request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data",
                "dermatology.data",
            )
            shutil.move(
                "dermatology.data",
                os.path.join(os.path.expanduser(self._data_dir), "dermatology.data"),
            )
        data = pd.read_csv(
            os.path.join(self._data_dir, "dermatology.data"), header=None, na_values="?"
        )
        # Convert categorical variables to one-hot encoding.
        # data = pd.get_dummies(data)
        data = data.dropna().to_numpy()
        data = torch.from_numpy(data).float()
        inputs, targets = data[:, :-1], (data[:, -1] - 1).long()
        return TensorDataset(inputs, targets)


class ECG5000ClassificationDataModule(ECG5000DataModule):
    """Data module for the ECG5000 dataset."""

    # Number of classes in the dataset.
    outputs_dim = 5
    outputs_dtype = torch.long
    inputs_dtype = torch.float32
    targets_dim = 1
    task = CLASSIFICATION_KEY

    def prepare_data(self) -> None:
        """Download and prepare the data"""
        # The train and test data are pandas dataframes
        train_data, test_data = super().prepare_data()

        # Separate the inputs and targets the first column is the target
        train_targets = train_data.iloc[:, 0]
        train_inputs = train_data.iloc[:, 1:]

        test_targets = test_data.iloc[:, 0]
        test_inputs = test_data.iloc[:, 1:]

        # Scale the targets to start from 0
        train_targets = train_targets - 1
        test_targets = test_targets - 1

        # Convert them into TensorDatasets
        self._train_dataset = TensorDataset(
            torch.from_numpy(train_inputs.to_numpy())
            .unsqueeze(1)
            .float()
            .permute(0, 2, 1),
            torch.from_numpy(train_targets.to_numpy()).long(),
        )

        self._test_dataset = TensorDataset(
            torch.from_numpy(test_inputs.to_numpy())
            .unsqueeze(1)
            .float()
            .permute(0, 2, 1),
            torch.from_numpy(test_targets.to_numpy()).long(),
        )

        # Calculate the mean and standard deviation of the training data
        self._data_mean, self._data_std = self._mean_std(self._train_dataset, index=0)

        # Calculate the maximum and minimum of the training data
        self._data_max, self._data_min = self._max_min(self._train_dataset, index=0)

        # Log the training data distribution and the proposed class weights
        class_counts = torch.bincount(self._train_dataset.tensors[1])
        logging.info(f"Training data class counts: {class_counts}")

        # Calculate the class weights
        class_weights = class_counts / class_counts.sum()
        class_weights = 1 / class_weights
        logging.info(f"Training data class weights: {class_weights}")

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
        y_hat = output[MEAN_PREDICTION_KEY].cpu()
        y_hat = torch.argmax(y_hat, dim=1)
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
            outputs = self._get_prediction(tester, inputs, targets, TEST_KEY)[0]

            # Visualise the data and the predictions
            # Create 10 subplots with the ECG data and the predictions
            # Denormalise the inputs
            inputs = inputs * self._data_std + self._data_mean

            # Plot the data
            for j in range(10):
                axs[i, j].plot(inputs[j, :, 0])
                axs[i, j].set_title(f"Prediction: {outputs[j]}, Target: {targets[j]}")

        plt.savefig(plots_file(save_path, specific_name), bbox_inches="tight")
        plt.close(fig)
        plt.clf()
