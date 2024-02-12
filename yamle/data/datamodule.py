import argparse
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union, Iterable

import torch
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule
from torch.utils.data import Subset, TensorDataset, random_split
from torchvision.transforms import Compose, ToTensor
import scienceplots
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from yamle.data.transforms import FromOneHot, JointCompose, Normalize
from yamle.defaults import (
    INPUT_KEY,
    MEAN_PREDICTION_KEY,
    TARGET_KEY,
    TEST_KEY,
    TRAIN_KEY,
    VALIDATION_KEY,
    CALIBRATION_KEY,
    CLASSIFICATION_KEY,
    REGRESSION_KEY,
)
from yamle.third_party.imagenet_c import VisionCorruption, TabularCorruption
from yamle.third_party.imagenet_c.extra import RandomImageNoise, RandomTabularNoise
from yamle.utils.file_utils import plots_file
from yamle.data.dataset_wrappers import SurrogateDataset

logging = logging.getLogger("pytorch_lightning")
plt.style.use("science")


class BaseDataModule(ABC):
    """General data module returning training, validation and test data loaders.

    Args:
        validation_portion (float): Portion of the training data to use for validation.
        test_portion (float): Portion of the training data to use for test if test data is not provided.
        calibration_portion (float): Portion of the training data to use for calibration.
        seed (int): Seed for the random number generator.
        data_dir (str): Path to the data directory.
        train_splits (Optional[int]): Number of splits to use for the training data.
        train_splits_proportions (Optional[List[float]]): Proportions of the training data to use for each split.
        train_size (Optional[int]): Size of the training data.
        train_tranform (Optional[List[str]]): Transformations to apply to the training data. Note that if the list is provided, it is ordered.
        test_transform (Optional[List[str]]): Transformations to apply to the test data. Note that if the list is provided, it is ordered.
        test_augmentations (Optional[List[str]]): Augmentations to apply to the test data. Note that if the list is provided, it is ordered.
        train_target_transform (Optional[List[str]]): Transformations to apply to the training targets. Note that if the list is provided, it is ordered.
        test_target_transform (Optional[List[str]]): Transformations to apply to the test targets. Note that if the list is provided, it is ordered.
        train_joint_transform (Optional[List[str]]): Transformations to apply to the training data as well as the targets. Note that if the list is provided, it is ordered.
        test_joint_transform (Optional[List[str]]): Transformations to apply to the test data as well as the targets. Note that if the list is provided, it is ordered.
        num_workers (Optional[int]): Number of workers to use for the data loaders. Defaults to None.
        batch_size (int): Batch size to use for the data loaders. Defaults to 32.
        pin_memory (bool): Whether to use pinned memory for the data loaders. Defaults to True.
    """

    data_shape = None
    inputs_dim = None
    inputs_dtype = torch.float32
    outputs_dim = None
    outputs_dtype = torch.float32
    targets_dim = None
    task = ""
    ignore_indices = []

    def __init__(
        self,
        validation_portion: float = 0.1,
        test_portion: float = 0.1,
        calibration_portion: float = 0.0,
        seed: int = 0,
        data_dir: Optional[str] = None,
        train_splits: Optional[int] = None,
        train_splits_proportions: Optional[List[float]] = None,
        train_size: Optional[int] = None,
        train_transform: Optional[List[str]] = None,
        test_transform: Optional[List[str]] = None,
        test_augmentations: Optional[List[str]] = None,
        train_target_transform: Optional[List[str]] = None,
        test_target_transform: Optional[List[str]] = None,
        train_joint_transform: Optional[List[str]] = None,
        test_joint_transform: Optional[List[str]] = None,
        num_workers: int = 0,
        batch_size: int = 32,
        pin_memory: bool = True,
    ) -> None:
        self._seed = seed
        self._data_dir = os.path.expanduser(data_dir)
        self._validation_portion = validation_portion
        self._test_portion = test_portion
        self._calibration_portion = calibration_portion
        assert (
            self._validation_portion >= 0 and self._validation_portion <= 1
        ), f"Validation portion must be between 0 and 1."

        assert (
            self._test_portion >= 0 and self._test_portion <= 1
        ), f"Test portion must be between 0 and 1."

        assert (
            self._calibration_portion >= 0 and self._calibration_portion <= 1
        ), f"Calibration portion must be between 0 and 1."

        assert (
            self._validation_portion + self._test_portion + self._calibration_portion
            <= 1
        ), f"Validation, test and calibration portions must sum to 1."

        self._train_splits = train_splits
        self._train_splits_proportions = train_splits_proportions
        self._train_size = train_size
        assert (
            self._train_splits is None or self._train_size is None
        ), f"Either train_splits or train_size must be None."
        self._train_transform = train_transform
        self._test_transform = test_transform
        self._train_target_transform = train_target_transform
        self._test_target_transform = test_target_transform
        self._train_joint_transform = train_joint_transform
        self._test_joint_transform = test_joint_transform

        self._train_dataset: SurrogateDataset = None
        self._validation_dataset: SurrogateDataset = None
        self._calibration_dataset: SurrogateDataset = None
        self._test_dataset: SurrogateDataset = None

        if num_workers is None:
            # Get the number of CPUs
            cpus = os.cpu_count()
            # Get the number of GPUs
            gpus = torch.cuda.device_count()
            num_workers = cpus // gpus if gpus > 0 else cpus
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._pin_memory = pin_memory

        self.available_transforms: List[str] = ["totensor"]
        self.available_test_augmentations: List[str] = []
        self.test_augmentations: List[str] = (
            test_augmentations if test_augmentations is not None else []
        )

    def get_transform(self, name: str) -> Callable:
        """Returns the transformation with the given name."""
        assert len(set(self.available_transforms)) == len(
            self.available_transforms
        ), f"Transforms must be unique. {self.available_transforms}."

        if name not in self.available_transforms and name not in self.available_test_augmentations:
            raise ValueError(f"Transformation {name} not supported.")

        if name == "totensor":
            return ToTensor()

    def get_transform_composition(
        self, names: List[str], joint: bool = False
    ) -> Compose:
        """Returns the composition of the transformations with the given names."""
        transform = []
        for name in names:
            transform.append(self.get_transform(name))
        return Compose(transform) if not joint else JointCompose(transform)

    def train_transform(self) -> Optional[Compose]:
        """Returns the training data transformation."""
        if self._train_transform is not None:
            return self.get_transform_composition(self._train_transform, joint=False)
        return None

    def validation_transform(self) -> Optional[Compose]:
        """Returns the validation data transformation."""
        return self.test_transform()

    def test_transform(self) -> Optional[Compose]:
        """Returns the test data transformation."""
        if self._test_transform is not None:
            return self.get_transform_composition(self._test_transform, joint=False)
        return None

    def test_augmentation(self, name: str) -> Callable:
        """Returns the augmentation with the given name."""
        assert len(set(self.available_test_augmentations)) == len(
            self.available_test_augmentations
        ), f"Augmentations must be unique. {self.available_test_augmentations}."
        if name is not None and name not in self.available_test_augmentations:
            raise ValueError(f"Test augmentation {name} not supported.")
        return self.get_transform_composition([name], joint=False)

    def calibration_transform(self) -> Optional[Compose]:
        """Returns the calibration data transformation."""
        return self.test_transform()

    def train_target_transform(self) -> Optional[Compose]:
        """Returns the training target transformation."""
        if self._train_target_transform is not None:
            return self.get_transform_composition(
                self._train_target_transform, joint=False
            )
        return None

    def validation_target_transform(self) -> Optional[Compose]:
        """Returns the validation target transformation."""
        return self.test_target_transform()

    def test_target_transform(self) -> Optional[Compose]:
        """Returns the test target transformation."""
        if self._test_target_transform is not None:
            return self.get_transform_composition(
                self._test_target_transform, joint=False
            )
        return None

    def calibration_target_transform(self) -> Optional[Compose]:
        """Returns the calibration target transformation."""
        return self.test_target_transform()

    def train_joint_transform(self) -> Optional[Compose]:
        """Returns the training joint transformation."""
        if self._train_joint_transform is not None:
            return self.get_transform_composition(
                self._train_joint_transform, joint=True
            )
        return None

    def validation_joint_transform(self) -> Optional[Compose]:
        """Returns the validation joint transformation."""
        return self.test_joint_transform()

    def test_joint_transform(self) -> Optional[Compose]:
        """Returns the test joint transformation."""
        if self._test_joint_transform is not None:
            return self.get_transform_composition(
                self._test_joint_transform, joint=True
            )
        return None

    def calibration_joint_transform(self) -> Optional[Compose]:
        """Returns the calibration joint transformation."""
        return self.test_joint_transform()

    def train_dataset(
        self, split: Optional[int] = None
    ) -> Union[SurrogateDataset, Subset]:
        """Returns the training dataset."""
        if self._train_dataset is not None:
            dataset = self._train_dataset
            if split is None and self._train_splits is not None:
                raise ValueError(
                    f"Split not specified. {self._train_splits} splits available."
                )
            if split is not None and self._train_splits is not None:
                assert (
                    self._train_splits is not None and split < self._train_splits
                ), f"Split {split} not available. Only {self._train_splits} splits available."
                if self._train_splits_proportions is not None:
                    assert (
                        len(self._train_splits_proportions) == self._train_splits
                    ), f"Number of train splits ({self._train_splits}) and number of train splits proportions ({len(self._train_splits_proportions)}) do not match."
                    assert (
                        sum(self._train_splits_proportions) == 1
                    ), f"Train splits proportions do not sum to 1."
                    lenghts = [
                        int(len(dataset) * p) for p in self._train_splits_proportions
                    ]
                    lenghts[-1] += len(dataset) - sum(lenghts)
                else:
                    lenghts = [len(dataset) // self._train_splits] * self._train_splits
                    lenghts[-1] += len(dataset) % self._train_splits
                dataset = random_split(
                    dataset,
                    lenghts,
                    generator=torch.Generator().manual_seed(self._seed),
                )[split]
            elif self._train_size is not None:
                dataset = Subset(
                    dataset,
                    torch.randperm(
                        len(dataset),
                        generator=torch.Generator().manual_seed(self._seed),
                    )[: self._train_size],
                )
            return dataset
        else:
            raise ValueError("Training dataset not initialized.")

    def train_dataset_size(self, split: Optional[int] = None) -> int:
        """Returns the size of the training dataset."""
        return len(self.train_dataset(split))

    def train_dataloader(
        self, shuffle: bool = True, split: Optional[int] = None
    ) -> torch.utils.data.DataLoader:
        """Returns the training data loader."""
        return torch.utils.data.DataLoader(
            self.train_dataset(split),
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def train_number_of_batches(self, split: Optional[int] = None) -> int:
        """Returns the number of batches in the training dataset."""
        return len(self.train_dataloader(split=split))

    def validation_dataset(self) -> Optional[SurrogateDataset]:
        """Returns the validation dataset."""
        return self._validation_dataset

    def validation_dataset_size(self) -> int:
        """Returns the size of the validation dataset."""
        return (
            len(self.validation_dataset())
            if self._validation_dataset is not None
            else 0
        )

    def validation_dataloader(
        self, shuffle: bool = False
    ) -> Optional[torch.utils.data.DataLoader]:
        """Returns the validation data loader."""
        return (
            torch.utils.data.DataLoader(
                self.validation_dataset(),
                batch_size=self._batch_size,
                shuffle=shuffle,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
            )
            if self._validation_dataset is not None
            else None
        )

    def validation_number_of_batches(self) -> int:
        """Returns the number of batches in the validation dataset."""
        return len(self.validation_dataloader())

    def test_dataset(self) -> SurrogateDataset:
        """Returns the test dataset."""
        if self._test_dataset is not None:
            return self._test_dataset
        else:
            raise ValueError("Test dataset not initialized.")

    def test_dataset_size(self) -> int:
        """Returns the size of the test dataset."""
        return len(self.test_dataset())

    def test_dataloader(self, shuffle: bool = False) -> torch.utils.data.DataLoader:
        """Returns the test data loader."""
        return torch.utils.data.DataLoader(
            self.test_dataset(),
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def test_number_of_batches(self) -> int:
        """Returns the number of batches in the test dataset."""
        return len(self.test_dataloader())

    def calibration_dataset(self) -> Optional[SurrogateDataset]:
        """Returns the calibration dataset."""
        return self._calibration_dataset

    def calibration_dataset_size(self) -> int:
        """Returns the size of the calibration dataset."""
        return (
            len(self.calibration_dataset())
            if self._calibration_dataset is not None
            else 0
        )

    def calibration_dataloader(
        self, shuffle: bool = False
    ) -> Optional[torch.utils.data.DataLoader]:
        """Returns the calibration data loader."""
        return (
            torch.utils.data.DataLoader(
                self.calibration_dataset(),
                batch_size=self._batch_size,
                shuffle=shuffle,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
            )
            if self._calibration_dataset is not None
            else None
        )

    def calibration_number_of_batches(self) -> int:
        """Returns the number of batches in the calibration dataset."""
        return len(self.calibration_dataloader())

    def total_dataset_size(self) -> int:
        """Returns the size of the total dataset."""
        return (
            self.train_dataset_size()
            + self.validation_dataset_size()
            + self.test_dataset_size()
            + self.calibration_dataset_size()
        )

    def sample_data(
        self, batch_size: int = 1, dataset: str = TRAIN_KEY
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample random data from training, validation or test dataset.

        It returns the input, target and index of the sampled data.
        """
        indices = torch.randperm(batch_size)
        x, y = [], []
        for index in indices:
            if dataset == TRAIN_KEY:
                x_, y_ = self._train_dataset[index]
            elif dataset == VALIDATION_KEY:
                x_, y_ = self._validation_dataset[index]
            elif dataset == TEST_KEY:
                x_, y_ = self._test_dataset[index]
            elif dataset == CALIBRATION_KEY:
                x_, y_ = self._calibration_dataset[index]
            else:
                raise ValueError(f"Dataset {dataset} not available.")
            # In some cases the y can be an integer, in this case we need to convert it to a tensor
            if not isinstance(y_, torch.Tensor):
                y_ = torch.tensor(y_)
            x.append(x_)
            y.append(y_)
        x = torch.stack(x, dim=0)
        y = torch.stack(y, dim=0)
        return x, y, indices

    @abstractmethod
    def prepare_data(self) -> None:
        """Download and prepare the data, the data is stored in `self._train_dataset`, `self._validation_dataset`, `self._test_dataset` and `self._calibration_dataset`."""
        pass

    def setup(self, *args: Any, **kwargs: Any) -> None:
        """Split the data into training, validation, calibration and test sets.

        The training and test sets need to be always provided, the validation and calibration sets are optional.
        The validation and calibration sets can be also provided in the base datamodule, then the portions are
        ignored.
        The splitting with respect to validation and calibration sets is done with respect to the training set.
        """

        if (self._validation_dataset is None and self._validation_portion > 0) or (
            self._calibration_dataset is None and self._calibration_portion > 0
        ):
            validation_portion = (
                self._validation_portion if self._validation_dataset is None else 0
            )
            calibration_portion = (
                self._calibration_portion if self._calibration_dataset is None else 0
            )

            validation_size = int(validation_portion * len(self._train_dataset))
            calibration_size = int(calibration_portion * len(self._train_dataset))
            train_size = len(self._train_dataset) - validation_size - calibration_size
            (
                train_dataset,
                validation_dataset,
                calibration_dataset,
            ) = random_split(
                self._train_dataset,
                [train_size, validation_size, calibration_size],
                generator=torch.Generator().manual_seed(self._seed),
            )

            split = False  # This checks if the dataset was split
            if len(validation_dataset) != 0 and self._validation_dataset is None:
                self._validation_dataset = validation_dataset
                split = True

            if len(calibration_dataset) != 0 and self._calibration_dataset is None:
                self._calibration_dataset = calibration_dataset
                split = True

            if split:
                self._train_dataset = train_dataset

        if not isinstance(self._train_dataset, SurrogateDataset):
            self._train_dataset = SurrogateDataset(
                self._train_dataset,
                transform=self.train_transform(),
                target_transform=self.train_target_transform(),
                joint_transform=self.train_joint_transform(),
            )
        if self._validation_dataset is not None and not isinstance(
            self._validation_dataset, SurrogateDataset
        ):
            self._validation_dataset = SurrogateDataset(
                self._validation_dataset,
                transform=self.test_transform(),
                target_transform=self.test_target_transform(),
                joint_transform=self.test_joint_transform(),
            )
        if self._calibration_dataset is not None and not isinstance(
            self._calibration_dataset, SurrogateDataset
        ):
            self._calibration_dataset = SurrogateDataset(
                self._calibration_dataset,
                transform=self.test_transform(),
                target_transform=self.test_target_transform(),
                joint_transform=self.test_joint_transform(),
            )
        if not isinstance(self._test_dataset, SurrogateDataset):
            self._test_dataset = SurrogateDataset(
                self._test_dataset,
                transform=self.test_transform(),
                target_transform=self.test_target_transform(),
                joint_transform=self.test_joint_transform(),
            )

        logging.info(f"Train dataset total size: {self.train_dataset_size()}")
        if self._train_splits is not None:
            for i in range(self._train_splits):
                logging.info(
                    f"Train dataset size for split {i}: {len(self.train_dataset(split=i))}"
                )
        logging.info(f"Validation dataset size: {self.validation_dataset_size()}")
        logging.info(f"Calibration dataset size: {self.calibration_dataset_size()}")
        logging.info(f"Test dataset size: {self.test_dataset_size()}")

    @torch.no_grad()
    def _get_prediction(
        self,
        tester: LightningModule,
        x: torch.Tensor,
        y: Union[torch.Tensor, int],
        phase: str = TRAIN_KEY,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the prediction, input and target for the given input and target."""
        # Need to set the tester to `eval` mode
        assert phase in [
            TRAIN_KEY,
            VALIDATION_KEY,
            CALIBRATION_KEY,
            TEST_KEY,
        ], f"Phase {phase} not recognized."
        if phase == TRAIN_KEY:
            tester.train()
        else:
            tester.eval()

    def plot(
        self, tester: LightningModule, save_path: str, specific_name: str = ""
    ) -> None:
        """if `self.can_be_plotted` is True, this method is used to plot the data and the model predictions."""
        pass

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add datamodel specific arguments to the general parser."""
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--datamodule_data_dir",
            type=str,
            default="~/.pytorch/",
            help="Path to the data directory.",
        )
        parser.add_argument(
            "--datamodule_train_splits",
            type=int,
            default=None,
            help="Number of splits to use for the training data.",
        )
        parser.add_argument(
            "--datamodule_train_splits_proportions",
            type=str,
            default=None,
            help="Proportions of the training data to use for each split.",
        )
        parser.add_argument(
            "--datamodule_train_size",
            type=int,
            default=None,
            help="Size of the training data.",
        )
        parser.add_argument(
            "--datamodule_validation_portion",
            type=float,
            default=0.1,
            help="Portion of the training data to use for validation.",
        )
        parser.add_argument(
            "--datamodule_calibration_portion",
            type=float,
            default=0.0,
            help="Portion of the training data to use for calibration.",
        )
        parser.add_argument(
            "--datamodule_test_portion",
            type=float,
            default=0.1,
            help="Portion of the training data to use for test if test data is not provided.",
        )
        parser.add_argument(
            "--datamodule_num_workers",
            type=int,
            default=None,
            help="Number of workers to use for the data loaders.",
        )
        parser.add_argument(
            "--datamodule_batch_size",
            type=int,
            default=32,
            help="Batch size to use for the data loaders.",
        )
        parser.add_argument(
            "--datamodule_pin_memory",
            type=int,
            default=1,
            choices=[0, 1],
            help="Whether to pin the memory for the data loaders.",
        )
        parser.add_argument(
            "--datamodule_train_transform",
            type=str,
            default=None,
            help="Transform to apply to the training data.",
        )
        parser.add_argument(
            "--datamodule_train_target_transform",
            type=str,
            default=None,
            help="Target transform to apply to the training data.",
        )
        parser.add_argument(
            "--datamodule_train_joint_transform",
            type=str,
            default=None,
            help="Joint transform to apply to the training data.",
        )
        parser.add_argument(
            "--datamodule_test_transform",
            type=str,
            default=None,
            help="Transform to apply to the test data.",
        )
        parser.add_argument(
            "--datamodule_test_target_transform",
            type=str,
            default=None,
            help="Target transform to apply to the test data.",
        )
        parser.add_argument(
            "--datamodule_test_joint_transform",
            type=str,
            default=None,
            help="Joint transform to apply to the test data.",
        )
        parser.add_argument(
            "--datamodule_test_augmentations",
            type=str,
            default=None,
            help="Augmentations to apply to the test data.",
        )

        return parser


class VisionDataModule(BaseDataModule):
    """Data module for the vision datasets.

    Args:
        validation_portion (float): Portion of the training data to use for validation.
        seed (int): Seed for the random number generator.
        data_dir (str): Path to the data directory.
        train_tranform (Callable): Transformations to apply to the training data. Default: `transforms.ToTensor(), transforms.Normalize(mean, str)`.
        test_transform (Callable): Transformations to apply to the test data. Default: `transforms.ToTensor(), transforms.Normalize(mean, str)`.
    """

    mean: Tuple[float, ...] = None
    std: Tuple[float, ...] = None

    inputs_dtype = torch.float32

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.available_transforms += ["normalize", "horizontalflip", "randomcrop"]
        self.available_test_augmentations = VisionCorruption.available_augmentations + [
            "randomgaussiannoise",
            "randomuniformnoise",
        ]

        if self._train_transform is None:
            self._train_transform = [
                "randomcrop",
                "horizontalflip",
                "totensor",
                "normalize",
            ]
        if self._test_transform is None:
            self._test_transform = ["totensor", "normalize"]

        if self.test_augmentations is None or len(self.test_augmentations) == 0:
            self.test_augmentations = VisionCorruption.available_augmentations

        self._data_mean: torch.Tensor = None  # Mean of the training data
        self._data_std: torch.Tensor = None  # Standard deviation of the training data

        self._data_min: torch.Tensor = None  # Minimum of the training data
        self._data_max: torch.Tensor = None  # Maximum of the training data

    def get_transform(self, name: str) -> Callable:
        """This is a helper function to get the transform by name."""
        transform = super().get_transform(name)
        if transform is not None:
            return transform
        elif name == "normalize":
            return transforms.Normalize(self.mean, self.std)
        elif name == "horizontalflip":
            return transforms.RandomHorizontalFlip()
        elif name == "randomcrop":
            assert (
                self.inputs_dim is not None
            ), f"`inputs_dim` must be set to use {name} transform."
            return transforms.RandomCrop(self.inputs_dim[1], padding=4)
        elif name == "randomgaussiannoise":
            return RandomImageNoise(
                size=self.inputs_dim,
                minimum=self._data_min,
                maximum=self._data_max,
                mean=self._data_mean,
                std=self._data_std,
                noise="gaussian",
            )
        elif name == "randomuniformnoise":
            return RandomImageNoise(
                size=self.inputs_dim,
                minimum=self._data_min,
                maximum=self._data_max,
                mean=self._data_mean,
                std=self._data_std,
                noise="uniform",
            )
        elif name in VisionCorruption.available_augmentations:
            return VisionCorruption(img_size=self.inputs_dim, corruption_name=name)

    def _denormalize(self, image: torch.Tensor) -> torch.Tensor:
        """Denormalize the image."""
        mean = torch.tensor(self.mean, device=image.device)
        std = torch.tensor(self.std, device=image.device)
        return image * std[:, None, None] + mean[:, None, None]

    def _mean_std_max_min(self, dataset: Iterable) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the mean, standard deviation, maximum and minimum per channel of the dataset.

        The data can be PIL images, numpy arrays or tensors.
        It is assumed that the first index is the input and the second index is the target.
        All the values are with respect to raw RGB values.
        """
        inputs = []
        # Collect all the inputs in a list as tensors
        # Collect at most 1000 inputs
        for i, (input, _) in enumerate(dataset):
            if isinstance(input, Image.Image):
                input = np.array(input)
            if isinstance(input, np.ndarray):
                input = torch.from_numpy(input)
            if isinstance(input, Image.Image):
                # Get the numpy array
                input = np.array(input)
                # Convert to tensor
                input = torch.from_numpy(input)
            if not isinstance(input, torch.Tensor):
                raise ValueError(f"Input type {type(input)} not supported.")

            # Convert to float
            input = input.float()

            # If the input is grayscale, add a channel dimension
            if input.dim() == 2:
                input = input.unsqueeze(0)

            inputs.append(input)
            
            if i == 999:
                break

        inputs = torch.stack(inputs, dim=0)

        # Check which dimension is channel dimension, it can be only 1 or 3
        if inputs.shape[1] not in [1, 3]:
            # Permute the channel dimension to the first dimension
            inputs = inputs.permute(0, 3, 1, 2)

        mean = inputs.mean(dim=(0, 2, 3))
        std = inputs.std(dim=(0, 2, 3))

        # Compute the maximum and minimum per channel
        maximum = []
        minimum = []
        for i in range(inputs.shape[1]):
            maximum.append(inputs[:, i].max())
            minimum.append(inputs[:, i].min())

        maximum = torch.stack(maximum, dim=0)
        minimum = torch.stack(minimum, dim=0)

        return mean, std, maximum, minimum

    def setup(
        self,
        augmentation: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Split the data into training, validation and test sets.

        Additionally for apply augmentation to the test data.
        Insert the augmentation into the existing test transformation at the first position.

        Args:
            augmentation (str): Name of the augmentation to apply to the test data.
        """
        if self._data_mean is None or self._data_std is None:
            # Do this only once
            (
                self._data_mean,
                self._data_std,
                self._data_max,
                self._data_min,
            ) = self._mean_std_max_min(self._train_dataset)

        super().setup(*args, **kwargs)
        if augmentation is None:
            return

        test_transform = self.test_transform()
        test_transform.transforms.insert(
            0,
            self.test_augmentation(augmentation),
        )
        self._test_dataset._transform = test_transform

    @torch.no_grad()
    def plot(
        self, tester: LightningModule, save_path: str, specific_name: str = ""
    ) -> None:
        """Plot random samples from the training, validation and test set to check if the data is correctly predicted."""
        fig, axs = plt.subplots(3, 5, figsize=(12, 4))

        # Create a batch of 5 random samples from the training set
        x, y, idxs = self.sample_data(5, dataset=TRAIN_KEY)
        y_hat, x, y = self._get_prediction(tester, x, y, phase=TRAIN_KEY)

        for i in range(5):
            axs[0, i].imshow(self._denormalize(x[i]).permute(1, 2, 0))
            axs[0, i].set_title(f"Train {idxs[i]}, y={y[i]}, y_hat={y_hat[i]}")
            axs[0, i].axis("off")

        if self._validation_dataset is not None:
            # Create a batch of 5 random samples from the validation set
            x, y, idxs = self.sample_data(5, dataset=VALIDATION_KEY)
            y_hat, x, y = self._get_prediction(tester, x, y, phase=VALIDATION_KEY)

            for i in range(5):
                axs[1, i].imshow(self._denormalize(x[i]).permute(1, 2, 0))
                axs[1, i].set_title(f"Val {idxs[i]}, y={y[i]}, y_hat={y_hat[i]}")
                axs[1, i].axis("off")

        # Create a batch of 5 random samples from the test set
        x, y, idxs = self.sample_data(5, dataset=TEST_KEY)
        y_hat, x, y = self._get_prediction(tester, x, y, phase=TEST_KEY)

        for i in range(5):
            axs[2, i].imshow(self._denormalize(x[i]).permute(1, 2, 0))
            axs[2, i].set_title(f"Test {idxs[i]}, y={y[i]}, y_hat={y_hat[i]}")
            axs[2, i].axis("off")

        plt.tight_layout()
        plt.savefig(plots_file(save_path, specific_name), bbox_inches="tight")
        plt.close(fig)


class VisionClassificationDataModule(VisionDataModule):
    """Data module for the vision classification datasets."""

    task = CLASSIFICATION_KEY

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
        y = torch.tensor(y).long().to(tester.device)
        if phase == TRAIN_KEY:
            output = tester.training_step([x, y], batch_idx=0)
        elif phase == VALIDATION_KEY:
            output = tester.validation_step([x, y], batch_idx=0)
        elif phase == TEST_KEY:
            output = tester.test_step([x, y], batch_idx=0)
        y_hat = output[MEAN_PREDICTION_KEY].cpu()
        x = output[INPUT_KEY].cpu()
        y = output[TARGET_KEY].cpu()
        y_hat = y_hat.argmax(dim=1)
        return y_hat, x, y


class VisionRegressionDataModule(VisionDataModule):
    """Data module for the vision regression datasets."""

    task = REGRESSION_KEY

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
        y = torch.tensor(y).float().to(tester.device)
        if phase == TRAIN_KEY:
            output = tester.training_step([x, y], batch_idx=0)
        elif phase == VALIDATION_KEY:
            output = tester.validation_step([x, y], batch_idx=0)
        elif phase == TEST_KEY:
            output = tester.test_step([x, y], batch_idx=0)
        y_hat = output[MEAN_PREDICTION_KEY].cpu()
        x = output[INPUT_KEY].cpu()
        y = output[TARGET_KEY].cpu()
        return y_hat, x, y

    @torch.no_grad()
    def plot(
        self, tester: LightningModule, save_path: str, specific_name: str = ""
    ) -> None:
        """Plot random samples from the training, validation and test set to check if the data is correctly predicted."""
        fig, axs = plt.subplots(3, 5, figsize=(12, 4))

        # Create a batch of 5 random samples from the training set
        x, y, idxs = self.sample_data(5, dataset=TRAIN_KEY)
        y_hat, x, y = self._get_prediction(tester, x, y, phase=TRAIN_KEY)
        mean, std = y_hat[:, 0], y_hat[:, 1].sqrt()

        for i in range(5):
            axs[0, i].imshow(self._denormalize(x[i]).permute(1, 2, 0))
            axs[0, i].set_title(
                f"Train {idxs[i]}, y={y[i].item():.2f}, y_hat={mean[i].item():.2f} ± {std[i].item():.2f}"
            )
            axs[0, i].axis("off")

        if self._validation_dataset is not None:
            # Create a batch of 5 random samples from the validation set
            x, y, idxs = self.sample_data(5, dataset=VALIDATION_KEY)
            y_hat, x, y = self._get_prediction(tester, x, y, phase=VALIDATION_KEY)
            mean, std = y_hat[:, 0], y_hat[:, 1].sqrt()

            for i in range(5):
                axs[1, i].imshow(self._denormalize(x[i]).permute(1, 2, 0))
                axs[1, i].set_title(
                    f"Val {idxs[i]}, y={y[i].item():.2f}, y_hat={mean[i].item():.2f} ± {std[i].item():.2f}"
                )
                axs[1, i].axis("off")

        # Create a batch of 5 random samples from the test set
        x, y, idxs = self.sample_data(5, dataset=TEST_KEY)
        y_hat, x, y = self._get_prediction(tester, x, y, phase=TEST_KEY)
        mean, std = y_hat[:, 0], y_hat[:, 1].sqrt()

        for i in range(5):
            axs[2, i].imshow(self._denormalize(x[i]).permute(1, 2, 0))
            axs[2, i].set_title(
                f"Test {idxs[i]}, y={y[i].item():.2f}, y_hat={mean[i].item():.2f} ± {std[i].item():.2f}"
            )
            axs[2, i].axis("off")

        plt.tight_layout()
        plt.savefig(plots_file(save_path, specific_name), bbox_inches="tight")
        plt.close(fig)


class RealWorldDataModule(BaseDataModule):
    """Data module for real world datasets.

    To test out-of-distribution robustness, the test dataset can be modified
    with tabular corruptions. The corruptions are applied to the test dataset only.
    """

    inputs_dtype = torch.float32
    outputs_dtype = torch.long
    targets_dim = 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._data_mean: torch.Tensor = None  # The mean of the features.
        # The standard deviation of the features.
        self._data_std: torch.Tensor = None

        self._data_max: torch.Tensor = None  # The max of the features.
        # The min of the features.
        self._data_min: torch.Tensor = None

        self.available_transforms += ["datanormalize", "targetnormalize", "fromonehot"]
        self.available_test_augmentations = (
            TabularCorruption.available_augmentations
            + ["randomgaussiannoise", "randomuniformnoise"]
        )

        if self._train_transform is None:
            self._train_transform = ["datanormalize"]
        if self._test_transform is None:
            self._test_transform = ["datanormalize"]

        if self.test_augmentations is None or len(self.test_augmentations) == 0:
            self.test_augmentations = TabularCorruption.available_augmentations

    def get_transform(self, name: str) -> Callable:
        """This is a helper function to get the transform by name."""
        transform = super().get_transform(name)
        if transform is not None:
            return transform

        if name == "datanormalize":
            return Normalize(self._data_mean, self._data_std)
        elif name == "targetnormalize":
            return Normalize(self._target_mean, self._target_std)
        elif name == "fromonehot":
            return FromOneHot()
        elif name == "randomgaussiannoise":
            return RandomTabularNoise(
                size=self.inputs_dim,
                maximum=self._data_max,
                minimum=self._data_min,
                mean=self._data_mean,
                std=self._data_std,
                noise="gaussian",
            )
        elif name == "randomuniformnoise":
            return RandomTabularNoise(
                size=self.inputs_dim,
                maximum=self._data_max,
                minimum=self._data_min,
                mean=self._data_mean,
                std=self._data_std,
                noise="uniform",
            )
        elif name in TabularCorruption.available_augmentations:
            return TabularCorruption(corruption_name=name)

    def _split_train_test(
        self, dataset: TensorDataset
    ) -> Tuple[TensorDataset, TensorDataset]:
        """Splits the dataset into training and testing sets."""
        train_size = int((1.0 - self._test_portion) * len(dataset))
        test_size = len(dataset) - train_size
        return random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(self._seed),
        )

    def _mean_std(
        self, dataset: TensorDataset, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A helper function to compute the mean and standard deviation for `index`th column of the dataset."""
        mean = torch.mean(
            torch.stack([dataset[i][index] for i in range(len(dataset))]), dim=0
        )
        std = torch.std(
            torch.stack([dataset[i][index] for i in range(len(dataset))]), dim=0
        )
        return mean, std

    def _max_min(
        self, dataset: TensorDataset, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A helper function to compute the max and min for `index`th column of the dataset."""
        max_ = torch.max(
            torch.stack([dataset[i][index] for i in range(len(dataset))]), dim=0
        ).values
        min_ = torch.min(
            torch.stack([dataset[i][index] for i in range(len(dataset))]), dim=0
        ).values
        return max_, min_

    def prepare_data(self) -> None:
        """Prepares the data for training, validation, and testing."""
        super().prepare_data()
        data = self._download_data()
        self._train_dataset, self._test_dataset = self._split_train_test(data)

        self._data_mean, self._data_std = self._mean_std(self._train_dataset, 0)
        self._data_max, self._data_min = self._max_min(self._train_dataset, 0)

    def setup(self, augmentation: Optional[str] = None) -> None:
        """Split the data into training, validation and test sets.

        Additionally for apply augmentation to the test data.
        Insert the augmentation into the existing test transformation at the first position.

        Args:
            augmentation (str): Name of the augmentation to apply to the training data. Default: None.
        """
        super().setup()
        if augmentation is None:
            return

        test_transform = self.test_transform()
        test_transform.transforms.insert(
            0,
            self.test_augmentation(augmentation),
        )
        self._test_dataset._transform = test_transform

    def _download_data(self) -> TensorDataset:
        """Downloads the dataset."""
        raise NotImplementedError("This method must be implemented by the subclass.")


class RealWorldRegressionDataModule(RealWorldDataModule):
    """Data module for real world regression datasets.

    Args:
        test_portion (float): Portion of the training data to use for testing.
    """

    inputs_dtype = torch.float32
    outputs_dtype = torch.float32
    targets_dim = 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The mean of the targets.
        self._target_mean: torch.Tensor = None
        # The standard deviation of the targets.
        self._target_std: torch.Tensor = None
        if self._train_target_transform is None:
            self._train_target_transform = ["targetnormalize"]
        if self._test_target_transform is None:
            self._test_target_transform = ["targetnormalize"]

    def prepare_data(self) -> None:
        """Prepares the data for training, validation, and testing."""
        super().prepare_data()
        self._target_mean, self._target_std = self._mean_std(self._train_dataset, 1)

    @torch.no_grad()
    def _get_prediction(
        self,
        tester: LightningModule,
        x: torch.Tensor,
        y: Union[torch.Tensor, int],
        phase: str = TRAIN_KEY,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the prediction of the model."""
        super()._get_prediction(tester, x, y, phase)
        x = x.to(tester.device)
        y = y.to(tester.device)
        if phase == TRAIN_KEY:
            output = tester.training_step([x, y], batch_idx=0)
        elif phase == VALIDATION_KEY:
            output = tester.validation_step([x, y], batch_idx=0)
        elif phase == TEST_KEY:
            output = tester.test_step([x, y], batch_idx=0)
        y_hat = output[MEAN_PREDICTION_KEY]
        x = output[INPUT_KEY].cpu()
        y = output[TARGET_KEY].cpu()
        # Convert variance to standard deviation.
        y_hat[:, 1] = torch.sqrt(y_hat[:, 1])
        return y_hat, x, y


class RealWorldClassificationDataModule(RealWorldDataModule):
    """Data module for real world classification datasets.

    Args:
        test_portion (float): Portion of the training data to use for testing.

    """

    inputs_dtype = torch.float32
    outputs_dtype = torch.long
    targets_dim = 1

    @torch.no_grad()
    def _get_prediction(
        self,
        tester: LightningModule,
        x: torch.Tensor,
        y: Union[torch.Tensor, int],
        phase: str = TRAIN_KEY,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the prediction of the model."""
        super()._get_prediction(tester, x, y, phase)
        x = x.to(tester.device)
        y = y.to(tester.device)
        if phase == TRAIN_KEY:
            output = tester.training_step([x, y], batch_idx=0)
        elif phase == VALIDATION_KEY:
            output = tester.validation_step([x, y], batch_idx=0)
        elif phase == TEST_KEY:
            output = tester.test_step([x, y], batch_idx=0)
        y_hat = output[MEAN_PREDICTION_KEY].cpu()
        x = output[INPUT_KEY].cpu()
        y = output[TARGET_KEY].cpu()
        return y_hat, x, y
