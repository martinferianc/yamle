from typing import Callable, Tuple, Any, Optional

from yamle.data.datamodule import BaseDataModule
import urllib.request
import shutil
import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from yamle.third_party.imagenet_c import TabularCorruption
from yamle.third_party.imagenet_c.extra import RandomTabularNoise
from yamle.data.transforms import Normalize


class ECG5000DataModule(BaseDataModule):
    """This is the base class for the EC5000 dataset.

    The link to download the dataset is: https://www.timeseriesclassification.com/description.php?Dataset=ECG5000
    """

    inputs_dim = (140, 1)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._data_mean: torch.Tensor = None  # The mean of the dataset
        self._data_std: torch.Tensor = None  # The standard deviation of the dataset

        self._data_max: torch.Tensor = None  # The maximum value of the dataset
        self._data_min: torch.Tensor = None  # The minimum value of the dataset

        self.available_transforms += ["datanormalize"]
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

    def get_transform(self, name: str) -> Callable[..., Any]:
        """This is a helper function to get the transform by name."""
        transform = super().get_transform(name)
        if transform is not None:
            return transform

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

        elif name == "datanormalize":
            return Normalize(self._data_mean, self._data_std)

        elif name in TabularCorruption.available_augmentations:
            return TabularCorruption(corruption_name=name)

    def setup(
        self,
        augmentation: Optional[str] = None,
    ) -> None:
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

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """This function is used to download the dataset."""
        super().prepare_data()
        if not os.path.exists(
            os.path.join(os.path.expanduser(self._data_dir), "ECG5000.zip")
        ):
            # Download the dataset
            urllib.request.urlretrieve(
                "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip",
                "ECG5000.zip",
            )

            shutil.move(
                "ECG5000.zip",
                os.path.join(os.path.expanduser(self._data_dir), "ECG5000.zip"),
            )
            # Unzip the dataset
            shutil.unpack_archive(
                os.path.join(os.path.expanduser(self._data_dir), "ECG5000.zip"),
                os.path.join(os.path.expanduser(self._data_dir), "ECG5000"),
            )

        data_dir = os.path.join(os.path.expanduser(self._data_dir), "ECG5000")

        # Separate the timesteps, each row is a sample
        # Read all of the data files and replace whitespace with commas
        train_data = pd.read_csv(
            os.path.join(data_dir, "ECG5000_TRAIN.txt"), sep="\s+", header=None
        )
        test_data = pd.read_csv(
            os.path.join(data_dir, "ECG5000_TEST.txt"), sep="\s+", header=None
        )

        return train_data, test_data

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
        """A helper function to compute the maximum and minimum for `index`th column of the dataset."""
        max_val = torch.max(
            torch.stack([dataset[i][index] for i in range(len(dataset))]), dim=0
        )[0]
        min_val = torch.min(
            torch.stack([dataset[i][index] for i in range(len(dataset))]), dim=0
        )[0]
        return max_val, min_val
