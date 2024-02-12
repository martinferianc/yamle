from typing import Any, Callable, Optional

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


class SurrogateDataset(Dataset):
    """This class is a dataset wrapper, ensuring that the transforms are applied to the data and targets
    after splitting the dataset into training and validation.

    Args:
        dataset (Dataset): Dataset to wrap.
        transform (Optional[Callable]): Transformations to apply to the data.
        target_transform (Optional[Callable]): Transformations to apply to the targets.
        joint_transform (Optional[Callable]): Transformations to apply to the input as well as the target.
    """

    def __init__(
        self,
        dataset: Dataset,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        joint_transform: Optional[Callable] = None,
    ) -> None:
        self._dataset = dataset
        self._transform = transform
        self._target_transform = target_transform
        self._joint_transform = joint_transform

    def __getitem__(self, index: int) -> Any:
        data, target = self._dataset[index]
        if self._joint_transform is not None:
            data, target = self._joint_transform(data, target)
        if self._transform is not None:
            data = self._transform(data)
        if self._target_transform is not None:
            target = self._target_transform(target)
        return data, target

    def __len__(self) -> int:
        return len(self._dataset)


class InputImagePaddingDataset(Dataset):
    """This class is a dataset wrapper, which can pad the input image to a given size.

    Args:
        dataset (Dataset): Dataset to wrap.
        padding (int): Padding to apply to the input image on all sides.
    """

    def __init__(self, dataset: Dataset, padding: int) -> None:
        self._dataset = dataset
        self._padding = padding

    def __getitem__(self, index: int) -> Any:
        data, target = self._dataset[index]
        assert isinstance(data, Image.Image), f"Data type {type(data)} is not supported"
        data = F.pad(data, self._padding)
        return data, target

    def __len__(self) -> int:
        return len(self._dataset)


class ImageRotationDataset(Dataset):
    """This class is a dataset wrapper for image rotation.

    It discards the target and replaces it with the rotation angle which should be predicted.
    This changes the task from anything to regression.

    Args:
        dataset (Dataset): Dataset to wrap.
        max_angle (float): Maximum angle to rotate the image by. Defaults to 90 degrees.
        min_angle (float): Minimum angle to rotate the image by. Defaults to 0 degrees.
        seed (int): Seed for the random number generator. Defaults to 42.
    """

    def __init__(
        self,
        dataset: Dataset,
        max_angle: float = 90,
        min_angle: float = 0,
        seed: int = 42,
    ) -> None:
        self._dataset = dataset
        self._max_angle = max_angle
        self._min_angle = min_angle
        self._seed = seed
        self._generator = torch.Generator().manual_seed(self._seed)

    def __getitem__(self, index: int) -> Any:
        data, _ = self._dataset[index]
        angle = torch.randint(
            self._min_angle, self._max_angle, (1,), generator=self._generator
        )
        # Scale the angle to between 0 and 1 through min-max scaling
        scaled_angle = (angle - self._min_angle) / (self._max_angle - self._min_angle)
        data = F.rotate(data, angle.item())
        return data, scaled_angle

    def __len__(self) -> int:
        return len(self._dataset)
