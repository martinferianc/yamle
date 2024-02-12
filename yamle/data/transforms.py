from typing import Any
import torch
from torch.utils.data import Subset
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode, Compose
from typing import Tuple, List

from yamle.defaults import TINY_EPSILON


class JointCompose(Compose):
    """This class is a wrapper for the torchvision Compose class, which applies transformations to
    both the data and the target.

    Args:
        transforms (List[Callable]): List of transformations to apply to the data and the target.
    """

    def __call__(self, x: Any, y: Any) -> Tuple[Any, Any]:
        for t in self.transforms:
            x, y = t(x, y)
        return x, y


class FromOneHot:
    """This class converts one-hot encoded targets to class labels."""

    def __call__(self, target: torch.Tensor) -> torch.Tensor:
        """Converts one-hot encoded targets to class labels.

        Args:
            target (torch.Tensor): One-hot encoded targets.
        """
        return torch.argmax(target, dim=1)


class JointResize:
    """Perform resizing if the input is larger or smaller than the limiting height/width.

    The input is interpolated using the bilinear interpolation method.
    The target is interpolated using the nearest neighbour interpolation method.

    Args:
        height (int): The limiting height.
        width (int): The limiting width.
    """

    def __init__(self, height: int, width: int) -> None:
        self._height = height
        self._width = width

    def __call__(
        self, img: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This method is called when the object is called as a function."""
        img = F.resize(
            img,
            (self._height, self._width),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        target = F.resize(
            target,
            (self._height, self._width),
            interpolation=InterpolationMode.NEAREST,
            antialias=True,
        )
        return img, target


class JointCenterCrop:
    """Perform center cropping if the input is larger than the limiting height/width.

    Args:
        height (int): The limiting height.
        width (int): The limiting width.
    """

    def __init__(self, height: int, width: int) -> None:
        self._height = height
        self._width = width

    def __call__(
        self, img: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This method is called when the object is called as a function."""
        img = F.center_crop(img, (self._height, self._width))
        target = F.center_crop(target, (self._height, self._width))
        return img, target


class JointNormalize:
    """Perform normalization on the input and leave the target unchanged.

    Args:
        mean (Tuple[float, float, float]): Mean values for each channel.
        std (Tuple[float, float, float]): Standard deviation values for each channel.

    """

    def __init__(
        self, mean: Tuple[float, float, float], std: Tuple[float, float, float]
    ) -> None:
        self._mean = mean
        self._std = std

    def __call__(
        self, img: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This method is called when the object is called as a function."""
        img = F.normalize(img, self._mean, self._std)
        return img, target


class JointToTensor:
    """Convert the input and target to tensors.

    Args:
        img_dtype (torch.dtype): The data type of the input. Default: torch.float32.
        target_dtype (torch.dtype): The data type of the target. Default: torch.long.
    """

    def __init__(
        self,
        img_dtype: torch.dtype = torch.float32,
        target_dtype: torch.dtype = torch.long,
    ) -> None:
        self._img_dtype = img_dtype
        self._target_dtype = target_dtype

    def __call__(
        self, img: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This method is called when the object is called as a function."""
        img = F.to_tensor(img).to(self._img_dtype)
        target = F.pil_to_tensor(target).to(self._target_dtype)
        return img, target


class JointTargetSqueeze:
    """Squeeze the target tensor.

    Args:
        dim (int): The dimension to squeeze.
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim

    def __call__(
        self, img: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This method is called when the object is called as a function."""
        target = torch.squeeze(target, dim=self._dim)
        return img, target


class TargetToUnit:
    """
    Converts the target image to meters.

    Args:
        scale (float): Scale factor to convert the target image.
    """

    def __init__(self, scale: float = 1.0) -> None:
        self._scale = scale

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        return y.float() / self._scale


class ClassificationDatasetSubset:
    """This class takes a subset of a dataset specified by a list of indices corresponding to the subset of the classes.

    It creates a `Subset` object from the `torch.utils.data` package. It goes through the entire dataset and checks
    which indices correspond to the subset of the classes. It then creates a list of indices corresponding to the subset.
    Finally, it creates a `Subset` object from the dataset and the list of indices.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to take a subset of.
        indices (List[int]): The indices of the subset of classes.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, indices: List[int]) -> None:
        self._dataset = dataset
        self._indices = indices
        self._subset = self._create_subset()

    def _create_subset(self) -> Subset:
        """Create a subset of the dataset."""
        subset_indices = []
        for i in range(len(self._dataset)):
            _, target = self._dataset[i]
            if target in self._indices:
                subset_indices.append(i)
        return Subset(self._dataset, subset_indices)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """This method is called when the object is called as a function."""
        x, y = self._subset[index]
        # Make also sure that the indices are in the range
        # of the number of classes in the subset dataset.
        y = self._indices.index(y)
        return x, y

    def __len__(self) -> int:
        """This method is called when the object is called as a function."""
        return len(self._subset)


class Normalize:
    """This class normalizes the data and targets to zero mean and unit variance, given the mean and standard deviation
    of the training data.

    Args:
        mean (torch.Tensor): Mean of the training data.
        std (torch.Tensor): Standard deviation of the training data.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self._mean = mean
        self._std = std

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Normalizes the data to zero mean and unit variance.

        Args:
            data (torch.Tensor): Data to normalize.
        """
        return (data - self._mean) / (self._std + TINY_EPSILON)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse normalization.

        Args:
            data (torch.Tensor): Data to inverse normalize.
        """
        return data * self._std + self._mean


class Denormalize:
    """Denormalize the input image.

    Args:
        mean (Tuple[float, float, float]): Mean values for each channel.
        std (Tuple[float, float, float]): Standard deviation values for each channel.
    """

    def __init__(
        self, mean: Tuple[float, float, float], std: Tuple[float, float, float]
    ) -> None:
        assert len(mean) == 3, "The mean must be a tuple of length 3."
        assert len(std) == 3, "The std must be a tuple of length 3."
        self._mean = mean
        self._std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """This method is called when the object is called as a function."""
        x = x * torch.tensor(self._std, device=x.device).view(3, 1, 1) + torch.tensor(
            self._mean, device=x.device
        ).view(3, 1, 1)
        return x