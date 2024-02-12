from typing import Tuple, Optional

import torchvision
from PIL import Image
from torch.utils.data import Dataset


class MedMNISTDatasetWrapper(Dataset):
    """This is a wrapper class for MedMNIST dataset.

    It enables the padding of the images to 32x32, which is required by the
    corruptions and might be required by the models. It also does unsqueezing
    of the target since they are in a numpy array of shape (1,).

    Args:
        dataset (Dataset): The MedMNIST dataset.
        pad_to_32 (bool): Whether to pad the images to 32x32. Defaults to True.
        target_normalization (bool): Whether to normalize the target to [0, 1]. Defaults to False.
                                     This is used for the ordinal regression task.
    """

    def __init__(
        self,
        dataset: Dataset,
        pad_to_32: bool = True,
        target_normalization: bool = False,
    ):
        self._dataset = dataset
        self._pad_to_32 = pad_to_32
        self._target_normalization = target_normalization
        self._min_max: Optional[Tuple[float, float]] = None
        if self._target_normalization:
            self._min_max = self._get_min_max()

    def __len__(self) -> int:
        return len(self._dataset)

    def _get_min_max(self) -> Tuple[float, float]:
        """This is a helper function to get the min and max of the target across the dataset."""
        m = float("inf")
        M = float("-inf")
        for _, target in self._dataset:
            m = min(m, target[0])
            M = max(M, target[0])
        return m, M

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        img, target = self._dataset[idx]
        if self._pad_to_32:
            img = torchvision.transforms.functional.pad(img, 2)  # 28 -> 32
        target = target[0]  # (1,) -> ()
        if self._target_normalization:
            target = (target - self._min_max[0]) / (self._min_max[1] - self._min_max[0])
        return img, target
