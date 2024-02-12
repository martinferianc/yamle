from typing import Tuple, Union
import numpy as np
import PIL
from PIL import Image
from yamle.third_party.imagenet_c.corruptions import *
from yamle.third_party.imagenet_c.tabular_corruptions import *
from torchvision.transforms import ToPILImage
import torch


class VisionCorruption:
    """Corrupts an image with a given corruption and severity.

    The image must be a square and the size must be a multiple of 2.

    Args:
        img_size (Tuple[int, int, int]): The size of the image. The image must be a square and the size must be a multiple of 2.
        corruption_name (str): The name of the corruption. Must be one of 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'.
        severity (int): The severity of the corruption. Must be in [1, 6],
    """

    corruption_tuple = (
        gaussian_noise,
        shot_noise,
        impulse_noise,
        defocus_blur,
        glass_blur,
        motion_blur,
        zoom_blur,
        snow,
        frost,
        fog,
        brightness,
        contrast,
        elastic_transform,
        pixelate,
        jpeg_compression,
        speckle_noise,
        gaussian_blur,
        spatter,
        saturate,
    )
    corruption_dict = {corr_func.__name__: corr_func for corr_func in corruption_tuple}
    corruption_names = list(corruption_dict.keys())
    
    # Perform a cartesian product of the corruption names and the severities
    available_augmentations = [
        f"{corruption_name}_{severity}"
        for corruption_name in corruption_names
        for severity in range(0, 5)
    ]

    def __init__(
        self, img_size: Tuple[int, int, int], corruption_name: str,
    ) -> None:
        # The only sizes the are supported are below 64 and multiples of 2
        _, height, width = img_size
        assert (
            height % 2 == 0 and height == width
        ), f"Image size {img_size} is not supported"
        severity = int(corruption_name.split("_")[-1])
        name = "_".join(corruption_name.split("_")[:-1])
        self._corruption_name = name
        self._severity = severity
        self._img_size = height

    def __call__(
        self, img: Union[np.ndarray, PIL.Image.Image, torch.Tensor]
    ) -> PIL.Image.Image:
        """This function corrupts the image.

        If the image is a numpy array or torch tensor, it will be converted to a PIL image.
        """
        if isinstance(img, torch.Tensor):
            img = ToPILImage()(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif not isinstance(img, PIL.Image.Image):
            raise TypeError(f"Image type {type(img)} is not supported")

        is_grayscale = img.mode == "L"
        # If it is grayscale, we need to convert it to RGB
        # to apply the corruption
        if is_grayscale:
            # Check if the mode is grayscale
            assert img.mode == "L", f"Cannot convert image with mode {img.mode} to RGB"
            img = img.convert("RGB")

        # The severities are in [1, 5] but the functions expect [0, 4]
        # This is why we add 1 to the severity
        # I am really unhappy with this indexing
        x = np.uint8(
            self.corruption_dict[self._corruption_name](
                x=img, img_size=self._img_size, severity=self._severity + 1
            )
        )
        # If the image was grayscale, we need to convert it back
        # to grayscale
        x = Image.fromarray(x)
        if is_grayscale:
            x = x.convert("L")
        return x
    
class TabularCorruption:
    """Corrupts a tabular dataset with a given corruption and severity.
    
    These augmentations are custom to this package and are not part of the original ImageNet-C corruptions.

    Args:
        corruption_name (str): The name of the corruption. Must be one of 'additive_gaussian_noise', 'multiplicative_gaussian_noise', 'additive_uniform_noise', 'multiplicative_uniform_noise', 'multiplicative_bernoulli_noise'.
        severity (int): The severity of the corruption. Must be in [1, 5],
    """

    corruption_tuple = (
        additive_gaussian_noise,
        multiplicative_gaussian_noise,
        additive_uniform_noise,
        multiplicative_uniform_noise,
        multiplicative_bernoulli_noise,
    )
    corruption_dict = {corr_func.__name__: corr_func for corr_func in corruption_tuple}
    corruption_names = list(corruption_dict.keys())
    
    # Perform a cartesian product of the corruption names and the severities
    available_augmentations = [
        f"{corruption_name}_{severity}"
        for corruption_name in corruption_names
        for severity in range(0, 5)
    ]

    def __init__(
        self, corruption_name: str,
    ) -> None:
        severity = int(corruption_name.split("_")[-1])
        name = "_".join(corruption_name.split("_")[:-1])
        self._corruption_name = name
        self._severity = severity

    def __call__(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """This function corrupts the tabular data.

        If the data is a numpy array or torch tensor, it will be converted to a numpy array.
        """
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        elif not isinstance(x, np.ndarray):
            raise TypeError(f"Data type {type(x)} is not supported")

        # The severities are in [1, 5] but the functions expect [0, 4]
        # This is why we add 1 to the
        # I am really unhappy with this indexing
        x = self.corruption_dict[self._corruption_name](
            x=x, severity=self._severity + 1
        )
        return torch.from_numpy(x).float()
        
    
