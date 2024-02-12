from typing import Tuple

from PIL import Image
import numpy as np
import torch


class RandomImageNoise:
    """This class creates an image where each pixel is uniformly distributed
    between 0 and 255.

    Args:
        size (Tuple[int, int, int]): The size of the image. The shape is `(channels, height, width)`.
        minimum (torch.Tensor): The minimum value of each pixel per channel.
        maximum (torch.Tensor): The maximum value of each pixel per channel.
        mean (torch.Tensor): The mean value of each pixel per channel.
        std (torch.Tensor): The standard deviation of each pixel per channel.
        noise (str): The type of noise to use. Can be one of `uniform` or `gaussian`. Default: `uniform`.
    """

    def __init__(
        self,
        size: Tuple[int, int, int],
        minimum: torch.Tensor,
        maximum: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        noise: str = "uniform",
    ) -> None:
        self._size = size
        # Broadcast all tensors to the same shape
        self._minimum = minimum.reshape(-1, 1, 1)
        self._maximum = maximum.reshape(-1, 1, 1)
        self._mean = mean.reshape(-1, 1, 1)
        self._std = std.reshape(-1, 1, 1)
        self._noise = noise

    def __call__(self, x: Image.Image) -> Image.Image:
        """Creates an image where each pixel is uniformly distributed between 0 and 255.

        Args:
            x (Image.Image): The input image.

        Returns:
            Image.Image: The output image.
        """
        noise = None
        if self._noise == "uniform":
            noise = np.random.uniform(
                self._minimum, self._maximum, size=self._size
            ).astype(np.uint8)
        elif self._noise == "gaussian":
            noise = np.random.normal(self._mean, self._std, size=self._size).astype(
                np.uint8
            )

        # Handle grayscale images (single channel)
        if self._size[0] == 1:
            noise = noise.squeeze(0)  # Remove channel dimension if grayscale

        # Handle color images
        if self._size[0] == 3:
            noise = noise.transpose(1, 2, 0)  # Transpose to (H, W, C)

        return Image.fromarray(noise)

class RandomTabularNoise:
    """This class creates a tabular noise where each feature is uniformly sampled between min and max.

    Args:
        size (Tuple[..., int]): The size of the tabular noise. The shape is `(features)`.
        minimum (torch.Tensor): The minimum value of each feature.
        maximum (torch.Tensor): The maximum value of each feature.
        mean (torch.Tensor): The mean value of each feature.
        std (torch.Tensor): The standard deviation of each feature.
        noise (str): The type of noise to use. Can be one of `uniform` or `gaussian`. Default: `uniform`.
    """

    def __init__(
        self,
        size: Tuple[int, ...],
        minimum: torch.Tensor,
        maximum: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        noise: str = "uniform",
    ) -> None:
        self._size = size
        self._minimum = minimum
        self._maximum = maximum
        self._mean = mean
        self._std = std

        assert noise in [
            "uniform",
            "gaussian",
        ], f"Unknown noise type {noise}. Must be one of `uniform` or `gaussian`."

        self._noise = noise

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Creates a tabular noise where each feature is uniformly sampled between min and max.

        Args:
            x (np.ndarray): The input tabular data.

        Returns:
            np.ndarray: The output tabular noise.
        """
        if self._noise == "uniform":
            noise = torch.rand(self._size, device=x.device)
            return self._minimum + noise * (self._maximum - self._minimum)

        elif self._noise == "gaussian":
            return torch.randn(self._size, device=x.device) * self._std + self._mean
