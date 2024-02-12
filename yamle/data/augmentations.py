from abc import ABC, abstractmethod
from typing import Tuple, Any, Union
import torch
import torch.nn as nn
from torchvision.transforms import RandomErasing as TorchRandomErasing


class Augmentation(ABC):
    """This is a base class for all augmentations. It is an abstract class and
    should not be used directly. Instead, use one of the concrete subclasses.

    Args:
        batch_proportion (float): The proportion of the batch to augment.
    """

    def __init__(self, batch_proportion: float) -> None:
        assert (
            0.0 <= batch_proportion <= 1.0
        ), f"batch_proportion must be in [0,1], but was {batch_proportion}"
        self._batch_proportion = batch_proportion

    @abstractmethod
    def augment(
        self, inputs: torch.Tensor, targets: torch.Tensor, model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "The `augment` method must be implemented by subclasses."
        )

    def __call__(
        self, inputs: torch.Tensor, targets: torch.Tensor, model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the augmentation to the inputs and returns the augmented inputs.

        The augmentation is only applied to a proportion of the batch, as specified
        by the `batch_proportion` argument to the constructor.

        Args:
            inputs (torch.Tensor): The inputs to augment.
            targets (torch.Tensor): The targets to augment.
            model (nn.Module): The model to use for the augmentation.

        Returns:
            torch.Tensor: The augmented inputs.
        """
        indices = torch.tensor(range(inputs.shape[0]))
        input_shape = inputs.shape
        targets_shape = targets.shape
        # Select a random subset of the batch to augment.
        augment_indices = indices[: int(self._batch_proportion * inputs.shape[0])]
        non_augment_indices = indices[int(self._batch_proportion * inputs.shape[0]) :]
        inputs_augmented, targets_augmented = self.augment(
            inputs[augment_indices], targets[augment_indices], model
        )
        non_augmented_inputs = inputs[non_augment_indices]
        non_augmented_targets = targets[non_augment_indices]
        inputs = torch.cat([inputs_augmented, non_augmented_inputs], dim=0)
        targets = torch.cat([targets_augmented, non_augmented_targets], dim=0)
        assert (
            inputs.shape == input_shape
        ), f"Expected inputs to have shape {input_shape}, but got {inputs.shape}"
        assert (
            targets.shape == targets_shape
        ), f"Expected targets to have shape {targets_shape}, but got {targets.shape}"
        return inputs, targets


class MixUp(Augmentation):
    """This is an implementation of the MixUp augmentation, as described in
    https://arxiv.org/abs/1710.09412.

    Args:
        alpha (float): The alpha parameter for the beta distribution used to
            sample the mixup weights.
    """

    def __init__(self, alpha: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert alpha > 0.0, f"alpha must be positive, but was {alpha}"
        self._alpha = alpha
        self._distribution = torch.distributions.beta.Beta(alpha, alpha)

    def augment(
        self, inputs: torch.Tensor, targets: torch.Tensor, model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the MixUp augmentation to the inputs and returns the augmented inputs.

        The targets are assumed to be one-hot encoded.

        It is applied such that:

        l ~ Beta(alpha, alpha)
        x' = l * x + (1 - l) * x'
        y' = l * y + (1 - l) * y'
        where x' and y' are the randomly shuffled inputs and targets, respectively.
        """
        assert (
            len(targets.shape) >= 2
        ), f"Expected targets to be one-hot encoded, but got shape {targets.shape}"
        indices = torch.randperm(inputs.shape[0]).to(inputs.device)
        inputs_shuffled = inputs[indices]
        targets_shuffled = targets[indices]

        mixup_weights = self._distribution.sample((inputs.shape[0],)).to(inputs.device)

        mixup_weights_input = mixup_weights.clone()
        mixup_weights_target = mixup_weights.clone()
        while len(mixup_weights_input.shape) < len(inputs.shape):
            mixup_weights_input = mixup_weights_input.unsqueeze(-1)
        while len(mixup_weights_target.shape) < len(targets.shape):
            mixup_weights_target = mixup_weights_target.unsqueeze(-1)
        inputs = (
            mixup_weights_input * inputs + (1 - mixup_weights_input) * inputs_shuffled
        )
        targets = (
            mixup_weights_target * targets
            + (1 - mixup_weights_target) * targets_shuffled
        )
        return inputs, targets


class CutOut(Augmentation):
    """This is an implementation of the CutOut augmentation, as described in
    https://arxiv.org/abs/1708.04552.

    Args:
        batch_proportion (float): The proportion of the batch to augment.
        size (int): The size of the square region to cut out.
    """

    def __init__(self, batch_proportion: float, size: int) -> None:
        super().__init__(batch_proportion)
        assert size > 0, f"size must be a positive integer, but was {size}"
        self._size = size

    def augment(
        self, inputs: torch.Tensor, targets: torch.Tensor, model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the CutOut augmentation to the inputs and returns the augmented inputs.

        The targets are not modified.

        It is applied such that a square region of size `size` is randomly cut out
        from each input image.

        Args:
            inputs (torch.Tensor): The inputs to augment.
            targets (torch.Tensor): The targets (labels) associated with the inputs. Not modified.
            model (nn.Module): The model to use for the augmentation (not used here). Not modified.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The augmented inputs and unchanged targets.
        """
        assert (
            len(inputs.shape) == 4
        ), f"Expected inputs to have shape (batch_size, channels, height, width), but got {inputs.shape}"
        assert (
            inputs.shape[0] == targets.shape[0]
        ), "Number of inputs and targets must match"
        _, _, height, width = inputs.shape

        x = torch.randint(0, width, (1,)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        y = torch.randint(0, height, (1,)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mask = torch.ones_like(inputs)
        mask[:, :, y : y + self._size, x : x + self._size] = 0.0
        augmented_inputs = inputs * mask

        return augmented_inputs, targets


class CutMix(Augmentation):
    """This is an implementation of CutMix augmentation, as described in the paper: https://arxiv.org/abs/1905.04899.

    Args:
        batch_proportion (float): The proportion of the batch to augment.
        alpha (float): The alpha parameter for the distribution used to sample the combination ratio.
    """

    def __init__(self, batch_proportion: float, alpha: float) -> None:
        super().__init__(batch_proportion)
        self._alpha = alpha
        self._distribution = torch.distributions.beta.Beta(alpha, alpha)

    def augment(
        self, inputs: torch.Tensor, targets: torch.Tensor, model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the CutMix augmentation to the inputs and returns the augmented inputs and targets.

        The targets are assumed to be one-hot encoded.

        It is applied such that:
        - Two random inputs are selected for combining.
        - A random box is cut out from one input and replaced with the corresponding box from another input.
        - The targets are adjusted to match the pixel ratio.

        Args:
            inputs (torch.Tensor): The inputs to augment.
            targets (torch.Tensor): The targets to augment.
            model (nn.Module): The model to use for the augmentation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The augmented inputs and targets.
        """
        assert (
            len(targets.shape) >= 2
        ), f"Expected targets to be one-hot encoded, but got shape {targets.shape}"
        indices = torch.randperm(inputs.shape[0]).to(inputs.device)
        inputs_shuffled = inputs[indices]
        targets_shuffled = targets[indices]

        lam = self._distribution.sample((1,)).to(inputs.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.shape, lam)

        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs_shuffled[:, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - (
            (bbx2 - bbx1) * (bby2 - bby1) / (inputs.shape[-1] * inputs.shape[-2])
        )

        mixed_targets = lam[:, None] * targets + (1 - lam[:, None]) * targets_shuffled

        return inputs, mixed_targets


class RandomErasing(Augmentation):
    """This is a wrapper class for the RandomErasing augmentation provided by torchvision.

    Args:
        batch_proportion (float): The proportion of the batch to augment.
        scale (Tuple[float, float]): The range of the proportion of erased area against the input image.
        ratio (Tuple[float, float]): The range of the aspect ratio of the erased area.
        value (int or tuple or str): The erasing value. Default is 0.
            - If an integer, it is used to erase all pixels.
            - If a tuple of length 3, it is used to erase R, G, B channels respectively.
            - If a string 'random', each pixel is erased with random values.
    """

    def __init__(
        self,
        batch_proportion: float,
        scale: Tuple[float, float],
        ratio: Tuple[float, float],
        value: Union[int, Tuple[int, int, int], str] = 0,
    ) -> None:
        super().__init__(batch_proportion)
        self._transform = TorchRandomErasing(
            p=1.0, scale=scale, ratio=ratio, value=value, inplace=False
        )

    def augment(
        self, inputs: torch.Tensor, targets: torch.Tensor, model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the Random Erasing augmentation to the inputs and returns the augmented inputs and targets.

        The targets are unchanged.

        Args:
            inputs (torch.Tensor): The inputs to augment.
            targets (torch.Tensor): The targets to augment.
            model (nn.Module): The model to use for the augmentation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The augmented inputs and unchanged targets.
        """
        assert (
            len(targets.shape) >= 2
        ), f"Expected targets to be one-hot encoded, but got shape {targets.shape}"
        augmented_inputs = self._transform(inputs)
        return augmented_inputs, targets


def rand_bbox(
    size: Tuple[int, int, int, int], lam: torch.Tensor
) -> Tuple[int, int, int, int]:
    """Generates a random bounding box coordinates for CutMix.

    Args:
        size (Tuple[int, int, int, int]): The shape of the input tensor.
        lam (torch.Tensor): The combination ratio.

    Returns:
        Tuple[int, int, int, int]: The bounding box coordinates (x1, y1, x2, y2).
    """
    width = size[3]
    height = size[2]
    cut_ratio = torch.sqrt(1.0 - lam)
    cut_w = (width * cut_ratio).int()
    cut_h = (height * cut_ratio).int()

    cx = torch.randint(0, width, size=(1,), device=lam.device)
    cy = torch.randint(0, height, size=(1,), device=lam.device)
    bbx1 = torch.clamp(cx - cut_w // 2, 0, width)
    bby1 = torch.clamp(cy - cut_h // 2, 0, height)
    bbx2 = torch.clamp(cx + cut_w // 2, 0, width)
    bby2 = torch.clamp(cy + cut_h // 2, 0, height)

    return bbx1, bby1, bbx2, bby2
