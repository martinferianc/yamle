from yamle.defaults import (
    LOSS_KEY,
    CLASSIFICATION_KEY,
    TRAIN_KEY,
    TARGET_KEY,
    PREDICTION_KEY,
    MEAN_PREDICTION_KEY,
    INPUT_KEY,
)
from yamle.losses.classification import CrossEntropyLoss, one_hot
from yamle.data.augmentations import CutMix, CutOut, MixUp, RandomErasing
from yamle.utils.operation_utils import average_predictions
from yamle.methods.method import BaseMethod
from typing import List, Dict, Any, Optional, Callable, Tuple
import torch
import argparse
import logging

logging = logging.getLogger("pytorch_lightning")


class AugmentationImageClassificationMethod(BaseMethod):
    """This is the base class for image classification methods with augmentations."""

    tasks = [CLASSIFICATION_KEY]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(
            self._loss, CrossEntropyLoss
        ), f"Loss must be CrossEntropyLoss, got {self._loss}"
        assert (
            self._loss._one_hot_target
        ), f"CrossEntropyLoss must have one_hot_target=True, got {self._loss._one_hot_target}"
        self._augmentation: Callable = None

    def _step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
        phase: str = TRAIN_KEY,
    ) -> Dict[str, Any]:
        """This method is used to perform a single training or validation step.

        The data is split into inputs and targets and the forward pass is performed.
        The predictions have the shame `(batch_size, num_members=1, num_outputs)` shape.
        An average of the predictions is also computed across the ensemble members.

        Args:
            batch (List[torch.Tensor]): The batch of data.
            batch_idx (int): The index of the batch.
        """
        x, y = batch
        y_one_hot = one_hot(y, self._outputs_dim)
        if self.training:
            x, y_one_hot = self._augmentation(x, y_one_hot, self.model)
            # Need to convert back to labels - the augmentation reshuflles the labels
            y = torch.argmax(y_one_hot, dim=1)
        y_hat = self._predict(x)
        loss = self._loss(y_hat, y_one_hot)
        y_hat_mean = average_predictions(y_hat, self._task)
        outputs = {}

        outputs[LOSS_KEY] = loss
        outputs[TARGET_KEY] = y.detach()
        outputs[INPUT_KEY] = x.detach()
        outputs[PREDICTION_KEY] = y_hat.detach()
        outputs[MEAN_PREDICTION_KEY] = y_hat_mean.detach()
        return outputs


class CutOutImageClassificationMethod(AugmentationImageClassificationMethod):
    """This is the base class for image classification methods with cutout.

    Args:
        batch_proportion (float): The proportion of the batch to cut out.
        cutout_size (int): The size of the cutout.
    """

    def __init__(
        self,
        batch_proportion: float = 0.5,
        cutout_size: int = 16,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._augmentation = CutOut(batch_proportion=batch_proportion, size=cutout_size)

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Add arguments for CutOutImageClassificationMethod."""
        parser = super(
            CutOutImageClassificationMethod, CutOutImageClassificationMethod
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_batch_proportion",
            type=float,
            default=0.5,
            help="The proportion of the batch to cut out.",
        )
        parser.add_argument(
            "--method_cutout_size",
            type=int,
            default=16,
            help="The size of the cutout.",
        )
        return parser


class MixUpImageClassificationMethod(AugmentationImageClassificationMethod):
    """This is the base class for image classification methods with mixup.

    Args:
        batch_proportion (float): The proportion of the batch to mixup.
        mixup_alpha (float): The alpha parameter for the beta distribution.
    """

    def __init__(
        self,
        batch_proportion: float = 0.5,
        mixup_alpha: float = 0.4,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._augmentation = MixUp(batch_proportion=batch_proportion, alpha=mixup_alpha)

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Add arguments for MixUpImageClassificationMethod."""
        parser = super(
            MixUpImageClassificationMethod, MixUpImageClassificationMethod
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_batch_proportion",
            type=float,
            default=0.5,
            help="The proportion of the batch to mixup.",
        )
        parser.add_argument(
            "--method_mixup_alpha",
            type=float,
            default=0.4,
            help="The alpha parameter for the beta distribution.",
        )
        return parser


class CutMixImageClassificationMethod(AugmentationImageClassificationMethod):
    """This is the base class for image classification methods with cutmix.

    Args:
        batch_proportion (float): The proportion of the batch to cutmix.
        cutmix_alpha (float): The alpha parameter for the beta distribution.
    """

    def __init__(
        self,
        batch_proportion: float = 0.5,
        cutmix_alpha: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._augmentation = CutMix(
            batch_proportion=batch_proportion, alpha=cutmix_alpha
        )

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Add arguments for CutMixImageClassificationMethod."""
        parser = super(
            CutMixImageClassificationMethod, CutMixImageClassificationMethod
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_batch_proportion",
            type=float,
            default=0.5,
            help="The proportion of the batch to cutmix.",
        )
        parser.add_argument(
            "--method_cutmix_alpha",
            type=float,
            default=1.0,
            help="The alpha parameter for the beta distribution.",
        )
        return parser


class RandomErasingImageClassificationMethod(AugmentationImageClassificationMethod):
    """This is the base class for image classification methods with random erasing.

    Args:
        batch_proportion (float): The proportion of the batch to randomly erase.
        random_erasing_scale (Tuple[float, float]): The range of the random erasing scale.
        random_erasing_ratio (Tuple[float, float]): The range of the random erasing ratio.
        random_erasing_value (float): The value to fill the erased area with.
    """

    def __init__(
        self,
        batch_proportion: float = 0.5,
        random_erasing_scale: Tuple[float, float] = (0.02, 0.33),
        random_erasing_ratio: Tuple[float, float] = (0.3, 3.3),
        random_erasing_value: float = 0.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._augmentation = RandomErasing(
            batch_proportion=batch_proportion,
            scale=random_erasing_scale,
            ratio=random_erasing_ratio,
            value=random_erasing_value,
        )

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Add arguments for RandomErasingImageClassificationMethod."""
        parser = super(
            RandomErasingImageClassificationMethod,
            RandomErasingImageClassificationMethod,
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_batch_proportion",
            type=float,
            default=0.5,
            help="The proportion of the batch to randomly erase.",
        )
        parser.add_argument(
            "--method_random_erasing_scale",
            type=float,
            nargs=2,
            default=[0.02, 0.33],
            help="The range of the random erasing scale.",
        )
        parser.add_argument(
            "--method_random_erasing_ratio",
            type=float,
            nargs=2,
            default=[0.3, 3.3],
            help="The range of the random erasing ratio.",
        )
        parser.add_argument(
            "--method_random_erasing_value",
            type=float,
            default=0.0,
            help="The value to fill the erased area with.",
        )
        return parser
