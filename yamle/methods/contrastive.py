import argparse
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import Normalize, RandAugment

from yamle.data.transforms import Denormalize
from yamle.defaults import (
    INPUT_KEY,
    LOSS_KEY,
    MEAN_PREDICTION_KEY,
    PRE_TRAINING_KEY,
    PREDICTION_KEY,
    TARGET_KEY,
    TRAIN_KEY,
)
from yamle.methods.method import BaseMethod
from yamle.losses.contrastive import NoiseContrastiveEstimatorLoss


class SimCLRVisionMethod(BaseMethod):
    """This is the base class for contrastive pre-training of vision models.

    It uses RandAugment to perform random augmentations on the input images to create two views.

    Args:
        random_augment_num_ops (int): The number of random augmentations to perform. Defaults to 2.
        random_augment_magnitude (int): The magnitude of the random augmentations. Defaults to 10.
    """

    def __init__(
        self,
        random_augment_num_ops: int = 2,
        random_augment_magnitude: int = 10,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(
            self._loss, NoiseContrastiveEstimatorLoss
        ), f"Loss must be NoiseContrastiveEstimatorLoss, got {self._loss}"
        assert self._datamodule._train_transform == [
            "totensor",
            "normalize",
        ], f"Train transform must be ['totensor', 'normalize'], got {self._datamodule._train_transform}"
        assert hasattr(
            self._datamodule, "mean"
        ), f"DataModule must have attribute 'mean'"
        assert hasattr(self._datamodule, "std"), f"DataModule must have attribute 'std'"
        # There is no need for augmentations in the pre-training phase.
        self._datamodule.available_test_augmentations = []
        assert (
            self._plotting_testing is False or self._plotting_training == 0
        ), f"Plotting testing must be False or 0, got {self._plotting_testing}"
        assert (
            self._plotting_training is False or self._plotting_training == 0
        ), f"Plotting training must be False or 0, got {self._plotting_training}"
        self._task = PRE_TRAINING_KEY
        self._augmentation = RandAugment(
            random_augment_num_ops, random_augment_magnitude
        )
        self._denormalize = Denormalize(
            mean=self._datamodule.mean, std=self._datamodule.std
        )
        self._normalize = Normalize(
            mean=self._datamodule.mean, std=self._datamodule.std
        )

    def _create_metrics(self, metrics_kwargs: Dict[str, Any]) -> None:
        """This method is used to create the metrics for the method."""
        metrics_kwargs["task"] = PRE_TRAINING_KEY
        super()._create_metrics(metrics_kwargs)

    def _predict(self, x: torch.Tensor, **forward_kwargs: Any) -> torch.Tensor:
        """This method is used to perform a forward pass of the model.

        It is done with respect to the number of hidden layers or how hidden layers are being defined
        in the underlying `self.model`.
        """
        _, stages = self.model(x, staged_output=True, **forward_kwargs)
        # We only need the last layer stage, the rest are discarded
        return stages[-1]

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to augment the input data to create the positive samples.

        We need to 1. denormalize the data, 2. convert it to uint8, 3. augment it, convert it back to float32, 4. normalize it
        """
        x = (self._denormalize(x) * 255).to(torch.uint8).cpu()
        x = (self._augmentation(x).float() / 255.0).to(self.device)
        x = self._normalize(x)
        if self._debug:
            # If we are debugging, we want to plot some of the augmented images
            # We only plot the first 4 images
            fig = plt.figure(figsize=(10, 10))
            for i in range(4):
                ax = fig.add_subplot(2, 2, i + 1)
                # denormalize the image
                img = self._denormalize(x[i].cpu())
                ax.imshow(img.permute(1, 2, 0))
            plt.savefig(os.path.join(self._save_path, "augmented.png"))
            plt.close(fig)

        return x

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
        x = batch[0]
        # Check if the data is vision data
        assert len(x.shape) == 4, f"Input data must be vision data, got {x.shape}"
        # We need to 1. denormalize the data, 2. convert it to uint8, 3. augment it, convert it back to float32, 4. normalize it
        x_positive = self._augment(x)
        y_hat = self._predict(x).unsqueeze(1)
        y_hat_positive = self._predict(x_positive).unsqueeze(1)

        # Construct negative samples
        y_hat_negative = []
        for i in range(y_hat.shape[0]):
            y_hat_negative.append(
                torch.cat(
                    (
                        y_hat[:i],
                        y_hat[i + 1 :],
                        y_hat_positive[:i],
                        y_hat_positive[i + 1 :],
                    ),
                    dim=0,
                )
            )
        y_hat_negative = torch.stack(y_hat_negative, dim=2)
        # The shape is (K, num_members=1, batch_size, num_outputs)
        # Permute the negative samples such that the
        # shape is (batch_size, num_members=1, K, num_outputs)
        y_hat_negative = y_hat_negative.permute(2, 1, 0, 3)

        # Compute the loss for the positive and negative samples
        loss = self._loss(y_hat, y_hat_positive, y_hat_negative)
        # Swap the positive and negative samples
        loss += self._loss(y_hat_positive, y_hat, y_hat_negative)
        loss /= 2.0

        outputs = {}
        outputs[LOSS_KEY] = loss
        outputs[TARGET_KEY] = None
        outputs[INPUT_KEY] = x.detach()
        outputs[PREDICTION_KEY] = None
        outputs[MEAN_PREDICTION_KEY] = None
        return outputs

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add arguments specific to this method to the parser."""
        parser = super(SimCLRVisionMethod, SimCLRVisionMethod).add_specific_args(
            parent_parser
        )
        parser.add_argument(
            "--method_random_augment_num_ops",
            type=int,
            default=2,
            help="The number of random augmentations to perform. Defaults to 2.",
        )
        parser.add_argument(
            "--method_random_augment_magnitude",
            type=int,
            default=10,
            help="The magnitude of the random augmentations. Defaults to 10.",
        )
        return parser
