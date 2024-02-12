from typing import Callable, Any, Tuple, Union

import torch
from pytorch_lightning import LightningModule

import matplotlib.pyplot as plt
import numpy as np
import os
from yamle.data.datamodule import BaseDataModule
from yamle.third_party.nyuv2 import NYUv2
from yamle.data.transforms import (
    JointResize,
    JointNormalize,
    JointToTensor,
    TargetToUnit,
)
from yamle.utils.file_utils import plots_file
from yamle.utils.operation_utils import regression_uncertainty_decomposition
from yamle.defaults import (
    DEPTH_ESTIMATION_KEY,
    PREDICTION_KEY,
    TRAIN_KEY,
    VALIDATION_KEY,
    TEST_KEY,
    AVERAGE_WEIGHTS_KEY,
)


class DepthEstimationDataModule(BaseDataModule):
    """Data module for depth estimation.

    Args:
        dataset (str): Name of the torchvision dataset. Currently supported are `nyudepthv2`.
    """

    mean = None
    std = None
    inputs_dim = None
    outputs_dim = None
    task = DEPTH_ESTIMATION_KEY
    inputs_dtype = torch.float32
    outputs_dtype = torch.float32

    def __init__(self, dataset: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if dataset not in ["nyudepthv2"]:
            raise ValueError("Dataset not supported.")

        self.available_test_augmentations = []
        self.available_transforms += [
            "jointtotensor",
            "jointresize",
            "jointnormalize",
            "targettometers",
        ]

        self._dataset = dataset
        if self._train_joint_transform is None:
            self._train_joint_transform = [
                "jointtotensor",
                "jointresize",
                "jointnormalize",
            ]
        if self._test_joint_transform is None:
            self._test_joint_transform = [
                "jointtotensor",
                "jointresize",
                "jointnormalize",
            ]

    def get_transform(self, name: str) -> Callable:
        """This is a helper function to get the transform by name."""
        transform = super().get_transform(name)
        if transform is not None:
            return transform

        if name == "jointtotensor":
            return JointToTensor(
                img_dtype=self.inputs_dtype, target_dtype=self.outputs_dtype
            )
        elif name == "jointresize":
            assert (
                self.inputs_dim is not None
            ), f"inputs_dim is not set for {self._dataset}"
            return JointResize(self.inputs_dim[2], self.inputs_dim[1])
        elif name == "jointnormalize":
            return JointNormalize(self.mean, self.std)
        elif name == "targettometers":
            return TargetToUnit(scale=10000.0)

    def _denormalize(self, image: torch.Tensor) -> torch.Tensor:
        """Denormalize the image."""
        mean = torch.tensor(self.mean)
        std = torch.tensor(self.std)
        return image * std[:, None, None] + mean[:, None, None]

    def _get_prediction(
        self,
        tester: LightningModule,
        x: torch.Tensor,
        y: Union[torch.Tensor, int],
        phase: str = TRAIN_KEY,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the prediction, input and target for the given input and target."""
        super()._get_prediction(tester, x, y, phase)
        x = x.to(tester.device).unsqueeze(0)
        y = y.to(tester.device).unsqueeze(0)
        if phase == TRAIN_KEY:
            output = tester.training_step([x, y], batch_idx=0)
        elif phase == VALIDATION_KEY:
            output = tester.validation_step([x, y], batch_idx=0)
        elif phase == TEST_KEY:
            output = tester.test_step([x, y], batch_idx=0)

        y_hat = output[PREDICTION_KEY]
        average_weights = (
            output[AVERAGE_WEIGHTS_KEY].cpu() if AVERAGE_WEIGHTS_KEY in output else None
        )
        (
            mean,
            predictive_variance,
            aleatoric_variance,
            epistemic_variance,
        ) = regression_uncertainty_decomposition(y_hat, weights=average_weights)
        return (
            mean.cpu().squeeze(0).squeeze(0),
            predictive_variance.cpu().squeeze(0),
            aleatoric_variance.cpu().squeeze(0),
            epistemic_variance.cpu().squeeze(0),
        )

    @torch.no_grad()
    def plot(
        self, tester: LightningModule, save_path: str, specific_name: str = ""
    ) -> None:
        """Plot random samples from the training and validation set to check if the data is correctly predicted"""

    def prepare_data(self) -> None:
        """Download and prepare the data, the data is stored in `self._train_dataset`, `self._validation_dataset` and `self._test_dataset`."""
        super().prepare_data()
        if self._dataset == "nyudepthv2":
            self._train_dataset = NYUv2(
                os.path.join(self._data_dir, "nyuv2"),
                train=True,
                download=True,
                task="depth",
            )
            self._test_dataset = NYUv2(
                os.path.join(self._data_dir, "nyuv2"),
                train=False,
                download=True,
                task="depth",
            )
        else:
            raise ValueError("Dataset not supported.")


class NYUv2DataModule(DepthEstimationDataModule):
    """Data module for NYUv2."""

    inputs_dim = (3, 320, 240)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    outputs_dim = 2
    targets_dim = (outputs_dim, 320, 240)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset="nyudepthv2", *args, **kwargs)

        if self._train_target_transform is None:
            self._train_target_transform = ["targettometers"]

        if self._test_target_transform is None:
            self._test_target_transform = ["targettometers"]

    @torch.no_grad()
    def plot(
        self, tester: LightningModule, save_path: str, specific_name: str = ""
    ) -> None:
        """Plot random samples from the training and validation set to check if the data is correctly predicted"""
        fig, axs = plt.subplots(2, 6, figsize=(20, 10))
        idx = np.random.randint(0, len(self._train_dataset))
        x, y = self._train_dataset[idx]
        predictions, total, aleatoric, epistemic = self._get_prediction(
            tester, x, y, TEST_KEY
        )
        axs[0, 0].imshow(self._denormalize(x).permute(1, 2, 0))
        axs[0, 0].set_title("Input")
        axs[0, 0].axis("off")
        axs[0, 1].imshow(y.squeeze(0), cmap="jet")
        axs[0, 1].set_title("Target")
        axs[0, 1].axis("off")
        axs[0, 2].imshow(predictions, cmap="jet")
        axs[0, 2].set_title("Prediction")
        axs[0, 2].axis("off")
        axs[0, 3].imshow(total)
        axs[0, 3].set_title("Total uncertainty")
        axs[0, 3].axis("off")
        axs[0, 4].imshow(aleatoric)
        axs[0, 4].set_title("Aleatoric uncertainty")
        axs[0, 4].axis("off")
        axs[0, 5].imshow(epistemic)
        axs[0, 5].set_title("Epistemic uncertainty")
        axs[0, 5].axis("off")

        if self._validation_dataset is not None:
            idx = np.random.randint(0, len(self._validation_dataset))
            x, y = self._validation_dataset[idx]
            predictions, total, aleatoric, epistemic = self._get_prediction(
                tester, x, y
            )

            axs[1, 0].imshow(self._denormalize(x).permute(1, 2, 0))
            axs[1, 0].set_title("Input")
            axs[1, 0].axis("off")
            axs[1, 1].imshow(y.squeeze(0), cmap="jet")
            axs[1, 1].set_title("Target")
            axs[1, 1].axis("off")
            axs[1, 2].imshow(predictions, cmap="jet")
            axs[1, 2].set_title("Prediction")
            axs[1, 2].axis("off")
            axs[1, 3].imshow(total)
            axs[1, 3].set_title("Total uncertainty")
            axs[1, 3].axis("off")
            axs[1, 4].imshow(aleatoric)
            axs[1, 4].set_title("Aleatoric uncertainty")
            axs[1, 4].axis("off")
            axs[1, 5].imshow(epistemic)
            axs[1, 5].set_title("Epistemic uncertainty")
            axs[1, 5].axis("off")

        plt.tight_layout()
        plt.savefig(plots_file(save_path, specific_name), bbox_inches="tight")
        plt.close(fig)
