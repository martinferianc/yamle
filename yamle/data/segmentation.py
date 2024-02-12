from typing import Callable, Any, Tuple, Union

import torch
from torchvision.datasets import Cityscapes
from pytorch_lightning import LightningModule

import matplotlib.pyplot as plt
import numpy as np
import os
from yamle.data.datamodule import BaseDataModule
from yamle.data.transforms import (
    JointToTensor,
    JointResize,
    JointTargetSqueeze,
    JointNormalize,
)
from yamle.utils.file_utils import plots_file
from yamle.utils.operation_utils import classification_uncertainty_decomposition
from yamle.defaults import (
    SEGMENTATION_KEY,
    MEAN_PREDICTION_KEY,
    PREDICTION_KEY,
    TRAIN_KEY,
    VALIDATION_KEY,
    TEST_KEY,
    INPUT_KEY,
    TARGET_KEY,
    AVERAGE_WEIGHTS_KEY,
)


class TorchvisionSegmentationDataModule(BaseDataModule):
    """Data module for the torchvision segmentation datasets.

    Args:
        dataset (str): Name of the torchvision dataset. Currently supported are `cityscapes`.
        seed (int): Seed for the random number generator.
        data_dir (str): Path to the data directory.
        train_tranform (Callable): Transformations to apply to the training data. Default: `transforms.ToTensor(), transforms.Normalize(mean, str)`.
        test_transform (Callable): Transformations to apply to the test data. Default: `transforms.ToTensor(), transforms.Normalize(mean, str)`.
    """

    mean = None
    std = None
    inputs_dim = None
    outputs_dim = None
    task = SEGMENTATION_KEY
    inputs_dtype = torch.float32
    outputs_dtype = torch.long

    def __init__(self, dataset: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if dataset not in ["cityscapes"]:
            raise ValueError("Dataset not supported.")
        self._dataset = dataset
        
        self.available_test_augmentations = []
        self.available_transforms += [
            "jointtotensor",
            "jointresize",
            "jointtargetsqueeze",
            "jointnormalize",
        ]
        
        if self._train_joint_transform is None:
            self._train_joint_transform = [
                "jointtotensor",
                "jointresize",
                "jointtargetsqueeze",
                "jointnormalize",
            ]
        if self._test_joint_transform is None:
            self._test_joint_transform = [
                "jointtotensor",
                "jointresize",
                "jointtargetsqueeze",
                "jointnormalize",
            ]

    def get_transform(self, name: str) -> Callable:
        """This is a helper function to get the transform by name."""
        transform = super().get_transform(name)
        if transform is not None:
            return transform
        
        if name == "jointtotensor":
            return JointToTensor()
        elif name == "jointresize":
            assert (
                self.inputs_dim is not None
            ), f"inputs_dim is not set for {self._dataset}"
            return JointResize(self.inputs_dim[2], self.inputs_dim[1])
        elif name == "jointtargetsqueeze":
            return JointTargetSqueeze(0)
        elif name == "jointnormalize":
            return JointNormalize(self.mean, self.std)

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
        """Returns the prediction of the model."""
        super()._get_prediction(tester, x, y, phase)
        x = x.to(tester.device).unsqueeze(0)
        y = y.to(tester.device).unsqueeze(0)
        if phase == TRAIN_KEY:
            output = tester.training_step([x, y], batch_idx=0)
        elif phase == VALIDATION_KEY:
            output = tester.validation_step([x, y], batch_idx=0)
        elif phase == TEST_KEY:
            output = tester.test_step([x, y], batch_idx=0)
        x = output[INPUT_KEY].cpu()
        y = output[TARGET_KEY].cpu()
        y_hat = output[PREDICTION_KEY]
        y_hat_mean = output[MEAN_PREDICTION_KEY]
        averaging_weights = output[AVERAGE_WEIGHTS_KEY].cpu() if AVERAGE_WEIGHTS_KEY in output else None
        labels = y_hat_mean.argmax(dim=1).cpu()
        total, aleatoric, epistemic = classification_uncertainty_decomposition(y_hat, probabilities=True, weights=averaging_weights)
        return (
            labels.cpu().squeeze(0),
            total.cpu().squeeze(0),
            aleatoric.cpu().squeeze(0),
            epistemic.cpu().squeeze(0),
        )

    @torch.no_grad()
    def plot(
        self, tester: LightningModule, save_path: str, specific_name: str = ""
    ) -> None:
        """Plot random samples from the training and validation set to check if the data is correctly predicted"""

    def prepare_data(self) -> None:
        """Download and prepare the data, the data is stored in `self._train_dataset`, `self._validation_dataset` and `self._test_dataset`."""
        super().prepare_data()
        if self._dataset == "cityscapes":
            self._train_dataset = Cityscapes(
                os.path.join(self._data_dir, "cityscapes"),
                split="train",
                mode="fine",
                target_type="semantic",
            )
            self._validation_dataset = Cityscapes(
                os.path.join(self._data_dir, "cityscapes"),
                split="val",
                mode="fine",
                target_type="semantic",
            )
            self._test_dataset = Cityscapes(
                os.path.join(self._data_dir, "cityscapes"),
                split="test",
                mode="fine",
                target_type="semantic",
            )
        else:
            raise ValueError("Dataset not supported.")


class TorchvisionSegmentationDataModuleCityscapes(TorchvisionSegmentationDataModule):
    """Data module for the Cityscapes dataset."""

    inputs_dim = (3, 512, 256)
    mean = [0.28689554, 0.32513303, 0.28389177]
    std = [0.18696375, 0.19017339, 0.18720214]
    ignore_indices = [i for i, c in enumerate(Cityscapes.classes) if c.ignore_in_eval]
    outputs_dim = len(Cityscapes.classes)
    targets_dim = (outputs_dim, 512, 256)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("cityscapes", **kwargs)

    def _decode_target_to_rgb(self, target: torch.Tensor) -> np.ndarray:
        """Decode the integer target to RGB.

        The target is a tensor of shape `(num_classes, height, width)` with integer values."""
        r = torch.zeros_like(target)
        g = torch.zeros_like(target)
        b = torch.zeros_like(target)
        for c in range(self.outputs_dim):
            r[target == c] = Cityscapes.classes[c].color[0]
            g[target == c] = Cityscapes.classes[c].color[1]
            b[target == c] = Cityscapes.classes[c].color[2]
        rgb = np.zeros((target.shape[0], target.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    @torch.no_grad()
    def plot(
        self, tester: LightningModule, save_path: str, specific_name: str = ""
    ) -> None:
        """Plot random samples from the training and validation set to check if the data is correctly predicted"""
        fig, axs = plt.subplots(2, 6, figsize=(20, 10))
        idx = np.random.randint(0, len(self._train_dataset))
        x, y = self._train_dataset[idx]
        labels, total, aleatoric, epistemic = self._get_prediction(
            tester, x, y, TRAIN_KEY
        )

        axs[0, 0].imshow(self._denormalize(x).permute(1, 2, 0))
        axs[0, 0].set_title("Input")
        axs[0, 0].axis("off")
        axs[0, 1].imshow(self._decode_target_to_rgb(y))
        axs[0, 1].set_title("Target")
        axs[0, 1].axis("off")
        axs[0, 2].imshow(self._decode_target_to_rgb(labels))
        axs[0, 2].set_title("Prediction")
        axs[0, 2].axis("off")
        axs[0, 3].imshow(total, cmap="jet")
        axs[0, 3].set_title("Total uncertainty")
        axs[0, 3].axis("off")
        axs[0, 4].imshow(aleatoric, cmap="jet")
        axs[0, 4].set_title("Aleatoric uncertainty")
        axs[0, 4].axis("off")
        axs[0, 5].imshow(epistemic, cmap="jet")
        axs[0, 5].set_title("Epistemic uncertainty")
        axs[0, 5].axis("off")

        if self._validation_dataset is not None:
            idx = np.random.randint(0, len(self._validation_dataset))
            x, y = self._validation_dataset[idx]
            labels, total, aleatoric, epistemic = self._get_prediction(
                tester, x, y, VALIDATION_KEY
            )

            axs[1, 0].imshow(self._denormalize(x).permute(1, 2, 0))
            axs[1, 0].set_title("Input")
            axs[1, 0].axis("off")
            axs[1, 1].imshow(self._decode_target_to_rgb(y))
            axs[1, 1].set_title("Target")
            axs[1, 1].axis("off")
            axs[1, 2].imshow(self._decode_target_to_rgb(labels))
            axs[1, 2].set_title("Prediction")
            axs[1, 2].axis("off")
            axs[1, 3].imshow(total, cmap="jet")
            axs[1, 3].set_title("Total uncertainty")
            axs[1, 3].axis("off")
            axs[1, 4].imshow(aleatoric, cmap="jet")
            axs[1, 4].set_title("Aleatoric uncertainty")
            axs[1, 4].axis("off")
            axs[1, 5].imshow(epistemic, cmap="jet")
            axs[1, 5].set_title("Epistemic uncertainty")
            axs[1, 5].axis("off")

        plt.tight_layout()
        plt.savefig(plots_file(save_path, specific_name), bbox_inches="tight")
        plt.close(fig)
