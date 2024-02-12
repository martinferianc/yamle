from abc import ABC, abstractmethod
from typing import Any, Optional, Union, List

import torch
import argparse

from yamle.defaults import SUPPORTED_TASKS


class BaseLoss(ABC):
    """This is the base class for all losses.

    Args:
        reduction_per_member (str): The reduction to be used for the loss. Given that the input has the shape
            (batch_size, num_members, ...), the loss will be reduced over the `num_members` dimension.
        reduction_per_sample (str): The reduction to be used for the loss. Given that the input has the shape
            (batch_size, num_members, ...), the loss will be reduced over the `batch_size` dimension.
        reduction_per_feature (str): The reduction to be used for the loss. Given that the input has the shape
            (batch_size, num_members, num_features, ...), the loss will be reduced over the `num_features` dimension.
    """
    
    tasks = SUPPORTED_TASKS

    def __init__(
        self,
        reduction_per_member: str = "mean",
        reduction_per_sample: str = "mean",
        reduction_per_feature: str = "none",
        task: str = None,
    ) -> None:
        self.set_reduction_per_member(reduction_per_member)
        self.set_reduction_per_sample(reduction_per_sample)
        self.set_reduction_per_feature(reduction_per_feature)
        
        assert task in self.tasks, f"Unknown task {task} for this loss. Must be one of {self.tasks}."

    @abstractmethod
    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """This method is used to compute the loss.

        Args:
            y_hat (torch.Tensor): The predictions.
            y (torch.Tensor): The ground truth.
        """
        raise NotImplementedError("This method is not implemented.")

    def _process_sample_loss(
        self,
        sample_loss: torch.Tensor,
        member: int,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """This method is used to process the loss per sample in the batch."""
        if weights is not None:
            if weights.dim() == 1:
                sample_loss = sample_loss * weights
            elif weights.dim() == 2:
                sample_loss = sample_loss * weights[:, member]
            else:
                raise ValueError(
                    f"Expected weights to be of dimension 1 or 2. Got {weights.dim()}."
                )
        if self._reduction_per_sample == "mean":
            return sample_loss.mean(dim=0)
        elif self._reduction_per_sample == "sum":
            return sample_loss.sum(dim=0)
        elif self._reduction_per_sample == "none":
            return sample_loss
        else:
            raise ValueError(
                f"Expected reduction per sample to be one of 'mean', 'sum' or 'none'. Got {self._reduction_per_sample}."
            )

    def _process_member_loss(
        self, member_loss: torch.Tensor, members: int
    ) -> torch.Tensor:
        """This method is used to process the loss per member."""
        if self._reduction_per_member == "mean":
            return member_loss / members
        elif self._reduction_per_member == "sum":
            return member_loss
        elif self._reduction_per_member == "none":
            return member_loss
        else:
            raise ValueError(
                f"Expected reduction per member to be one of 'mean', 'sum' or 'none'. Got {self._reduction_per_member}."
            )

    def _process_feature_loss(
        self, feature_loss: torch.Tensor, dim: Union[int, List[int]]
    ) -> torch.Tensor:
        """This method is used to process the loss per feature."""
        if self._reduction_per_feature == "mean":
            return torch.mean(feature_loss, dim=dim)
        elif self._reduction_per_feature == "sum":
            return torch.sum(feature_loss, dim=dim)
        elif self._reduction_per_feature == "none":
            return feature_loss
        else:
            raise ValueError(
                f"Expected reduction per feature to be one of 'mean', 'sum' or 'none'. Got {self._reduction_per_feature}."
            )

    def __repr__(self) -> str:
        return f"BaseLoss(reduction_per_member={self._reduction_per_member}, reduction_per_sample={self._reduction_per_sample}, reduction_per_feature={self._reduction_per_feature})"

    def set_reduction_per_member(self, reduction_per_member: str) -> None:
        """This method is used to set the reduction per member."""
        assert reduction_per_member in [
            "mean",
            "sum",
            "none",
        ], f"Expected reduction per member to be one of 'mean', 'sum' or 'none'. Got {reduction_per_member}."
        self._reduction_per_member = reduction_per_member

    def set_reduction_per_sample(self, reduction_per_sample: str) -> None:
        """This method is used to set the reduction per sample."""
        assert reduction_per_sample in [
            "mean",
            "sum",
            "none",
        ], f"Expected reduction per sample to be one of 'mean', 'sum' or 'none'. Got {reduction_per_sample}."
        self._reduction_per_sample = reduction_per_sample

    def set_reduction_per_feature(self, reduction_per_feature: str) -> None:
        """This method is used to set the reduction per feature."""
        assert reduction_per_feature in [
            "mean",
            "sum",
            "none",
        ], f"Expected reduction per feature to be one of 'mean', 'sum' or 'none'. Got {reduction_per_feature}."
        self._reduction_per_feature = reduction_per_feature

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the loss specific arguments to the parent parser."""
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--loss_reduction_per_member",
            type=str,
            default="sum",
            help="The reduction to be used for the loss. Given that the input has the shape (batch_size, num_members, ...), the loss will be reduced over the `num_members` dimension.",
        )
        parser.add_argument(
            "--loss_reduction_per_sample",
            type=str,
            default="mean",
            help="The reduction to be used for the loss. Given that the input has the shape (batch_size, num_members, ...), the loss will be reduced over the `batch_size` dimension.",
        )
        parser.add_argument(
            "--loss_reduction_per_feature",
            type=str,
            default="none",
            help="The reduction to be used for the loss. Given that the input has the shape (batch_size, num_members, ...), the loss will be reduced over the `...` dimension.",
        )
        return parser


class DummyLoss(BaseLoss):
    """This is a dummy loss class which always returns 0.0 tensor."""

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """This method is used to compute the loss.

        In this case it will just return 0 and put it on some device.
        """
        # Iterate through all the arguments and if one of them is a tensor, get it's device
        device = "cpu"
        for arg in args:
            if isinstance(arg, torch.Tensor):
                device = arg.device
                break
        return torch.tensor(0.0).to(device)

    def __repr__(self) -> str:
        return f"DummyLoss()"
