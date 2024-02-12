from yamle.methods.uncertain_method import MemberMethod
from yamle.models.specific.ensemble import Ensemble
from yamle.models.operations import output_activation
from yamle.losses.classification import one_hot
from typing import Any, Tuple, List, Dict
from yamle.defaults import (
    LOSS_KEY,
    TARGET_KEY,
    TARGET_PER_MEMBER_KEY,
    MEAN_PREDICTION_KEY,
    PREDICTION_KEY,
    PREDICTION_PER_MEMBER_KEY,
    CLASSIFICATION_KEY,
    REGRESSION_KEY,
    SEGMENTATION_KEY,
    MEMBERS_DIM,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging

logging = logging.getLogger("pytorch_lightning")


class EnsembleMethod(MemberMethod):
    """This is a Method class for the Ensemble model.

    It uses the `Ensemble` model to wrap around the original model and then uses the base method to train the members
    one by one in cooperation with the `EnsembleTrainer` class.

    Args:
        num_members (int): The number of members in the ensemble.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Wrap the model with the ensemble model.
        self.model = Ensemble(self.model, kwargs["num_members"])

    def _predict(self, x: torch.Tensor, **forward_kwargs: Any) -> torch.Tensor:
        """This method is used to perform a forward pass of the model.

        In validation it is done with respect to all models that have been trained.
        """
        if self.training:
            return super()._predict(x)
        else:
            outputs = []
            for i in range(self.model._num_members):
                outputs.append(self.model(x, current_member=i, **forward_kwargs))

            return torch.stack(outputs, dim=MEMBERS_DIM)

    def increment_current_member(self) -> None:
        """This method is used to increment the current member index."""
        self.model.increment_current_member()


class SnapsotEnsembleMethod(EnsembleMethod):
    """This is a Method class for the Snapshot Ensemble method.

    It uses the `Ensemble` model to wrap around the original model and then uses the base method to train the network
    via the cyclic learning rate scheduler. Each time the learning rate hits the minimum, the current model is saved
    as the next member, while the learning rate is reset to the maximum value and the main model is trained
    further.

    Args:
        num_members (int): The number of members in the ensemble.
    """

    def _predict(self, x: torch.Tensor, **forward_kwargs: Any) -> torch.Tensor:
        """This method is used to perform a forward pass of the model.

        In validation it is done with respect to all models that have been trained.
        In training only the first member is used.
        """
        if self.training:
            return super(EnsembleMethod, self)._predict(
                x, current_member=0, **forward_kwargs
            )
        else:
            return super()._predict(x, **forward_kwargs)

    def get_parameters(self, recurse: bool = True) -> List[torch.nn.Parameter]:
        """A helper function to get the parameters of a single ensemble member.

        In this case, get always the first one.
        """
        return list(self.model.parameters(index=0, recurse=recurse))

    def on_train_epoch_end(self) -> None:
        """This method is called at the end of each training epoch.

        In this case, if the learning rate cycle has been completed, the current model's weights are copied into
        the next member of the ensemble.
        """
        if (self.current_epoch) % self._cycle_length == 0 and self.current_epoch != 0:
            self.increment_current_member()
            self.model[self.model.currently_trained_member.item()].load_state_dict(
                self.model[0].state_dict()
            )

        super().on_train_epoch_end()

    def configure_optimizers(
        self,
    ) -> Tuple[
        List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]
    ]:
        """This method is used to configure the optimizers and the learning rate schedulers."""
        optimizer, scheduler = super().configure_optimizers()
        logging.warn(
            "The SnapshotEnsembleMethod always uses the cosine cyclic learning rate scheduler."
        )
        assert (
            self.trainer.max_epochs % self._num_members == 0
        ), "The number of epochs must be divisible by the number of members."
        self._cycle_length = self.trainer.max_epochs // self._num_members
        scheduler = [
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer[0], T_0=self._cycle_length, T_mult=1, eta_min=0, last_epoch=-1
            )
        ]
        return optimizer, scheduler


class GradientBoostingEnsembleMethod(EnsembleMethod):
    """This is a Method class for the Gradient Boosting Ensemble method.

    It uses the `Ensemble` model to wrap around the original model and then uses the base method to train the network.

    Args:
        num_members (int): The number of members in the ensemble.
        shrinkage (float): The shrinkage parameter for the gradient boosting. Defaults to 1.0.
    """

    def __init__(
        self, num_members: int, shrinkage: float = 1.0, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(num_members, *args, **kwargs)
        assert (
            0 <= shrinkage <= 1
        ), f"The shrinkage parameter must be in the range [0,1]. Got {shrinkage}."
        self._shrinkage = shrinkage
        logging.warning("The loss function is set to the MSELoss.")
        self._loss = nn.MSELoss(reduction="none")

        logging.warning(
            "For all the ensemble members, disabling the output activation. Needs to be done manually."
        )
        for i in range(self._num_members):
            self.model[i]._output_activation.disable()

    def _predict(self, x: torch.Tensor, **forward_kwargs: Any) -> torch.Tensor:
        """This method is used to perform a forward pass of the model."""
        outputs = []
        for i in range(self.model._num_members):
            outputs.append(
                self.model(x, current_member=i, **forward_kwargs) * self._shrinkage
            )

        return torch.stack(outputs, dim=MEMBERS_DIM)

    def _training_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """This method is used to perform a single training step.

        It assumes that the batch has a shape `(batch_size, num_features)`.
        It assumes that the output of the model has a shape `(batch_size, n_samples, num_classes)`.
        By default the `n_samples` is set to 1 and squeezed out.
        """
        x, y = batch
        y_hat = self._predict(x)
        y_hat_current_member = y_hat[:, self.model.currently_trained_member.item()]
        residual = self._residual_error(
            y_hat, y, self.model.currently_trained_member.item()
        )
        loss = self._loss(y_hat_current_member, residual).sum(dim=1).mean()

        y_permember = torch.stack([y] * self._num_members, dim=MEMBERS_DIM)
        y_hat_permember = output_activation(y_hat, self._task, dim=2)

        y_hat_mean = y_hat.sum(dim=1)
        y_hat_mean = output_activation(y_hat_mean, self._task, dim=1)

        output = {
            LOSS_KEY: loss,
            PREDICTION_KEY: y_hat_permember.detach(),
            TARGET_KEY: y.detach(),
            TARGET_PER_MEMBER_KEY: y_permember.detach(),
            PREDICTION_PER_MEMBER_KEY: y_hat_permember.detach(),
            MEAN_PREDICTION_KEY: y_hat_mean.detach(),
        }
        return output

    def _residual_error(
        self, y_hat: torch.Tensor, y: torch.Tensor, member: int
    ) -> torch.Tensor:
        """This method is used to compute the residual error.

        Args:
            y_hat (torch.Tensor): The predicted values.
            y (torch.Tensor): The ground truth values.
            member (int): The current member of the ensemble.
        """
        if self._task in [SEGMENTATION_KEY, CLASSIFICATION_KEY]:
            num_classes = y_hat.shape[2]
            y = one_hot(y, num_classes)

        if member == 0:
            return y

        if self._task in [SEGMENTATION_KEY, CLASSIFICATION_KEY]:
            y_hat = y_hat[:, :member].sum(dim=1)
            y_hat = F.softmax(y_hat, dim=1)
        elif self._task == REGRESSION_KEY:
            y_hat = y_hat[:, :member, 0].sum(dim=1)
        return y - y_hat

    def _validation_test_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """This method is used to perform a single validation step.

        It assumes that the batch has a shape `(batch_size, num_features)`.
        It assumes that the output of the model has a shape `(batch_size, n_samples, num_classes)`.
        By default the `n_samples` is set to 1 and squeezed out.
        """
        x, y = batch
        y_hat = self._predict(x)
        loss = torch.zeros(1, device=self.device)
        y_permember = torch.stack([y] * self._num_members, dim=MEMBERS_DIM)
        y_hat_permember = output_activation(y_hat, self._task, dim=2)

        y_hat_mean = y_hat.sum(dim=1)
        y_hat_mean = output_activation(y_hat_mean, self._task, dim=1)
        output = {
            LOSS_KEY: loss,
            PREDICTION_KEY: y_hat_permember.detach(),
            TARGET_KEY: y.detach(),
            TARGET_PER_MEMBER_KEY: y_permember.detach(),
            PREDICTION_PER_MEMBER_KEY: y_hat_permember.detach(),
            MEAN_PREDICTION_KEY: y_hat_mean.detach(),
        }
        return output

    def _test_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """This method is used to perform a single test step."""
        return self._validation_test_step(batch, batch_idx)

    def _validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """This method is used to perform a single validation step."""
        return self._validation_test_step(batch, batch_idx)

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = super(
            GradientBoostingEnsembleMethod, GradientBoostingEnsembleMethod
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_shrinkage",
            type=float,
            default=1.0,
            help="The shrinkage parameter for the gradient boosting.",
        )
        return parser
