from yamle.utils.operation_utils import average_predictions
from yamle.defaults import (
    TINY_EPSILON,
    LOSS_KEY,
    LOSS_KL_KEY,
    TARGET_KEY,
    PREDICTION_KEY,
    MEAN_PREDICTION_KEY,
    TRAIN_KEY,
    PREDICTION_PER_MEMBER_KEY,
    TARGET_PER_MEMBER_KEY,
    VALIDATION_KEY,
    TEST_KEY,
    MEMBERS_DIM,
    INPUT_KEY,
    AVERAGE_WEIGHTS_KEY,
    MIN_TENDENCY,
)
from yamle.utils.regularizer_utils import disable_regularizer
from yamle.evaluation.metrics.algorithmic import metrics_factory
from yamle.methods.uncertain_method import MemberMethod
from typing import Any, List, Dict, Optional
import torch
import torchmetrics
import argparse
import logging

logging = logging.getLogger("pytorch_lightning")


class DUNMethod(MemberMethod):
    """This class is the extension of the base method for which the prediciton is performed through the method of:
    Depth Uncertainty in Neural Networks where the `_output` layer is used repatedly to get the prediction
    per each hidden layer.

    Args:
        alpha (float): The alpha parameter for the KL divergence.
        warm_starting_epochs (int): The number of epochs to train the model without changing the depth weights.
    """

    def __init__(
        self, alpha: float, warm_starting_epochs: int, *args: Any, **kwargs: Any
    ) -> None:
        if not hasattr(kwargs["model"], "_depth"):
            raise ValueError(
                "The model should have a `_depth` attribute which is the number of hidden layers."
            )

        kwargs["num_members"] = kwargs["model"]._depth + 1
        kwargs["metrics_kwargs"]["num_members"] = kwargs["model"]._depth + 1
        self._depth = kwargs["model"]._depth

        super().__init__(*args, **kwargs)

        logging.warning(
            f"The number of members is set to {self._num_members} because of the depth of the model."
        )

        self._loss.set_reduction_per_member("sum")
        self._loss.set_reduction_per_sample("sum")
        logging.warning(
            f"The reduction per member is set to {self._loss._reduction_per_member} and the reduction per sample is set to {self._loss._reduction_per_sample}."
        )
        assert (
            alpha > 0
        ), f"The alpha parameter should be greater than 0, but it is {alpha}."
        self._alpha = alpha

        if hasattr(self.model, "_alphas"):
            raise ValueError("The model should not have an `_alphas` attribute.")

        if hasattr(self.model, "_prior_betas"):
            raise ValueError("The model should not have an `_prior_betas` attribute.")

        self._warm_starting_epochs = warm_starting_epochs

        self.model._alphas = torch.nn.Parameter(
            torch.zeros(self._num_members, requires_grad=True) / (self._num_members)
        )
        self._alphas_container = torch.empty((0, self._num_members))

        self.model.register_buffer(
            "_prior_betas", torch.ones(self._num_members) / self._num_members
        )

        disable_regularizer(self.model._alphas)
        self.model.add_method_specific_layers(method="dun")

    @property
    def alphas(self) -> torch.Tensor:
        """This method is used to get the alphas of the model."""
        if self.current_epoch < self._warm_starting_epochs and self.training:
            return self.prior_betas
        return torch.softmax(self.model._alphas, dim=0)

    @property
    def prior_betas(self) -> torch.Tensor:
        """This method is used to get the prior betas of the model."""
        return self.model._prior_betas

    def _kl_divergence(self) -> torch.Tensor:
        """This method is used to compute the kl divergence between the prior and the posterior."""
        return torch.sum(
            self.alphas * torch.log(self.alphas / self.prior_betas + TINY_EPSILON)
        )

    def _create_metrics(self, metrics_kwargs: Dict[str, Any]) -> None:
        """This method is used to create the metrics to be used for training, validation and testing."""
        self.metrics = {
            TRAIN_KEY: metrics_factory(**metrics_kwargs, per_member=True),
            VALIDATION_KEY: metrics_factory(**metrics_kwargs, per_member=True),
            TEST_KEY: metrics_factory(**metrics_kwargs, per_member=True),
        }
        self._add_additional_metrics(
            {LOSS_KL_KEY: torchmetrics.MeanMetric()}, tendencies=[MIN_TENDENCY]
        )

    def _predict(self, x: torch.Tensor, **forward_kwargs: Any) -> torch.Tensor:
        """This method is used to perform a forward pass of the model.

        It is done with respect to the number of hidden layers or how hidden layers are being defined
        in the underlying `self.model`.
        """
        last_layer, stages = self.model(x, staged_output=True, **forward_kwargs)
        # Since the last layer uses the last hidden layer
        # we can remove it
        stages = stages[:-1]
        outputs = []
        for i, h in enumerate(stages):
            h = self.model._reshaping_layers[i](h)
            h = self.model.final_layer(h)
            outputs.append(h)
        outputs.append(last_layer)
        return torch.stack(outputs, dim=MEMBERS_DIM)

    def _step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
        phase: str = TRAIN_KEY,
    ) -> Dict[str, torch.Tensor]:
        """This method is used to perform a single step."""
        x, y = batch
        y = torch.stack([y for _ in range(self._num_members)], dim=MEMBERS_DIM)

        y_hat = self._predict(x)
        outputs = {}
        if self.training:
            N_train = self._datamodule.train_dataset_size()
            batch_size = x.shape[0]
            loss = -(N_train / batch_size) * self._loss_per_member(
                y_hat, y, weights_per_member=self.alphas
            )
            kl_divergence = self._kl_divergence()
            loss = loss - self._alpha * kl_divergence
            loss = -loss / N_train
        else:
            loss = torch.tensor(0.0, device=y_hat.device)
            kl_divergence = torch.tensor(0.0, device=y_hat.device)

        y_hat_permember = y_hat.detach()
        y_permember = y.detach()

        y = y[:, 0]
        y_hat_mean = average_predictions(
            y_hat,
            self._task,
            weights=torch.stack([self.alphas for _ in range(y_hat.shape[0])], dim=0),
        )

        outputs[LOSS_KEY] = loss
        outputs[LOSS_KL_KEY] = kl_divergence.detach()
        outputs[TARGET_KEY] = y.detach()
        outputs[INPUT_KEY] = x.detach()
        outputs[PREDICTION_KEY] = y_hat.detach()
        outputs[MEAN_PREDICTION_KEY] = y_hat_mean.detach()
        outputs[PREDICTION_PER_MEMBER_KEY] = y_hat_permember.detach()
        outputs[TARGET_PER_MEMBER_KEY] = y_permember.detach()
        outputs[AVERAGE_WEIGHTS_KEY] = torch.stack(
            [self.alphas for _ in range(y_hat.shape[0])], dim=0
        ).detach()
        return outputs

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        logging.info(f"Alphas: {self.alphas}")
        self._alphas_container = torch.cat(
            [self._alphas_container, self.alphas.unsqueeze(0).detach().cpu()], dim=0
        )

    def state_dict(self) -> Dict[str, Any]:
        """This method is used to get the state of the method."""
        state_dict = super().state_dict()
        state_dict["alphas_container"] = self._alphas_container
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """This method is used to load the state of the method."""
        super().load_state_dict(state_dict)
        self._alphas_container = state_dict["alphas_container"]

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the specific arguments for the DUN method."""
        parser = super(DUNMethod, DUNMethod).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_alpha",
            type=float,
            default=1.0,
            help="The alpha to be used for the trade-off between the likelihood and the prior.",
        )
        parser.add_argument(
            "--method_warm_starting_epochs",
            type=int,
            default=0,
            help="The number of epochs to use the prior betas.",
        )
        return parser
