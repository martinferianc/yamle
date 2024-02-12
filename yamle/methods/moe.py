from yamle.utils.operation_utils import average_predictions
from yamle.defaults import (
    LOSS_KEY,
    TARGET_KEY,
    TINY_EPSILON,
    PREDICTION_KEY,
    MEAN_PREDICTION_KEY,
    PREDICTION_PER_MEMBER_KEY,
    TARGET_PER_MEMBER_KEY,
    TRAIN_KEY,
    VALIDATION_KEY,
    TEST_KEY,
    MEMBERS_DIM,
    INPUT_KEY,
    MIN_TENDENCY,
)
from yamle.evaluation.metrics.algorithmic import metrics_factory
from yamle.methods.uncertain_method import MemberMethod
from yamle.models.operations import (
    Conv2dExtractor,
    LinearExtractor,
    ParallelModel,
    OutputActivation,
)
from yamle.models.visual_transformer import SpatialPositionalEmbedding
from typing import Any, List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
import torchmetrics
import math

logging = logging.getLogger("pytorch_lightning")


class MultiHeadEnsembleMethod(MemberMethod):
    """This class is the extension of the base method which accepts a single input and has multiple heads.
    The prediction is performed by averaging the predictions of the heads.

    Args:
        head_expansion_factor (float): The hidden size expansion factor for the heads. Default: 1.
        head_depth (int): The depth of the heads. Default: 1.
    """

    def __init__(
        self, head_expansion_factor: float, head_depth: int, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._head_expansion_factor = head_expansion_factor
        self._head_depth = head_depth

        self._output_inputs_dim = None
        self._output_outputs_dim = None
        if isinstance(self.model._output, nn.Linear):
            self._output_inputs_dim = self.model._output.in_features
            self._output_outputs_dim = self.model._output.out_features
        elif isinstance(self.model._output, nn.Conv2d):
            self._output_inputs_dim = self.model._output.in_channels
            self._output_outputs_dim = self.model._output.out_channels

        # Replace the output layer with respect to independent heads
        # The output layer is a parallel model with multiple heads
        # The output of the heads is addded in the `MEMBERS_DIM` dimension
        self.model._output = ParallelModel(
            nn.ModuleList(
                [
                    LinearExtractor(
                        self._output_inputs_dim,
                        self._head_expansion_factor,
                        self._output_outputs_dim,
                        self._head_depth,
                        norm=True,
                        end_activation=False,
                        end_normalization=False,
                    )
                    for _ in range(self._num_members)
                ]
            ),
            single_source=True,
            outputs_dim=MEMBERS_DIM,
        )
        self.model._output_activation = OutputActivation(
            self._task, self.model._output_activation._dim + 1
        )

    def _create_metrics(self, metrics_kwargs: Dict[str, Any]) -> None:
        """This method is used to create the metrics to be used for training, validation and testing."""
        self.metrics = {
            TRAIN_KEY: metrics_factory(**metrics_kwargs, per_member=True),
            VALIDATION_KEY: metrics_factory(**metrics_kwargs, per_member=True),
            TEST_KEY: metrics_factory(**metrics_kwargs, per_member=True),
        }

    def _step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
        phase: str = TRAIN_KEY,
    ) -> Dict[str, torch.Tensor]:
        """This method is used to perform a single training step.

        It assumes that the batch has a shape `(batch_size, num_features)`.
        It assumes that the output of the model has a shape `(batch_size, n_samples, num_classes)`.
        """
        x, y = batch
        y = torch.stack([y for _ in range(self._num_members)], dim=MEMBERS_DIM)

        y_hat = self._predict(x, unsqueeze=False)
        outputs = {}
        loss = self._loss_per_member(y_hat, y)
        y_hat_permember = y_hat.detach()
        y_permember = y.detach()
        y = y[:, 0]
        y_hat_mean = average_predictions(y_hat, self._task)

        outputs[LOSS_KEY] = loss
        outputs[TARGET_KEY] = y.detach()
        outputs[INPUT_KEY] = x.detach()
        outputs[PREDICTION_KEY] = y_hat.detach()
        outputs[MEAN_PREDICTION_KEY] = y_hat_mean.detach()
        outputs[PREDICTION_PER_MEMBER_KEY] = y_hat_permember.detach()
        outputs[TARGET_PER_MEMBER_KEY] = y_permember.detach()
        return outputs

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the specific arguments for this method."""
        parser = super(
            MultiHeadEnsembleMethod, MultiHeadEnsembleMethod
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_head_expansion_factor",
            type=float,
            default=1,
            help="The hidden size expansion factor for the heads.",
        )
        parser.add_argument(
            "--method_head_depth", type=int, default=1, help="The depth of the heads."
        )
        return parser


class MixtureOfExpertsMethod(MultiHeadEnsembleMethod):
    """This class is the extension of the multi-head method which accepts a single input and has multiple heads.
    The prediction is performed by averaging the predictions of the heads and the model is trained as the
    mixture of experts model. A gating network is used to determine the weights of the heads.

    Args:
        gating_expansion_factor (float): The hidden size expansion factor for the gating network. Default: 1.
        gating_depth (int): The depth of the gating network. Default: 1.
        alpha (float): The alpha parameter for weighting the importance loss. Default: 1.0.
        beta (float): The beta parameter for weighting the load-balancing loss. Default: 1.0.
        k (int): How many experts to sample from. Default: 1.
        noisy_gate (bool): Whether to use noisy gate or not. Default: False.
    """

    def __init__(
        self,
        gating_expansion_factor: float,
        gating_depth: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        k: int = 1,
        noisy_gate: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert alpha >= 0, f"alpha must be non-negative, got {alpha}"
        assert beta >= 0, f"beta must be non-negative, got {beta}"
        self._alpha = alpha
        self._beta = beta
        self._gating_expansion_factor = gating_expansion_factor
        self._gating_depth = gating_depth
        self._k = k
        self._noisy_gate = noisy_gate

        # Find out what is the input layer type to build the gating network
        self._input_inputs_dim = None
        self._input_outputs_dim = None
        if isinstance(self.model._input, nn.Linear):
            self._input_inputs_dim = self.model._input.in_features
            self._input_outputs_dim = self.model._input.out_features
        elif isinstance(self.model._input, nn.Conv2d):
            self._input_inputs_dim = self.model._input.in_channels
            self._input_outputs_dim = self.model._input.out_channels
        elif isinstance(self.model._input, SpatialPositionalEmbedding):
            self._input_inputs_dim = self.model._input._to_patch_embedding[2].in_features
            self._input_outputs_dim = self.model._input._to_patch_embedding[
                2
            ].out_features
        self._input_layer_type = type(self.model._input)

        if self._input_layer_type in [nn.Conv2d, SpatialPositionalEmbedding]:
            self.model._gating = nn.Sequential(
                Conv2dExtractor(
                    self._input_inputs_dim,
                    self._gating_expansion_factor,
                    self._input_outputs_dim,
                    self._gating_depth,
                    norm=True,
                    end_activation=True,
                    end_normalization=True,
                    end_pooling=True,
                ),
                nn.Linear(
                    int(self._input_outputs_dim * self._gating_expansion_factor),
                    self._num_members
                    if not self._noisy_gate
                    else 2 * self._num_members,
                ),
            )

        elif self._input_layer_type == nn.Linear:
            self.model._gating = nn.Sequential(
                nn.Flatten(),
                LinearExtractor(
                    self._input_inputs_dim,
                    self._gating_expansion_factor,
                    self._input_outputs_dim,
                    self._gating_depth,
                    norm=True,
                    end_activation=True,
                    end_normalization=True,
                ),
                nn.Linear(
                    self._input_outputs_dim,
                    self._num_members
                    if not self._noisy_gate
                    else 2 * self._num_members,
                ),
            )
        self._initialise_gating_network()
        self.register_buffer("_mean", torch.tensor([0.0]))
        self.register_buffer("_std", torch.tensor([1.0]))

    def _initialise_gating_network(self) -> None:
        """Initialise all modules in the gating network with 0 mean and 1e-3 variance."""
        for m in self.model._gating.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-3)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=1e-3)
                nn.init.zeros_(m.bias)

    def _create_metrics(self, metrics_kwargs: Dict[str, Any]) -> None:
        """This method is used to create the metrics for the method."""
        super()._create_metrics(metrics_kwargs)
        self._add_additional_metrics(
            {
                f"{LOSS_KEY}_importance": torchmetrics.MeanMetric(),
                f"{LOSS_KEY}_load_balancing": torchmetrics.MeanMetric(),
            },
            tendencies=[MIN_TENDENCY, MIN_TENDENCY],
        )

    def _predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """This method is used to perform a single prediction step.

        It assumes that the batch has a shape `(batch_size, num_features)`.
        It assumes that the output of the model has a shape `(batch_size, n_samples, num_classes)`.
        """
        y_hat = self.model(x)
        gating_logits = self.model._gating(x)
        return y_hat, gating_logits

    def _top_k_gates(
        self, gating_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """This method is used to sample the top k gating values.

        Args:
            gating (torch.Tensor): The gating network output.

        Returns:
            gates (torch.Tensor): The top k gating values.
            clean_logits (torch.Tensor): The clean logits of the top k gating values.
            noisy_logits (torch.Tensor): The noisy logits of the top k gating values.
            std (torch.Tensor): The standard deviation of the top k gating values.
            top_k_logits (torch.Tensor): The top k gating logits.
        """
        noisy_logits = None
        std = None
        clean_logits = gating_logits
        if self._noisy_gate:
            if self.training:
                clean_logits = gating_logits[:, : self._num_members]
                std = F.softplus(gating_logits[:, self._num_members :]) + TINY_EPSILON
                noisy_logits = clean_logits + torch.randn_like(clean_logits) * std
                gating_logits = noisy_logits
            else:
                clean_logits = gating_logits[:, : self._num_members]
                gating_logits = gating_logits[:, : self._num_members]

        top_logits, top_indices = torch.topk(
            gating_logits, min(self._k + 1, self._num_members), dim=1
        )
        top_k_logits = top_logits[:, : self._k]
        top_k_indices = top_indices[:, : self._k]
        top_k_probs = F.softmax(top_k_logits, dim=1)
        zeros = torch.zeros_like(gating_logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_probs)
        return gates, clean_logits, noisy_logits, std, top_logits

    def _gates_load(
        self,
        gates: torch.Tensor,
        logits: torch.Tensor,
        noisy_logits: torch.Tensor,
        std: torch.Tensor,
        noisy_top_logits: torch.Tensor,
    ) -> torch.Tensor:
        """This method is used to sample the top k gating values.

        Args:
            gating (torch.Tensor): The gating network output.

        Returns:
            torch.Tensor: The top k gating values.
        """
        if not self._noisy_gate or self.evaluation:
            return (gates > 0.0).sum(dim=0)
        else:
            B = logits.size(0)
            M = noisy_top_logits.size(1)
            top_values_flat = noisy_top_logits.flatten()

            threshold_positions_if_in = (
                torch.arange(B, device=logits.device) * M + self._k
            )
            threshold_if_in = torch.unsqueeze(
                torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
            )
            is_in = torch.gt(noisy_logits, threshold_if_in)
            threshold_positions_if_out = threshold_positions_if_in - 1
            threshold_if_out = torch.unsqueeze(
                torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
            )

            def normal_cdf(x: torch.Tensor) -> torch.Tensor:
                return 0.5 * (
                    1.0
                    + torch.erf(x - self._mean)
                    / (self._std * math.sqrt(2.0) + TINY_EPSILON)
                )

            prob_if_in = normal_cdf((logits - threshold_if_in) / (std + TINY_EPSILON))
            prob_if_out = normal_cdf((logits - threshold_if_out) / (std + TINY_EPSILON))
            prob = torch.where(is_in, prob_if_in, prob_if_out)
            return prob

    def _gates_importance(self, gates: torch.Tensor) -> torch.Tensor:
        """Importance loss for each expert"""
        return gates.sum(dim=0)

    def _cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        """Coefficient of variation squared"""
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + TINY_EPSILON)

    def _loss_load_balancing(
        self,
        gates: torch.Tensor,
        logits: torch.Tensor,
        noisy_logits: torch.Tensor,
        std: torch.Tensor,
        noisy_top_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Auxiliary loss in addition to the main loss"""
        return self._cv_squared(
            self._gates_load(gates, logits, noisy_logits, std, noisy_top_logits)
        )

    def _loss_importance(self, gates: torch.Tensor) -> torch.Tensor:
        """Auxiliary loss in addition to the main loss"""
        return self._cv_squared(self._gates_importance(gates))

    def _step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
        phase: str = TRAIN_KEY,
    ) -> Dict[str, torch.Tensor]:
        """This method is used to perform a single training step.

        It assumes that the batch has a shape `(batch_size, num_features)`.
        It assumes that the output of the model has a shape `(batch_size, n_samples, num_classes)`.
        """
        x, y = batch

        y_hat, gating_logits = self._predict(x)
        (
            gating_weights,
            clean_logits,
            noisy_logits,
            std,
            noisy_top_k_logits,
        ) = self._top_k_gates(gating_logits)

        loss_gating_weights = gating_weights
        outputs = {}
        while len(loss_gating_weights.shape) < len(y_hat.shape):
            loss_gating_weights = loss_gating_weights.unsqueeze(-1)

        loss = self._loss(
            (y_hat * loss_gating_weights).sum(dim=MEMBERS_DIM).unsqueeze(1), y
        )
        loss_importance = self._loss_importance(gating_weights)
        loss_load_balancing = self._loss_load_balancing(
            gating_weights, clean_logits, noisy_logits, std, noisy_top_k_logits
        )
        loss = loss + self._alpha * loss_importance + self._beta * loss_load_balancing
        y_permember = torch.stack(
            [y for _ in range(self._num_members)], dim=MEMBERS_DIM
        ).detach()
        y_hat_permember = y_hat.detach()
        y_hat_mean = average_predictions(
            y_hat, self._task, weights=F.softmax(clean_logits, dim=1)
        )
        gating_weights
        outputs[LOSS_KEY] = loss
        outputs[f"{LOSS_KEY}_importance"] = loss_importance
        outputs[f"{LOSS_KEY}_load_balancing"] = loss_load_balancing
        outputs[TARGET_KEY] = y.detach()
        outputs[INPUT_KEY] = x.detach()
        outputs[PREDICTION_KEY] = y_hat.detach()
        outputs[MEAN_PREDICTION_KEY] = y_hat_mean.detach()
        outputs[PREDICTION_PER_MEMBER_KEY] = y_hat_permember.detach()
        outputs[TARGET_PER_MEMBER_KEY] = y_permember.detach()
        return outputs

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the specific arguments to the parser."""
        parser = super(
            MixtureOfExpertsMethod, MixtureOfExpertsMethod
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_gating_expansion_factor",
            type=float,
            default=1.0,
            help="The expansion factor of the gating network.",
        )
        parser.add_argument(
            "--method_gating_depth",
            type=int,
            default=2,
            help="The depth of the gating network.",
        )
        parser.add_argument(
            "--method_alpha",
            type=float,
            default=1.0,
            help="The weight of the loss diversity.",
        )
        parser.add_argument(
            "--method_beta",
            type=float,
            default=1.0,
            help="The weight of the loss load balancing.",
        )
        parser.add_argument(
            "--method_k", type=int, default=2, help="The number of experts to use."
        )
        parser.add_argument(
            "--method_noisy_gate",
            type=int,
            default=0,
            choices=[0, 1],
            help="Whether to use noisy gate or not.",
        )

        return parser
