from yamle.utils.optimization_utils import (
    LinearScalarScheduler,
    AVAILABLE_SCALAR_SCHEDULERS,
)
from yamle.utils.operation_utils import average_predictions
from yamle.utils.regularizer_utils import disable_regularizer
from yamle.models.operations import (
    OutputActivation,
    ParallelModel,
    LinearExtractor,
    Conv2dExtractor,
    ReshapeInput,
    ReshapeOutput,
)
from yamle.models.specific.mixmo import MixMoBlock, UnmixingBlock
from yamle.models.specific.mixvit import MixVitWrapper
from yamle.models.specific.datamux import Multiplexer, Demultiplexer
from yamle.models.visual_transformer import SpatialPositionalEmbedding
from yamle.models.specific.mimmo import MIMMMOWrapper
from yamle.methods.uncertain_method import MemberMethod
from typing import List, Dict, Any, Tuple, Optional
from yamle.evaluation.metrics.algorithmic import metrics_factory
from yamle.defaults import (
    TINY_EPSILON,
    LOSS_KEY,
    TARGET_KEY,
    TARGET_PER_MEMBER_KEY,
    MEAN_PREDICTION_KEY,
    PREDICTION_KEY,
    PREDICTION_PER_MEMBER_KEY,
    TRAIN_KEY,
    VALIDATION_KEY,
    TEST_KEY,
    MEMBERS_DIM,
    INPUT_KEY,
    AVERAGE_WEIGHTS_KEY,
    MIN_TENDENCY,
)
import torch
import torch.nn as nn
import argparse
import logging
import torchmetrics
import copy

from yamle.utils.specific.mimo_experiments.plotting_utils import (
    plot_input_layer_norm_bar,
    plot_output_layer_norm_bar,
    plot_weight_trajectories,
    plot_overlap_between_members,
)

logging = logging.getLogger("pytorch_lightning")


class MIMOMethod(MemberMethod):
    """This class is the extension of the base method for MIMO methods.

    The difference is in having to change the prediction to concatenate the `num_members` dimension.
    into the first feature dimension.

    Args:
        initialise_encoder_members_same (bool): Whether to initialise the members in the encoder with the same weights.
        num_batch_repetitions (int): The number of times some samples are repeated in the batch.
        input_repetition_probability (Optional[float]): The probability that the inputs are identical for the ensemble members.
        repeat_evaluation (bool): Whether to repeat samples in the evaluation.
    """

    def __init__(
        self,
        initialise_encoder_members_same: bool = False,
        num_batch_repetitions: int = 1,
        input_repetition_probability: Optional[float] = None,
        repeat_evaluation: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert (
            num_batch_repetitions >= 1
        ), "The number of batch duplication should be greater than or equal to 1."

        self._num_batch_repetitions = num_batch_repetitions
        self._initialise_encoder_members_same = initialise_encoder_members_same

        self._input_repetition_probability = input_repetition_probability

        # Remember all the input and output layer information.
        self._input_inputs_dim = None
        self._input_outputs_dim = None
        if isinstance(self.model._input, nn.Linear):
            self._input_inputs_dim = self.model._input.in_features
            self._input_outputs_dim = self.model._input.out_features
            self._input_layer_kwargs = {
                "in_features": self._input_inputs_dim,
                "out_features": self._input_outputs_dim,
                "bias": self.model._input.bias is not None,
            }
        elif isinstance(self.model._input, nn.Conv2d):
            self._input_inputs_dim = self.model._input.in_channels
            self._input_outputs_dim = self.model._input.out_channels
            self._input_layer_kwargs = {
                "in_channels": self._input_inputs_dim,
                "out_channels": self._input_outputs_dim,
                "kernel_size": self.model._input.kernel_size,
                "stride": self.model._input.stride,
                "padding": self.model._input.padding,
                "dilation": self.model._input.dilation,
                "groups": self.model._input.groups,
                "bias": self.model._input.bias is not None,
                "padding_mode": self.model._input.padding_mode,
            }
        elif isinstance(self.model._input, SpatialPositionalEmbedding):
            self._input_inputs_dim = self.model._input._to_patch_embedding[
                2
            ].in_features
            self._input_outputs_dim = self.model._input._to_patch_embedding[
                2
            ].out_features
            self._input_layer_kwargs = {
                "patch_size": self.model._input._patch_size,
                "inputs_dim": self.model._input._inputs_dim,
                "embedding_dim": self.model._input._embedding_dim,
                "dropout": self.model._input._dropout,
                "num_cls_tokens": self.model._input._num_cls_tokens,
                "positional_embedding": self.model._input._positional_embedding
                is not None,
            }

        self._input_layer_type = type(self.model._input)

        self._output_inputs_dim = None
        self._output_outputs_dim = None
        if isinstance(self.model._output, nn.Linear):
            self._output_inputs_dim = self.model._output.in_features
            self._output_outputs_dim = self.model._output.out_features
            self._output_layer_kwargs = {
                "in_features": self._output_inputs_dim,
                "out_features": self._output_outputs_dim,
                "bias": self.model._output.bias is not None,
            }
        elif isinstance(self.model._output, nn.Conv2d):
            self._output_inputs_dim = self.model._output.in_channels
            self._output_outputs_dim = self.model._output.out_channels
            self._output_layer_kwargs = {
                "in_channels": self._output_inputs_dim,
                "out_channels": self._output_outputs_dim,
                "kernel_size": self.model._output.kernel_size,
                "stride": self.model._output.stride,
                "padding": self.model._output.padding,
                "dilation": self.model._output.dilation,
                "groups": self.model._output.groups,
                "bias": self.model._output.bias is not None,
                "padding_mode": self.model._output.padding_mode,
            }

        self._output_layer_type = type(self.model._output)
        self._input_layer_overlap_container = torch.zeros((1, 1))
        self._output_layer_overlap_container = torch.zeros((1, 1))

        self._post_init()

        self._repeat_evaluation = repeat_evaluation

    def _create_metrics(self, metrics_kwargs: Dict[str, Any]) -> None:
        """This method is used to create the metrics to be used for training, validation and testing."""
        self.metrics = {
            TRAIN_KEY: metrics_factory(**metrics_kwargs, per_member=True),
            VALIDATION_KEY: metrics_factory(**metrics_kwargs, per_member=True),
            TEST_KEY: metrics_factory(**metrics_kwargs, per_member=True),
        }

    def _post_init(self) -> None:
        """This method is called after the initialisation of the method."""
        self._replace_input_layer(self._num_members)
        self._replace_output_layer(self._num_members)

    def _train_batch_repetition(
        self, batch: Tuple[torch.Tensor, torch.Tensor], num_members: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A helper method to repeat some samples in the training batch to create a new batch.

        Steps:
            Randomly select `batch_size`/`batch_repetition` samples from the batch.
            Duplicate the selected samples `batch_repetition` times to create a new batch of size `batch_size`.
            Concatenate the new batch with the original batch to create a new batch of size `num_members*batch_size`.
        """
        x, y = batch
        batch_size = x.shape[0]
        indices = torch.tile(
            torch.arange(batch_size), [self._num_batch_repetitions]
        ).to(x.device)
        # Shuffle the indices to create a new batch
        main_shuffle = indices[torch.randperm(indices.shape[0])]
        to_shuffle = (
            torch.tensor(len(main_shuffle), device=main_shuffle.device).float()
            * (1.0 - self._input_repetition_probability)
        ).long()
        shuffle_indices = [
            torch.cat(
                [
                    main_shuffle[:to_shuffle][torch.randperm(to_shuffle)],
                    main_shuffle[to_shuffle:],
                ],
                dim=0,
            )
            for _ in range(num_members)
        ]
        x = torch.stack([x[indices]
                        for indices in shuffle_indices], dim=MEMBERS_DIM)
        y = torch.stack([y[indices]
                        for indices in shuffle_indices], dim=MEMBERS_DIM)
        return (x, y), torch.stack(shuffle_indices, dim=MEMBERS_DIM)

    def _validation_test_batch_repetition(
        self, batch: Tuple[torch.Tensor, torch.Tensor], num_members: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A helper method to repeat the samples in the `batch` `num_members` times."""
        x, y = batch
        if self._repeat_evaluation:
            # If `True` the samples are repeated for all the members and they are the same
            batch = torch.stack(
                [x for _ in range(num_members)], dim=MEMBERS_DIM
            ), torch.stack([y for _ in range(num_members)], dim=MEMBERS_DIM)
        else:
            # If `False` the samples are randomly shuffled for each member
            xs = []
            ys = []
            for _ in range(num_members):
                indices = torch.randperm(x.shape[0])
                xs.append(x[indices])
                ys.append(y[indices])
            batch = torch.stack(xs, dim=MEMBERS_DIM), torch.stack(
                ys, dim=MEMBERS_DIM)
        return batch

    def _replace_input_layer(self, num_members: int) -> None:
        """A helper function to replace the first layer with one where the input dimension is multiplied by the number of members."""
        input_layer: nn.Module = None
        input_layer_kwargs = copy.deepcopy(self._input_layer_kwargs)
        if self._input_layer_type == nn.Linear:
            input_layer_kwargs["in_features"] = (
                input_layer_kwargs["in_features"] * num_members
            )
            input_layer = torch.nn.Linear(**input_layer_kwargs)

        elif self._input_layer_type == nn.Conv2d:
            input_layer_kwargs["in_channels"] = (
                input_layer_kwargs["in_channels"] * num_members
            )
            input_layer = torch.nn.Conv2d(**input_layer_kwargs)
        elif self._input_layer_type == SpatialPositionalEmbedding:
            input_layer_kwargs["inputs_dim"] = (
                input_layer_kwargs["inputs_dim"][0] * num_members,
                *input_layer_kwargs["inputs_dim"][1:],
            )
            input_layer_kwargs["num_cls_tokens"] = (
                input_layer_kwargs["num_cls_tokens"] * num_members
            )
            input_layer = SpatialPositionalEmbedding(**input_layer_kwargs)
        else:
            raise ValueError(
                f"Input layer type {self._input_layer_type} not supported."
            )

        self.model._input = nn.Sequential(ReshapeInput(), input_layer).to(
            next(self.model.parameters()).device
        )
        self._input_layer = self.model._input[1]
        if self._input_layer_type == SpatialPositionalEmbedding:
            self._input_layer = self.model._input[1]._to_patch_embedding[2]
            self.model._input.get_cls_token_indices = self.model._input[
                1
            ].get_cls_token_indices

        if self._initialise_encoder_members_same:
            for member in range(1, num_members):
                self._initialise_input_layer_weights_same(
                    source_member=0, target_member=member
                )

    def _initialise_input_layer_weights_same(
        self, source_member: int, target_member: int
    ) -> None:
        """A helper method to initialise the weights of the input layer from the `source_member` to the `target_member`.

        The initialisation is done by copying the weights of the `source_member` to the `target_member`.
        """
        assert (
            source_member != target_member
        ), f"The source member and the target member should be different. Got {source_member} and {target_member}."
        assert self._input_layer_type in [
            nn.Linear,
            nn.Conv2d,
        ], f"The input layer type should be either `torch.nn.Linear` or `torch.nn.Conv2d`, but it is {self._input_layer_type}."
        self._input_layer.weight.data[
            :,
            target_member
            * self._input_inputs_dim: (target_member + 1)
            * self._input_inputs_dim,
        ] = self._input_layer.weight.data[
            :,
            source_member
            * self._input_inputs_dim: (source_member + 1)
            * self._input_inputs_dim,
        ].clone()

    def _replace_output_layer(self, num_members: int) -> None:
        """Replace the last layer with one where the output dimension is multiplied by the number of members."""
        output_layer: nn.Module = None
        output_layer_kwargs = copy.deepcopy(self._output_layer_kwargs)
        if self._output_layer_type == nn.Linear:
            output_layer_kwargs["out_features"] = (
                output_layer_kwargs["out_features"] * num_members
            )
            # If the input is SpatialPositionalEmbedding,
            # and the pooling type is `cls`,
            # then the output layer also needs to be multiplied
            # with respect to `num_cls_tokens`.
            if (
                self._input_layer_type == SpatialPositionalEmbedding
                and self.model._pooling == "cls"
            ):
                output_layer_kwargs["in_features"] = (
                    output_layer_kwargs["in_features"]
                    * self._input_layer_kwargs["num_cls_tokens"]
                    * num_members
                )
            output_layer = torch.nn.Linear(**output_layer_kwargs)
        elif self._output_layer_type == nn.Conv2d:
            output_layer_kwargs["out_channels"] = (
                output_layer_kwargs["out_channels"] * num_members
            )
            output_layer = torch.nn.Conv2d(**output_layer_kwargs)
        else:
            raise ValueError(
                f"The last layer of the model should be a `torch.nn.Linear` layer, but it is a \
                {self._output_layer_type}."
            )
        self.model._output = nn.Sequential(
            output_layer, ReshapeOutput(num_members=num_members)
        ).to(next(self.model.parameters()).device)

        self._output_layer = self.model._output[0]

        self.model._output_activation = OutputActivation(
            self.model._output_activation._task, dim=2
        )

    def _step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
        phase: str = TRAIN_KEY,
        num_members: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """A helper method to perform a single step whether training, validation or test."""
        x, y = batch
        y_hat = self._predict(x, unsqueeze=False)
        loss = self._loss_per_member(y_hat, y)
        # We are in training and the input has not been repeated.
        y_permember = y
        y_hat_permember = y_hat
        if self.training:
            y_hat = y_hat.reshape(-1, 1, *y_hat.shape[2:])
            x = x.reshape(-1, *x.shape[2:])
        else:
            y = y[:, 0]  # They are all the same.
            x = x[:, 0]  # They are all the same.
        y = y.reshape(-1, *y.shape[2:])
        y_hat_mean = average_predictions(y_hat, self._task)
        output = {
            LOSS_KEY: loss,
            PREDICTION_KEY: y_hat.detach(),
            TARGET_KEY: y.detach(),
            TARGET_PER_MEMBER_KEY: y_permember.detach(),
            PREDICTION_PER_MEMBER_KEY: y_hat_permember.detach(),
            MEAN_PREDICTION_KEY: y_hat_mean.detach(),
            INPUT_KEY: x.detach(),
        }
        return output

    def _validation_test_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        optimizer_idx: int = 0,
        phase: str = VALIDATION_KEY,
        num_members: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """This method is used to perform a single validation or testing step.

        It assumes that the batch has a shape `(batch_size, num_features)`.
        The inputs need to be repeated `num_members` times. Such that the input then has a shape
        `(batch_size * num_members, num_features)`.
        The output of the model has a shape `(batch_size, num_members, num_classes)`.
        """
        new_batch = self._validation_test_batch_repetition(
            batch, num_members=num_members
        )
        return self._step(
            new_batch,
            batch_idx,
            optimizer_idx=optimizer_idx,
            phase=phase,
            num_members=num_members,
        )

    def _training_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """This method is used to perform a single training step.

        It assumes that the batch has a shape `(batch_size, num_features)`.
        It assumes that the output of the model has a shape `(batch_size, n_samples, num_classes)`.
        """
        new_batch, _ = self._train_batch_repetition(
            batch, num_members=self._num_members
        )
        output = self._step(
            new_batch,
            batch_idx,
            optimizer_idx=optimizer_idx,
            phase=TRAIN_KEY,
            num_members=self._num_members,
        )
        return output

    def _validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """This method is used to perform a single validation step.

        It assumes that the batch has a shape `(batch_size, num_features)`.
        It assumes that the output of the model has a shape `(batch_size, n_samples, num_classes)`.
        """
        return self._validation_test_step(
            batch,
            batch_idx,
            optimizer_idx=None,
            phase=VALIDATION_KEY,
            num_members=self._num_members,
        )

    def _test_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """This method is used to perform a single test step.

        It assumes that the batch has a shape `(batch_size, num_features)`.
        It assumes that the output of the model has a shape `(batch_size, n_samples, num_classes)`.
        """
        return self._validation_test_step(
            batch,
            batch_idx,
            optimizer_idx=None,
            phase=TEST_KEY,
            num_members=self._num_members,
        )

    def _plots(self) -> None:
        output_weights_per_member = []
        input_weights_per_member = []
        for member in range(self._num_members):
            output_weights_per_member.append(
                self._output_layer.weight.data[
                    member
                    * self._output_outputs_dim: (member + 1)
                    * self._output_outputs_dim
                ]
            )
            input_weights_per_member.append(
                self._input_layer.weight.data[
                    :,
                    member
                    * self._input_inputs_dim: (member + 1)
                    * self._input_inputs_dim,
                ]
            )

        input_overlap = plot_input_layer_norm_bar(
            input_weights_per_member, self._save_path, self.current_epoch
        )
        output_overlap = plot_output_layer_norm_bar(
            output_weights_per_member, self._save_path, self.current_epoch
        )

        self._input_layer_overlap_container = torch.cat(
            (
                self._input_layer_overlap_container,
                input_overlap.unsqueeze(0).unsqueeze(0).detach().cpu(),
            ),
            dim=0,
        )

        self._output_layer_overlap_container = torch.cat(
            (
                self._output_layer_overlap_container,
                output_overlap.unsqueeze(0).unsqueeze(0).detach().cpu(),
            ),
            dim=0,
        )
        plot_overlap_between_members(
            self._input_layer_overlap_container, self._save_path, input=True
        )
        plot_overlap_between_members(
            self._output_layer_overlap_container, self._save_path, input=False
        )

    def on_train_epoch_end(self) -> None:
        """This method is called at the end of the training epoch."""
        super().on_train_epoch_end()
        if self._plotting_training:
            self._plots()

    def state_dict(self) -> Dict[str, Any]:
        """This method returns the state dict of the MIMO method."""
        state_dict = super().state_dict()
        state_dict[
            "input_layer_overlap_container"
        ] = self._input_layer_overlap_container
        state_dict[
            "output_layer_overlap_container"
        ] = self._output_layer_overlap_container
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """This method loads the state dict of the MIMO method."""
        super().load_state_dict(state_dict)
        self._input_layer_overlap_container = state_dict[
            "input_layer_overlap_container"
        ]
        self._output_layer_overlap_container = state_dict[
            "output_layer_overlap_container"
        ]

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method adds the specific arguments for the MIMO method."""
        parser = super(MIMOMethod, MIMOMethod).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_initialise_encoder_members_same",
            type=int,
            choices=[0, 1],
            default=0,
            help="Whether to initialise the members of the ensemble with the same weights.",
        )
        parser.add_argument(
            "--method_num_batch_repetitions",
            type=int,
            default=2,
            help="The number of times some samples are repeated in the batch.",
        )
        parser.add_argument(
            "--method_input_repetition_probability",
            type=float,
            default=0.0,
            help="The value to start the input repetition probability.",
        )
        parser.add_argument(
            "--method_repeat_evaluation",
            type=int,
            choices=[0, 1],
            default=1,
            help="Whether to repeat the samples in the evaluation.",
        )
        return parser


class MIMMOMethod(MIMOMethod):
    """This class is the extension of the MIMO method in which we will try to find the depth for each ensemble member.

    Args:
        alpha (float): The alpha value to regularize the depth loss term.
        prior (str): The prior to use for the depth weights.
        do_not_optimize_depth_weights (bool): Whether to optimize the depth weights or not.
        additional_heads (bool): Whether to enable specific heads for each layer.
        available_heads (List[bool]): Toggles to hard enable or disable heads.
        warm_starting_epochs (int): The number of epochs to train the model without changing the member or depth weights.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        prior: str = "uniform",
        do_not_optimize_depth_weights: bool = False,
        additional_heads: bool = False,
        available_heads: List[bool] = None,
        warm_starting_epochs: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        model_depth = kwargs["model"]._depth
        if available_heads is None:
            available_heads = [True] * (model_depth)
        available_heads = [bool(x) for x in available_heads]
        assert (
            len(available_heads) == model_depth
        ), f"The length of the available heads should be {model_depth} but it is {len(available_heads)}."

        kwargs["metrics_kwargs"]["num_members"] = (
            len(available_heads) * kwargs["metrics_kwargs"]["num_members"]
        )

        super().__init__(*args, **kwargs)
        if not hasattr(self.model, "_depth"):
            raise ValueError(
                "The model should have a `_depth` attribute which is the number of hidden layers."
            )

        self._depth = sum(available_heads)
        self._available_heads = available_heads
        self._additional_heads = additional_heads
        self._do_not_optimize_depth_weights = do_not_optimize_depth_weights

        self.model.add_method_specific_layers(
            method="mimmo",
            heads=additional_heads,
            num_members=self._num_members,
            available_heads=available_heads[:-1],
        )

        self._loss.set_reduction_per_member("sum")
        self._loss.set_reduction_per_sample("sum")
        logging.warning(
            f"The reduction per member is set to {self._loss._reduction_per_member} and the reduction per sample is set to {self._loss._reduction_per_sample}."
        )

        # Add the alpha and beta learnable parameters to the model
        if hasattr(self.model, "_depth_weights"):
            raise ValueError(
                "The model should not have a `_depth_weights` attribute.")
        if hasattr(self.model, "_prior_depth_weights"):
            raise ValueError(
                "The model should not have a `_prior_depth_weights` attribute."
            )

        assert alpha >= 0.0, f"The alpha value needs to be positive, got {alpha}."
        self._alpha = alpha

        assert (
            warm_starting_epochs >= 0
        ), f"Warm starting epochs should be non-negative, got {warm_starting_epochs}."
        self._warm_starting_epochs = warm_starting_epochs

        self.model._depth_weights = torch.nn.Parameter(
            torch.zeros((self._depth, self._num_members)), requires_grad=True
        )
        disable_regularizer(self.model._depth_weights)

        if prior == "uniform":
            self.model.register_buffer(
                "_prior_depth_weights",
                torch.ones_like(self.model._depth_weights) / self._depth,
            )
        else:
            raise NotImplementedError(
                f"Prior {prior} not implemented. Only `uniform` and `early` are supported."
            )

        self._depth_weight_container = (
            self.model._prior_depth_weights.clone().unsqueeze(0)
        )
        self.model._available_heads = self._available_heads
        self.model._additional_heads = self._additional_heads

        self.model = MIMMMOWrapper(
            self.model, evaluation_depth_weights_function=self._evaluation_depth_weights
        )

    def _create_metrics(self, metrics_kwargs: Dict[str, Any]) -> None:
        """This method creates the metrics for the method."""
        super()._create_metrics(metrics_kwargs)
        self._add_additional_metrics(
            {
                f"{LOSS_KEY}_kl_depth": torchmetrics.MeanMetric(),
                f"{LOSS_KEY}_individual": torchmetrics.MeanMetric(),
            },
            tendencies=[MIN_TENDENCY, MIN_TENDENCY],
        )

    @property
    def _prior_depth_weights(self) -> torch.Tensor:
        """This method returns the prior depth weights."""
        return self.model._prior_depth_weights

    def _train_depth_weights(self) -> torch.Tensor:
        """This method returns the depth weights in the step function."""
        if (
            self.current_epoch < self._warm_starting_epochs and self.training
        ) or self._do_not_optimize_depth_weights:
            return self._prior_depth_weights
        return self._evaluation_depth_weights()

    def _evaluation_depth_weights(self) -> torch.Tensor:
        """This method returns the depth weights in the step function."""
        return torch.softmax(self.model._depth_weights, dim=0)

    @property
    def _depth_weights(self) -> torch.Tensor:
        """This method returns the depth weights in the step function."""
        if self.training:
            return self._train_depth_weights()
        return self._evaluation_depth_weights()

    def _predict(self, x: torch.Tensor, **forward_kwargs: Any) -> torch.Tensor:
        """This method is used to perform a forward pass of the model.

        It is done with respect to the number of hidden layers or how hidden layers are being defined
        in the underlying `self.model`.
        """
        return self.model(x, **forward_kwargs)

    def _loss_per_depth_per_member(
        self, y_hat: torch.Tensor, y: torch.Tensor, depth_weights: torch.Tensor
    ) -> torch.Tensor:
        """This method computes the loss per depth per member."""
        loss = 0.0
        for i in range(self._depth):
            for j in range(self._num_members):
                # Compute the loss for each depth and member
                # Add the loss to the total loss
                loss += (
                    self._loss(y_hat[:, i, j].unsqueeze(1), y[:, i, j])
                    * depth_weights[i, j]
                )
        return loss

    def _loss_kl_depth_weights(self) -> torch.Tensor:
        """This method computes the KL divergence loss for the depth weights."""
        if (
            self.current_epoch < self._warm_starting_epochs
            or self._do_not_optimize_depth_weights
        ):
            return torch.tensor(0.0, device=self.device)
        depth_weights = torch.softmax(self.model._depth_weights, dim=0)
        return torch.sum(
            depth_weights
            * torch.log(
                depth_weights / (self._prior_depth_weights + TINY_EPSILON)
                + TINY_EPSILON
            )
        )

    def _step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
        phase: str = TRAIN_KEY,
        num_members: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """A helper method to perform a single step whether training, validation or test."""
        output = {}
        x, y = batch
        # N_train = self._datamodule.train_dataset_size(split=optimizer_idx)
        batch_size = x.shape[0]
        # Repeat the labels for the depth dimension
        # The labels have the shape `(batch_size, depth, num_members)`
        # The predictions have the shape `(batch_size, depth, num_members, predictions)`
        y_depth = torch.stack([y] * self._depth, dim=1)
        y_hat = self._predict(x)

        # Construct the weights for the loss
        # The depth weights have the shape `(depth, num_members)`
        depth_weights = self._depth_weights

        # loss = -(N_train/batch_size) * self._loss_per_depth_per_member(
        #    y_hat, y_depth, depth_weights)
        loss = -(1 / batch_size) * self._loss_per_depth_per_member(
            y_hat, y_depth, depth_weights
        )
        loss_kl_depth = self._loss_kl_depth_weights()

        # output[f"{LOSS_KEY}_individual"] = -loss.detach()/N_train
        output[f"{LOSS_KEY}_individual"] = -loss.detach()
        # output[f"{LOSS_KEY}_kl_depth"] = (
        #    self._alpha * loss_kl_depth).detach()/N_train
        output[f"{LOSS_KEY}_kl_depth"] = (
            self._alpha * loss_kl_depth
        ).detach()

        loss = loss - self._alpha * loss_kl_depth
        # loss = -loss/N_train
        loss = -loss

        # Weight the predictions per depth and member and then sum across the depth dimension
        depth_weights = depth_weights.unsqueeze(0)
        while len(depth_weights.shape) < len(y_hat.shape):
            depth_weights = depth_weights.unsqueeze(-1)
        # Normalize the weights per depth to sum to 1
        depth_weights = depth_weights / \
            torch.sum(depth_weights, dim=1, keepdim=True)

        # Divide the depth weights by the number of members
        depth_weights = depth_weights / self._num_members

        # Reshape the depth weights and predictions to have the shape `(batch_size, depth*num_members, predictions)`
        depth_weights = depth_weights.reshape(
            1, depth_weights.shape[1] * depth_weights.shape[2]
        )
        y_hat = y_hat.reshape(
            y_hat.shape[0], y_hat.shape[1] * y_hat.shape[2], *y_hat.shape[3:]
        )
        y_depth = y_depth.reshape(
            y_depth.shape[0], y_depth.shape[1] *
            y_depth.shape[2], *y_depth.shape[3:]
        )

        # We are in training and the input has not been repeated.
        y_permember = y_depth
        y_hat_permember = y_hat
        if self.training:
            y_hat = y_hat.reshape(-1, 1, *y_hat.shape[2:])
            y = y_depth.reshape(-1, 1, *y_depth.shape[2:])
            x = x.reshape(-1, *x.shape[2:])
            average_weights = None
        else:
            y = y[:, 0]  # They are all the same.
            x = x[:, 0]  # They are all the same.
            # Repeat the average weights for all samples in batch
            average_weights = depth_weights.repeat(batch_size, 1)
        y = y.reshape(-1, *y.shape[2:])

        y_hat_mean = average_predictions(
            y_hat, self._task, weights=average_weights)
        output.update(
            {
                LOSS_KEY: loss,
                PREDICTION_KEY: y_hat.detach(),
                TARGET_KEY: y.detach(),
                TARGET_PER_MEMBER_KEY: y_permember.detach(),
                PREDICTION_PER_MEMBER_KEY: y_hat_permember.detach(),
                MEAN_PREDICTION_KEY: y_hat_mean.detach(),
                INPUT_KEY: x.detach(),
                AVERAGE_WEIGHTS_KEY: average_weights.detach()
                if average_weights is not None
                else None,
            }
        )
        return output

    def _plots(self) -> None:
        super()._plots()
        plot_weight_trajectories(self._depth_weight_container, self._save_path)

    def on_train_epoch_end(self) -> None:
        """A helper method to log the weights of the ensemble."""
        depth_weights = self._evaluation_depth_weights()
        logging.info(
            f"Depth weights at the end of training epoch: {depth_weights}")
        self._depth_weight_container = torch.cat(
            (self._depth_weight_container, depth_weights.unsqueeze(0).detach().cpu()),
            dim=0,
        )
        super().on_train_epoch_end()

    def state_dict(self) -> Dict[str, Any]:
        """A helper method to save the weights of the ensemble."""
        state_dict = super().state_dict()
        state_dict["depth_weight_container"] = self._depth_weight_container
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """A helper method to load the weights of the ensemble."""
        super().load_state_dict(state_dict)
        self._depth_weight_container = state_dict["depth_weight_container"]

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """The method to add additional arguments."""
        parser = super(MIMMOMethod, MIMMOMethod).add_specific_args(
            parent_parser)
        parser.add_argument(
            "--method_alpha",
            type=float,
            default=1.0,
            help="The alpha value.",
        )
        parser.add_argument(
            "--method_prior",
            type=str,
            default="uniform",
            help="The prior for the depth weights.",
        )
        parser.add_argument(
            "--method_additional_heads",
            type=int,
            default=1,
            choices=[0, 1],
            help="Whether to use a single head or a head per depth.",
        )
        parser.add_argument(
            "--method_do_not_optimize_depth_weights",
            type=int,
            default=0,
            choices=[0, 1],
            help="Whether to optimize the depth weights.",
        )
        parser.add_argument(
            "--method_available_heads",
            type=str,
            default=None,
            help="The available depth heads for the method.",
        )
        parser.add_argument(
            "--method_warm_starting_epochs",
            type=int,
            default=0,
            help="The number of epochs to warm start the model.",
        )
        return parser


class MixMoMethod(MIMOMethod):
    """This is a module which applies the MixMo regularization to a model.

    As proposed in: "MixMo: Mixing Multiple Inputs for Multiple Outputs via Deep Subnetworks"

    Show that binary mixing in features - particularly with rectangular patches from CutMix -
    enhances results by making subnetworks stronger and more diverse.

    Args:
        alpha (float): The alpha parameter for the Dirichlet distribution.
        r (float): The `r` parameter applied to the weighting factor.
        initial_p (float): Initial probability of applying linear or cutmix augmentation.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        r: float = 0.5,
        initial_p: float = 0.5,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert (
            0 < alpha
        ), f"The alpha parameter needs to be greater than 0. Got {alpha}."
        assert 0 < r, f"The r parameter needs to be greater than 0. Got {r}."
        assert (
            0 <= initial_p <= 1
        ), f"The p parameter needs to be between 0 and 1. Got {initial_p}."
        self._alpha = alpha
        self._r = r
        self._p = initial_p
        self._initial_p = initial_p

        self._dirichlet = torch.distributions.dirichlet.Dirichlet(
            torch.tensor([alpha] * self._num_members)
        )

    def _replace_input_layer(self, num_members: int) -> None:
        # Replace the first layer with one where the input dimension is multiplied by the number of
        # members.
        if self._input_layer_type == nn.Linear:
            self.model._input = ParallelModel(
                [
                    torch.nn.Linear(
                        **self._input_layer_kwargs,
                    )
                    for _ in range(num_members)
                ]
            )

        elif self._input_layer_type == nn.Conv2d:
            self.model._input = ParallelModel(
                [
                    torch.nn.Conv2d(
                        **self._input_layer_kwargs,
                    )
                    for _ in range(num_members)
                ]
            )
        else:
            raise ValueError(
                f"The first layer of the model should be either a `torch.nn.Linear` or a \
                `torch.nn.Conv2d` layer, but it is a {type(self.model._input)}."
            )

        self.model._input = MixMoBlock(num_members, self.model._input)

    def _plots(self) -> None:
        output_weights_per_member = []
        input_weights_per_member = []
        for member in range(self._num_members):
            output_weights_per_member.append(
                self.model._output[0].weight.data[
                    member
                    * self._output_outputs_dim: (member + 1)
                    * self._output_outputs_dim
                ]
            )
            input_weights_per_member.append(
                self.model._input._input[member].weight.data
            )

        input_overlap = plot_input_layer_norm_bar(
            input_weights_per_member, self._save_path, self.current_epoch
        )
        output_overlap = plot_output_layer_norm_bar(
            output_weights_per_member, self._save_path, self.current_epoch
        )

        self._input_layer_overlap_container = torch.cat(
            (
                self._input_layer_overlap_container,
                input_overlap.unsqueeze(0).unsqueeze(0).detach().cpu(),
            ),
            dim=0,
        )

        self._output_layer_overlap_container = torch.cat(
            (
                self._output_layer_overlap_container,
                output_overlap.unsqueeze(0).unsqueeze(0).detach().cpu(),
            ),
            dim=0,
        )
        plot_overlap_between_members(
            self._input_layer_overlap_container, self._save_path, input=True
        )
        plot_overlap_between_members(
            self._output_layer_overlap_container, self._save_path, input=False
        )

    def on_train_epoch_start(self) -> None:
        """A method which is called at the start of the training epoch."""
        if self.current_epoch > (11 / 12) * self.trainer.max_epochs:
            self._p = (
                self._initial_p
                * (self.trainer.max_epochs - self.current_epoch)
                / (self.trainer.max_epochs * (1 / 12))
            )
        else:
            self._p = self._initial_p
        self.log("p", self._p, on_epoch=True, on_step=False)
        super().on_train_epoch_start()

    def _step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
        phase: str = TRAIN_KEY,
        num_members: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """A helper method to perform a single step whether training, validation or test."""
        x, y = batch

        K = None
        w_r = None
        # Sample K
        if self.training:
            K = self._dirichlet.sample((x.shape[0],)).to(x.device)

        input_kwargs = {"K": K, "p": self._p}
        y_hat = self._predict(x, input_kwargs=input_kwargs, unsqueeze=False)

        if K is not None:
            w_r = num_members * (
                (K) ** (1 / self._r)
                / torch.sum(K ** (1 / self._r) + TINY_EPSILON, dim=1, keepdim=True)
            )
        y_permember = y
        y_hat_permember = y_hat
        loss = self._loss_per_member(y_hat, y, weights_per_sample=w_r)
        # We are in training and the input has not been repeated.
        if self.training:
            y_hat = y_hat.reshape(-1, 1, *y_hat.shape[2:])
            x = x.reshape(-1, 1, *x.shape[2:])
        else:
            y = y[:, 0]  # They are all the same.
            x = x[:, 0]  # They are all the same.
        y = y.reshape(-1)
        y_hat_mean = average_predictions(y_hat, self._task)
        output = {
            LOSS_KEY: loss,
            PREDICTION_KEY: y_hat.detach(),
            TARGET_KEY: y.detach(),
            TARGET_PER_MEMBER_KEY: y_permember.detach(),
            PREDICTION_PER_MEMBER_KEY: y_hat_permember.detach(),
            MEAN_PREDICTION_KEY: y_hat_mean.detach(),
            INPUT_KEY: x.detach(),
        }
        if self.training:
            output["new_batch"] = batch
        return output

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method adds the specific arguments for the MixMo method."""
        parser = super(MixMoMethod, MixMoMethod).add_specific_args(
            parent_parser)
        parser.add_argument(
            "--method_alpha",
            type=float,
            default=2.0,
            help="The alpha parameter for the MixMo regularization.",
        )
        parser.add_argument(
            "--method_r",
            type=float,
            default=3,
            help="The r parameter for the MixMo regularization.",
        )
        parser.add_argument(
            "--method_initial_p",
            type=float,
            default=0.5,
            help="The initial value for the p parameter.",
        )
        return parser


class MixVitMethod(MIMOMethod):
    """This is a module which applies MixToken augmentation to a vision transformer.

    Args:
        depth (int): The depth at which to add the source attribution.
    """

    def __init__(self, depth: int, *args: Any, **kwargs: Any) -> None:
        logging.warning(
            "MixVitMethod is only supported for 2 members. It is not trivial to extend it to more members."
        )
        kwargs["num_members"] = 2
        super().__init__(*args, **kwargs)
        self._input_layer_overlap_container = None
        self._depth = depth
        self.model = MixVitWrapper(self.model, self._depth)

        self._input_layer_overlap_container = None

    def _post_init(self) -> None:
        pass

    def _plots(self) -> None:
        output_weights_per_member = []
        for member in range(self._num_members):
            output_weights_per_member.append(
                self.model._vit._output.weight.data[
                    member
                    * self._output_outputs_dim: (member + 1)
                    * self._output_outputs_dim
                ]
            )

        output_overlap = plot_output_layer_norm_bar(
            output_weights_per_member, self._save_path, self.current_epoch
        )

        self._output_layer_overlap_container = torch.cat(
            (
                self._output_layer_overlap_container,
                output_overlap.unsqueeze(0).unsqueeze(0).detach().cpu(),
            ),
            dim=0,
        )

        plot_overlap_between_members(
            self._output_layer_overlap_container, self._save_path, input=False
        )

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method adds the specific arguments for the MixMo method."""
        parser = super(MixVitMethod, MixVitMethod).add_specific_args(
            parent_parser)
        parser.add_argument(
            "--method_depth",
            type=int,
            default=2,
            help="The depth at which to add the source attribution.",
        )
        return parser


class UnMixMoMethod(MixMoMethod):
    """This is a module which applies the unmixing regularization to the model in addition to MixMo.

    The key concept is that instead of processing the inputs just through a convolutional or a linear layer,
    each input is mixed and then unmixed selectively depending on the mixing mask.

    Precisely it focuses on `fadeout` unmixing which has demonstrated to be more effective than
    full unmixing. The `fadeout` unmixing is a gradual unmixing of the inputs.

    Args:
        m_start_value (Optional[float]): The initial value for the `m` parameter. Defaults to 0.
        m_end_value (Optional[float]): The final value for the `m` parameter. Defaults to 1.
        m_start_epoch (Optional[int]): The epoch at which the `m` parameter starts to increase. Defaults to 0.
        m_end_epoch (Optional[int]): The epoch at which the `m` parameter reaches its final value. Defaults to 100.
    """

    def __init__(
        self,
        m_start_value: float = 0,
        m_end_value: float = 1,
        m_start_epoch: int = 0,
        m_end_epoch: int = 100,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        assert (
            kwargs["num_members"] == 2
        ), "The number of members should be 2 for the UnMixMo method."
        kwargs["initial_p"] = 1.0
        logging.warning(
            "The UnMixMo method uses probability 1.0 for cutmix mixing, which is constant throughout training."
        )
        super().__init__(*args, **kwargs)
        assert (
            m_start_value <= m_end_value
        ), f"The start value {m_start_value} should be less than or equal to the end value {m_end_value}."
        self._m_scheduler = LinearScalarScheduler(
            start_value=m_start_value,
            end_value=m_end_value,
            start_epoch=m_start_epoch,
            end_epoch=m_end_epoch,
        )

    def _replace_output_layer(self, num_members: int) -> None:
        """Replace the last layer with one where the output dimension is multiplied by the number of members."""
        output_layer: nn.Module = None
        if self._output_layer_type == nn.Linear:
            output_layer = UnmixingBlock(
                mixmo_block=self.model._input,
                in_features=self._output_layer_kwargs["in_features"],
                out_features=self._output_layer_kwargs["out_features"],
                num_members=num_members,
                outputs_dim=1,
            )
            # If the last two layers in `model._layers` are `nn.AdaptiveAvgPool2d` and `nn.Flatten`, then remove them.
            if (
                len(self.model._layers) >= 2
                and isinstance(self.model._layers[-1], nn.Flatten)
                and isinstance(self.model._layers[-2], nn.AdaptiveAvgPool2d)
            ):
                self.model._layers = self.model._layers[:-2]
        else:
            raise ValueError(
                f"The last layer of the model should be a `torch.nn.Linear` layer, but it is a \
                {self._output_layer_type}."
            )
        self.model._output = output_layer.to(
            next(self.model.parameters()).device)

        self.model._output_activation = OutputActivation(
            self.model._output_activation._task, dim=2
        )

    def _plots(self) -> None:
        output_weights_per_member = []
        input_weights_per_member = []
        for member in range(self._num_members):
            output_weights_per_member.append(
                self.model._output._output[member].weight.data
            )
            input_weights_per_member.append(
                self.model._input._input[member].weight.data
            )

        input_overlap = plot_input_layer_norm_bar(
            input_weights_per_member, self._save_path, self.current_epoch
        )
        output_overlap = plot_output_layer_norm_bar(
            output_weights_per_member, self._save_path, self.current_epoch
        )

        self._input_layer_overlap_container = torch.cat(
            (
                self._input_layer_overlap_container,
                input_overlap.unsqueeze(0).unsqueeze(0).detach().cpu(),
            ),
            dim=0,
        )

        self._output_layer_overlap_container = torch.cat(
            (
                self._output_layer_overlap_container,
                output_overlap.unsqueeze(0).unsqueeze(0).detach().cpu(),
            ),
            dim=0,
        )
        plot_overlap_between_members(
            self._input_layer_overlap_container, self._save_path, input=True
        )
        plot_overlap_between_members(
            self._output_layer_overlap_container, self._save_path, input=False
        )

    def on_train_epoch_start(self) -> None:
        """A method which is called at the start of the training epoch."""
        self.model._output.set_m(self._m_scheduler.get_value())
        # Use the grandparent class to avoid calling the MixMoMethod on_train_epoch_start method.
        super(MixMoMethod, self).on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        """A method which is called at the end of the training epoch."""
        # Use the grandparent class to avoid calling the MixMoMethod on_train_epoch_end method.
        self._m_scheduler.step()
        self.log("m", self._m_scheduler.get_value(),
                 on_epoch=True, on_step=False)
        super(MixMoMethod, self).on_train_epoch_end()

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict["m_scheduler"] = self._m_scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._m_scheduler.load_state_dict(state_dict["m_scheduler"])

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method adds the specific arguments for the UnMixMo method."""
        parser = super(UnMixMoMethod, UnMixMoMethod).add_specific_args(
            parent_parser)
        parser.add_argument(
            "--method_m_start_value",
            type=float,
            default=0,
            help="The initial value for the m parameter.",
        )
        parser.add_argument(
            "--method_m_end_value",
            type=float,
            default=1,
            help="The final value for the m parameter.",
        )
        parser.add_argument(
            "--method_m_start_epoch",
            type=int,
            default=0,
            help="The epoch at which the m parameter starts to increase.",
        )
        parser.add_argument(
            "--method_m_end_epoch",
            type=int,
            default=100,
            help="The epoch at which the m parameter reaches its final value.",
        )
        return parser


class DataMUXMethod(MIMOMethod):
    """This is a module which applies the DataMUX multiplexing and demultiplexing to the input and output of the model.

    As proposed in: DataMUX: Data Multiplexing for Neural Networks.

    The key concept is that instead of processing the inputs just through a convolutional or a linear layer,
    each input has a dedicated encoder and the output is processed through a separate decoder to give predictions.

    Args:
        coder_expansion_factor (int): The expansion factor for the coder. Defaults to 1.
        coder_depth (int): The depth of the coder. Defaults to 1.
    """

    def __init__(
        self,
        coder_expansion_factor: int = 1,
        coder_depth: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._coder_expansion_factor = coder_expansion_factor
        self._coder_depth = coder_depth
        super().__init__(*args, **kwargs)

    def _replace_input_layer(self, num_members: int) -> None:
        """A helper method to replace the input layer with a DataMUXInput layer."""
        coder: List[nn.Module] = None
        inputs_dim = self._input_inputs_dim
        input_layer: nn.Module = None

        if self._input_layer_type == nn.Linear:
            coder = [
                LinearExtractor(
                    inputs_dim=inputs_dim,
                    expansion_factor=self._coder_expansion_factor,
                    depth=self._coder_depth,
                    norm=True,
                    outputs_dim=int(inputs_dim * self._coder_expansion_factor),
                    end_activation=True,
                    activation="ReLU",
                    end_normalization=True,
                )
                for _ in range(num_members)
            ]
            inputs_dim = int(
                inputs_dim * self._coder_expansion_factor * num_members)

            input_layer_kwargs = copy.deepcopy(self._input_layer_kwargs)
            input_layer_kwargs["in_features"] = inputs_dim
            input_layer = nn.Linear(**input_layer_kwargs)
            # If the model has _flatten layer, we need to remove it.
            # We assume that the _flatten layer is the first layer before the input layer.
            if hasattr(self.model, "_flatten"):
                self.model._flatten = nn.Identity()

        elif self._input_layer_type == nn.Conv2d:
            coder = [
                Conv2dExtractor(
                    input_channels=inputs_dim,
                    expansion_factor=self._coder_expansion_factor,
                    depth=self._coder_depth,
                    norm=True,
                    output_channels=int(
                        inputs_dim * self._coder_expansion_factor),
                    convolution="conv2d",
                    end_activation=True,
                    activation="ReLU",
                    end_normalization=True,
                )
                for _ in range(num_members)
            ]

            inputs_dim = int(
                inputs_dim * self._coder_expansion_factor * num_members)

            input_layer_kwargs = copy.deepcopy(self._input_layer_kwargs)
            input_layer_kwargs["in_channels"] = inputs_dim
            input_layer = nn.Conv2d(**input_layer_kwargs)
        else:
            raise ValueError(
                f"The input layer should be either a Linear or a Conv2d layer. Got {self.model._input}."
            )

        multiplexer = Multiplexer(
            precoder=None,
            coder=coder,
            postcoder=None,
            reduction="cat",
            reduction_normalization=None,
            feature_regularizer=None,
            inputs_dim=1,
            outputs_dim=1,
        )

        self.model._input = nn.Sequential(multiplexer, input_layer).to(
            next(self.model.parameters()).device
        )

    def _replace_output_layer(self, num_members: int) -> None:
        """A helper method to replace the output layer with a DataMUXOutput layer."""
        demultiplexer: nn.Module = None
        if self._output_layer_type == nn.Linear:
            demultiplexer = Demultiplexer(
                parallel_layers=[
                    LinearExtractor(
                        inputs_dim=self._output_inputs_dim,
                        expansion_factor=1,
                        depth=1,
                        norm=False,
                        end_normalization=False,
                        end_activation=False,
                        activation="ReLU",
                        outputs_dim=self._output_outputs_dim,
                    )
                    for _ in range(num_members)
                ],
                outputs_dim=1,
            )
        else:
            raise ValueError(
                f"The output layer should be a Linear layer. Got {self.model._output}."
            )

        self.model._output = demultiplexer.to(
            next(self.model.parameters()).device)

        self.model._output_activation = OutputActivation(
            self.model._output_activation._task, dim=2
        )

    def _plots(self) -> None:
        output_weights_per_member = []
        input_weights_per_member = []
        for member in range(self._num_members):
            output_weights_per_member.append(
                self.model._output._parallel_layers[member]._model[0].weight.data
            )
            input_weights_per_member.append(
                self.model._input[1].weight.data[
                    :,
                    member
                    * self._input_inputs_dim: (member + 1)
                    * self._input_inputs_dim,
                ]
            )

        input_overlap = plot_input_layer_norm_bar(
            input_weights_per_member, self._save_path, self.current_epoch
        )
        output_overlap = plot_output_layer_norm_bar(
            output_weights_per_member, self._save_path, self.current_epoch
        )

        self._input_layer_overlap_container = torch.cat(
            (
                self._input_layer_overlap_container,
                input_overlap.unsqueeze(0).unsqueeze(0).detach().cpu(),
            ),
            dim=0,
        )

        self._output_layer_overlap_container = torch.cat(
            (
                self._output_layer_overlap_container,
                output_overlap.unsqueeze(0).unsqueeze(0).detach().cpu(),
            ),
            dim=0,
        )
        plot_overlap_between_members(
            self._input_layer_overlap_container, self._save_path, input=True
        )
        plot_overlap_between_members(
            self._output_layer_overlap_container, self._save_path, input=False
        )

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Adds the specific arguments for the DataMUXMethod."""
        parser = super(DataMUXMethod, DataMUXMethod).add_specific_args(
            parent_parser)
        parser.add_argument(
            "--method_coder_expansion_factor",
            type=int,
            default=1,
            help="The expansion factor for the coder. Defaults to 1.",
        )
        parser.add_argument(
            "--method_coder_depth",
            type=int,
            default=1,
            help="The depth of the coder. Defaults to 1.",
        )
        return parser
