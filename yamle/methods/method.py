import argparse
import copy
import logging
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim.optimizer import Optimizer

from yamle.data.datamodule import BaseDataModule
from yamle.defaults import (
    CLASSIFICATION_KEY,
    INPUT_KEY,
    LOSS_KEY,
    LOSS_REGULARIZER_KEY,
    MEAN_PREDICTION_KEY,
    MEMBERS_DIM,
    PREDICTION_KEY,
    SUPPORTED_TASKS,
    TARGET_KEY,
    TEST_KEY,
    TRAIN_KEY,
    VALIDATION_KEY,
)
from yamle.evaluation.metrics.algorithmic import (
    metrics_factory,
    METRIC_TENDENCY,
    MAX_TENDENCY,
    MIN_TENDENCY,
    METRICS_TO_DESCRIPTION,
)
from yamle.regularizers.regularizer import BaseRegularizer
from yamle.utils.operation_utils import average_predictions
from yamle.utils.optimization_utils import (
    get_optimizer,
    get_scheduler,
    recover_frozen_weights,
    split_optimizer_parameters,
)
from yamle.utils.file_utils import save_pickle, predictions_file

logging = logging.getLogger("pytorch_lightning")


class BaseMethod(LightningModule):
    """This class is the base class for all methods in the project.

    It assumes that the output of the model has a shape `(batch_size, 1, num_classes)` for training.
    This corresponds to a single Monte Carlo sample.

    Args:
        model (nn.Module): The model to be trained.
        loss (nn.Module): The loss function to be used.
        regularizer (BaseRegularizer): The regularizer to be used.
        learning_rate (float): The learning rate to be used for training.
        regularizer_weight (float): The weight of the regularizer.
        momentum (float): The momentum to be used for training.
        task (str): The task to be performed. Can be either `classification` or `regression`.
        optimizer (str): The optimizer to be used for training. Can be either `adam` or `sgd`.
        scheduler (str): The learning rate scheduler to be used for training.
        scheduler_step_size (int): The step size to be used for the learning rate scheduler.
        scheduler_gamma (float): The gamma to be used for the learning rate scheduler.
        scheduler_factor (float): The factor to be used for the learning rate scheduler.
        scheduler_patience (int): The patience to be used for the learning rate scheduler.
        seed (int): The seed to be used for training.
        inputs_dim (Tuple[int, ...]): The shape of the inputs to the model.
        inputs_dtype (torch.dtype): The dtype of the inputs to the model.
        outputs_dim (int): The number of outputs of the model.
        targets_dim (int): The feature dimension of the targets.
        outputs_dtype (torch.dtype): The dtype of the outputs of the model.
        datamodule (Optional[BaseDataModule]): The datamodule to be used for training or testing.
        plotting_training (bool): Whether to plot sanity checks or not during training.
        plotting_testing (bool): Whether to plot sanity checks or not during testing.
        save_path (Optional[str]): The path to save files to.
        save_test_predictions (bool): Whether to save the test predictions or not.
        metrics_kwargs (Dict[str, Any]): The keyword arguments to be passed to the metrics.
        model_kwargs (Dict[str, Any]): The keyword arguments to be passed to the model.
    """

    tasks = SUPPORTED_TASKS

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        regularizer: BaseRegularizer,
        learning_rate: float = 1e-3,
        regularizer_weight: float = 0.0,
        momentum: float = 0.9,
        task: str = CLASSIFICATION_KEY,
        optimizer: str = "adam",
        scheduler: str = "step",
        scheduler_step_size: int = 10,
        scheduler_gamma: float = 0.1,
        scheduler_factor: float = 0.1,
        scheduler_patience: int = 10,
        seed: int = 42,
        inputs_dim: Tuple[int, ...] = (1, 28, 28),
        inputs_dtype: torch.dtype = torch.float32,
        outputs_dim: int = 10,
        targets_dim: int = 1,
        outputs_dtype: torch.dtype = torch.float32,
        datamodule: Optional[BaseDataModule] = None,
        plotting_training: bool = False,
        plotting_testing: bool = False,
        save_path: Optional[str] = None,
        save_test_predictions: bool = False,
        metrics: Optional[List[str]] = None,
        metrics_kwargs: Dict[str, Any] = {},
        model_kwargs: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> None:
        super(BaseMethod, self).__init__()
        assert (
            task in self.tasks
        ), f"Task {task} is not supported, supported tasks are {self.tasks}"
        self.save_hyperparameters(
            ignore=[
                "model",
                "loss",
                "regularizer",
                "task",
                "inputs_dim",
                "outputs_dim",
                "targets_dim",
                "datamodule",
                "plotting_training",
                "plotting_testing",
                "save_test_predictions",
                "save_path",
                "metrics_kwargs",
                "inputs_dtype",
                "outputs_dtype",
                "metrics",
                "model_kwargs",
                "seed",
            ]
        )
        self.model = model
        self._loss = loss
        self._regularizer = regularizer
        self._task = task
        self._inputs_dim = inputs_dim
        self._inputs_dtype = inputs_dtype
        self._outputs_dim = outputs_dim
        self._targets_dim = targets_dim
        self._outputs_dtype = outputs_dtype
        self._seed = seed
        self._datamodule = datamodule
        self._plotting_training = plotting_training
        self._plotting_testing = plotting_testing
        self._save_path = save_path
        self._debug = False
        self._metrics = metrics
        self._create_metrics(metrics_kwargs)
        self._fit_counter = 0
        self._test_counter = 0
        self._training_step_counter = 0

        self.training_step_exception = False
        self.validation_step_exception = False
        self.test_step_exception = False
        self._model_kwargs = model_kwargs

        self._check_overriding()

        self._save_test_predictions = save_test_predictions
        self._save_cache = None

        self.test_name: Optional[str] = None  # The name of the current test set

    def _create_metrics(self, metrics_kwargs: Dict[str, Any]) -> None:
        """This method is used to create the metrics to be used for training, validation and testing."""
        self.metrics = {
            TRAIN_KEY: metrics_factory(**metrics_kwargs),
            VALIDATION_KEY: metrics_factory(**metrics_kwargs),
            TEST_KEY: metrics_factory(**metrics_kwargs),
        }

    def _add_additional_metrics(
        self,
        metrics: Dict[str, torchmetrics.Metric],
        data: Optional[List[str]] = None,
        tendencies: Optional[List[str]] = None,
        descriptions: Optional[List[str]] = None,
    ) -> None:
        """This method is used to add additional metrics to the metrics dictionary.

        It works only for metrics which are not per member.
        The metrics are deep copied to avoid any issues with the metrics.

        Args:
            metrics (Dict[str, torchmetrics.Metric]): The metrics to be added.
            data (Optional[List[str]]): The data to which the metrics should be added. Defaults to None.
                The default value means that the metrics are added to all data.
            tendencies (Optional[List[str]]): The tendencies of the metrics. Defaults to MIN_TENDENCY if not provided.
            descriptions (Optional[List[str]]): The descriptions of the metrics. Defaults to "" if not provided.
        """
        if data is None:
            data = [TRAIN_KEY, VALIDATION_KEY, TEST_KEY]
        for key in data:
            self.metrics[key].update(copy.deepcopy(metrics))
        if tendencies is not None:
            assert len(tendencies) == len(
                metrics
            ), "The number of tendencies must be equal to the number of metrics"
            assert all(
                [t in [MAX_TENDENCY, MIN_TENDENCY] for t in tendencies]
            ), "The tendencies must be one of max or min"
        else:
            tendencies = [MIN_TENDENCY for _ in range(len(metrics))]
        for i, metric_name in enumerate(metrics.keys()):
            if metric_name in METRIC_TENDENCY.keys():
                logging.warning(
                    f"Metric {metric_name} has a default tendency of {METRIC_TENDENCY[metric_name]}, rewriting to {tendencies[i]}"
                )
            METRIC_TENDENCY[metric_name] = tendencies[i]
        if descriptions is not None:
            assert len(descriptions) == len(
                metrics
            ), "The number of descriptions must be equal to the number of metrics"
        else:
            descriptions = ["" for _ in range(len(metrics))]
        for i, metric_name in enumerate(metrics.keys()):
            if metric_name in METRICS_TO_DESCRIPTION.keys():
                logging.warning(
                    f"Metric {metric_name} has a default description of {METRICS_TO_DESCRIPTION[metric_name]}, rewriting to {descriptions[i]}"
                )
            METRICS_TO_DESCRIPTION[metric_name] = descriptions[i]

    def _metrics_to_device(self, device: torch.device) -> None:
        """This method is used to move the metrics to the correct device."""
        self.metrics[TRAIN_KEY].to(device)
        self.metrics[VALIDATION_KEY].to(device)
        self.metrics[TEST_KEY].to(device)

    def _on_fit_or_test_start(self) -> None:
        """A method that is called when training or testing starts."""
        self._metrics_to_device(self.device)
        self.reset_metrics(TRAIN_KEY, complete=True)
        self.reset_metrics(VALIDATION_KEY, complete=True)
        self.reset_metrics(TEST_KEY, complete=True)
        self.model.reset()

    def on_fit_start(self) -> None:
        """This method is used to set the metrics to the correct device."""
        super(BaseMethod, self).on_fit_start()
        self.metrics[TRAIN_KEY][LOSS_REGULARIZER_KEY] = torchmetrics.MeanMetric()
        self._on_fit_or_test_start()
        self._fit_counter += 1

    def on_test_start(self) -> None:
        """This method is used to set the metrics to the correct device."""
        super(BaseMethod, self).on_test_start()
        self._on_fit_or_test_start()
        self._test_counter += 1

    def _predict(
        self, x: torch.Tensor, unsqueeze: bool = True, **forward_kwargs: Any
    ) -> torch.Tensor:
        """This method is used to perform a forward pass of the model.

        It unsqueezes the output of the model to have a shape `(batch_size, 1, num_outputs)`.

        Args:
            x (torch.Tensor): The input to the model.
            **forward_kwargs (Any): The keyword arguments to be passed to the forward pass of the model.
        """
        output = self.model(x, **forward_kwargs)
        if unsqueeze:
            output = output.unsqueeze(MEMBERS_DIM)
        return output

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
        y_hat = self._predict(x)
        loss = self._loss(y_hat, y)
        y_hat_mean = average_predictions(y_hat, self._task)
        outputs = {}

        outputs[LOSS_KEY] = loss
        outputs[TARGET_KEY] = y.detach()
        outputs[INPUT_KEY] = x.detach()
        outputs[PREDICTION_KEY] = y_hat.detach()
        outputs[MEAN_PREDICTION_KEY] = y_hat_mean.detach()
        return outputs

    def _clean_outputs_dict(self, outputs: Dict[str, Any]) -> None:
        """This method is used to perform post processing on the output.

        Args:
            outputs (Dict[str, Any]): The output of the step method.
        """
        keys = list(outputs.keys())
        for key in keys:
            outputs.pop(key)

    def training_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """This method is used to perform a single training step.

        This method should not be overridden.
        It can catch exceptions if they are raised inside _training_step.
        """
        try:
            outputs = self._training_step(batch, batch_idx)
            self.training_step_exception = False
            return outputs
        except Exception as e:
            logging.warning(
                f"Exception {e} raised during training step. Continuing training."
            )
            self.training_step_exception = True
            return {
                LOSS_KEY: torch.tensor(0.0, device=self.device, requires_grad=False)
            }

    def _training_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """This method is used to perform a single training step.

        This method should be overridden.
        """
        return self._step(batch, batch_idx=batch_idx, phase=TRAIN_KEY)

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """This method is used to perform a single validation step.

        This method should not be overridden.
        It can catch exceptions if they are raised inside _validation_step.
        """
        try:
            outputs = self._validation_step(batch, batch_idx)
            self.validation_step_exception = False
            return outputs
        except Exception as e:
            logging.warning(
                f"Exception {e} raised during validation step. Continuing validation."
            )
            self.validation_step_exception = True
            return {
                LOSS_KEY: torch.tensor(0.0, device=self.device, requires_grad=False)
            }

    def _validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """This method is used to perform a single validation step.

        This method should be overridden.
        """
        return self._step(batch, batch_idx=batch_idx, phase=VALIDATION_KEY)

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """This method is used to perform a single test step.

        This method should not be overridden.
        It can catch exceptions if they are raised inside _test_step.
        """
        try:
            outputs = self._test_step(batch, batch_idx)
            self.test_step_exception = False
            return outputs
        except Exception as e:
            logging.warning(f"Exception {e} raised during test step. Continuing testing.")
            self.test_step_exception = True
            return {
                LOSS_KEY: torch.tensor(0.0, device=self.device, requires_grad=False)
            }

    def _test_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """This method is used to perform a single test step.

        This method should be overridden.
        """
        return self._step(batch, batch_idx=batch_idx, phase=TEST_KEY)

    def _check_overriding(self) -> None:
        """This method is called in the `__init__` method to check if the methods are overridden."""
        # Check that training_step, validation_step and test_step are not overridden
        if (
            self.training_step.__func__ != BaseMethod.training_step
            or self.validation_step.__func__ != BaseMethod.validation_step
            or self.test_step.__func__ != BaseMethod.test_step
        ):
            raise RuntimeError(
                "training_step, validation_step and test_step should not be overridden."
            )

    def on_before_backward(self, loss: torch.Tensor) -> None:
        """This method is called before the backward pass, but after the loss has been computed.

        By default regularizer term is added to the loss.
        """
        if self.training_step_exception:
            return
        super(BaseMethod, self).on_before_backward(loss)
        regularizer = self._regularizer(self.model)
        loss += self.hparams.regularizer_weight * regularizer
        self.metrics[TRAIN_KEY][LOSS_REGULARIZER_KEY].update(
            regularizer.detach().item()
        )

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        """This method is used to perform the optimizer step.

        The optimzier is not stepped if an exception is raised during the training step.
        """
        if self.training_step_exception:
            # Execute optimizer closure
            if optimizer_closure is not None:
                optimizer_closure()
            # Make sure to zero any gradients
            optimizer.zero_grad()
            return
        super(BaseMethod, self).optimizer_step(
            epoch, batch_idx, optimizer, optimizer_closure
        )

    def on_train_batch_end(
        self, outputs: Dict[str, Any], batch: List[torch.Tensor], batch_idx: int
    ) -> None:
        """This method is used to update the metrics at the end of each training batch.

        Weight decay is also performed at the end of each training batch, if it is selected.
        """
        if self.training_step_exception:
            self._clean_outputs_dict(outputs)
            return
        self._training_step_counter += 1
        super(BaseMethod, self).on_train_batch_end(outputs, batch, batch_idx)
        lightning_optimizer = self.optimizers()
        current_lr = lightning_optimizer.optimizer.param_groups[0]["lr"]
        self._regularizer.on_after_training_step(
            model=self.model,
            weight=self.hparams.regularizer_weight,
            lr=current_lr,
            step=self.global_step,
        )

        recover_frozen_weights(self.model)
        self._clean_outputs_dict(outputs)

    def on_validation_batch_end(
        self, outputs: Dict[str, Any], batch: List[torch.Tensor], batch_idx: int
    ) -> None:
        """This method is used to update the metrics at the end of each validation batch."""
        if self.validation_step_exception:
            self._clean_outputs_dict(outputs)
            return
        super(BaseMethod, self).on_validation_batch_end(outputs, batch, batch_idx)
        self._clean_outputs_dict(outputs)

    def on_test_batch_end(
        self, outputs: Dict[str, Any], batch: List[torch.Tensor], batch_idx: int
    ) -> None:
        """This method is used to update the metrics at the end of each test batch."""
        if self.test_step_exception:
            self._clean_outputs_dict(outputs)
            return
        super(BaseMethod, self).on_test_batch_end(outputs, batch, batch_idx)

        if self._save_test_predictions:
            if self._save_cache is None:
                self._save_cache = {
                    key: outputs[key].cpu().detach() for key in outputs.keys()
                }
                # Unsqueeze any 0-dim tensors
                for key in self._save_cache.keys():
                    if self._save_cache[key].ndim == 0:
                        self._save_cache[key] = self._save_cache[key].unsqueeze(0)
            else:
                for key in self._save_cache.keys():
                    value = outputs[key].cpu().detach()
                    if value.ndim == 0:
                        value = value.unsqueeze(0)
                    self._save_cache[key] = torch.cat(
                        [self._save_cache[key], value], dim=0
                    )

        self._clean_outputs_dict(outputs)

    def on_train_epoch_start(self) -> None:
        """This method is used to set the model in training mode at the beginning of each training epoch."""
        super(BaseMethod, self).on_train_epoch_start()
        self.train()

    def on_validation_epoch_start(self) -> None:
        """This method is used to set the model in evaluation mode at the beginning of each validation epoch."""
        super(BaseMethod, self).on_validation_epoch_start()
        self.eval()

    def on_test_epoch_start(self) -> None:
        """This method is used to set the model in evaluation mode at the beginning of each test epoch."""
        super(BaseMethod, self).on_test_epoch_start()
        self.eval()

    def on_train_epoch_end(self) -> None:
        """This method is used to:

        Reset the model at the end of each training epoch.
        Step the learning rate schedulers if automatic optimization is not selected.
        Plot the training results if plotting is selected.
        Apply the regularizer at the end of each training epoch if a regularizer is selected.
        """
        super(BaseMethod, self).on_train_epoch_end()
        self.model.reset()
        if not self.automatic_optimization:
            # Step learning rate schedulers,
            # in manual optimization we have to do this ourselves
            # Check if schedulers are iterable
            if isinstance(self.lr_schedulers(), Iterable):
                for scheduler in self.lr_schedulers():
                    scheduler.step()
            else:
                self.lr_schedulers().step()

        if self._plotting_training:
            self._datamodule.plot(
                self, self._save_path, f"train_epoch_{self.current_epoch}"
            )

        self._regularizer.on_after_train_epoch(
            model=self.model,
            weight=self.hparams.regularizer_weight,
            lr=self.optimizers().param_groups[0]["lr"],
            epoch=self.current_epoch,
        )

    def on_validation_epoch_end(self) -> None:
        """This method is used to reset the model at the end of each validation epoch."""
        super(BaseMethod, self).on_validation_epoch_end()
        self.model.reset()

    def on_test_epoch_end(self) -> None:
        """This method is used to reset the model at the end of each test epoch."""
        super(BaseMethod, self).on_test_epoch_end()
        self.model.reset()
        if self._plotting_testing:
            self._datamodule.plot(
                self, self._save_path, f"test_epoch_{self._test_counter}"
            )

        if self._save_test_predictions:
            save_pickle(
                self._save_cache,
                predictions_file(self._save_path, special_name=self.test_name),
            )
            self._save_cache = None

    def on_after_model_load(self) -> None:
        """This method is used after the model is loaded."""
        pass

    def on_before_model_load(self) -> None:
        """This method is used before the model is loaded."""
        pass

    def on_before_method_load(self) -> None:
        """This method is used before the method is loaded."""
        pass

    def on_after_method_load(self) -> None:
        """This method is used after the method is loaded."""
        pass

    def reset_metrics(self, prefix: str, complete: bool = False) -> None:
        """This method is used to reset the metrics.
        The metrics are not reset at the end of training, validation and testing, because they are logged externally.

        Args:
            prefix (str): The prefix of the metrics to be reset.
            complete (bool): If True, the metrics are reset completely. If False, only the values are reset.
        """
        if complete or self.current_epoch < self.trainer.max_epochs - 1:
            for metric in self.metrics[prefix].values():
                metric.reset()

    def get_parameters(self, recurse: bool = True) -> List[torch.nn.Parameter]:
        """This method is used to get the parameters of the model."""
        return list(self.model.parameters(recurse=recurse))

    def get_named_parameters(
        self, recurse: bool = True
    ) -> List[Tuple[str, torch.nn.Parameter]]:
        """This method is used to get the named parameters of the model."""
        return list(self.model.named_parameters(recurse=recurse))

    def configure_optimizers(
        self,
    ) -> Tuple[
        List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]
    ]:
        """This method is used to configure the optimizers to be used for training.

        Additionally, it is used to configure the learning rate schedulers.
        """
        assert self.hparams.optimizer in [
            "adam",
            "sgd",
            "sgld",
            "psgld",
        ], f"Optimizer {self.hparams.optimizer} not supported."
        # Ensure the model is in training mode
        self.model.train()
        parameters = self.get_parameters()
        # Split the parameters for different optimizers and schedulers
        parameters = split_optimizer_parameters(parameters)
        optimizers_and_schedulers = []
        for i, params in enumerate(parameters):
            optimizer_config = {
                "lr": self.hparams.learning_rate
                if i == 0
                else getattr(self.hparams, f"learning_rate_{i}"),
                "weight_decay": 0.0,  # Weight decay is handled by the regularizer
            }
            optimizer_name = (
                self.hparams.optimizer
                if i == 0
                else getattr(self.hparams, f"optimizer_{i}")
            )
            if (
                optimizer_name == "sgd"
                or optimizer_name == "sgld"
                or optimizer_name == "psgld"
            ):
                optimizer_config["momentum"] = (
                    self.hparams.momentum
                    if i == 0
                    else getattr(self.hparams, f"momentum_{i}")
                )
            optimizer = get_optimizer(optimizer_name, params, optimizer_config)

            scheduler_config = {}
            scheduler_name = (
                self.hparams.scheduler
                if i == 0
                else getattr(self.hparams, f"scheduler_{i}")
            )
            if scheduler_name == "plateau":
                scheduler_config["factor"] = (
                    self.hparams.scheduler_factor
                    if i == 0
                    else getattr(self.hparams, f"scheduler_factor_{i}")
                )
                scheduler_config["patience"] = (
                    self.hparams.scheduler_patience
                    if i == 0
                    else getattr(self.hparams, f"scheduler_patience_{i}")
                )
            elif scheduler_name == "exponential":
                scheduler_config["gamma"] = (
                    self.hparams.scheduler_gamma
                    if i == 0
                    else getattr(self.hparams, f"scheduler_gamma_{i}")
                )
            elif scheduler_name in ["linear", "cosine"]:
                scheduler_config["max_epochs"] = self.trainer.max_epochs
            elif scheduler_name == "step":
                scheduler_config["step_size"] = (
                    self.hparams.scheduler_step_size
                    if i == 0
                    else getattr(self.hparams, f"scheduler_step_size_{i}")
                )
                scheduler_config["gamma"] = (
                    self.hparams.scheduler_gamma
                    if i == 0
                    else getattr(self.hparams, f"scheduler_gamma_{i}")
                )
            scheduler = get_scheduler(scheduler_name, optimizer, scheduler_config)

            optimizers_and_schedulers.append(
                {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            )

        return optimizers_and_schedulers

    def state_dict(self) -> Dict[str, Any]:
        """This method is used to get the state dict of the method."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """This method is used to load the state dict of the method."""
        pass

    def analyse(self, save_path: str) -> None:
        """This method is used to analyse the method.

        The analysis is used on a trained method before the evaluation.
        Implement here any analysis that should be performed on the trained method/model.
        The `save_path` is the path to the directory where the analysis should be saved.
        """
        pass

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        """This method is used to perform the backward pass.

        Args:
            loss (torch.Tensor): The loss to be used for the backward pass.
        """
        if self.training_step_exception:
            return
        super().backward(loss, *args, **kwargs)
        # Make sure to zero any nan gradients
        for param in self.get_parameters():
            if param.grad is not None:
                param.grad[param.grad != param.grad] = 0.0

    def on_after_backward(self) -> None:
        """This method is used to perform any operation after the backward pass

        A regularizer might perform some operations after the backward pass.
        """
        super().on_after_backward()

        self._regularizer.on_after_backward(
            model=self.model, epoch=self.current_epoch, step=self.global_step
        )

    @property
    def evaluation(self):
        """This property is used to get if the method is in evaluation mode."""
        return not self.training

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add method specific arguments to the parent parser."""
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--method_learning_rate",
            type=float,
            default=1e-3,
            help="The learning rate to be used for training.",
        )
        parser.add_argument(
            "--method_regularizer_weight",
            type=float,
            default=0.0,
            help="The weight of the regularizer to be used for training.",
        )
        parser.add_argument(
            "--method_momentum",
            type=float,
            default=0.9,
            help="The momentum to be used for training.",
        )
        parser.add_argument(
            "--method_optimizer",
            type=str,
            default="adam",
            choices=["adam", "sgd", "sgld", "psgld"],
            help="The optimizer to be used for training.",
        )
        parser.add_argument(
            "--method_scheduler",
            type=str,
            default="cosine",
            choices=["none", "cosine", "step", "plateau", "exponential", "linear"],
            help="The scheduler to be used for training.",
        )
        parser.add_argument(
            "--method_scheduler_step_size",
            type=int,
            default=10,
            help="The step size to be used for the step learning rate scheduler.",
        )
        parser.add_argument(
            "--method_scheduler_gamma",
            type=float,
            default=0.1,
            help="The gamma to be used for the step learning rate scheduler.",
        )
        parser.add_argument(
            "--method_scheduler_factor",
            type=float,
            default=0.1,
            help="The factor to be used for the plateau learning rate scheduler.",
        )
        parser.add_argument(
            "--method_scheduler_patience",
            type=int,
            default=10,
            help="The patience to be used for the plateau learning rate scheduler.",
        )
        parser.add_argument(
            "--method_plotting_training",
            type=int,
            choices=[0, 1],
            default=0,
            help="If set, the plots are created during training.",
        )
        parser.add_argument(
            "--method_plotting_testing",
            type=int,
            choices=[0, 1],
            default=0,
            help="If set, the plots are created during testing.",
        )
        parser.add_argument(
            "--method_metrics",
            type=str,
            nargs="+",
            default=None,
            help="The metrics to be used for training, validation and testing.",
        )
        parser.add_argument(
            "--method_save_test_predictions",
            type=int,
            choices=[0, 1],
            default=0,
            help="If set, the test predictions are saved.",
        )
        return parser
