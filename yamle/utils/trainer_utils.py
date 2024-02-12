import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from pytorch_lightning import Callback, Trainer
from syne_tune import Reporter

from yamle.defaults import (
    LOSS_KEY,
    MAX_TENDENCY,
    MEAN_PREDICTION_KEY,
    MEMBERS_DIM,
    MIN_TENDENCY,
    PREDICTION_KEY,
    PREDICTION_PER_MEMBER_KEY,
    TARGET_KEY,
    TARGET_PER_MEMBER_KEY,
    TEST_KEY,
    TRAIN_KEY,
    VALIDATION_KEY,
    AVERAGE_WEIGHTS_KEY,
    POSITIVE_INFINITY,
    NEGATIVE_INFINITY
)
from yamle.evaluation.metrics.algorithmic import (
    INDIVIDUAL_PREDICTIONS_AND_MEAN_METRICS,
    MAIN_METRIC,
    MEAN_METRIC_SPLITTING_KEY,
    METRIC_TENDENCY,
    PERMEMBER_METRIC_SPLITTING_KEY,
    PREDICTION_METRICS,
    parse_metric,
)
from yamle.methods.method import BaseMethod
from yamle.utils.file_utils import (
    model_best_on_validation_file,
    model_initial_file,
    model_train_epoch_file,
    save_model,
)
from yamle.utils.optimization_utils import split_optimizer_parameters

logging = logging.getLogger("pytorch_lightning")


class ValidationReporterMonitorCallback(Callback):
    """This callback reports all the metrics back to the Syne-Tune Reporter.

    This is happening after each validation epoch.

    Args:
        reporter (Reporter): The Syne-Tune Reporter.
        epoch_offset (int): The offset for the epoch number. Defaults to `None.`
    """

    def __init__(self, reporter: Reporter, epoch_offset: Optional[int] = None) -> None:
        super().__init__()
        self._reporter = reporter
        self._epoch_offset = epoch_offset if epoch_offset is not None else 0

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: BaseMethod) -> None:
        """This method reports the validation metric of choice to the Syne-Tune Reporter.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer.
            pl_module (BaseMethod): The PyTorch Lightning Module.
        """
        # If we are in the sanity check, we do not report anything.
        if trainer.sanity_checking:
            return
        logged_metrics = {}
        for key, value in trainer.logged_metrics.items():
            if f"{VALIDATION_KEY}/" not in key:
                continue
            key = key.replace("/", "_")
            nan_value = (
                NEGATIVE_INFINITY if METRIC_TENDENCY[parse_metric(key.replace(f"{VALIDATION_KEY}_", ""))] == MAX_TENDENCY else POSITIVE_INFINITY
            ) # It is the worst possible value for the tendency
            logged_metrics[key] = torch.nan_to_num(value, nan=nan_value).item()
        self._reporter(
            step=trainer.current_epoch + self._epoch_offset,
            **logged_metrics,
            epoch=trainer.current_epoch + self._epoch_offset + 1,
        )


class GradientNormMonitorCallback(Callback):
    """This callback logs the norm of the gradients of the method.

    This is happening after each training batch with a given frequency.

    Args:
        norm (int): The norm to be used. Defaults to 2.
        frequency (int): The frequency of the logging. Defaults to 10.
    """

    def __init__(self, norm: int = 2, frequency: int = 10) -> None:
        super().__init__()
        assert norm > 0, f"Norm must be greater than 0. Got {norm}."
        self._norm = norm
        assert frequency > 0, f"Frequency must be greater than 0. Got {frequency}."
        self._frequency = frequency
        self._counter = 0

    @torch.no_grad()
    def on_after_backward(self, trainer: Trainer, pl_module: BaseMethod) -> None:
        """This method logs the norm of the gradients of the method.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer.
            pl_module (BaseMethod): The PyTorch Lightning Module.
        """
        # If we are in the sanity check, we do not report anything.
        self._counter += 1
        if (
            trainer.sanity_checking
            or self._counter % self._frequency == 0
            or self._counter == 1
            or pl_module.training_step_exception
        ):
            return
        total_norm = 0.0
        parameters = pl_module.get_parameters()
        grads = [p.grad for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(g, self._norm) for g in grads]), self._norm
        )
        pl_module.log("grad_norm", total_norm, on_step=True, on_epoch=False)


class GradientValueClippingCallback(Callback):
    """This callback clips the value of the gradient to a min and max."""

    def __init__(self, value: Optional[float] = None) -> None:
        super().__init__()
        self._value = value

    @torch.no_grad()
    def on_after_backward(self, trainer: Trainer, pl_module: BaseMethod) -> None:
        """This method clips the value of the gradient to a min and max.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer.
            pl_module (BaseMethod): The PyTorch Lightning Module.
        """
        if (
            self._value is None
            or trainer.sanity_checking
            or pl_module.training_step_exception
        ):
            return
        for p in pl_module.model.parameters():
            if p.grad is not None:
                p.grad.clamp_(-self._value, self._value)


class L1L2MonitorCallback(Callback):
    """This callback logs the L1 and L2 norm of the parameters of the method.

    This is happening after each training epoch.
    """

    @torch.no_grad()
    def on_train_epoch_end(self, trainer: Trainer, pl_module: BaseMethod) -> None:
        """This method logs the L1 and L2 norm of the parameters of the method.

        It logs the L1 and L2 norm of the parameters of the method. It calculates the total norm
        as well as per model part norm.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer.
            pl_module (BaseMethod): The PyTorch Lightning Module.
        """
        l1 = 0.0
        l2 = 0.0
        for name, p in pl_module.get_named_parameters():
            if p.requires_grad:
                l1_ = torch.abs(p).sum()
                l2_ = torch.pow(p, 2).sum()
                name = name.replace(".", "/")
                pl_module.log(
                    f"model/{name}/l1_norm", l1_, on_step=False, on_epoch=True
                )
                pl_module.log(
                    f"model/{name}/l2_norm", l2_, on_step=False, on_epoch=True
                )
                l1 += l1_
                l2 += l2_

        pl_module.log("model/l1_norm", l1, on_step=False, on_epoch=True)
        pl_module.log("model/l2_norm", l2, on_step=False, on_epoch=True)


class WeightHistogramMonitorCallback(Callback):
    """This callback logs the histogram of the weights of the method.

    This is happening after each training epoch.
    """

    @torch.no_grad()
    def on_train_epoch_end(self, trainer: Trainer, pl_module: BaseMethod) -> None:
        """This method logs the histogram of the weights of the method."""
        tensorboard = (
            pl_module.logger.experiment
            if hasattr(pl_module, "logger") and hasattr(pl_module.logger, "experiment")
            else None
        )
        if tensorboard is not None:
            for name, module in pl_module.model.named_modules():
                name = name.replace(".", "/")
                if hasattr(module, "weight") and isinstance(
                    module.weight, torch.nn.Parameter
                ):
                    tensorboard.add_histogram(
                        f"model/{name}/weight",
                        module.weight.data,
                        pl_module.current_epoch,
                    )
                if hasattr(module, "bias") and isinstance(
                    module.bias, torch.nn.Parameter
                ):
                    tensorboard.add_histogram(
                        f"model/{name}/bias", module.bias.data, pl_module.current_epoch
                    )


class RegularizedWeightsMonitorCallback(Callback):
    """This callback logs the regularized weights of the method on the start of the training."""

    def on_fit_start(self, trainer: Trainer, pl_module: BaseMethod) -> None:
        if hasattr(pl_module, "_regularizer") and pl_module._regularizer is not None:
            regularized, non_regularized = pl_module._regularizer.get_names(
                pl_module.model
            )
            logging.info(f"Regularized parameters: {regularized}")
            logging.info(f"Non-regularized parameters: {non_regularized}")


class SplitParametersMonitorCallback(Callback):
    """This callback logs how the parameters are split between different optimizers at the start of the training."""

    def on_fit_start(self, trainer: Trainer, pl_module: BaseMethod) -> None:
        parameters = split_optimizer_parameters(pl_module.get_named_parameters())
        for i in range(len(parameters)):
            params = parameters[i][0]["params"]
            logging.info(
                f"Parameters for optimizer {i}: {[p[0] for p in params if p[1].requires_grad]}"
            )


class NoOptimizationMonitorCallback(Callback):
    """This callback logs which parameters are not optimized at the start of the training."""

    def on_fit_start(self, trainer: Trainer, pl_module: BaseMethod) -> None:
        logging.info(
            f"Non-optimized parameters: {[p[0] for p in pl_module.get_named_parameters() if not p[1].requires_grad]}"
        )


class LoggingCallback(Callback):
    """This callback updates, logs and resets all the metrics.

    This is happening after each training, validation or test epoch.
    """

    def _log_metrics(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
        prefix: str,
        metrics: nn.ModuleDict,
        log_to_tensorboard: bool = True,
    ) -> None:
        """This method is used to log the metrics.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer.
            pl_module (BaseMethod): The PyTorch Lightning Module.
            prefix (str): The prefix to be used for the logged metrics.
            metrics (nn.ModuleDict): The metrics to be logged.
            log_to_tensorboard (bool, optional): Whether to log the metrics to tensorboard. Defaults to `True`.
        """
        if trainer.sanity_checking:
            return
        single_string = ""
        for metric_name, metric in metrics.items():
            nan_value = (
                NEGATIVE_INFINITY
                if METRIC_TENDENCY[parse_metric(metric_name)] == MAX_TENDENCY
                else POSITIVE_INFINITY
            ) # It is the worst possible value for the tendency
            if log_to_tensorboard:
                pl_module.log(
                    f"{prefix}/{metric_name}",
                    torch.nan_to_num(metric.compute(), nan=nan_value).item(),
                    on_step=False,
                    on_epoch=True,
                )
            single_string += (
                f"{metric_name}: {torch.nan_to_num(metric.compute(), nan=nan_value).item():.4f}, "
            )
        logging.info(
            f"Epoch: [{pl_module.current_epoch + 1}/{trainer.max_epochs}]: {prefix}: {single_string[:-2]}"
        )

    def _log_progress_bar(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
        prefix: str,
        metrics: nn.ModuleDict,
    ) -> None:
        """This method is used to log the main metrics.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer.
            pl_module (BaseMethod): The PyTorch Lightning Module.
            prefix (str): The prefix to be used for the logged metrics.
            metrics (nn.ModuleDict): The metrics to be logged.
            log_to_tensorboard (bool, optional): Whether to log the metrics to tensorboard. Defaults to `True`.
        """
        pl_module.log(
            LOSS_KEY,
            metrics[LOSS_KEY].compute(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )
        pl_module.log(
            MAIN_METRIC[pl_module._task],
            metrics[MAIN_METRIC[pl_module._task]].compute(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )

    @torch.no_grad()
    def _update_prediction_metrics(
        self, metrics: nn.ModuleDict, outputs: Dict[str, Any]
    ) -> None:
        """This method is used to update the metrics.

        Note that the losses are updated in the training loop directly and not here.
        """
        y = outputs[TARGET_KEY]
        y_hat = outputs[PREDICTION_KEY]
        y_hat_mean = outputs[MEAN_PREDICTION_KEY]
        y_permember = (
            outputs[TARGET_PER_MEMBER_KEY] if TARGET_PER_MEMBER_KEY in outputs else None
        )
        y_hat_permember = (
            outputs[PREDICTION_PER_MEMBER_KEY]
            if PREDICTION_PER_MEMBER_KEY in outputs
            else None
        )
        averaging_weights = (
            outputs[AVERAGE_WEIGHTS_KEY] if AVERAGE_WEIGHTS_KEY in outputs else None
        )

        # Both permember labels and predictions need to be defined in order to update the permember metrics
        assert (y_permember is None and y_hat_permember is None) or (
            y_permember is not None and y_hat_permember is not None
        ), f"Both {TARGET_PER_MEMBER_KEY} and {PREDICTION_PER_MEMBER_KEY} need to be defined in order to update the permember metrics."

        for key in metrics.keys():
            parsed_metric = parse_metric(key)
            if parsed_metric in PREDICTION_METRICS:
                if PERMEMBER_METRIC_SPLITTING_KEY in key:
                    member = int(key.split(PERMEMBER_METRIC_SPLITTING_KEY)[1])
                    if y_permember is not None and y_hat_permember is not None:
                        # Test if the member is in the range of the number of members
                        if member >= y_permember.shape[1]:
                            logging.warning(
                                f"Member {member} is not in the range of the number of members {y_permember.shape[1]}."
                            )
                            continue
                        metrics[key].update(
                            y_hat_permember[:, member], y_permember[:, member]
                        )
                    else:
                        if member >= y_hat.shape[1]:
                            logging.warning(
                                f"Member {member} is not in the range of the number of members {y_hat.shape[1]}."
                            )
                            continue
                        metrics[key].update(y_hat[:, member], y)
                elif MEAN_METRIC_SPLITTING_KEY in key:
                    if y_permember is not None and y_hat_permember is not None:
                        y_hat_permember_ = y_hat_permember.reshape(
                            -1, *y_hat_permember.shape[2:]
                        )
                        y_permember_ = y_permember.reshape(-1, *y_permember.shape[2:])
                        metrics[key].update(y_hat_permember_, y_permember_)
                    else:
                        members = y_hat.shape[1]
                        y_hat_ = y_hat.reshape(-1, *y_hat.shape[2:])
                        # This is going to guarantee that the labels are organised
                        # such that [y_0, y_0, y_1, y_1, ...] if there are 2 members
                        y_ = torch.stack([y] * members, dim=MEMBERS_DIM).reshape(
                            -1, *y.shape[1:]
                        )
                        metrics[key].update(y_hat_, y_)
                elif not isinstance(
                    metrics[key], INDIVIDUAL_PREDICTIONS_AND_MEAN_METRICS
                ):
                    metrics[key].update(y_hat_mean, y)
                else:
                    metrics[key].update(y_hat_mean, y, y_hat, averaging_weights)

    def _update_losses(self, metrics: nn.ModuleDict, outputs: Dict[str, Any]) -> None:
        """This method is used to update the loss containers for in metrics."""
        keys = list(outputs.keys())
        for output_key in keys:
            if output_key.startswith(LOSS_KEY):
                if output_key in metrics:
                    metrics[output_key].update(outputs[output_key].detach(), 1)
                else:
                    raise ValueError(
                        f"Loss: {output_key} not found. It does not have a tracking metric associated with it."
                    )

    def _update_other_metrics(
        self, metrics: nn.ModuleDict, outputs: Dict[str, Any]
    ) -> None:
        """This method is used to update the other metrics."""
        keys = list(outputs.keys())
        for output_key in keys:
            if output_key in metrics:
                metrics[output_key].update(
                    outputs[output_key].detach()
                    if isinstance(outputs[output_key], torch.Tensor)
                    else outputs[output_key],
                    1,
                )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        """This method is used to update the training metrics after each training step."""
        assert (
            pl_module.training
        ), f"The model is not in training mode. Got training={pl_module.training}."
        if not pl_module.training_step_exception:
            self._update_prediction_metrics(pl_module.metrics[TRAIN_KEY], outputs)
            self._update_losses(pl_module.metrics[TRAIN_KEY], outputs)
            self._update_other_metrics(pl_module.metrics[TRAIN_KEY], outputs)
        self._log_progress_bar(
            trainer, pl_module, TRAIN_KEY, pl_module.metrics[TRAIN_KEY]
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
        outputs: Dict[str, Any],
        batch: List[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """This method is used to update the validation metrics after each validation step."""
        assert (
            not pl_module.training
        ), f"The model is in training mode. Got training={pl_module.training}."
        if not pl_module.validation_step_exception:
            self._update_prediction_metrics(pl_module.metrics[VALIDATION_KEY], outputs)
            self._update_losses(pl_module.metrics[VALIDATION_KEY], outputs)
            self._update_other_metrics(pl_module.metrics[VALIDATION_KEY], outputs)
        self._log_progress_bar(
            trainer, pl_module, VALIDATION_KEY, pl_module.metrics[VALIDATION_KEY]
        )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
        outputs: Dict[str, Any],
        batch: List[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """This method is used to update the test metrics after each test step."""
        assert (
            not pl_module.training
        ), f"The model is in training mode. Got training={pl_module.training}."
        if not pl_module.test_step_exception:
            self._update_prediction_metrics(pl_module.metrics[TEST_KEY], outputs)
            self._update_losses(pl_module.metrics[TEST_KEY], outputs)
            self._update_other_metrics(pl_module.metrics[TEST_KEY], outputs)
        self._log_progress_bar(
            trainer, pl_module, TEST_KEY, pl_module.metrics[TEST_KEY]
        )

    def on_train_epoch_start(self, trainer: Trainer, pl_module: BaseMethod) -> None:
        """This method is used at the start of each training epoch."""
        pl_module.reset_metrics(TRAIN_KEY, complete=True)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: BaseMethod) -> None:
        """This method is used to log the training metrics at the end of each training epoch."""
        self._log_metrics(trainer, pl_module, TRAIN_KEY, pl_module.metrics[TRAIN_KEY])
        pl_module.reset_metrics(TRAIN_KEY, complete=True)

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: BaseMethod
    ) -> None:
        """This method is used at the start of each validation epoch."""
        pl_module.reset_metrics(VALIDATION_KEY, complete=True)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: BaseMethod) -> None:
        """This method is used to log the validation metrics at the end of each validation epoch."""
        self._log_metrics(
            trainer, pl_module, VALIDATION_KEY, pl_module.metrics[VALIDATION_KEY]
        )
        pl_module.reset_metrics(VALIDATION_KEY, complete=True)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: BaseMethod) -> None:
        """This method is used at the start of each test epoch."""
        pl_module.reset_metrics(TEST_KEY, complete=True)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: BaseMethod) -> None:
        """This method is used to log the test metrics at the end of each test epoch.

        The reset is performed manually in the Trainer class.
        """
        self._log_metrics(
            trainer,
            pl_module,
            TEST_KEY,
            pl_module.metrics[TEST_KEY],
            log_to_tensorboard=False,
        )


class ValidationModelSavingCallback(Callback):
    """This callback saves the best model given a validation metric.

    The metric is decided by the main metric given by the task. It comes
    from `MAIN_METRIC`.

    Args:
        task (str): The task of the method.
    """

    def __init__(self, task: str) -> None:
        super().__init__()
        self._best_performance: float = None
        self._main_metric = MAIN_METRIC[task]
        self._tendency = METRIC_TENDENCY[self._main_metric]

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: BaseMethod) -> None:
        """This method saves the best model given a validation metric."""
        if trainer.sanity_checking:
            return
        value = trainer.logged_metrics[f"{VALIDATION_KEY}/{self._main_metric}"]
        if self._best_performance is None:
            self._best_performance = value
        if (self._tendency == MIN_TENDENCY and value < self._best_performance) or (
            self._tendency == MAX_TENDENCY and value > self._best_performance
        ):
            logging.info(
                f"Saving best model on validation with {self._main_metric} of {value}. Previous best was {self._best_performance}."
            )
            self._best_performance = value
            save_model(
                pl_module.model, model_best_on_validation_file(pl_module._save_path)
            )


class TrainingModelSavingCallback(Callback):
    """This callback saves the model at the end of the training epoch."""

    def on_train_epoch_end(self, trainer: Trainer, pl_module: BaseMethod) -> None:
        """This method saves the model at the end of the training epoch."""
        save_model(
            pl_module.model,
            model_train_epoch_file(pl_module._save_path, trainer.current_epoch),
        )


class InitialModelSavingCallback(Callback):
    """This callback saves the model at the start of the training."""

    def on_fit_start(self, trainer: Trainer, pl_module: BaseMethod) -> None:
        """This method saves the model at the start of the training."""
        save_model(pl_module.model, model_initial_file(pl_module._save_path))
