import argparse
import logging
import time
from typing import Any, Dict, List, Literal, Optional

import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import torch
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from syne_tune import Reporter

from yamle.data.datamodule import BaseDataModule
from yamle.defaults import (
    ALL_DATASETS_KEY,
    CLASSIFICATION_KEY,
    FIT_TIME_KEY,
    TEST_KEY,
    TEST_TIME_KEY,
    VALIDATION_KEY,
    CALIBRATION_KEY,
)
from yamle.methods.method import BaseMethod
from yamle.utils.file_utils import store_metrics
from yamle.utils.trainer_utils import (
    GradientNormMonitorCallback,
    GradientValueClippingCallback,
    InitialModelSavingCallback,
    L1L2MonitorCallback,
    LoggingCallback,
    NoOptimizationMonitorCallback,
    RegularizedWeightsMonitorCallback,
    SplitParametersMonitorCallback,
    TrainingModelSavingCallback,
    ValidationModelSavingCallback,
    ValidationReporterMonitorCallback,
    WeightHistogramMonitorCallback,
)

logging = logging.getLogger("pytorch_lightning")


class BaseTrainer:
    """This class defines a base trainer which given a method and data loaders performs training and evaluation.

    Args:
        save_path (str): The path to the experiment folder.
        datamodule (BaseDataModule): The datamodule to be used for training or evaluation.
        epochs (int): The number of epochs to train for.
        accelerator (str): The accelerator to be used for training or evaluation. (cpu, gpu, ddp, ddp2, ddp_spawn, auto)
        devices (List[int]): The devices to be used for training or evaluation.
        precision (int): The precision to be used for training or evaluation.
        method (BaseMethod): The method to be used for training or evaluation.
        gradient_clip_norm_value (float): The gradient clipping value when clipping by norm. Defaults to 0.0.
        gradient_clip_value (float): The gradient clipping value when clipping by value. Defaults to 5.0.
        mode (str): The mode of the trainer. (train, eval, tune)
        st_checkpoint_dir (Optional[str]): The path to the Syne-Tune checkpoint directory. Defaults to None.
        debug (bool): Whether to run in debug mode. Defaults to False.
        compile (bool): Whether to compile the model. Defaults to True.
        task (str): The task to be performed.
        no_saving (bool): Whether to not do any kind of saving. Defaults to False.
        no_initial_saving (bool): Whether to save the initial model. Defaults to True.
        no_validation_saving (bool): Whether to save the validation model. Defaults to True.
        no_every_train_epoch_saving (bool): Whether to save the model after every epoch. Defaults to True.
        no_augmentation_testing (bool): Whether to use augmentation during testing. Defaults to True.
        profiler (str): The profiler to be used for debugging. (simple, advanced, None)
    """

    def __init__(
        self,
        save_path: str,
        datamodule: BaseDataModule,
        epochs: int,
        accelerator: str,
        devices: List[int],
        precision: int,
        method: BaseMethod,
        gradient_clip_norm_value: float = 0.0,
        gradient_clip_value: float = 5.0,
        mode: Literal["train", "eval", "tune"] = "train",
        st_checkpoint_dir: Optional[str] = None,
        debug: bool = False,
        compile: bool = True,
        task: str = CLASSIFICATION_KEY,
        no_saving: bool = False,
        no_initial_saving: bool = True,
        no_validation_saving: bool = True,
        no_every_train_epoch_saving: bool = True,
        no_augmentation_testing: bool = True,
        profiler: Optional[str] = None,
    ) -> None:
        self._save_path = save_path
        self._datamodule = datamodule
        self._epochs = epochs
        self._accelerator = accelerator
        self.devices = devices
        self._method = method
        if compile:
            self._method.model = torch.compile(self._method.model)
        self._gradient_clip_norm_value = gradient_clip_norm_value
        self._gradient_clip_value = gradient_clip_value
        assert mode in [
            "train",
            "eval",
            "tune",
        ], "The mode must be one of `train`, `eval` or `tune`."
        self._mode = mode
        self._reporter = Reporter()
        self._total_elapsed_epochs = 0
        self._st_checkpoint_dir = st_checkpoint_dir
        self._debug = debug
        self._task = task
        self._precision = precision
        self._no_saving = no_saving
        self._no_initial_saving = no_initial_saving
        self._no_validation_saving = no_validation_saving
        self._no_every_train_epoch_saving = no_every_train_epoch_saving
        self._no_augmentation_testing = no_augmentation_testing
        self._profiler = profiler
        self._initialize_trainer()

    def _initialize_trainer(
        self,
        epochs: Optional[int] = None,
    ) -> None:
        """This method initializes the PyTorch Lightning trainer."""
        if epochs is None:
            epochs = self._epochs
        tb_logger = (
            pl.loggers.TensorBoardLogger(save_dir=self._save_path, name="logs")
            if self._mode == "train"
            else False
        )
        is_gpu_available = torch.cuda.is_available()
        if self._accelerator == "cpu" and is_gpu_available:
            logging.warning(
                "GPU is available but CPU is used for training. This might cause performance issues."
            )
        if self._accelerator == "cpu":
            assert (
                len(self.devices) <= 1
            ), f"Only one device can be used with accelerator `cpu` but {len(self.devices)} devices were specified."
            self.devices = 1 if len(self.devices) == 0 else self.devices[0]
        if (
            self._accelerator in ["auto", "gpu"]
            and is_gpu_available
            and len(self.devices) == 0
        ):
            # If no device is specified, use the default gpu.
            self.devices = [torch.cuda.current_device()]
        elif (
            self._accelerator in ["auto", "gpu"]
            and not is_gpu_available
            and len(self.devices) == 0
        ):
            self.devices = None
        enable_progress_bar = True if self._st_checkpoint_dir is None else False

        callbacks = [
            RegularizedWeightsMonitorCallback(),
            GradientValueClippingCallback(self._gradient_clip_value),
            LoggingCallback(),
        ]
        if (
            self._mode == "train"
            and not self._no_validation_saving
            and not self._no_saving
        ):
            callbacks += [ValidationModelSavingCallback(self._task)]
        if (
            self._mode == "train"
            and not self._no_every_train_epoch_saving
            and not self._no_saving
        ):
            callbacks += [TrainingModelSavingCallback()]
        if (
            self._mode == "train"
            and not self._no_initial_saving
            and not self._no_saving
        ):
            callbacks += [InitialModelSavingCallback()]
        if self._mode == "train":
            callbacks.append(
                pl_callbacks.LearningRateMonitor(
                    logging_interval="epoch", log_momentum=True
                )
            )
            callbacks.append(GradientNormMonitorCallback(norm=2.0))
            callbacks.append(L1L2MonitorCallback())
            callbacks.append(WeightHistogramMonitorCallback())
            callbacks.append(SplitParametersMonitorCallback())
            callbacks.append(NoOptimizationMonitorCallback())
        elif self._mode == "tune":
            callbacks.append(
                ValidationReporterMonitorCallback(
                    self._reporter, self._total_elapsed_epochs
                )
            )

        if enable_progress_bar:
            progress_bar = RichProgressBar(
                theme=RichProgressBarTheme(
                    description="green_yellow",
                    progress_bar="green1",
                    progress_bar_finished="green1",
                    progress_bar_pulse="#6206E0",
                    batch_progress="green_yellow",
                    time="grey82",
                    processing_speed="grey82",
                    metrics="grey82",
                )
            )
            callbacks.append(progress_bar)

        self._trainer = pl.Trainer(
            max_epochs=epochs,
            devices=self.devices,
            gradient_clip_val=self._gradient_clip_norm_value,
            accelerator=self._accelerator,
            logger=tb_logger,
            enable_progress_bar=enable_progress_bar,
            enable_checkpointing=False,
            benchmark=True,
            sync_batchnorm=True,
            fast_dev_run=self._debug,
            precision=self._precision,
            callbacks=callbacks,
            deterministic=True,
            profiler=self._profiler,
        )
        self._trainer_kwargs = {
            "max_epochs": epochs,
            "devices": self.devices,
            "gradient_clip_val": self._gradient_clip_norm_value,
            "accelerator": self._accelerator,
            "logger": tb_logger,
            "enable_progress_bar": enable_progress_bar,
            "enable_checkpointing": False,
            "benchmark": True,
            "sync_batchnorm": True,
            "fast_dev_run": self._debug,
            "precision": self._precision,
            "callbacks": callbacks,
            "deterministic": True,
            "profiler": self._profiler,
        }
        self._total_elapsed_epochs += epochs

        self._set_method_debug_mode_and_save_path()

    def _set_method_debug_mode_and_save_path(self) -> None:
        """This method sets the method to debug mode."""
        self._method._debug = self._debug
        for module in self._method.model.modules():
            module._debug = self._debug
            module._save_path = self._save_path

    def fit(self, results: Optional[Dict[str, Any]] = None) -> float:
        """This method trains the method and the embedded model.

        Returns the time it took to train the model.
        """
        train_dataloader = self._datamodule.train_dataloader()
        validation_dataloader = self._datamodule.validation_dataloader()
        start_time = time.time()
        self._trainer.fit(self._method, train_dataloader, validation_dataloader)
        end_time = time.time()
        if self._profiler is not None:
            logging.info("Terminating after profiling.")
            exit()
        if results is not None:
            results[ALL_DATASETS_KEY][FIT_TIME_KEY] = end_time - start_time
        return end_time - start_time

    @property
    def interrupted(self) -> bool:
        """This property returns whether the training was interrupted."""
        return self._trainer.interrupted

    def test(self, results: Dict[str, Any]) -> float:
        """This method tests the method and the embedded model.

        The results are stored in the given dictionary.
        Returns the time it took to test the model.
        """
        testing_options = [None]

        if not self._no_augmentation_testing:
            testing_options = testing_options + self._datamodule.test_augmentations

        # At first test the model on validation data if available
        # We assume that no augmentation can be applied to the validation data
        start_time = time.time()
        self._method.reset_metrics(prefix=TEST_KEY, complete=True)
        self._datamodule.setup(augmentation=None)
        validation_data_loader = self._datamodule.validation_dataloader()
        if validation_data_loader is not None:
            self._method.reset_metrics(prefix=TEST_KEY, complete=True)
            logging.info("Testing on validation data.")
            self._method.test_name = VALIDATION_KEY
            self._trainer.test(self._method, validation_data_loader)
            store_metrics(
                results, metrics=self._method.metrics[TEST_KEY], prefix=VALIDATION_KEY
            )

        calibration_data_loader = self._datamodule.calibration_dataloader()
        if calibration_data_loader is not None:
            self._method.reset_metrics(prefix=TEST_KEY, complete=True)
            logging.info("Testing on calibration data.")
            self._method.test_name = CALIBRATION_KEY
            self._trainer.test(self._method, calibration_data_loader)
            store_metrics(
                results, metrics=self._method.metrics[TEST_KEY], prefix=CALIBRATION_KEY
            )

        # Then test the model on test data.
        # We assume that augmentation can be applied to the test data.
        for test in testing_options:
            self._method.reset_metrics(prefix=TEST_KEY, complete=True)
            logging.info("Testing with augmentation: %s", test)
            self._datamodule.setup(augmentation=test)
            test_data_loader = self._datamodule.test_dataloader()
            prefix = TEST_KEY if test is None else f"{TEST_KEY}_{test}"
            self._method.test_name = prefix
            self._trainer.test(self._method, test_data_loader)
            store_metrics(
                results, metrics=self._method.metrics[TEST_KEY], prefix=prefix
            )

        end_time = time.time()
        if ALL_DATASETS_KEY not in results:
            results[ALL_DATASETS_KEY] = {}
        results[ALL_DATASETS_KEY][TEST_TIME_KEY] = end_time - start_time
        self._method.test_name = None 
        return end_time - start_time

    @torch.no_grad()
    def calibrate(self) -> float:
        """This is a helper function which runs calibration data through the model.

        Note that the Trainer is not used here, the model is not trained - the gradients are not updated.
        We need to manually run the calibration data through the `validation_step` method of the method.

        Returns the time it took to calibrate the model.
        """
        self._method.eval()
        calibration_dataloader = self._datamodule.calibration_dataloader()
        # Throw an exception if the dataloader is empty.
        if calibration_dataloader is None or len(calibration_dataloader) == 0:
            raise RuntimeError(
                "The calibration dataloader is empty. Please check your dataloader."
            )
        start_time = time.time()
        for i, batch in enumerate(calibration_dataloader):
            self._method.validation_step(batch, i)
        end_time = time.time()
        return end_time - start_time

    def fine_tune(self, epochs: int) -> float:
        """This method fine-tunes the method and the embedded model.

        It does it for additional epochs with respect to a new trainer.
        The fine-tuning is done on the calibration data.

        Returns the time it took to fine-tune the model.
        """
        self._initialize_trainer(epochs)
        calibration_dataloader = self._datamodule.calibration_dataloader()
        if calibration_dataloader is None or len(calibration_dataloader) == 0:
            raise RuntimeError(
                "The calibration dataloader is empty. Please check your dataloader."
            )
        start_time = time.time()
        self._trainer.fit(self._method, calibration_dataloader, calibration_dataloader)
        end_time = time.time()
        return end_time - start_time

    @staticmethod
    def add_specific_args(
        parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method adds trainer arguments to the given parser.

        Args:
            parser (ArgumentParser): The parser to which the arguments should be added.
        """
        parser.add_argument(
            "--trainer_epochs",
            type=int,
            default=10,
            help="The number of epochs to be used for training.",
        )
        parser.add_argument(
            "--trainer_devices",
            type=str,
            default="[]",
            help="The devices to be used for training.",
        )
        parser.add_argument(
            "--trainer_accelerator",
            type=str,
            default="auto",
            choices=["cpu", "gpu", "ddp", "ddp2", "ddp_spawn", "auto"],
            help="The accelerator to be used for training.",
        )
        parser.add_argument(
            "--trainer_gradient_clip_norm_value",
            type=float,
            default=0.0,
            help="The gradient clipping value.",
        )
        parser.add_argument(
            "--trainer_gradient_clip_value",
            type=float,
            default=5.0,
            help="The gradient clipping value.",
        )
        parser.add_argument(
            "--trainer_mode",
            type=str,
            default="train",
            choices=["train", "eval", "tune"],
            help="The mode of the trainer.",
        )
        parser.add_argument(
            "--trainer_debug",
            type=int,
            default=0,
            choices=[0, 1],
            help="If set to 1, the trainer will be run in debug mode.",
        )
        parser.add_argument(
            "--trainer_precision",
            type=int,
            default=32,
            choices=[16, 32],
            help="The precision to be used for training.",
        )
        parser.add_argument(
            "--trainer_compile",
            type=int,
            default=0,
            choices=[0, 1],
            help="If set to 1, the model will be compiled before training.",
        )

        parser.add_argument(
            "--trainer_no_initial_saving",
            type=int,
            default=1,
            choices=[0, 1],
            help="If set to 1, the initial model will not be saved.",
        )
        parser.add_argument(
            "--trainer_no_validation_saving",
            type=int,
            default=1,
            choices=[0, 1],
            help="If set to 1, the model will not be saved during validation.",
        )
        parser.add_argument(
            "--trainer_no_every_train_epoch_saving",
            type=int,
            default=1,
            choices=[0, 1],
            help="If set to 1, the model will not be saved after every epoch.",
        )
        parser.add_argument(
            "--trainer_no_augmentation_testing",
            type=int,
            default=0,
            choices=[0, 1],
            help="If set to 1, no augmentation will be used during testing.",
        )
        parser.add_argument(
            "--trainer_profiler",
            type=str,
            default=None,
            choices=["simple", "advanced", None],
            help="The profiler to be used for debugging.",
        )
        return parser

    def __repr__(self) -> str:
        return self._trainer.__repr__()
