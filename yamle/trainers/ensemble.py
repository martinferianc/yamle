from typing import Dict, Any
import argparse
import logging
from typing import Any
import time

from yamle.defaults import TRAIN_DATA_SPLIT_KEY, FIT_TIME_KEY, ALL_DATASETS_KEY
from yamle.trainers.trainer import BaseTrainer

logging = logging.getLogger("pytorch_lightning")


class BaggingTrainer(BaseTrainer):
    """This class defines a bagging trainer which given a method and multiple data splits performs training and evaluation.

    The difference is that the data is split across training splits and the training is performed
    in parallel for each training split.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._train_splits = self._datamodule._train_splits

    def fit(self) -> None:
        """This method trains the method and the embedded model."""
        train_dataloaders = {}
        validation_dataloader = self._datamodule.validation_dataloader()
        for i in range(self._train_splits):
            train_dataloaders[
                f"{TRAIN_DATA_SPLIT_KEY}{i}"
            ] = self._datamodule.train_dataloader(
                split=None if self._train_splits is None else i
            )
        self._trainer.fit(self._method, train_dataloaders, validation_dataloader)


class EnsembleTrainer(BaggingTrainer):
    """This class defines an ensemble trainer which given a method and data loaders performs training and evaluation.

    The difference is that the training is performed repeatedly for all ensemble members.
    If training data splits are provided, the training is performed with respect to the data splits
    which are separate for each ensemble member.

    If the training is defines as parallel, this trainer returns multiple train loaders to the method.
    Then it is the method's responsibility to train the ensemble members in parallel.

    Args:
        parallel (bool): Whether to train the ensemble members in parallel.
    """

    def __init__(self, parallel: bool = False, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._parallel = parallel

    def fit(self, results: Dict[str, Any]) -> float:
        """This method trains the method and the embedded model.

        It also measures the time taken for training and returns it.
        """
        assert hasattr(
            self._method, "_num_members"
        ), "The method does not have a `num_members` attribute. Maybe not an ensemble model?"
        if self._train_splits is not None:
            assert (
                self._train_splits == self._method._num_members
            ), f"The number of training splits ({len(self._train_splits)}) does not match the number of ensemble members ({self._method._num_members})."
        total_time = 0
        if not self._parallel:
            assert hasattr(
                self._method, "increment_current_member"
            ), "The method does not have a `increment_current_member` method. Maybe not an ensemble model?"
            validation_dataloader = self._datamodule.validation_dataloader()
            for i in range(self._method._num_members):
                logging.info(f"Training member {i+1} of {self._method._num_members}.")
                train_dataloader = self._datamodule.train_dataloader(
                    split=None if self._train_splits is None else i
                )
                start_time = time.time()
                self._trainer.fit(self._method, train_dataloader, validation_dataloader)
                end_time = time.time()
                # The trainer needs to be reinitialized after each training round.
                if i < self._method._num_members - 1:
                    self._method.increment_current_member()
                    self._initialize_trainer()
                total_time += end_time - start_time
            if results is not None:
                results[ALL_DATASETS_KEY][FIT_TIME_KEY] = total_time
        else:
            total_time = super().fit(results)
        return total_time

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method adds the specific arguments for the ensemble trainer."""
        parser = super(EnsembleTrainer, EnsembleTrainer).add_specific_args(
            parent_parser
        )
        parser.add_argument(
            "--trainer_parallel",
            type=int,
            choices=[0, 1],
            default=0,
            help="Whether to train the ensemble members in parallel.",
        )
        return parser
