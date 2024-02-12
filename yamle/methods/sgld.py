import argparse
from typing import Any, Dict, List, Optional
import torch
from yamle.defaults import LOSS_KEY, TRAIN_KEY
from yamle.methods.ensemble import EnsembleMethod


class SGLDMethod(EnsembleMethod):
    """This is a Method class for the Stochastic Gradient Langevin Dynamics Optimizer.

    It uses the `Ensemble` model to wrap around the original model and then uses the base method to train the network
    via the SGLD optimizer.

    At predefined epoch intervals, the weights of the main model
    are copied into the next member in the ensemble. This represents the posterior distribution of the weights.
    The last sample is always at the last epoch.

    Args:
        sampling_epochs (List[int], optional): Epochs at which to sample (default: [0, 1, 2, 3, 4, 5, 10, 20, 50, 100]).
    """

    def __init__(
        self,
        sampling_epochs: List[int] = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._sampling_epochs = sampling_epochs
        assert self._num_members == len(
            self._sampling_epochs
        ), f"Number of sampling epochs ({len(self._sampling_epochs)}) must match number of ensemble members ({self._num_members})."
        assert self.hparams.optimizer in [
            "sgld",
            "psgld",
        ], f"Optimizer must be 'sgld' or 'psgld', not {self.hparams.optimizer}."

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

    def _step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
        phase: str = TRAIN_KEY,
    ) -> Dict[str, Any]:
        output = super().step(batch, batch_idx, optimizer_idx, phase)
        # A batch is an approximation of the whole dataset, so we need to scale the loss
        # by the size of the dataset (training set size)
        output[LOSS_KEY] = output[LOSS_KEY] * self._datamodule.train_dataset_size()
        return output

    def on_train_epoch_end(self) -> None:
        """This method is called at the end of each training epoch.

        In this case, if the current epoch can be found in the sampling epochs, the current member is incremented.
        """
        assert (
            self._sampling_epochs[-1] == self.trainer.max_epochs - 1
        ), f"Last sampling epoch ({self._sampling_epochs[-1]}) must match last epoch ({self.trainer.max_epochs - 1})."
        # The -1 is the default model at index 0, we don't need to do anything
        if self.current_epoch in self._sampling_epochs[:-1]:
            self.increment_current_member()
            self.model[self.model.currently_trained_member.item()].load_state_dict(
                self.model[0].state_dict()
            )
        super().on_train_epoch_end()

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method adds arguments specific to this method to the parser."""
        parser = super(SGLDMethod, SGLDMethod).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_sampling_epochs",
            type=str,
            default="[0,1,2,3,4,5,10,20,50,100]",
            help="Epochs at which to sample (default: [0,1,2,3,4,5,10,20,50,100]).",
        )
        return parser
