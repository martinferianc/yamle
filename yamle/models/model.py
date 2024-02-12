from typing import Any, Union, Dict, Tuple
import abc
import torch
import torch.nn as nn
import argparse

from yamle.defaults import SUPPORTED_TASKS


class BaseModel(nn.Module, abc.ABC):
    """This is the base class for all the models.

    By default it should have an input and output layer in `_input` and `_output` respectively.
    All the intermediate layers should be in `_layers`.
    The depth of the model should be in `_depth`.

    Args:
        inputs_dim (Tuple[int,...]): The input dimensions.
        outputs_dim (int): The output dimension.
        task (str): The task to perform.
    """

    tasks = SUPPORTED_TASKS

    def __init__(
        self,
        inputs_dim: Tuple[int, ...],
        outputs_dim: int,
        task: str,
        seed: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._inputs_dim = inputs_dim
        self._outputs_dim = outputs_dim
        assert (
            task in self.tasks
        ), f"The task {task} is not supported. Supported tasks are {self.tasks}."
        self._task = task
        self._output: nn.Module = None
        self._input: nn.Module = None
        self._output_activation: nn.Module = None
        self._layers: Union[nn.ModuleList, nn.Sequential] = None

        self._added_method_specific_layers = False
        self._method: str = None
        self._method_kwargs: Dict[str, Any] = None
        self._depth: int = None
        self._seed = seed

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass of the model."""
        raise NotImplementedError("The forward method must be implemented.")

    @abc.abstractmethod
    def final_layer(self, x: torch.Tensor, **output_kwargs: Any) -> torch.Tensor:
        """This function is used to get the final layer output."""
        raise NotImplementedError("The final_layer method must be implemented.")

    @classmethod
    def add_specific_args(
        cls, parent_parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        """This method adds model arguments to the given parser."""
        return argparse.ArgumentParser(parents=[parent_parser], add_help=False)

    def reset(self) -> None:
        """This method is used to reset the model e.g. at the start of a new epoch."""
        pass

    def replace_layers_for_quantization(self) -> None:
        """Fuses all the operations in the network.

        In this function we only need to fuse layers that are not in the blocks.
        e.g. the reshaping layers added by the method.
        """
        pass

    def add_method_specific_layers(self, method: str, **kwargs: Any) -> None:
        """This method is used to add method specific layers to the model.

        Args:
            method (str): The method to use.
        """
        self._added_method_specific_layers = True
        self._method = method
        self._method_kwargs = kwargs
