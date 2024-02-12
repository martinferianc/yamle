from typing import Optional, Iterator, Tuple
import torch
import torch.nn as nn
import copy
import logging

logging = logging.getLogger("pytorch_lightning")


class Ensemble(nn.Module):
    """This class defines an ensemble of any models.

    During training, only the model with the `current_trained_member` is trained.

    The copying of the model is done via `copy.deepcopy`.
    The initialization of the model ensemble is done through the `reset_parameters` method.

    Args:
        model (nn.Module): The model to initially copy for the ensemble.
        num_members (int): The number of members in the ensemble.
    """

    def __init__(self, model: nn.Module, num_members: int) -> None:
        super(Ensemble, self).__init__()
        assert (
            num_members > 0
        ), "The number of members in the ensemble must be positive."
        self._models = nn.ModuleList([copy.deepcopy(model) for _ in range(num_members)])
        for m in self._models:
            for l in m.modules():
                if hasattr(l, "reset_parameters"):
                    l.reset_parameters()
                else:
                    logging.warn(
                        f"Module {l} does not have a `reset_parameters` method."
                    )
        self._num_members = num_members
        self.register_buffer("currently_trained_member", torch.tensor(0))

    def forward(
        self, x: torch.Tensor, current_member: Optional[int] = None
    ) -> torch.Tensor:
        """This method is used to perform a forward pass through the current model."""
        if current_member is not None:
            assert (
                current_member >= 0 and current_member < self._num_members
            ), "The current member index is out of bounds."
            return self._models[current_member](x)
        return self._models[self.currently_trained_member.item()](x)

    def parameters(
        self, recurse: bool = True, index: Optional[int] = None
    ) -> Iterator[nn.Parameter]:
        """This method is used to get the parameters of the current model or all models."""
        if self.training:
            if index is not None:
                assert (
                    index >= 0 and index < self._num_members
                ), f"The index is out of bounds. It must be between 0 and {self._num_members - 1}."
                return self[index].parameters(recurse=recurse)
            return self[self.currently_trained_member.item()].parameters(
                recurse=recurse
            )
        else:
            return self._models.parameters(recurse=recurse)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, index: Optional[int] = None
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        """This method is used to get the named parameters of the current model or all models."""
        if self.training:
            if index is not None:
                assert (
                    index >= 0 and index < self._num_members
                ), f"The index is out of bounds. It must be between 0 and {self._num_members - 1}."
                return self[index].named_parameters(prefix=prefix, recurse=recurse)
            return self[self.currently_trained_member.item()].named_parameters(
                prefix=prefix, recurse=recurse
            )
        else:
            return self._models.named_parameters(prefix=prefix, recurse=recurse)

    def increment_current_member(self) -> None:
        """This method is used to increment the current member index."""
        self.currently_trained_member.data.add_(1)

    def reset(self) -> None:
        """This method is used to reset a model after an epoch."""
        pass

    def __getitem__(self, index: int) -> nn.Module:
        """This method is used to get the model at the given index."""
        return self._models[index]

    def __len__(self) -> int:
        """This method is used to get the number of models in the ensemble."""
        return len(self._models)
