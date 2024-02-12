from typing import List, Dict, Any, Union, Tuple, Optional

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import math

from yamle.defaults import (
    FROZEN_MASK_KEY,
    FROZEN_DATA_KEY,
    OPTIMIZER_ID_KEY,
    DISABLED_OPTIMIZATION_KEY,
    TINY_EPSILON,
)
from yamle.models.specific.sgld import SGLD, pSGLD


def get_optimizer(
    name: str, parameters: List[nn.Parameter], optimizer_config: Dict[str, Any]
) -> torch.optim.Optimizer:
    optimizer: torch.optim.Optimizer = None
    if name == "adam":
        optimizer = torch.optim.Adam(
            parameters,
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config["weight_decay"],
        )
    elif name == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=optimizer_config["lr"],
            momentum=optimizer_config["momentum"],
            weight_decay=optimizer_config["weight_decay"],
        )
    elif name == "sgld":
        optimizer = SGLD(
            parameters, lr=optimizer_config["lr"], momentum=optimizer_config["momentum"]
        )
    elif name == "psgld":
        optimizer = pSGLD(parameters, lr=optimizer_config["lr"])
    else:
        raise ValueError(
            f"Optimizer {name} is not supported. Please use one of the following: adam, sgd."
        )
    return optimizer


def get_scheduler(
    name: str, optimizer: torch.optim.Optimizer, scheduler_config: Dict[str, Any]
) -> torch.optim.lr_scheduler._LRScheduler:
    scheduler: torch.optim.lr_scheduler._LRScheduler = None
    if name == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1)
    elif name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config["mode"],
            factor=scheduler_config["factor"],
            patience=scheduler_config["patience"],
            verbose=True,
        )
    elif name == "linear":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1 - epoch / scheduler_config["max_epochs"]
        )
    elif name == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=scheduler_config["gamma"]
        )
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_config["max_epochs"], eta_min=0
        )
    elif name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config["step_size"],
            gamma=scheduler_config["gamma"],
        )
    else:
        raise ValueError(
            f"Scheduler {name} is not supported. Please use one of the following: step, cosine, linear, power_growth, sine."
        )
    return scheduler


class ScalarScheduler(ABC):
    """This is a general class for scalar schedulers."""

    @abstractmethod
    def step(self) -> None:
        """This method is used to update the scheduler."""
        pass

    @abstractmethod
    def get_value(self) -> float:
        """This method is used to get the current value of the scheduler."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """This method is used to reset the scheduler."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """This method is used to get the state of the scheduler."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """This method is used to load the state of the scheduler."""
        pass


class LinearScalarScheduler(ScalarScheduler):
    """This class defines a linear scheduler for a scalar value.

    Until the start epoch the return value will be `start_value`.
    After the start epoch the return value will be linearly increased until the end epoch.
    After the end epoch the return value will be the `end_value`.

    Args:
        start_value (float): The initial value of the scheduler.
        start_epoch (int): The epoch to start the scheduler.
        end_value (float): The final value of the scheduler.
        end_epoch (int): The epoch to end the scheduler.
    """

    def __init__(
        self, start_value: float, start_epoch: int, end_value: float, end_epoch: int
    ) -> None:
        self._start_value = start_value
        self._start_epoch = start_epoch
        self._end_value = end_value
        self._end_epoch = end_epoch
        self._current_epoch = 0
        # This is used to set a hard value for the scheduler, ignoring the schedule.
        self._hard_value: float = None

    def step(self) -> None:
        """This method is used to update the scheduler."""
        self._current_epoch += 1

    def set_hard_value(self, value: float) -> None:
        """This method is used to set a hard value for the scheduler, ignoring the schedule."""
        if self._hard_value is not None:
            raise ValueError("The hard value is already set.")
        self._hard_value = value

    def get_value(self) -> float:
        """This method is used to get the current value of the scheduler."""
        if self._hard_value is not None:
            return self._hard_value
        elif self._current_epoch < self._start_epoch:
            return self._start_value
        elif self._current_epoch >= self._end_epoch:
            return self._end_value
        else:
            return self._start_value + (self._end_value - self._start_value) * (
                self._current_epoch - self._start_epoch
            ) / (self._end_epoch - self._start_epoch - 1 + TINY_EPSILON)

    def reset(self) -> None:
        """This method is used to reset the scheduler."""
        self._current_epoch = 0

    def state_dict(self) -> Dict[str, Any]:
        """This method is used to get the state of the scheduler."""
        state_dict = super().state_dict()
        state_dict.update(
            {
                "current_epoch": self._current_epoch,
                "start_value": self._start_value,
                "start_epoch": self._start_epoch,
                "end_value": self._end_value,
                "end_epoch": self._end_epoch,
                "hard_value": self._hard_value,
            }
        )
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """This method is used to load the state of the scheduler."""
        self._current_epoch = state_dict["current_epoch"]
        self._start_value = state_dict["start_value"]
        self._start_epoch = state_dict["start_epoch"]
        self._end_value = state_dict["end_value"]
        self._end_epoch = state_dict["end_epoch"]
        self._hard_value = state_dict["hard_value"]


class PowerGrowthScalarScheduler(ScalarScheduler):
    """This class defines an exponential scheduler for a scalar value.

    Until the start epoch the return value will be `start_value`.
    Given a `power` value,

    Args:
        start_value (float): The initial value of the scheduler.
        start_epoch (int): The epoch to start the scheduler.
        end_value (float): The final value of the scheduler.
        end_epoch (int): The epoch to end the scheduler.
        gamma (float): The exponential growth factor.
    """

    def __init__(
        self,
        start_value: float,
        start_epoch: int,
        end_value: float,
        end_epoch: int,
        power: float = 1.0,
    ) -> None:
        assert (
            start_value <= end_value
        ), f"The start value {start_value} must be smaller than the end value {end_value}."
        self._start_value = start_value
        self._start_epoch = start_epoch
        self._end_value = end_value
        self._end_epoch = end_epoch
        self._current_epoch = 0
        # This is used to set a hard value for the scheduler, ignoring the schedule.
        self._hard_value: float = None
        self._power = power

    def step(self) -> None:
        """This method is used to update the scheduler."""
        self._current_epoch += 1

    def set_hard_value(self, value: float) -> None:
        """This method is used to set a hard value for the scheduler, ignoring the schedule."""
        if self._hard_value is not None:
            raise ValueError("The hard value is already set.")
        self._hard_value = value

    def get_value(self) -> float:
        """This method is used to get the current value of the scheduler."""
        if self._hard_value is not None:
            return self._hard_value
        elif self._current_epoch < self._start_epoch:
            return self._start_value
        elif self._current_epoch >= self._end_epoch:
            return self._end_value
        else:
            # Generate a value between 0 and 1.
            value = (self._current_epoch - self._start_epoch) / (
                self._end_epoch - self._start_epoch - 1
            )
            value **= self._power
            return self._start_value + (self._end_value - self._start_value) * value

    def reset(self) -> None:
        """This method is used to reset the scheduler."""
        self._current_epoch = 0

    def state_dict(self) -> Dict[str, Any]:
        """This method is used to get the state of the scheduler."""
        state_dict = super().state_dict()
        state_dict.update(
            {
                "current_epoch": self._current_epoch,
                "start_value": self._start_value,
                "start_epoch": self._start_epoch,
                "end_value": self._end_value,
                "end_epoch": self._end_epoch,
                "hard_value": self._hard_value,
            }
        )
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """This method is used to load the state of the scheduler."""
        self._current_epoch = state_dict["current_epoch"]
        self._start_value = state_dict["start_value"]
        self._start_epoch = state_dict["start_epoch"]
        self._end_value = state_dict["end_value"]
        self._end_epoch = state_dict["end_epoch"]
        self._hard_value = state_dict["hard_value"]
        self._power = state_dict["power"]


class SineScalarScheduler(ScalarScheduler):
    """This class defines a sine scheduler for a scalar value.

    Until the start epoch the return value will be `start_value`.
    After the start epoch the return value will be increased until the end epoch.
    After the end epoch the return value will be the `end_value`.

    Args:
        start_value (float): The initial value of the scheduler.
        start_epoch (int): The epoch to start the scheduler.
        end_value (float): The final value of the scheduler.
        end_epoch (int): The epoch to end the scheduler.
    """

    def __init__(
        self, start_value: float, start_epoch: int, end_value: float, end_epoch: int
    ) -> None:
        assert (
            start_value <= end_value
        ), f"The start value {start_value} must be less than the end value {end_value}."
        self._start_value = start_value
        self._start_epoch = start_epoch
        self._end_value = end_value
        self._end_epoch = end_epoch
        self._current_epoch = 0
        # This is used to set a hard value for the scheduler, ignoring the schedule.
        self._hard_value: float = None

    def step(self) -> None:
        """This method is used to update the scheduler."""
        self._current_epoch += 1

    def set_hard_value(self, value: float) -> None:
        """This method is used to set a hard value for the scheduler, ignoring the schedule."""
        if self._hard_value is not None:
            raise ValueError("The hard value is already set.")
        self._hard_value = value

    def get_value(self) -> float:
        """This method is used to get the current value of the scheduler."""
        if self._hard_value is not None:
            return self._hard_value
        elif self._current_epoch < self._start_epoch:
            return self._start_value
        elif self._current_epoch >= self._end_epoch:
            return self._end_value
        else:
            return (
                self._start_value
                + (self._end_value - self._start_value)
                * (
                    torch.sin(
                        torch.tensor(
                            (self._current_epoch - self._start_epoch)
                            / (self._end_epoch - self._start_epoch)
                            * (math.pi / 2)
                        )
                    )
                )
            ).item()

    def reset(self) -> None:
        """This method is used to reset the scheduler."""
        self._current_epoch = 0

    def state_dict(self) -> Dict[str, Any]:
        """This method is used to get the state of the scheduler."""
        state_dict = super().state_dict()
        state_dict.update(
            {
                "current_epoch": self._current_epoch,
                "start_value": self._start_value,
                "start_epoch": self._start_epoch,
                "end_value": self._end_value,
                "end_epoch": self._end_epoch,
                "hard_value": self._hard_value,
            }
        )
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """This method is used to load the state of the scheduler."""
        self._current_epoch = state_dict["current_epoch"]
        self._start_value = state_dict["start_value"]
        self._start_epoch = state_dict["start_epoch"]
        self._end_value = state_dict["end_value"]
        self._end_epoch = state_dict["end_epoch"]
        self._hard_value = state_dict["hard_value"]


class CosineScalarScheduler(ScalarScheduler):
    """This class defines a cosine scheduler for a scalar value.

    Until the start epoch the return value will be `start_value`.
    After the start epoch the return value will be increased until the end epoch.
    After the end epoch the return value will be the `end_value`.

    Args:
        start_value (float): The initial value of the scheduler.
        start_epoch (int): The epoch to start the scheduler.
        end_value (float): The final value of the scheduler.
        end_epoch (int): The epoch to end the scheduler.
    """

    def __init__(
        self, start_value: float, start_epoch: int, end_value: float, end_epoch: int
    ) -> None:
        assert (
            start_value >= end_value
        ), f"The start value {start_value} must be greater than the end value {end_value}."
        self._start_value = start_value
        self._start_epoch = start_epoch
        self._end_value = end_value
        self._end_epoch = end_epoch
        self._current_epoch = 0
        # This is used to set a hard value for the scheduler, ignoring the schedule.
        self._hard_value: float = None

    def step(self) -> None:
        """This method is used to update the scheduler."""
        self._current_epoch += 1

    def set_hard_value(self, value: float) -> None:
        """This method is used to set a hard value for the scheduler, ignoring the schedule."""
        if self._hard_value is not None:
            raise ValueError("The hard value is already set.")
        self._hard_value = value

    def get_value(self) -> float:
        """This method is used to get the current value of the scheduler."""
        if self._hard_value is not None:
            return self._hard_value
        elif self._current_epoch < self._start_epoch:
            return self._start_value
        elif self._current_epoch >= self._end_epoch:
            return self._end_value
        else:
            return (
                self._start_value
                + (self._end_value - self._start_value)
                * (
                    1
                    + torch.cos(
                        torch.tensor(
                            (self._current_epoch - self._start_epoch)
                            / (self._end_epoch - self._start_epoch)
                            * math.pi
                        )
                    )
                )
                / 2
            ).item()

    def reset(self) -> None:
        """This method is used to reset the scheduler."""
        self._current_epoch = 0

    def state_dict(self) -> Dict[str, Any]:
        """This method is used to get the state of the scheduler."""
        state_dict = super().state_dict()
        state_dict.update(
            {
                "current_epoch": self._current_epoch,
                "start_value": self._start_value,
                "start_epoch": self._start_epoch,
                "end_value": self._end_value,
                "end_epoch": self._end_epoch,
                "hard_value": self._hard_value,
            }
        )
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """This method is used to load the state of the scheduler."""
        self._current_epoch = state_dict["current_epoch"]
        self._start_value = state_dict["start_value"]
        self._start_epoch = state_dict["start_epoch"]
        self._end_value = state_dict["end_value"]
        self._end_epoch = state_dict["end_epoch"]
        self._hard_value = state_dict["hard_value"]


AVAILABLE_SCALAR_SCHEDULERS = {
    "linear": LinearScalarScheduler,
    "powergrowth": PowerGrowthScalarScheduler,
    "sine": SineScalarScheduler,
    "cosine": CosineScalarScheduler,
}


@torch.no_grad()
def recover_frozen_weights(model: nn.Module) -> None:
    """This function is used to recover frozen weights after an optimization step.

    The parameters that do have a `FROZEN_MASK_KEY` and `FROZEN_DATA_KEY` attribute will be recovered. The
    `FROZEN_MASK_KEY` is assumed to be a 1D tensor with the same number of elements as the parameter. The `FROZEN_DATA_KEY`
    has the same shape as the parameter. The `FROZEN_MASK_KEY` is used to select the elements that will be recovered.
    It should only contain `True` or `False` values. The `True` values are the elements that will be recovered from
    the `FROZEN_DATA_KEY`.

    Args:
        model (nn.Module): The model to recover the frozen weights.
    """
    for param in model.parameters():
        if hasattr(param, FROZEN_MASK_KEY) and hasattr(param, FROZEN_DATA_KEY):
            frozen_mask = getattr(param, FROZEN_MASK_KEY)
            frozen_data = getattr(param, FROZEN_DATA_KEY)
            assert torch.all(frozen_mask == 0) or torch.all(
                frozen_mask == 1
            ), f"The mask ({frozen_mask}) should only contain 0 or 1."
            assert (
                frozen_mask.numel() == param.numel()
            ), f"The number of elements of the mask ({frozen_mask.numel()}) and the parameter ({param.numel()}) are different."
            assert (
                frozen_mask.numel() == frozen_data.numel()
            ), f"The number of elements of the mask ({frozen_mask.numel()}) and the frozen data ({frozen_data.numel()}) are different."
            original_shape = param.shape
            current_data = param.data.view(-1)
            frozen_data = frozen_data.view(-1)
            frozen_mask = frozen_mask.view(-1).bool()
            current_data[frozen_mask] = frozen_data[frozen_mask]
            param.data = current_data.view(original_shape).contiguous()


def freeze_weights(
    parameters: Union[nn.Parameter, List[nn.Parameter]],
    masks: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
) -> None:
    """This function is used to freeze weights of a model.

    The masks are used to select the weights that will be frozen. The masks should have the same number of elements as the
    parameters. The masks should only contain `True` or `False` values. The `True` values are the elements that will be
    frozen. If no mask is provided, all the weights will be frozen.

    Args:
        parameters (Union[nn.Parameter, List[nn.Parameter]]): The parameters to freeze.
        masks (Optional[Union[torch.Tensor, List[torch.Tensor]]], optional): The masks to select the weights to freeze. Defaults to None.
    """
    if masks is None:
        masks = [torch.ones_like(param) for param in parameters]
    if isinstance(parameters, nn.Parameter):
        parameters = [parameters]
    if isinstance(masks, torch.Tensor):
        masks = [masks]
    assert len(parameters) == len(
        masks
    ), f"The number of parameters ({len(parameters)}) and masks ({len(masks)}) are different."
    for param, mask in zip(parameters, masks):
        assert (
            mask.numel() == param.numel()
        ), f"The number of elements of the mask ({mask.numel()}) and the parameter ({param.numel()}) are different."
        assert torch.all(mask == 0) or torch.all(
            mask == 1
        ), f"The mask ({mask}) should only contain 0 or 1."
        frozen_data = param.data.detach().clone()
        setattr(param, FROZEN_MASK_KEY, mask)
        setattr(param, FROZEN_DATA_KEY, frozen_data)


def split_optimizer_parameters(
    parameters: Union[nn.Parameter, List[nn.Parameter], List[Tuple[str, nn.Parameter]]]
) -> List[Dict[str, Union[List[nn.Parameter], List[Tuple[str, nn.Parameter]]]]]:
    """This function is used to split the parameters of a model into multiple dictionaries depending on an id.

    Given all model parameters, this function looks at if `OPTIMIZER_ID_KEY` is set as an attribute of the parameter.
    If it is set it will add the parameter to the dictionary with the corresponding `id`. If the `OPTIMIZER_ID_KEY` is
    not assigned it is assumed that the parameter is used by the first optimizer.

    Returns:
        A list of dictionaries with the parameters split.
    """
    id_parameter_mapping = {}
    if isinstance(parameters, nn.Parameter):
        parameters = [parameters]
    for param in parameters:
        if isinstance(param, tuple):
            name = param[0]
            p = param[1]
        else:
            p = param

        if not isinstance(p, nn.Parameter):
            raise ValueError(f"The parameter {p} is not a valid parameter.")

        if hasattr(p, OPTIMIZER_ID_KEY):
            optimizer_id = getattr(p, OPTIMIZER_ID_KEY)
        else:
            optimizer_id = 0  # Default optimizer id

        if optimizer_id not in id_parameter_mapping:
            id_parameter_mapping[optimizer_id] = []

        if isinstance(param, tuple):
            id_parameter_mapping[optimizer_id].append((name, p))
        else:
            id_parameter_mapping[optimizer_id].append(p)
    # Sort the parameters by id, key
    id_parameter_mapping = [
        v for _, v in sorted(id_parameter_mapping.items(), key=lambda item: item[0])
    ]
    return [[{"params": param_list}] for param_list in id_parameter_mapping]


def set_optimizer_id(
    parameters: Union[List[nn.Parameter], nn.Parameter], optimizer_id: int
) -> None:
    """This function is used to set the `OPTIMIZER_ID_KEY` attribute to the parameters.

    Args:
        parameters (List[nn.Parameter]): The parameters to set the optimizer id.
        optimizer_id (int): The optimizer id to set.
    """
    if isinstance(parameters, nn.Parameter):
        parameters = [parameters]
    for param in parameters:
        assert not hasattr(
            param, OPTIMIZER_ID_KEY
        ), f"The parameter {param} already has an optimizer id."
        setattr(param, OPTIMIZER_ID_KEY, optimizer_id)


def disable_optimization(parameters: Union[List[nn.Parameter], nn.Parameter]) -> None:
    """This function is used to disable the optimization of the parameters.

    Args:
        parameters (List[nn.Parameter]): The parameters to disable the optimization.
    """
    if isinstance(parameters, nn.Parameter):
        parameters = [parameters]
    for param in parameters:
        assert isinstance(
            param, nn.Parameter
        ), f"The parameter {param} is not a valid parameter."
        param.requires_grad = False
        setattr(param, DISABLED_OPTIMIZATION_KEY, True)


def enable_optimization(parameters: Union[List[nn.Parameter], nn.Parameter]) -> None:
    """This function is used to enable the optimization of the parameters.

    Args:
        parameters (List[nn.Parameter]): The parameters to enable the optimization.
    """
    if isinstance(parameters, nn.Parameter):
        parameters = [parameters]
    for param in parameters:
        assert isinstance(
            param, nn.Parameter
        ), f"The parameter {param} is not a valid parameter."
        param.requires_grad = True
        setattr(param, DISABLED_OPTIMIZATION_KEY, False)
