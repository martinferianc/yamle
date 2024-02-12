from typing import List, Union

import torch
import torch.nn as nn

from yamle.defaults import DISABLED_PRUNING_KEY


def enable_pruning(m: Union[nn.Parameter, List[nn.Parameter]]) -> None:
    """Enable pruning for the given parameters.

    Args:
        m (Union[nn.Parameter, List[nn.Parameter]]): The parameters to enable pruning for.
    """
    if isinstance(m, nn.Parameter):
        setattr(m, DISABLED_PRUNING_KEY, False)
    elif isinstance(m, list):
        for param in m:
            setattr(param, DISABLED_PRUNING_KEY, False)
    else:
        raise ValueError(
            f"The parameters should be either a list of parameters or a single parameter. Got {type(m)}."
        )


def disable_pruning(m: Union[nn.Parameter, List[nn.Parameter]]) -> None:
    """Disable pruning for the given parameters.

    Args:
        m (Union[nn.Parameter, List[nn.Parameter]]): The parameters to disable pruning for.
    """
    if isinstance(m, nn.Parameter):
        setattr(m, DISABLED_PRUNING_KEY, True)
    elif isinstance(m, list):
        for param in m:
            setattr(param, DISABLED_PRUNING_KEY, True)
    else:
        raise ValueError(
            f"The parameters should be either a list of parameters or a single parameter. Got {type(m)}."
        )


def is_layer_prunable(layer: nn.Module) -> bool:
    """Check if a layer is prunable.

    Args:
        layer (nn.Module): The layer to check.
    """
    return (
        isinstance(layer, nn.Linear)
        or issubclass(type(layer), nn.Linear)
        or isinstance(layer, nn.Conv2d)
        or issubclass(type(layer), nn.Conv2d)
    )


def is_parameter_prunable(param: Union[nn.Parameter, List[nn.Parameter]]) -> bool:
    """Check if a parameter is prunable.

    Args:
        param (Union[nn.Parameter, torch.Tensor]): The parameter to check.
    """
    if isinstance(param, nn.Parameter):
        if not hasattr(param, DISABLED_PRUNING_KEY):
            return True
        return hasattr(param, DISABLED_PRUNING_KEY) and not getattr(
            param, DISABLED_PRUNING_KEY
        )
    elif isinstance(param, list):
        for p in param:
            if not is_parameter_prunable(p):
                return False
        return True
    else:
        raise ValueError(
            f"The parameters should be either a list of parameters or a single parameter. Got {type(param)}."
        )


def get_all_prunable_weights(module: nn.Module) -> torch.Tensor:
    """Get all the prunable weights in the model.

    All the parameters of the prunable layers will be flattened into a single vector.
    These weights will be returned in a single Tensor.

    Args:
        m (nn.Module): The model to get the weights from.
    """
    weights = []
    for m in module.modules():
        if is_layer_prunable(m):
            for p in m.parameters():
                if is_parameter_prunable(p):
                    weights.append(p.data.view(-1))
    return torch.cat(weights)


def get_all_prunable_modules(module: nn.Module) -> List[nn.Module]:
    """Get all the prunable layers in the model.

    Args:
        m (nn.Module): The model to get the layers from.
    """
    layers = []
    for m in module.modules():
        if is_layer_prunable(m):
            layers.append(m)
    return layers
