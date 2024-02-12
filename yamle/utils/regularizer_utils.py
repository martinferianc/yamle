from typing import Union, List

import torch.nn as nn
from yamle.defaults import DISABLED_REGULARIZER_KEY


def disable_regularizer(parameters: Union[List[nn.Parameter], nn.Parameter]) -> None:
    """This method is used to disable weight decay for the given parameters."""
    if isinstance(parameters, nn.Parameter):
        setattr(parameters, DISABLED_REGULARIZER_KEY, True)
    elif isinstance(parameters, list):
        for param in parameters:
            setattr(param, DISABLED_REGULARIZER_KEY, True)
    else:
        raise ValueError(
            f"The parameters should be either a list of parameters or a single parameter. Got {type(parameters)}."
        )


def enable_regularizer(parameters: Union[List[nn.Parameter], nn.Parameter]) -> None:
    """This method is used to enable weight decay for the given parameters."""
    if isinstance(parameters, nn.Parameter):
        setattr(parameters, DISABLED_REGULARIZER_KEY, False)
    elif isinstance(parameters, list):
        for param in parameters:
            setattr(param, DISABLED_REGULARIZER_KEY, False)
    else:
        raise ValueError(
            f"The parameters should be either a list of parameters or a single parameter. Got {type(parameters)}."
        )
