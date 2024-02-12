from typing import Tuple, Optional
import torch
from pytorch_lightning import LightningModule
from yamle.defaults import (
    MODULE_INPUT_SHAPE_KEY,
    MODULE_OUTPUT_SHAPE_KEY,
    MODULE_NAME_KEY,
)


def forward_shape_hook(
    module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
) -> None:
    """This function is used to cache the input and output shapes of a module.

    The shapes will be stored in the module as `MODULE_INPUT_SHAPE_KEY` and `MODULE_OUTPUT_SHAPE_KEY`.

    Args:
        module (torch.nn.Module): The module to cache the input and output shapes of.
        input (torch.Tensor): The input to the module.
        output (torch.Tensor): The output of the module.

    """
    setattr(
        module,
        MODULE_INPUT_SHAPE_KEY,
        [x.shape if isinstance(x, torch.Tensor) else None for x in input],
    )
    if isinstance(output, torch.Tensor):
        setattr(module, MODULE_OUTPUT_SHAPE_KEY, [output.shape])
    elif isinstance(output, (tuple, list)):
        setattr(
            module,
            MODULE_OUTPUT_SHAPE_KEY,
            [
                x.shape if x is not None and isinstance(x, torch.Tensor) else None
                for x in output
            ],
        )
    else:
        setattr(module, MODULE_OUTPUT_SHAPE_KEY, [None])


@torch.no_grad()
def get_sample_input_and_target(
    method: LightningModule, batch_size: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This method is used to get the sample input and target of the model."""

    input_shape = method._inputs_dim
    if batch_size is not None:
        input_shape = (batch_size, *input_shape[1:])

    if method._inputs_dtype == torch.float:
        x = torch.ones(input_shape).to(next(method.model.parameters()).device)
    elif method._inputs_dtype == torch.long:
        x = torch.randint(0, 1, input_shape).to(next(method.model.parameters()).device)
    else:
        raise ValueError(f"Input dtype {method._inputs_dtype} is not supported.")

    output_shape = method._targets_dim
    batch_size = method._inputs_dim[0] if batch_size is None else batch_size
    output_shape = (
        (batch_size, *output_shape)
        if isinstance(output_shape, (tuple, list))
        else (batch_size, output_shape)
    )
    if method._outputs_dtype == torch.float:
        y = torch.randn(output_shape).to(next(method.model.parameters()).device)
    elif method._outputs_dtype == torch.long:
        y = torch.randint(0, 1, output_shape).to(next(method.model.parameters()).device)
        if method._targets_dim == 1:
            y = y.view(-1)
    else:
        raise ValueError(f"Output dtype {method._outputs_dtype} is not supported.")
    return x, y


def get_input_shape_from_model(model: torch.nn.Module) -> Tuple[int, ...]:
    """This method is used to get the input shape of the model."""
    return getattr(model, MODULE_INPUT_SHAPE_KEY, None)


def get_output_shape_from_model(model: torch.nn.Module) -> Tuple[int, ...]:
    """This method is used to get the output shape of the model."""
    return getattr(model, MODULE_OUTPUT_SHAPE_KEY, None)


@torch.no_grad()
def trace_input_output_shapes(method: LightningModule) -> None:
    """This method is used to trace the input and output shapes of the model.

    Additionally, it will name all the modules in the model.
    """
    method.eval()
    hooks = []
    for m in method.model.modules():
        hooks.append(m.register_forward_hook(forward_shape_hook))

    batch = get_sample_input_and_target(method)

    method.test_step(batch, batch_idx=0)
    for hook in hooks:
        hook.remove()
    method.train()

    name_all_modules(method.model)


def name_all_modules(model: torch.nn.Module) -> None:
    """This method is used to name all the modules in the model."""
    for name, module in model.named_modules():
        setattr(module, MODULE_NAME_KEY, name)
