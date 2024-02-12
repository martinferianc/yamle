"""
Taken/adapted from: https://github.com/sovrasov/flops-counter.pytorch
Copyright (C) 2019-2021 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
"""
import logging
import time
import platform
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import copy
from fvcore.nn import FlopCountAnalysis
from ptflops.pytorch_engine import (
    compute_average_flops_cost,
    print_model_with_flops,
    remove_batch_counter_hook_function,
    remove_flops_counter_hook_function,
    reset_flops_count,
    start_flops_count,
)
from ptflops.pytorch_ops import (
    CUSTOM_MODULES_MAPPING,
    MODULES_MAPPING,
    bn_flops_counter_hook,
    conv_flops_counter_hook,
    empty_flops_counter_hook,
    linear_flops_counter_hook,
    relu_flops_counter_hook,
)
from pytorch_lightning import LightningModule
from torchinfo import summary

from yamle.defaults import (
    ALL_DATASETS_KEY,
    MODULE_HARDWARE_PROPERTIES_KEY,
    MODULE_OUTPUT_SHAPE_KEY,
    PROFILING_TIME_KEY,
    MODULE_FLOPS_KEY,
    MODULE_PARAMS_KEY,
    MODULE_CUMULATIVE_FLOPS_KEY,
    MODULE_CUMULATIVE_PARAMS_KEY,
    MODULE_NAME_KEY,
)
from yamle.utils.tracing_utils import (
    get_input_shape_from_model,
    get_sample_input_and_target,
)
from yamle.models.operations import (
    Add,
    Multiply,
    MatrixMultiplication,
    OutputActivation,
    Reduction,
    ScalarAdder,
    ScalarMultiplier,
)
from yamle.models.transformer import PositionalEncoding
from yamle.models.specific.be import Conv2dBE, LinearBE
from yamle.models.specific.mcdropout import (
    Conv2dDropConnect,
    Dropout1d,
    Dropout2d,
    Dropout3d,
    LinearDropConnect,
    StochasticDepth,
)
from yamle.models.specific.rbnn import Conv2dRBNN, LinearRBNN
from yamle.models.specific.sngp import RFF
from yamle.models.specific.svi import (
    Conv2dSVIFlipOut,
    Conv2dSVILRT,
    Conv2dSVIRT,
    LinearSVIFlipOut,
    LinearSVILRT,
    LinearSVIRT,
)
from yamle.models.specific.temperature_scaling import TemperatureScaler
from yamle.models.model import BaseModel

logging = logging.getLogger("pytorch_lightning")


@torch.no_grad()
def model_complexity(
    model: nn.Module,
    method: LightningModule,
    devices: Optional[torch.device] = None,
    results: Optional[Dict[str, Any]] = None,
    print_per_layer_stat: bool = True,
    verbose: bool = True,
    ignore_modules: List = [],
    flops_units: Optional[str] = None,
    param_units: Optional[str] = None,
    output_precision: int = 2,
    profiling_runs: int = 100,
) -> None:
    """Computes the flops and parameters of a model.

    For each custom method it is necesary to implement a custom fucntion that is going to count the number of FLOPs in the given operation.
    This is true only for operations in which an actual computation is performed. For example, for routing modules which put operations together
    such as `ResidualLayer` it is not necessary to implement a custom function since the FLOPs are computed by summing the FLOPs of the individual
    operations.

    When implementing a function it is important to note that the `__flops__` attribute already exists and with each forward pass,
    the FLOPS need to be added and not overwritten. For example, for a `Linear` layer the FLOPs are computed as follows:

    ```
    def linear_flops_counter_hook(module, input, output):
        input = input[0]
        # pytorch checks dimensions, so here we don't care much
        output_last_dim = output.shape[-1]
        bias_flops = output_last_dim if module.bias is not None else 0
        module.__flops__ += int(np.prod(input.shape) * output_last_dim + bias_flops)
    ```

    The function additionally naively profiles the model both on CPU and GPU if available.

    Args:
        model (nn.Module): Model to compute the complexity of.
        method (LightningModule): Lightning module to get the input shape, dtype and the test step function.
        devices (Optional[List[str]], optional): Device ids to use for the profiling. Defaults to None.
        print_per_layer_stat (bool, optional): Whether to print the complexity of each layer. Defaults to True.
        verbose (bool, optional): Whether to print the complexity of the model. Defaults to True.
        ignore_modules (List, optional): List of modules to ignore. Defaults to [].
        flops_units (Optional[str], optional): Units to use for the FLOPs. Defaults to None.
        param_units (Optional[str], optional): Units to use for the parameters. Defaults to None.
        output_precision (int, optional): Precision to use for the output. Defaults to 2.
        profiling_runs (int, optional): Number of runs to average the profiling over. Defaults to 100.
    """

    logging.write = logging.info
    batch_size = 1
    assert (
        batch_size == 1
    ), "Batch size must be 1 for complexity computation. The code has been adapted exactly for this case."
    method.eval()
    flops_model = add_flops_counting_methods(model)
    flops_model.start_flops_count(
        ost=logging, verbose=verbose, ignore_list=ignore_modules
    )
    batch = get_sample_input_and_target(method, batch_size=batch_size)
    method.test_step(batch, batch_idx=0)
    flops_model.__batch_counter__ = batch_size
    flops, params = flops_model.compute_average_flops_cost()
    setattr(model, MODULE_FLOPS_KEY, flops)
    setattr(model, MODULE_PARAMS_KEY, params)
    if print_per_layer_stat:
        print_model_with_flops(
            flops_model,
            flops,
            params,
            ost=logging,
            flops_units=flops_units,
            param_units=param_units,
            precision=output_precision,
        )
    flops_model.stop_flops_count()
    accumulate_flops_and_params_over_containers(flops_model)

    class ModelWrapper(nn.Module):
        """Wrapper for the model to be analyzed by FVCore/torchsummary.

        We need to use the `._predict` method of the method to get the output of the model
        such that it uses all the modules of the model.
        """

        def __init__(
            self,
            model: nn.Module,
            predict_fn: Callable[[torch.Tensor, Any, Any], torch.Tensor],
        ):
            super().__init__()
            self._model = model
            self._predict = predict_fn

        def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            return self._predict(x, *args, **kwargs)

    input_shape = list(
        get_input_shape_from_model(flops_model)[0]
    )  # This gets the input shape of the model
    # Replace the batch size with 1
    input_shape[0] = 1
    x = torch.randn(*input_shape).to(next(flops_model.parameters()).device)

    fvcore_flops = FlopCountAnalysis(
        ModelWrapper(flops_model, method._predict), x
    ).total()
    model_stats = summary(
        ModelWrapper(flops_model, method._predict), input_data=x, verbose=2
    )
    logging.info(model_stats)

    # Make all the hardware information available to the user
    # by storing it in the model (this function can be run befor training to figure out the hardware cost)
    if hasattr(model, MODULE_HARDWARE_PROPERTIES_KEY):
        logging.warning(
            f"Overwriting the hardware properties of the model, formerly stored in {MODULE_HARDWARE_PROPERTIES_KEY}: {getattr(model, MODULE_HARDWARE_PROPERTIES_KEY)}"
        )

    hardware_properties = {
        "flops": flops,
        "fvcore_flops": fvcore_flops,
        "torchinfo_flops": model_stats.total_mult_adds,
        "params": params,
        "torchinfo_params": model_stats.total_params,
        "torchinfo_forwardbackward_params": model_stats.total_output_bytes,
    }
    setattr(model, MODULE_HARDWARE_PROPERTIES_KEY, hardware_properties)

    # Perform the profiling on the CPU and GPU if available
    if isinstance(devices, (tuple, list)) and len(devices) > 0:
        # Convert the ids into cuda:0, cuda:1, ...
        devices = [f"cuda:{device}" for device in devices]
    elif torch.cuda.is_available():
        # If no devices are specified but cuda is available, use cuda:0
        devices = ["cuda:0"]
    elif devices is None:
        devices = []

    devices.append("cpu")
    times = []
    device_descriptions = []
    original_device = next(model.parameters()).device
    """
    with torch.no_grad():
        for device in devices:
            # Move the model to the device
            model.to(device)
            # Move all the data in the batch to the device
            batch = [tensor.to(device) for tensor in batch]
            model.eval()
            total_time = 0
            # This is not absolutely correct since test step
            # can also perform different operations e.g. loss computation
            # Average over 100 iterations
            for _ in range(profiling_runs):
                start = time.time()
                method.test_step(batch, batch_idx=0)
                end = time.time()
                total_time += end - start
            times.append(total_time / profiling_runs)
            device_descriptions.append(
                torch.cuda.get_device_name(device)
                if device.startswith("cuda")
                else platform.processor()
            )
    model.to(original_device)
    """
    # Store all the results from the different libraries in the results dictionary
    # Store the results in the dictionary
    if results is not None:
        results[ALL_DATASETS_KEY] = {}
        results[ALL_DATASETS_KEY]["flops"] = flops
        results[ALL_DATASETS_KEY]["fvcore_flops"] = fvcore_flops
        results[ALL_DATASETS_KEY]["torchinfo_flops"] = model_stats.total_mult_adds
        results[ALL_DATASETS_KEY]["params"] = params
        results[ALL_DATASETS_KEY]["torchinfo_params"] = model_stats.total_params
        results[ALL_DATASETS_KEY][
            "torchinfo_forwardbackward_params"
        ] = model_stats.total_output_bytes
        results[ALL_DATASETS_KEY][PROFILING_TIME_KEY] = {}
        """
        for i, device in enumerate(devices):
            results[ALL_DATASETS_KEY][PROFILING_TIME_KEY][device] = {}
            results[ALL_DATASETS_KEY][PROFILING_TIME_KEY][device]["time"] = times[i]
            results[ALL_DATASETS_KEY][PROFILING_TIME_KEY][device][
                "description"
            ] = device_descriptions[i]
        """

    trace_cumulative_flops_params(method)


def accumulate_flops_and_params_over_containers(module: nn.Module) -> None:
    """This is a helper function which accumulates the flops and params of a container module.

    The container modules and ``nn.Sequential`` or ``nn.ModuleList``. which do not incur any FLOPs or params.
    We sum the FLOPs and params of the modules in the container and store them in the attributes of the container.
    """
    for name, module in module.named_modules():
        if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
            # Delete the attributes of the container
            if hasattr(module, MODULE_FLOPS_KEY):
                delattr(module, MODULE_FLOPS_KEY)
            if hasattr(module, MODULE_PARAMS_KEY):
                delattr(module, MODULE_PARAMS_KEY)
            # Recursively accumulate the flops and params of the container
            for m in module.modules():
                if m is not module:
                    accumulate_flops_and_params_over_containers(m)
            # For all the submodules of the container, sum the flops and params
            # Except other nn.Sequential or nn.ModuleList which are not counted
            flops = sum(
                [
                    0
                    if isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList)
                    else getattr(m, MODULE_FLOPS_KEY, 0)
                    for m in module.modules()
                ]
            )
            params = sum(
                [
                    0
                    if isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList)
                    else getattr(m, MODULE_PARAMS_KEY, 0)
                    for m in module.modules()
                ]
            )
            setattr(module, MODULE_FLOPS_KEY, flops)
            setattr(module, MODULE_PARAMS_KEY, params)


def add_flops_counting_methods(net_main_module: nn.Module) -> nn.Module:
    """Adds flops counting methods to the model."""
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(
        net_main_module
    )

    net_main_module.reset_flops_count()

    return net_main_module


def stop_flops_count(self) -> None:
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.
    This is the only method which is changed in comparison to the original
    repository, we simply avoid deleting the flops variable
    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)


def reduction_flops_counter_hook(
    module: Reduction, input: torch.Tensor, output: torch.Tensor
) -> None:
    """Computes the flops of a reduction module."""
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(output.shape),
    )

    if module._alignment:
        # It computes self attention given that the input has a shape of (batch_size, seq_len, flattened_dim)
        # The attention is computed on the seq_len dimension
        # The attention is computed as a matrix multiplication of (flattened_dim, flattened_dim)
        input = input[0].view(input[0].size(0), input[0].size(1), -1)
        input_shape = input.shape
        if input_shape[1] > 1:
            B, L, D = input_shape
            setattr(
                module,
                MODULE_FLOPS_KEY,
                getattr(module, MODULE_FLOPS_KEY) + B * L**2 * D,
            )


def positional_encoding_flops_counter_hook(
    module: PositionalEncoding, input: torch.Tensor, output: torch.Tensor
) -> None:
    """Computes the flops of a positional encoding module."""
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(output.shape),
    )


def temperature_flops_counter_hook(
    module: TemperatureScaler, input: torch.Tensor, output: torch.Tensor
) -> None:
    """Computes the flops of a temperature scaling module."""
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(output.shape),
    )


def linearsvirt_flops_counter_hook(
    module: LinearSVIRT, input: torch.Tensor, output: torch.Tensor
) -> None:
    """Computes the flops of a linear module."""
    linear_flops_counter_hook(module, input, output)
    # To account for sampling of the weights
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(output.shape) * 2,
    )
    if module.bias is not None:
        setattr(
            module,
            MODULE_FLOPS_KEY,
            getattr(module, MODULE_FLOPS_KEY) + np.prod(module.bias.shape) * 2,
        )


def matrix_multiplication_flops_counter_hook(
    module: MatrixMultiplication, input: torch.Tensor, output: torch.Tensor
) -> None:
    """Computes the flops of a matrix multiplication module.

    It takes two matrices as input and computes the matrix multiplication."""
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(output.shape) * 2,
    )


def ln_flops_counter_hook(
    module: nn.LayerNorm, input: torch.Tensor, output: torch.Tensor
) -> None:
    """Computes the flops of a layer normalization module."""
    input = input[0]

    norm_flops = np.prod(input.shape)
    if module.elementwise_affine:
        norm_flops *= 2
    setattr(
        module, MODULE_FLOPS_KEY, getattr(module, MODULE_FLOPS_KEY) + int(norm_flops)
    )


def linearsvilrt_flops_counter_hook(
    module: LinearSVILRT, input: torch.Tensor, output: torch.Tensor
) -> None:
    """Computes the flops of a linear module."""
    linear_flops_counter_hook(module, input, output)
    linear_flops_counter_hook(module, input, output)
    # To account for the sampling of the output
    module.__flops__ += np.prod(output.shape)


def linearsviflipout_flops_counter_hook(
    module: LinearSVIFlipOut, input: torch.Tensor, output: torch.Tensor
) -> None:
    """Computes the flops of a linear module."""
    linear_flops_counter_hook(module, input, output)
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(output.shape) * 2,
    )
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(input.shape),
    )


def linearbe_flops_counter_hook(
    module: LinearBE, input: torch.Tensor, output: torch.Tensor
) -> None:
    """Computes the flops of a linear module."""
    linear_flops_counter_hook(module, input, output)
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(output.shape),
    )
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(input[0].shape),
    )


def lineardropconnect_flops_counter_hook(
    module: LinearDropConnect, input: torch.Tensor, output: torch.Tensor
) -> None:
    """Computes the flops of a linear module."""
    linear_flops_counter_hook(module, input, output)
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(module.weight.shape),
    )
    if module.bias is not None:
        setattr(
            module,
            MODULE_FLOPS_KEY,
            getattr(module, MODULE_FLOPS_KEY) + np.prod(module.bias.shape),
        )


def linearrbnn_flops_counter_hook(
    module: LinearRBNN, input: torch.Tensor, output: torch.Tensor
) -> None:
    """Computes the flops of a linear module."""
    linearbe_flops_counter_hook(module, input, output)
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(module.r.shape),
    )
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(module.s.shape),
    )


def convsvirt_flops_counter_hook(
    module: Conv2dSVIRT, input: torch.Tensor, output: torch.Tensor
) -> None:
    """Computes the flops of a convolutional module."""
    conv_flops_counter_hook(module, input, output)
    # To account for sampling of the weights
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(module.weight.shape) * 2,
    )
    if module.bias is not None:
        setattr(
            module,
            MODULE_FLOPS_KEY,
            getattr(module, MODULE_FLOPS_KEY) + np.prod(module.bias.shape) * 2,
        )


def convsvilrt_flops_counter_hook(
    module: Conv2dSVILRT, input: torch.Tensor, output: torch.Tensor
) -> None:
    """Computes the flops of a convolutional module."""
    conv_flops_counter_hook(module, input, output)
    conv_flops_counter_hook(module, input, output)
    # To account for the sampling of the output
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(output.shape),
    )


def convsviflipout_flops_counter_hook(
    module: Conv2dSVIFlipOut, input: torch.Tensor, output: torch.Tensor
) -> None:
    """Computes the flops of a convolutional module."""
    conv_flops_counter_hook(module, input, output)
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(output.shape) * 2,
    )
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(input[0].shape),
    )


def convbe_flops_counter_hook(
    module: Conv2dBE, input: torch.Tensor, output: torch.Tensor
) -> None:
    """Computes the flops of a convolutional module."""
    conv_flops_counter_hook(module, input, output)
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(output.shape),
    )
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(input[0].shape),
    )


def convdropconnect_flops_counter_hook(
    module: Conv2dDropConnect, input: torch.Tensor, output: torch.Tensor
) -> None:
    """Computes the flops of a convolutional module."""
    conv_flops_counter_hook(module, input, output)
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(module.weight.shape),
    )
    if module.bias is not None:
        setattr(
            module,
            MODULE_FLOPS_KEY,
            getattr(module, MODULE_FLOPS_KEY) + np.prod(module.bias.shape),
        )


def convrbnn_flops_counter_hook(
    module: Conv2dRBNN, input: torch.Tensor, output: torch.Tensor
) -> None:
    """Computes the flops of a convolutional module."""
    convbe_flops_counter_hook(module, input, output)
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(module.r.shape),
    )
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(module.s.shape),
    )



def rff_flops_counter_hook(
    module: RFF, input: torch.Tensor, output: torch.Tensor
) -> None:
    """Computes the flops of a random feature module."""
    input_shape = input[0].shape
    output_shape = output.shape
    W_shape = module.W.shape

    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY)
        + np.prod(input_shape) * W_shape[0]
        + W_shape[0],
    )
    setattr(
        module, MODULE_FLOPS_KEY, getattr(module, MODULE_FLOPS_KEY) + 2 * W_shape[1]
    )
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + np.prod(output_shape),
    )
    output_shape = getattr(module, MODULE_OUTPUT_SHAPE_KEY, None)
    setattr(
        module,
        MODULE_FLOPS_KEY,
        getattr(module, MODULE_FLOPS_KEY) + output_shape[0] * output_shape[0][0],
    )


CUSTOM_MODULES_MAPPING[nn.Flatten] = empty_flops_counter_hook
CUSTOM_MODULES_MAPPING[nn.Tanh] = relu_flops_counter_hook
CUSTOM_MODULES_MAPPING[nn.Sigmoid] = relu_flops_counter_hook
CUSTOM_MODULES_MAPPING[nn.Softplus] = relu_flops_counter_hook
CUSTOM_MODULES_MAPPING[nn.Softmax] = relu_flops_counter_hook
CUSTOM_MODULES_MAPPING[nn.Dropout] = relu_flops_counter_hook
CUSTOM_MODULES_MAPPING[nn.Dropout2d] = relu_flops_counter_hook
CUSTOM_MODULES_MAPPING[nn.Dropout3d] = relu_flops_counter_hook
CUSTOM_MODULES_MAPPING[Dropout1d] = relu_flops_counter_hook
CUSTOM_MODULES_MAPPING[Dropout2d] = relu_flops_counter_hook
CUSTOM_MODULES_MAPPING[Dropout3d] = relu_flops_counter_hook
CUSTOM_MODULES_MAPPING[nn.Identity] = empty_flops_counter_hook
CUSTOM_MODULES_MAPPING[nn.LayerNorm] = ln_flops_counter_hook

CUSTOM_MODULES_MAPPING[Add] = relu_flops_counter_hook
CUSTOM_MODULES_MAPPING[Multiply] = relu_flops_counter_hook
CUSTOM_MODULES_MAPPING[MatrixMultiplication] = matrix_multiplication_flops_counter_hook
CUSTOM_MODULES_MAPPING[ScalarAdder] = relu_flops_counter_hook
CUSTOM_MODULES_MAPPING[ScalarMultiplier] = relu_flops_counter_hook
CUSTOM_MODULES_MAPPING[Reduction] = reduction_flops_counter_hook
CUSTOM_MODULES_MAPPING[OutputActivation] = relu_flops_counter_hook
CUSTOM_MODULES_MAPPING[TemperatureScaler] = temperature_flops_counter_hook
CUSTOM_MODULES_MAPPING[LinearSVIRT] = linearsvirt_flops_counter_hook
CUSTOM_MODULES_MAPPING[LinearSVILRT] = linearsvilrt_flops_counter_hook
CUSTOM_MODULES_MAPPING[LinearSVIFlipOut] = linearsviflipout_flops_counter_hook
CUSTOM_MODULES_MAPPING[LinearBE] = linearbe_flops_counter_hook
CUSTOM_MODULES_MAPPING[LinearDropConnect] = lineardropconnect_flops_counter_hook
CUSTOM_MODULES_MAPPING[LinearRBNN] = linearrbnn_flops_counter_hook
CUSTOM_MODULES_MAPPING[Conv2dSVIRT] = convsvirt_flops_counter_hook
CUSTOM_MODULES_MAPPING[Conv2dSVILRT] = convsvilrt_flops_counter_hook
CUSTOM_MODULES_MAPPING[Conv2dSVIFlipOut] = convsviflipout_flops_counter_hook
CUSTOM_MODULES_MAPPING[Conv2dBE] = convbe_flops_counter_hook
CUSTOM_MODULES_MAPPING[Conv2dDropConnect] = convdropconnect_flops_counter_hook
CUSTOM_MODULES_MAPPING[Conv2dRBNN] = convrbnn_flops_counter_hook
CUSTOM_MODULES_MAPPING[StochasticDepth] = relu_flops_counter_hook


def cumulative_flops_params_hook(
    module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
) -> Optional[torch.Tensor]:
    """This function is used to cache the cumulative flops and params of a module.

    The cumulative flops and params will be stored in the module as ``CUMULATIVE_MODULE_FLOPS_KEY`` and ``CUMULATIVE_PARAMS_KEY``.

    The method passes flops and params with respect to the input, output tensors which are passed to the module.
    The input/output tensor(s) hold a dictionary of all the previous flops and params
    where the name of the module is the key and the value is the flops and params of the module.
    This way we can check to make sure that if the module has multiple inputs, the flops and params are only accumulated once.
    The module names are unique, so we do not need to worry about overwriting the flops and params of a module.
    The current module then adds its flops and params to the dictionary of the output tensor(s).
    The accumulation is only with respect to modules which are in the ``MODULES_MAPPING`` or ``CUSTOM_MODULES_MAPPING``,
    otherwise warning is raised.

    Args:
        module (torch.nn.Module): The module to cache the cumulative flops and params of.
        input (torch.Tensor): The input to the module.
        output (torch.Tensor): The output of the module.
    """
    # For nested modules the input alone does not have all the attributes
    # Therefore we need to look into the output as well to add any flops or params
    # for any nested modules
    input_output_flops_and_params = []
    for data in [input, output]:
        if isinstance(data, (tuple, list)):
            for x in data:
                input_output_flops_and_params.append(getattr(x, "_flops_params", {}))
        else:
            input_output_flops_and_params.append(getattr(data, "_flops_params", {}))

    # Add all the extra keys with respect to all inputs and outputs
    allkeys = set()
    for dictionary in input_output_flops_and_params:
        for key in dictionary.keys():
            if key not in allkeys:
                allkeys.add(key)

    # Create a single dictionary with all the keys for both flops and params
    flops_and_params = {}
    for key in allkeys:
        for dictionary in input_output_flops_and_params:
            if key in flops_and_params and key in dictionary:
                assert (
                    flops_and_params[key][0] == dictionary[key][0]
                ), f"Module {key} has different flops in different inputs. {flops_and_params[key][0]} != {dictionary[key][0]}"
                assert (
                    flops_and_params[key][1] == dictionary[key][1]
                ), f"Module {key} has different params in different inputs. {flops_and_params[key][1]} != {dictionary[key][1]}"

            if key not in flops_and_params and key in dictionary:
                flops_and_params[key] = copy.deepcopy(dictionary[key])

    def _cumulative_flops_and_params(
        dictionary: Dict[str, Tuple[int, int]]
    ) -> Tuple[int, int]:
        """Computes the cumulative flops and params of a dictionary."""
        flops = 0
        params = 0
        for key, value in dictionary.items():
            flops += value[0]
            params += value[1]
        return flops, params

    def _set_module_output_flops_and_params(
        module: torch.nn.Module,
        out: torch.Tensor,
        dictionary: Dict[str, Tuple[int, int]],
    ) -> None:
        """Sets the flops and params of the module."""
        if isinstance(out, (tuple, list)):
            for x in out:
                setattr(x, "_flops_params", dictionary)
        else:
            setattr(out, "_flops_params", dictionary)
        flops, params = _cumulative_flops_and_params(dictionary)
        assert (
            isinstance(flops, int) or flops.is_integer()
        ), f"Flops {flops} is not an integer for module {getattr(module, MODULE_NAME_KEY, module.__class__.__name__)}."
        assert (
            isinstance(params, int) or params.is_integer()
        ), f"Params {params} is not an integer for module {getattr(module, MODULE_NAME_KEY, module.__class__.__name__)}."
        setattr(module, MODULE_CUMULATIVE_FLOPS_KEY, flops)
        setattr(module, MODULE_CUMULATIVE_PARAMS_KEY, params)

    if not isinstance(
        module, tuple(MODULES_MAPPING.keys()) + tuple(CUSTOM_MODULES_MAPPING.keys())
    ):
        logging.warning(
            f"Module {getattr(module, MODULE_NAME_KEY, module.__class__.__name__)} is not in the MODULES_MAPPING or CUSTOM_MODULES_MAPPING, the flops and params are not updated."
        )
    elif isinstance(module, (nn.Sequential, nn.ModuleList)):
        logging.warning(
            f"Module {getattr(module, MODULE_NAME_KEY, module.__class__.__name__)} is a sequential module, the flops and params are not updated."
        )
    else:
        # We add the flops and params of the current module to the dictionary
        # We update the cumulative flops and params of the module
        flops_and_params[getattr(module, MODULE_NAME_KEY)] = (
            getattr(module, MODULE_FLOPS_KEY, 0),
            getattr(module, MODULE_PARAMS_KEY, 0),
        )
    _set_module_output_flops_and_params(module, output, flops_and_params)


@torch.no_grad()
def trace_cumulative_flops_params(method: LightningModule) -> None:
    """This method is used to trace the cumulative flops and params of a model."""
    method.eval()
    hooks = []
    # Apply the hooks to all submodules of the model except the model itself
    for m in method.model.modules():
        if not isinstance(m, BaseModel):
            hooks.append(m.register_forward_hook(cumulative_flops_params_hook))

    batch = get_sample_input_and_target(method)
    method.test_step(batch, batch_idx=0)
    for hook in hooks:
        hook.remove()
    method.train()
