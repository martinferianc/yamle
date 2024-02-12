import torch
import torch.nn as nn
from yamle.utils.tracing_utils import get_input_shape_from_model
import onnx

import logging

logging = logging.getLogger("pytorch_lightning")


def export_onnx(model: nn.Module, path: str) -> None:
    """This method is used to export the model to ONNX.

    Args:
        model (nn.Module): The model to export.
        path (str): The path to save the model.
    """
    model.eval()
    input_shape = list(get_input_shape_from_model(model)[0])
    input_shape[0] = 1
    x = torch.randn(*input_shape).to(next(model.parameters()).device)
    
    # Make a pass to count the number of outputs
    outputs = model(x)
    num_outputs = len(outputs) if isinstance(outputs, (list, tuple)) else 1
    
    output_names = [f"output_{i}" for i in range(num_outputs)]
    dynamic_axes = {
        "input": {0: "batch_size"},
    }
    for i in range(num_outputs):
        dynamic_axes[output_names[i]] = {0: "batch_size"}
    
    logging.info("Exporting model to ONNX.")
    torch.onnx.export(
        model,
        x,
        path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # Perform sanity check
    logging.info("Checking ONNX model.")
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    logging.info("ONNX model is valid.")
