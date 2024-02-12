import argparse
import copy
import logging
from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer

from yamle.defaults import FLOAT_MODEL_KEY, QUANTIZED_KEY, QUANTIZED_MODEL_KEY
from yamle.models.operations import Add
from yamle.models.specific.mcdropout import Dropout1d, Dropout2d, Dropout3d
from yamle.quantization.models.operations import QuantizableAdd
from yamle.quantization.models.specific.mcdropout import (
    QuantisedDropout1d,
    QuantisedDropout2d,
    QuantisedDropout3d,
)

logging = logging.getLogger("pytorch_lightning")


class BaseQuantizer(ABC):
    """This is the base class for all quantization methods.

    The quantizer's call method will be used to quantize the model.

    Args:
        activation_bits (int): The number of bits to use for the activation.
        weight_bits (int): The number of bits to use for the weight.
    """

    def __init__(self, activation_bits: int, weight_bits: int) -> None:
        assert (
            0 <= activation_bits <= 8
        ), "The number of bits for the activation must be between 0 and 8. Got {activation_bits}."
        assert (
            0 <= weight_bits <= 8
        ), "The number of bits for the weight must be between 0 and 8. Got {weight_bits}."
        self._activation_bits = activation_bits
        self._weight_bits = weight_bits

    @abstractmethod
    def __call__(self, trainer: Trainer, method: LightningModule) -> None:
        """This method is used to quantize the model.

        A copy of the model is saved before quantization.
        First the model is prepared for quantization.
        Then the trainer is queried for the dataloader - this can be used to calibrate the model or fine-tune it.
        Then the model is quantized.
        The original model is kept such that it can be recovered.
        """
        raise NotImplementedError("This method needs to be implemented.")

    @abstractmethod
    def prepare(self, *args: Any, **kwargs: Any) -> None:
        """This method is used to prepare the model for quantization."""
        raise NotImplementedError("This method needs to be implemented.")

    @abstractmethod
    def get_qconfig(self) -> Any:
        """This method is used to get the quantization configuration."""
        raise NotImplementedError("This method needs to be implemented.")

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        """This method is used to clean up the model after quantization."""
        pass

    def save_original_model(self, method: LightningModule) -> None:
        """This method is used to create a copy of the original model."""
        # Give a warning if original model already exists
        if hasattr(method, FLOAT_MODEL_KEY):
            logging.warning("Original model already exists. Overwriting it.")
        model_copy = copy.deepcopy(method.model.cpu())
        setattr(method, FLOAT_MODEL_KEY, model_copy)

    def save_quantized_model(self, method: LightningModule) -> None:
        """This method is used to save the quantized model."""
        # Give a warning if quantized model already exists
        if hasattr(method, QUANTIZED_MODEL_KEY):
            logging.warning("Quantized model already exists. Overwriting it.")
        q_model = copy.deepcopy(method.model.cpu())
        setattr(method, QUANTIZED_MODEL_KEY, q_model)

    def recover(self, method: LightningModule) -> None:
        """This method is used to recover the original model."""
        if hasattr(method, QUANTIZED_KEY) and getattr(method, QUANTIZED_KEY):
            setattr(method, QUANTIZED_MODEL_KEY, getattr(method, "model"))
            setattr(method, "model", getattr(method, FLOAT_MODEL_KEY))
            delattr(method, FLOAT_MODEL_KEY)
            setattr(method, QUANTIZED_KEY, False)

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the pruner specific arguments to the parent parser."""
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--quantizer_activation_bits",
            type=int,
            default=8,
            help="The number of bits to use for the activation.",
        )
        parser.add_argument(
            "--quantizer_weight_bits",
            type=int,
            default=8,
            help="The number of bits to use for the weight.",
        )
        return parser

    def replace_layers_for_quantization(self, model: nn.Module) -> None:
        """This function takes a model and replaces any special layers with their quantizable counterparts. e.g. Add -> FloatFunctional.add"""

        def _recursive_replace(module: nn.Module) -> None:
            for name, child in module.named_children():
                if isinstance(child, Add):
                    setattr(module, name, QuantizableAdd())
                elif isinstance(child, Dropout1d):
                    setattr(module, name, QuantisedDropout1d(module._p))
                elif isinstance(child, Dropout2d):
                    setattr(module, name, QuantisedDropout2d(module._p))
                elif isinstance(child, Dropout3d):
                    setattr(module, name, QuantisedDropout3d(module._p))
                else:
                    _recursive_replace(child)

        _recursive_replace(model)

        for m in model.modules():
            if hasattr(m, "replace_layers_for_quantization"):
                m.replace_layers_for_quantization()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class DummyQuantizer(BaseQuantizer):
    """This is a dummy quantizer that does not perform any quantization."""

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def prepare(self) -> None:
        pass

    def get_qconfig(self) -> None:
        pass
