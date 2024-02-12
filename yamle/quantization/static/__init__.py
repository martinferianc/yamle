import logging
from typing import Any

import torch
from pytorch_lightning import LightningModule, Trainer
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import HistogramObserver

from yamle.defaults import QUANTIZED_KEY
from yamle.quantization.quantizer import BaseQuantizer

logging = logging.getLogger("pytorch_lightning")


class StaticQuantizer(BaseQuantizer):
    """This is the static quantizer class.

    It performs static post-training quantization on the model.
    It does it with respect to a specific number of bits for the activation and weight.
    The quantization is simulated and the model is not actually quantized.

    """

    def __call__(self, trainer: Trainer, method: LightningModule) -> None:
        """This method is used to quantize the model.

        A copy of the model is saved before quantization, just in case.
        First the model is prepared for quantization.
        Then the trainer is queried for the dataloader - this can be used to calibrate the model or fine-tune it.
        Then the the fake quantization is applied to the model and the observer is disabled to simulate quantization.
        The original model is kept such that it can be recovered.
        """
        self.save_original_model(method)
        self.prepare(trainer, method)

        trainer.calibrate()

        method.model.apply(torch.ao.quantization.enable_fake_quant)
        method.model.apply(torch.ao.quantization.disable_observer)

        self.save_quantized_model(method)
        logging.info("Model quantized.")
        logging.info(method.model)
        setattr(method, QUANTIZED_KEY, True)

    def prepare(self, trainer: Trainer, method: LightningModule) -> None:
        """This method is used to prepare the model for quantization."""
        method.model.eval()
        self.replace_layers_for_quantization(method.model)
        method.model.qconfig = self.get_qconfig()
        method.model.train()
        torch.quantization.prepare_qat(method.model, inplace=True)
        method.model.apply(torch.ao.quantization.disable_fake_quant)
        logging.info("Model prepared for quantization.")
        logging.info(method.model)

    def get_qconfig(self) -> Any:
        """This method is used to get the quantization configuration.

        We use the number of activation and weight bits to create the quantization configuration.
        """
        # Else specify the qconfig manually based on the activation and weight bits
        activation_bits = self._activation_bits
        weight_bits = self._weight_bits

        activation_fq = FakeQuantize.with_args(
            observer=HistogramObserver,
            quant_min=0,
            quant_max=int(2**activation_bits - 1),
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,  # Since this is in simulation, we don't want to reduce the range
        )
        weight_fq = FakeQuantize.with_args(
            observer=HistogramObserver,
            quant_min=-int((2**weight_bits) / 2),
            quant_max=int((2**weight_bits) / 2 - 1),
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False,  # Since this is in simulation, we don't want to reduce the range
        )

        qconfig = torch.quantization.QConfig(activation=activation_fq, weight=weight_fq)
        return qconfig
