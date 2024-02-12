import argparse
import logging
from typing import Any

import torch
from pytorch_lightning import LightningModule, Trainer
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MovingAverageMinMaxObserver


from yamle.defaults import QUANTIZED_KEY
from yamle.quantization.quantizer import BaseQuantizer

logging = logging.getLogger("pytorch_lightning")


class QATQuantizer(BaseQuantizer):
    """This is the quantization-aware training quantizer class.

    It performs quantization-aware training on the model.
    In contrast to the static quantizer, the quantizer uses the calibration
    (validation) dataset to fine-tune the model.
    It uses the same optimiser as the one used for training the model.

    Args:
        learning_rate (float): The learning rate to use for the fine-tuning.
        epochs (int): The number of epochs to use for the fine-tuning.
    """

    def __init__(
        self, learning_rate: float, epochs: int, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._learning_rate = learning_rate
        self._epochs = epochs

    def __call__(self, trainer: Trainer, method: LightningModule) -> None:
        """This method is used to quantize the model.

        A copy of the model is saved before quantization.
        First the model is prepared for quantization.
        Then the trainer is queried to fine-tune the model.
        Then the model is quantized.
        The original model is kept such that it can be recovered.
        """
        self.save_original_model(method)
        self.prepare(trainer, method)
        trainer.fine_tune(self._epochs)

        method.model.apply(torch.ao.quantization.disable_observer)

        self.save_quantized_model(method)
        logging.info("Model quantized.")
        logging.info(method.model)
        setattr(method, QUANTIZED_KEY, True)
        self.cleanup(method, trainer)

    def prepare(self, trainer: Trainer, method: LightningModule) -> None:
        """This method is used to prepare the model for quantization.

        It caches the original hyperparameters for the optimisation and replaces the hyperparameters
        with the ones for the fine-tuning.
        """
        method.model.eval()
        self.replace_layers_for_quantization(method.model)
        method.model.qconfig = self.get_qconfig()
        method.model.train()
        torch.quantization.prepare_qat(method.model, inplace=True)
        logging.info("Model prepared for quantization.")
        logging.info(method.model)

        # Cache the original hyperparameters
        self._original_hyperparameters = {
            "learning_rate": method.hparams.learning_rate,
            "epochs": trainer._epochs,
        }

        # Replace the hyperparameters
        method.hparams.learning_rate = self._learning_rate
        trainer._epochs = self._epochs

    def cleanup(
        self, method: LightningModule, trainer: Trainer, *args: Any, **kwargs: Any
    ) -> None:
        """This method is used to clean up the model after quantization."""
        super().cleanup(*args, **kwargs)

        # Recover the original hyperparameters
        method.hparams.learning_rate = self._original_hyperparameters["learning_rate"]
        trainer._epochs = self._original_hyperparameters["epochs"]

        del self._original_hyperparameters

    def get_qconfig(self) -> Any:
        """This method is used to get the quantization configuration.

        We use the number of activation and weight bits to create the quantization configuration.
        """
        # Else specify the qconfig manually based on the activation and weight bits
        activation_bits = self._activation_bits
        weight_bits = self._weight_bits

        activation_fq = FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=int(2**activation_bits - 1),
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,  # Since this is in simulation, we don't want to reduce the range
        )
        weight_fq = FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=-int((2**weight_bits) / 2),
            quant_max=int((2**weight_bits) / 2 - 1),
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False,  # Since this is in simulation, we don't want to reduce the range
        )
        return torch.quantization.QConfig(activation=activation_fq, weight=weight_fq)

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add specific arguments to the parser."""
        parser = super(QATQuantizer, QATQuantizer).add_specific_args(parent_parser)
        parser.add_argument(
            "--quantizer_learning_rate",
            type=float,
            default=1e-3,
            help="The learning rate to use for the fine-tuning.",
        )
        parser.add_argument(
            "--quantizer_epochs",
            type=int,
            default=1,
            help="The number of epochs to use for the fine-tuning.",
        )
        return parser
