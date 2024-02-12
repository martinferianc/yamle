import argparse
import logging
from typing import Any, Dict

import torch.nn as nn
from backpack import backpack, extend, extensions
from tqdm import tqdm
import torch

from yamle.defaults import CLASSIFICATION_KEY, MEMBERS_DIM
from yamle.methods.uncertain_method import MCSamplingMethod
from yamle.models.specific.laplace import LaplaceLinear
from yamle.regularizers.weight import L2Regularizer
import math

logging = logging.getLogger("pytorch_lightning")


class LaplaceMethod(MCSamplingMethod):
    """This class implements the Laplace method.

    It performs Laplace approximation in the neural network to approximate the posterior distribution.

    It is important the regularisation is the `L2Regularizer` and the `regularizer_weight` is non-zero.
    These act as the prior distribution.

    At the moment, it only supports the `last_layer` mode and the `classification` task.
    This is because the `backpack` library supports only the `CrossEntropyLoss` but not the `GaussianNLLLoss`.

    Args:
        mode (str): The mode of the method. It can be `last_layer` for now.
    """

    tasks = [CLASSIFICATION_KEY]

    def __init__(
        self,
        mode: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert mode in [
            "last_layer",
        ], f"Mode {mode} is not supported. It must be one of ['last_layer']."
        self._mode = mode
        assert isinstance(
            self._regularizer, L2Regularizer
        ), f"The regularizer must be of type `L2Regularizer`. It is {self.regularizer}."
        assert (
            self.hparams.regularizer_weight > 0
        ), f"The regularizer weight must be greater than 0. It is {self.hparams.regularizer_weight}."
        self._precision = self.hparams.regularizer_weight

        assert isinstance(
            self.model._output, nn.Linear
        ), f"The output layer must be a linear layer. It is {self.model._output}."

        self._U: torch.Tensor = None
        self._V: torch.Tensor = None

    @property
    def hessian_computed(self) -> bool:
        """This property returns whether the Hessian is computed."""
        return self._U is not None and self._V is not None

    def _predict(self, x: torch.Tensor, **forward_kwargs: Any) -> torch.Tensor:
        """This method is used to perform a forward pass of the model."""
        outputs = []
        num_members = (
            self.training_num_members
            if self.training or not self.hessian_computed
            else self._num_members
        )
        for _ in range(num_members):
            outputs.append(super(MCSamplingMethod, self)._predict(x, **forward_kwargs))
        outputs = torch.cat(outputs, dim=MEMBERS_DIM)
        return outputs

    def _replace_output_layer(self) -> None:
        """This is a helper function to replace the output layer with the Laplace linear layer."""
        if isinstance(self.model._output, nn.Linear):
            device = self.model._output.weight.device
            self.model._output = LaplaceLinear(
                U=self._U,
                V=self._V,
                weight=self.model._output.weight,
                bias=self.model._output.bias,
            ).to(device)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """This function loads the state dictionary of the method."""
        super().load_state_dict(state_dict)
        self._U = state_dict["_U"]
        self._V = state_dict["_V"]

    def on_after_model_load(self) -> None:
        """This function is called after the model is loaded."""
        super().on_after_model_load()
        self._replace_output_layer()

    def state_dict(self) -> Dict[str, Any]:
        """This function returns the state dictionary of the method."""
        state_dict = super().state_dict()
        state_dict["_U"] = self._U
        state_dict["_V"] = self._V
        return state_dict

    def on_fit_end(self) -> None:
        """This method is used to compute the factorised Hessian at the end of the training.

        It uses the K-FAC approximation to compute the factorised Hessian.
        """
        super().on_fit_end()
        loss = nn.CrossEntropyLoss(
            reduction="sum"
        )  # Very important to set the reduction to sum
        loss = extend(loss)

        extend(self.model._output)
        loader = self._datamodule.train_dataloader()
        W = self.model._output.weight
        self.model._output_activation.disable()  # Need to disable the output activation because the inputs are logits
        A, B = None, None
        for batch in tqdm(loader, desc="Computing the factorised Hessian"):
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            output = output.view(output.shape[0], -1)
            self.model.zero_grad()
            with backpack(
                extensions.KFAC(),
            ):
                loss_value = loss(output, y.long())
                loss_value.backward()

            A_, B_ = W.kfac
            if A is None:
                A = A_
                B = B_
            else:
                A += A_
                B += B_
        # Average the factorised Hessian with the number of batches
        A = A / len(loader)
        B = B / len(loader)
        self.model._output_activation.enable()

        self._U = torch.inverse(A + math.sqrt(self._precision) * torch.eye(A.shape[0]))
        self._V = torch.inverse(B + math.sqrt(self._precision) * torch.eye(B.shape[1]))
        self._replace_output_layer()

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This function adds the arguments for the Laplace method."""
        parser = super(LaplaceMethod, LaplaceMethod).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_mode",
            type=str,
            default="last_layer",
            choices=["last_layer"],
            help="The mode of the method.",
        )
        return parser
