from typing import Any, Callable, Dict
import torch.nn as nn
import argparse

from yamle.methods.uncertain_method import SVIMethod
from yamle.models.specific.svi import (
    LinearSVILRT,
    LinearSVIFlipOut,
    LinearSVIRT,
    LinearSVILRTVD,
)
from yamle.models.specific.svi import (
    replace_with_flipout_svi,
    replace_with_svi_lrt,
    replace_with_svi_lrtvd,
    replace_with_svi_rt,
)


class SVIReparameterizationMethod(SVIMethod):
    """This class is the extension of the base method for stochastic variational inference methods.
    Implemented with respect to Local Reparameterization Trick (LRT) or the simple Reparameterization Trick (RT).

    It is assumed that the posterior should be mean-field and that the prior should be a Gaussian.

    Args:
        prior_mean (float): The mean of the prior. Only used if the method is `lrt`, `rt` or `flipout_gaussian`.
        log_variance (float): The initial value of the log variance of the weights. Only used if the method is `lrt`, `rt` or `flipout_gaussian`.
        prior_log_variance (float): The log variance of the prior. Only used if the method is `lrt`, `rt` or `flipout_gaussian`.
        p (float): The probability in the `DropConnect` layer. Only used if the method is `flipout_dropconnect`.
        mode (str): Whether the `last` layer or the `all` layers should be used for the inference.
        method (str): Whether to use the `lrt`, `rt`, `flipout_dropconnect` or `flipout_gaussian` method.
    """

    def __init__(
        self,
        prior_mean: float,
        log_variance: float,
        prior_log_variance: float,
        p: float,
        mode: str,
        method: str,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        assert mode in ["all", "last"]
        assert method in [
            "lrt",
            "rt",
            "lrtvd",
            "flipout_dropconnect",
            "flipout_gaussian",
        ]
        self._prior_mean = prior_mean
        self._log_variance = log_variance
        self._prior_log_variance = prior_log_variance
        self._mode = mode

        replacing_function: Callable
        additional_kwargs: Dict[str, Any] = {}
        if method == "lrt":
            replacing_function = replace_with_svi_lrt
        elif method == "lrtvd":
            replacing_function = replace_with_svi_lrtvd
        elif method == "rt":
            replacing_function = replace_with_svi_rt
        elif method in ["flipout_dropconnect", "flipout_gaussian"]:
            replacing_function = replace_with_flipout_svi
            if method == "flipout_dropconnect":
                additional_kwargs["p"] = p
                additional_kwargs["method"] = "dropconnect"
            else:
                additional_kwargs["method"] = "gaussian"

        assert isinstance(
            self.model._output, nn.Linear
        ), "The output layer should be a `nn.Linear` layer to enable replacing it."

        if self._mode == "all":
            self.model = replacing_function(
                self.model,
                self._prior_mean,
                self._log_variance,
                self._prior_log_variance,
                **additional_kwargs
            )
        elif self._mode == "last":
            if method == "lrt":
                self.model._output = LinearSVILRT(
                    self.model._output.in_features,
                    self.model._output.out_features,
                    self.model._output.bias is not None,
                    self._prior_mean,
                    self._log_variance,
                    self._prior_log_variance,
                )
            elif method == "lrtvd":
                self.model._output = LinearSVILRTVD(
                    self.model._output.in_features,
                    self.model._output.out_features,
                    self.model._output.bias is not None,
                    self._prior_mean,
                    self._log_variance,
                    self._prior_log_variance,
                )
            elif method == "rt":
                self.model._output = LinearSVIRT(
                    self.model._output.in_features,
                    self.model._output.out_features,
                    self.model._output.bias is not None,
                    self._prior_mean,
                    self._log_variance,
                    self._prior_log_variance,
                )
            elif method in ["flipout_dropconnect", "flipout_gaussian"]:
                self.model._output = LinearSVIFlipOut(
                    self.model._output.in_features,
                    self.model._output.out_features,
                    self.model._output.bias is not None,
                    self._prior_mean,
                    self._log_variance,
                    self._prior_log_variance,
                    **additional_kwargs
                )

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = super(
            SVIReparameterizationMethod, SVIReparameterizationMethod
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_prior_mean",
            type=float,
            default=0.0,
            help="The mean of the prior. Only used if the method is `lrt`, `rt` or `flipout_gaussian`.",
        )
        parser.add_argument(
            "--method_log_variance",
            type=float,
            default=-5.0,
            help="The initial value of the log variance of the weights. Only used if the method is `lrt`, `rt` or `flipout_gaussian`.",
        )
        parser.add_argument(
            "--method_prior_log_variance",
            type=float,
            default=-5.0,
            help="The log variance of the prior. Only used if the method is `lrt`, `rt` or `flipout_gaussian`.",
        )
        parser.add_argument(
            "--method_p",
            type=float,
            default=0.5,
            help="The probability of the dropconnect. Only used if method is `flipout_dropconnect`.",
        )
        parser.add_argument(
            "--method_mode",
            type=str,
            default="all",
            help="Whether the `last` layer or the `all` layers should be used for the inference.",
        )
        return parser


class SVILRTMethod(SVIReparameterizationMethod):
    """This class is the extension of the base method for stochastic variational inference methods.
    Implemented with respect to Local Reparameterization Trick (LRT) and Gaussian prior.
    """

    def __init__(self, **kwargs):
        super().__init__(method="lrt", **kwargs)


class SVILRTVDMethod(SVIReparameterizationMethod):
    """This class is the extension of the base method for stochastic variational inference methods.
    Implemented with respect to Local Reparameterization Trick (LRT) and Variational Dropout prior.
    """

    def __init__(self, **kwargs):
        super().__init__(method="lrtvd", **kwargs)


class SVIRTMethod(SVIReparameterizationMethod):
    """This class is the extension of the base method for stochastic variational inference methods.
    Implemented with respect to Reparameterization Trick (RT) and Gaussian prior.
    """

    def __init__(self, **kwargs):
        super().__init__(method="rt", **kwargs)


class SVIFlipOutRTMethod(SVIReparameterizationMethod):
    """This class implements the SVI method using the FlipOut trick with Gaussian prior and reparameterization trick."""

    def __init__(self, **kwargs):
        super().__init__(method="flipout_gaussian", **kwargs)


class SVIFlipOutDropConnectMethod(SVIReparameterizationMethod):
    """This class implements the SVI method using the FlipOut trick with DropConnect prior and reparameterization trick."""

    def __init__(self, **kwargs):
        super().__init__(method="flipout_dropconnect", **kwargs)
