import argparse
from typing import Any, Dict, Optional, Tuple

import torch.nn as nn

from yamle.methods.uncertain_method import MCSamplingMethod
from yamle.models.specific.mcdropout import (
    Dropout1d,
    Dropout2d,
    Dropout3d,
    LinearDropConnect,
    StandOut,
    disable_dropout_replacement,
    replace_with_dropblock,
    replace_with_dropconnect,
    replace_with_dropout,
    replace_with_standout,
    replace_with_stochastic_depth,
    count_linear_conv,
    count_residual,
    count_conv2d,
    count_lstm,
)


class MCDropoutMethod(MCSamplingMethod):
    """This class is the extension of the base method for which the prediciton is performed using Monte Carlo dropout.

    The dropout layers are added either to all layers or only to the last layer. Dropout is always on.

    Args:
        p (float): The dropout probability to be used for Monte Carlo dropout.
        mode (str): Where to add dropout layers, can be either `all`, `last`, `partial` or `custom`.
        no_input_replacement (bool): Whether to replace the input with dropout.
        conv_filter_dropout (bool): Whether to place 2D dropout on the convolutional filters.
        depth_portion_to_replace_start (float): The depth portion of the model to start replacing with dropout.
        depth_portion_to_replace_end (float): The depth portion of the model to end replacing with dropout.
        depth_indices (Tuple[int]): The indices of the layers to replace with dropout.
    """

    def __init__(
        self,
        p: float = 0.5,
        mode: str = "all",
        no_input_replacement: bool = False,
        conv_filter_dropout: bool = False,
        depth_portion_to_replace_start: Optional[float] = None,
        depth_portion_to_replace_end: Optional[float] = None,
        depth_indices: Optional[Tuple[int]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(MCDropoutMethod, self).__init__(*args, **kwargs)
        assert mode in [
            "all",
            "last",
            "partial",
            "custom",
        ], f"mode must be either `all`, `last`, `partial` or `custom`, got {mode}"
        assert 0 <= p <= 1, f"p must be in [0, 1], got {p}"
        if mode == "partial":
            assert (
                depth_portion_to_replace_start is not None
                and depth_portion_to_replace_end is not None
            ), f"`depth_portion_to_replace_start` and `depth_portion_to_replace_end` must be provided for `partial` mode."
        if mode == "custom":
            assert (
                depth_indices is not None
            ), f"`depth_indices` must be provided for `custom` mode."

        self._p = p
        self._mode = mode
        if no_input_replacement:
            disable_dropout_replacement(self.model._input)

        dropout_mapping: Dict[nn.Module, nn.Module] = {
            nn.Linear: Dropout1d,
            nn.Conv1d: Dropout1d,
            nn.Conv2d: Dropout2d if conv_filter_dropout else Dropout1d,
            nn.Conv3d: Dropout3d if conv_filter_dropout else Dropout1d,
            nn.Dropout: Dropout1d,
            nn.Dropout2d: Dropout2d,
            nn.Dropout3d: Dropout3d,
        }

        if mode == "all":
            replace_with_dropout(self.model, p, dropout_mapping)
        elif mode == "last":
            self.model._output = nn.Sequential(Dropout1d(p=p), self.model._output)
        elif mode == "partial":
            max_count = count_linear_conv(self)
            start_count = int(depth_portion_to_replace_start * max_count)
            end_count = int(depth_portion_to_replace_end * max_count)
            replace_with_dropout(
                self.model, p, dropout_mapping, (start_count, end_count)
            )
            if max_count == 0:
                # Maybe an LSTM model
                max_count = count_lstm(self)
                start_count = int(depth_portion_to_replace_start * max_count)
                end_count = int(depth_portion_to_replace_end * max_count)
                replace_with_dropout(
                    self.model, p, dropout_mapping, (start_count, end_count)
                )
        elif mode == "custom":
            # Check if the model is an LSTM model
            max_count = count_lstm(self)
            if max_count != 0:
                replace_with_dropout(
                    self.model, p, dropout_mapping, custom_indices=depth_indices
                )
            else:
                max_count = count_linear_conv(self)
                replace_with_dropout(
                    self.model, p, dropout_mapping, custom_indices=depth_indices
                )
        else:
            raise NotImplementedError(
                f"mode {mode} is not implemented for MCDropoutMethod"
            )

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the specific arguments for the class."""
        parser = super(MCDropoutMethod, MCDropoutMethod).add_specific_args(
            parent_parser
        )
        parser.add_argument(
            "--method_p",
            type=float,
            default=0.5,
            help="The dropout probability to be used for Monte Carlo dropout.",
        )
        parser.add_argument(
            "--method_mode",
            type=str,
            choices=["all", "last", "partial", "custom"],
            default="all",
            help="Where to add dropout layers, can be either `all`, `last` or `partial`.",
        )
        parser.add_argument(
            "--method_no_input_replacement",
            type=int,
            choices=[0, 1],
            default=1,
            help="Whether to place dropout on the input.",
        )
        parser.add_argument(
            "--method_conv_filter_dropout",
            type=int,
            choices=[0, 1],
            default=0,
            help="Whether to place 2D dropout on the convolutional filters.",
        )
        parser.add_argument(
            "--method_depth_portion_to_replace_start",
            type=float,
            default=None,
            help="The depth portion of the model to start replacing with dropout.",
        )
        parser.add_argument(
            "--method_depth_portion_to_replace_end",
            type=float,
            default=None,
            help="The depth portion of the model to end replacing with dropout.",
        )
        parser.add_argument(
            "--method_depth_indices",
            type=str,
            nargs="+",
            default=None,
            help="The indices of the layers to replace with dropout.",
        )

        return parser


class MCStandOutMethod(MCSamplingMethod):
    """This class is the extension of the base method for which the prediciton is performed using StandOut.

    The dropout layers are added either to all layers or only to the last layer. Dropout is always on.

    Args:
        alpha (float): The alpha parameter to be used for StandOut.
        beta (float): The beta parameter to be used for StandOut.
        mode (str): Where to add dropout layers, can be either `all`, `last` or `partial`.
        no_input_replacement (bool): Whether to replace the input with dropout.
        depth_portion_to_replace_start (float): The depth portion of the model to start replacing with dropout.
        depth_portion_to_replace_end (float): The depth portion of the model to end replacing with dropout.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        mode: str = "all",
        no_input_replacement: bool = False,
        depth_portion_to_replace_start: Optional[float] = None,
        depth_portion_to_replace_end: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(MCStandOutMethod, self).__init__(*args, **kwargs)
        assert mode in [
            "all",
            "last",
            "partial",
        ], f"mode must be either `all`, `last` or `partial`, got {mode}"
        if mode == "partial":
            assert (
                depth_portion_to_replace_start is not None
                and depth_portion_to_replace_end is not None
            ), f"`depth_portion_to_replace_start` and `depth_portion_to_replace_end` must be provided for `partial` mode."

        if no_input_replacement:
            disable_dropout_replacement(self.model._input)

        if mode == "all":
            replace_with_standout(self.model, alpha, beta)
        elif mode == "last":
            self.model._output = StandOut(self.model._output, alpha, beta)
        elif mode == "partial":
            max_count = count_linear_conv(self)
            start_count = int(depth_portion_to_replace_start * max_count)
            end_count = int(depth_portion_to_replace_end * max_count)
            replace_with_standout(self.model, alpha, beta, (start_count, end_count))
        else:
            raise NotImplementedError(
                f"mode {mode} is not implemented for MCStandOutMethod"
            )

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the specific arguments for the class."""
        parser = super(MCStandOutMethod, MCStandOutMethod).add_specific_args(
            parent_parser
        )
        parser.add_argument(
            "--method_alpha",
            type=float,
            default=0.5,
            help="The alpha parameter to be used for StandOut.",
        )
        parser.add_argument(
            "--method_beta",
            type=float,
            default=0.5,
            help="The beta parameter to be used for StandOut.",
        )
        parser.add_argument(
            "--method_mode",
            type=str,
            default="all",
            choices=["all", "last", "partial"],
            help="Where to add dropout layers, can be either `all`, `last` or `partial`.",
        )
        parser.add_argument(
            "--method_no_input_replacement",
            type=int,
            choices=[0, 1],
            default=0,
            help="Whether to replace the input with dropout.",
        )
        parser.add_argument(
            "--method_depth_portion_to_replace_start",
            type=float,
            default=None,
            help="The depth portion of the model to start replacing with dropout.",
        )
        parser.add_argument(
            "--method_depth_portion_to_replace_end",
            type=float,
            default=None,
            help="The depth portion of the model to end replacing with dropout.",
        )
        return parser


class MCDropConnectMethod(MCSamplingMethod):
    """This class is the extension of the base method for which the prediciton is performed using Monte Carlo dropconnect.

    Note that, dropconnect or dropping of the weights is always on.

    Args:
        p (float): The dropconnect probability to be used for Monte Carlo dropconnect.
        mode (str): Where to add dropconnect layers, can be either `all`, `partial` or `last`.
        no_input_replacement (bool): Whether to replace the input with dropout.
        depth_portion_to_replace_start (float): The depth portion of the model to start replacing with dropout.
        depth_portion_to_replace_end (float): The depth portion of the model to end replacing with dropout.
    """

    def __init__(
        self,
        p: float = 0.5,
        mode: str = "all",
        no_input_replacement: bool = False,
        depth_portion_to_replace_start: Optional[float] = None,
        depth_portion_to_replace_end: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(MCDropConnectMethod, self).__init__(*args, **kwargs)
        assert mode in [
            "all",
            "partial",
            "last",
        ], f"mode must be either `all`, `partial` or `last`, got {mode}"
        assert 0 <= p <= 1, f"p must be in [0, 1], got {p}"
        if mode == "partial":
            assert (
                depth_portion_to_replace_start is not None
                and depth_portion_to_replace_end is not None
            ), f"`depth_portion_to_replace_start` and `depth_portion_to_replace_end` must be provided for `partial` mode."

        if no_input_replacement:
            disable_dropout_replacement(self.model._input)
        self._p = p
        self._mode = mode
        if self._mode == "all":
            replace_with_dropconnect(self.model, self._p)
        elif self._mode == "last":
            assert isinstance(
                self.model._output, nn.Linear
            ), "The last layer must be a linear layer."
            self.model._output = LinearDropConnect(
                self.model._output.in_features,
                self.model._output.out_features,
                self.model._output.bias is not None,
                self._p,
            )
        elif self._mode == "partial":
            max_count = count_linear_conv(self)
            start_count = int(depth_portion_to_replace_start * max_count)
            end_count = int(depth_portion_to_replace_end * max_count)
            replace_with_dropconnect(self.model, self._p, (start_count, end_count))
        else:
            raise NotImplementedError(
                f"mode {self._mode} is not implemented for MCDropConnectMethod"
            )

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the specific arguments for the class."""
        parser = super(MCDropConnectMethod, MCDropConnectMethod).add_specific_args(
            parent_parser
        )
        parser.add_argument(
            "--method_p",
            type=float,
            default=0.5,
            help="The dropconnect probability to be used for Monte Carlo dropconnect.",
        )
        parser.add_argument(
            "--method_mode",
            type=str,
            default="all",
            choices=["all", "partial", "last"],
            help="Where to add dropconnect layers, can be either `all`, `partial` or `last`.",
        )
        parser.add_argument(
            "--method_no_input_replacement",
            type=int,
            choices=[0, 1],
            default=0,
            help="Whether to replace the input with dropout.",
        )
        parser.add_argument(
            "--method_depth_portion_to_replace_start",
            type=float,
            default=None,
            help="The depth portion of the model to start replacing with dropout.",
        )
        parser.add_argument(
            "--method_depth_portion_to_replace_end",
            type=float,
            default=None,
            help="The depth portion of the model to end replacing with dropout.",
        )
        return parser


class MCDropBlockMethod(MCSamplingMethod):
    """This class is the extension of the base method for which the prediciton is performed using Monte Carlo Drop Block.

    After every convolutional layer, a DropBlock layer is added. It drops a contiguous region of the feature map.

    Args:
        p (float): The dropblock probability to be used for Monte Carlo Drop Block.
        mode (str): Where to add dropblock layers, can be either `all` of `partial`.
        block_size_percentage (float): The percentage of the block size to be dropped relative to the input size.
        no_input_replacement (bool): Whether to replace the input with dropblock.
        depth_portion_to_replace_start (float): The depth portion of the model to start replacing with dropout.
        depth_portion_to_replace_end (float): The depth portion of the model to end replacing with dropout.
    """

    def __init__(
        self,
        p: float = 0.5,
        mode: str = "all",
        block_size_percentage: float = 0.1,
        no_input_replacement: bool = False,
        depth_portion_to_replace_start: Optional[float] = None,
        depth_portion_to_replace_end: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(MCDropBlockMethod, self).__init__(*args, **kwargs)
        assert mode in [
            "all",
            "partial",
        ], f"mode must be either `all` or `partial`, got {mode}"
        assert 0 <= p <= 1, f"p must be in [0, 1], got {p}"
        assert (
            0 <= block_size_percentage <= 1
        ), f"block_size_percentage must be in [0, 1], got {block_size_percentage}"
        if mode == "partial":
            assert (
                depth_portion_to_replace_start is not None
                and depth_portion_to_replace_end is not None
            ), f"`depth_portion_to_replace_start` and `depth_portion_to_replace_end` must be provided for `partial` mode."

        if no_input_replacement:
            disable_dropout_replacement(self.model._input)
        self._p = p
        self._block_size_percentage = block_size_percentage

        if mode == "all":
            replace_with_dropblock(self.model, self._block_size_percentage, self._p)
        elif mode == "partial":
            max_count = count_conv2d(self)
            start_count = int(depth_portion_to_replace_start * max_count)
            end_count = int(depth_portion_to_replace_end * max_count)
            replace_with_dropblock(
                self.model,
                self._block_size_percentage,
                self._p,
                (start_count, end_count),
            )

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the specific arguments for the class."""
        parser = super(MCDropBlockMethod, MCDropBlockMethod).add_specific_args(
            parent_parser
        )
        parser.add_argument(
            "--method_p",
            type=float,
            default=0.5,
            help="The dropblock probability to be used for Monte Carlo Drop Block.",
        )
        parser.add_argument(
            "--method_mode",
            type=str,
            default="all",
            choices=["all", "partial"],
            help="Where to add dropblock layers, can be either `all` of `partial`.",
        )
        parser.add_argument(
            "--method_block_size_percentage",
            type=float,
            default=0.1,
            help="The percentage of the block size to be dropped relative to the input size.",
        )
        parser.add_argument(
            "--method_no_input_replacement",
            type=int,
            choices=[0, 1],
            default=0,
            help="Whether to replace the input with dropblock.",
        )
        parser.add_argument(
            "--method_depth_portion_to_replace_start",
            type=float,
            default=None,
            help="The depth portion of the model to start replacing with dropout.",
        )
        parser.add_argument(
            "--method_depth_portion_to_replace_end",
            type=float,
            default=None,
            help="The depth portion of the model to end replacing with dropout.",
        )
        return parser


class MCStochasticDepthMethod(MCSamplingMethod):
    """This class is the extension of the base method for which the prediciton is performed using Monte Carlo Stochastic Depth.

    Stochastic depth is added to every residual block where the residual function is randomly dropped for each sample.
    It is assumed that the residual blocks are created one after another with respect to the depth, such that
    the probability of dropping a residual block depends on the depth of the residual block.

    Args:
        p (float): The initial stochastic depth probability `p_L` to be used for Monte Carlo Stochastic Depth.
    """

    def __init__(self, p: float = 0.5, *args: Any, **kwargs: Any) -> None:
        super(MCStochasticDepthMethod, self).__init__(*args, **kwargs)
        assert 0 <= p <= 1, f"p must be in [0, 1], got {p}"
        self._p = p

        L = count_residual(self)
        replace_with_stochastic_depth(self.model, L, self._p)

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = super(
            MCStochasticDepthMethod, MCStochasticDepthMethod
        ).add_specific_args(parent_parser)
        parser.add_argument(
            "--method_p",
            type=float,
            default=0.5,
            help="The initial stochastic depth probability to be used for Monte Carlo Stochastic Depth.",
        )
        return parser
