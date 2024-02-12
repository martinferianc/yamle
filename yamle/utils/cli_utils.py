import argparse

from yamle.data import AVAILABLE_DATAMODULES
from yamle.methods import AVAILABLE_METHODS
from yamle.losses import AVAILABLE_LOSSES
from yamle.models import AVAILABLE_MODELS
from yamle.pruning import AVAILABLE_PRUNERS
from yamle.quantization import AVAILABLE_QUANTIZERS
from yamle.regularizers import AVAILABLE_REGULARIZERS
from yamle.trainers import AVAILABLE_TRAINERS


def add_shared_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """This function adds the shared arguments between training and evaluation to the given parser."""
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="An optional label to be added to the experiment name.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="fc",
        choices=AVAILABLE_MODELS.keys(),
        help="The model to be used for training.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="base",
        choices=AVAILABLE_METHODS.keys(),
        help="The method to be used for testing.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=None,
        choices=AVAILABLE_LOSSES.keys(),
        help="The loss to be used for training.",
    )
    parser.add_argument(
        "--regularizer",
        type=str,
        default=None,
        choices=AVAILABLE_REGULARIZERS.keys(),
        help="The regularizer to be used for training.",
    )
    parser.add_argument(
        "--datamodule",
        type=str,
        default="mnist",
        choices=AVAILABLE_DATAMODULES.keys(),
        help="The data to be used for training.",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="base",
        choices=AVAILABLE_TRAINERS.keys(),
        help="The trainer to be used for training.",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="none",
        choices=AVAILABLE_PRUNERS.keys(),
        help="The pruner to be used for evaluation.",
    )
    parser.add_argument(
        "--quantizer",
        type=str,
        default="none",
        choices=AVAILABLE_QUANTIZERS.keys(),
        help="The quantizer to be used for evaluation.",
    )
    parser.add_argument_group("Experiment")
    parser.add_argument(
        "--seed", type=int, default=42, help="The seed to be used for training."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="experiments",
        help="The directory where the experiment results are stored.",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="The directory where the experiment results are stored and loaded from.",
    )
    parser.add_argument(
        "--no_saving",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to skip the saving step.",
    )
    parser.add_argument(
        "--st_checkpoint_dir",
        type=str,
        default=None,
        help="The directory where the Syne Tune checkpoint is stored.",
    )

    parser.add_argument(
        "--onnx_export",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to export the model to ONNX.",
    )

    return parser


def add_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """This function adds the training arguments to the given parser."""
    parser.add_argument(
        "--no_evaluation",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to skip the evaluation step.",
    )
    return parser


def add_test_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """This function adds the testing arguments to the given parser."""
    return parser
