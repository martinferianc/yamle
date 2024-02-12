from typing import Any, Dict
import argparse

from yamle.data.datamodule import BaseDataModule


def extract_kwargs(args: argparse.Namespace, prefix: str) -> Dict[str, Any]:
    """This method is used to extract the kwargs from the args.

    Args:
        args (argparse.Namespace): The arguments.
        prefix (str): The prefix to be used for the arguments.

    Returns:
        Dict[str, Any]: The extracted arguments.
    """
    kwargs = {}
    for key, value in vars(args).items():
        if key.startswith(prefix):
            kwargs[key[len(prefix) :]] = value
    return kwargs


def prepare_datamodule_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """This method is used to prepare the datamodule kwargs."""
    datamodule_kwargs = extract_kwargs(args, "datamodule_")
    datamodule_kwargs["seed"] = args.seed
    return datamodule_kwargs


def prepare_model_kwargs(
    args: argparse.Namespace, datamodule: BaseDataModule
) -> Dict[str, Any]:
    """This method is used to prepare the model kwargs."""
    model_kwargs = extract_kwargs(args, "model_")
    model_kwargs["task"] = datamodule.task
    model_kwargs["outputs_dim"] = datamodule.outputs_dim
    model_kwargs["inputs_dim"] = datamodule.inputs_dim
    model_kwargs["seed"] = args.seed
    return model_kwargs


def prepare_loss_kwargs(
    args: argparse.Namespace, datamodule: BaseDataModule
) -> Dict[str, Any]:
    """This method is used to prepare the loss kwargs."""
    loss_kwargs = extract_kwargs(args, "loss_")
    loss_kwargs["task"] = datamodule.task
    return loss_kwargs


def prepare_regularizer_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """This method is used to prepare the regularizer kwargs."""
    regularizer_kwargs = extract_kwargs(args, "regularizer_")
    return regularizer_kwargs


def prepare_metrics_kwargs(
    args: argparse.Namespace, datamodule: BaseDataModule
) -> Dict[str, Any]:
    """This method is used to prepare the metrics kwargs."""
    metrics_kwargs = {}
    metrics_kwargs["task"] = datamodule.task
    metrics_kwargs["outputs_dim"] = datamodule.outputs_dim
    metrics_kwargs["ignore_indices"] = datamodule.ignore_indices
    metrics_kwargs["num_members"] = (
        1 if not hasattr(args, "method_num_members") else args.method_num_members
    )
    metrics_kwargs["metrics"] = args.method_metrics
    return metrics_kwargs


def prepare_method_kwargs(
    args: argparse.Namespace, datamodule: BaseDataModule
) -> Dict[str, Any]:
    """This method is used to prepare the method kwargs."""
    method_kwargs = extract_kwargs(args, "method_")
    method_kwargs["seed"] = args.seed
    method_kwargs["task"] = datamodule.task
    method_kwargs["outputs_dim"] = datamodule.outputs_dim
    method_kwargs["targets_dim"] = datamodule.targets_dim
    method_kwargs["outputs_dtype"] = datamodule.outputs_dtype
    method_kwargs["inputs_dim"] = (args.datamodule_batch_size, *datamodule.inputs_dim)
    method_kwargs["inputs_dtype"] = datamodule.inputs_dtype
    method_kwargs["datamodule"] = datamodule
    method_kwargs["save_path"] = args.save_path
    method_kwargs["metrics_kwargs"] = prepare_metrics_kwargs(args, datamodule)
    method_kwargs["model_kwargs"] = prepare_model_kwargs(args, datamodule)
    return method_kwargs


def prepare_trainer_kwargs(
    args: argparse.Namespace, datamodule: BaseDataModule
) -> Dict[str, Any]:
    """This method is used to prepare the trainer kwargs."""
    trainer_kwargs = extract_kwargs(args, "trainer_")
    trainer_kwargs["save_path"] = args.save_path
    trainer_kwargs["st_checkpoint_dir"] = args.st_checkpoint_dir
    trainer_kwargs["datamodule"] = datamodule
    trainer_kwargs["task"] = datamodule.task
    trainer_kwargs["no_saving"] = args.no_saving
    return trainer_kwargs


def prepare_test_trainer_kwargs(
    args: argparse.Namespace, datamodule: BaseDataModule
) -> Dict[str, Any]:
    """This method is used to prepare the trainer kwargs."""
    test_trainer_kwargs = prepare_trainer_kwargs(args, datamodule)
    test_trainer_kwargs["precision"] = 32
    return test_trainer_kwargs


def prepare_pruner_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """This method is used to prepare the pruner kwargs."""
    pruner_kwargs = extract_kwargs(args, "pruner_")
    return pruner_kwargs


def prepare_quantizer_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """This method is used to prepare the quantizer kwargs."""
    quantizer_kwargs = extract_kwargs(args, "quantizer_")
    return quantizer_kwargs
