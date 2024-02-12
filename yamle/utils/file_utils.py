from typing import Any, Dict, Callable, Optional, List, Union
import os
import sys
from pathlib import Path
import logging
import pickle
import torch.nn as nn
import torch
import shutil
import argparse
from pytorch_lightning import LightningModule
import importlib
import datetime
import time
import pandas as pd
from yamle.defaults import QUANTIZED_MODEL_KEY, MAX_TENDENCY, POSITIVE_INFINITY, NEGATIVE_INFINITY
from yamle.evaluation.metrics.algorithmic import (
    METRIC_TENDENCY,
    parse_metric,
)
from yamle.utils.export_utils import export_onnx


def import_config_function_from_file(config_file: str) -> Callable:
    """This method is used to import a config function from a file.

    Note that the function name needs to be exactly `configuration_space`.
    """
    config_file_path = Path(config_file)
    config_file_dir = config_file_path.parent
    config_file_name = config_file_path.stem
    sys.path.append(str(config_file_dir))
    config_file = importlib.import_module(config_file_name)
    config_function = getattr(config_file, "configuration_space")
    return config_function


def current_time() -> str:
    """This method is used to get the current time to the millisecond."""
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")


def log_file(save_path: str) -> str:
    """This method is used to create the log file."""
    return os.path.join(save_path, "log.log")


def config_logger(log_path: str) -> None:
    """This method is used to configure the logger."""
    log_path = log_file(log_path)
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger("pytorch_lightning").addHandler(fh)


def create_experiment_folder(
    save_path: str, src_folder: str, cache_scripts: bool = True
) -> str:
    """This method is used to create the experiment folder and archite any code in the the src_folder."""
    # Check if the experiment folder exists
    counter = 0
    original_save_path = save_path
    while os.path.exists(save_path):
        # Wait randomly up to 5 seconds to avoid deadlock
        time.sleep(torch.randint(0, 5, (1,)).item())
        save_path = original_save_path + f"-{counter}"
        counter += 1

    Path(save_path).mkdir(parents=True, exist_ok=True)
    if cache_scripts:
        save_scripts(save_path, src_folder)
    return save_path


def save_scripts(save_path: str, src_folder: str) -> None:
    """This method is used to copy all the files in the the `src_folder` to the save_path while preserving the folder structure relative the to `src_folder`."""
    script_folder = os.path.join(save_path, "scripts")
    os.mkdir(script_folder)
    # Create a directory with respect to the script path
    # and copy all the files in the src folder while preserving the folder structure relative the to `src_folder`
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                file_path = os.path.relpath(file_path, src_folder)
                file_path = os.path.join(script_folder, file_path)
                Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
                shutil.copy(os.path.join(root, file), file_path)


def get_experiment_name(args: argparse.Namespace, mode: str = "train") -> str:
    """This method is used to get the experiment name."""
    experiment_name = (
        f"{current_time()}-{mode}-{args.model}-{args.datamodule}-{args.method}"
    )
    experiment_name += f"-{args.label}" if args.label is not None else ""

    if hasattr(args, "st_checkpoint_dir") and args.st_checkpoint_dir is not None:
        # Get the trial id out of the path
        trial_id = args.st_checkpoint_dir.split("/")[-2]
        # Add the trial id as the first part of the experiment name
        experiment_name = f"{trial_id}-{experiment_name}"
    return experiment_name


def argparse_to_dictionary(args: argparse.Namespace) -> Dict[str, Any]:
    """This method is used to convert an argparse.Namespace to a dictionary."""
    return vars(args)


def _string_args_to_list_or_tuple(args: argparse.Namespace) -> argparse.Namespace:
    """This method is used to convert an argparse.Namespace string arguments to a list or tuple.

    If an argument is a string and it looks like a list or tuple, it will be converted to a list or tuple.
    A list starts with `[` and ends with `]` and it is separated by `,`. A tuple starts with `(` and ends with `)`
    and it is separated by `,`.
    """
    for key, value in vars(args).items():
        if isinstance(value, str) or (
            isinstance(value, list)
            and len(value) == 1
            and isinstance(value[0], str)
            and (
                value[0].startswith("[")
                and value[0].endswith("]")
                or value[0].startswith("(")
                and value[0].endswith(")")
            )
        ):
            is_list = False
            is_tuple = False
            if isinstance(value, list):
                value = value[0]
            if value.startswith("[") and value.endswith("]"):
                is_list = True
            elif value.startswith("(") and value.endswith(")"):
                is_tuple = True
            if is_list or is_tuple:
                value = value[1:-1].split(",")
                value = [v.strip() for v in value]
                # If some of the valus are floats, convert them to floats
                value = _convert_list_of_strings_to_list_of_ints_or_floats(value)
                # If the list only contains '' or "" remove them
                if len(value) == 1 and (value[0] == "" or value[0] == '"'):
                    value = []
                setattr(args, key, value)
            if is_tuple:
                setattr(args, key, tuple(value))
    return args


def _convert_list_of_strings_to_list_of_ints_or_floats(
    values: List[str],
) -> Union[List[int], List[float], List[str]]:
    """This method is used to convert a list of strings to a list of ints or floats."""
    # Check if all the values are ints
    try:
        return [int(v) for v in values]
    except ValueError:
        pass
    try:
        return [float(v) for v in values]
    except ValueError:
        pass
    return values


def argparse_to_command(args: argparse.Namespace) -> str:
    """This is a method to convert an argparse.Namespace to a command arguments executable in the terminal."""
    command_arguments = ""
    # Sort the command arguments by key
    d = vars(args)
    d = {k: d[k] for k in sorted(d)}
    for key, value in d.items():
        if isinstance(value, list) or isinstance(value, tuple):
            value_type = type(value)
            value = ",".join([str(v) for v in value])
            if value_type == list:
                value = f"[{value}]"
            if value_type == tuple:
                value = f"({value})"
        if isinstance(value, str):
            value = f'"{value}"'
        elif value is None:
            continue
        command_arguments += f" --{key} {value}"
    return command_arguments


def argparse_to_config_file(
    args: argparse.Namespace,
    config_file: str,
    ignore_arguments: List[str] = [
        "save_path",
        "load_path",
        "st_checkpoint_dir",
        "label",
        "seed",
        "trainer_devices",
    ],
    replace_arguments: Dict[str, str] = {"trainer_mode": "train"},
    import_lines: List[str] = [],
) -> str:
    """This method is used to convert an argparse.Namespace to a config file.

    It creates a function `def config_space():` and returns the config dictionary
    with an argument per line.

    Args:
        args (argparse.Namespace): The arguments.
        config_file (str): The config file path.
        ignore_arguments (List[str], optional): The arguments to ignore. Defaults to [].
        replace_arguments (Dict[str, str], optional): The arguments to replace. Defaults to {}.
        import_lines (List[str], optional): The import lines to add to the config file. Defaults to [].
    """
    with open(config_file, "w") as f:
        f.write("from typing import Dict, Any\n")
        for import_line in import_lines:
            f.write(f"{import_line}\n")
        f.write("\n\n")
        f.write("def configuration_space() -> Dict[str, Any]:\n")
        f.write("    config_space = {\n")
        # Sort the command arguments by key
        d = vars(args)
        d = {k: d[k] for k in sorted(d)}
        for key, value in d.items():
            if key in ignore_arguments and key not in replace_arguments:
                continue
            if isinstance(value, list) or isinstance(value, tuple):
                value_type = type(value)
                value = ",".join([str(v) for v in value])
                if value_type == list:
                    value = f"[{value}]"
                if value_type == tuple:
                    value = f"({value})"
            if isinstance(value, str):
                value = f'"{value}"'
            elif value is None:
                continue
            elif key in replace_arguments:
                value = replace_arguments[key]
            f.write(f'        "{key}": {value},\n')
        f.write("    }\n")
        f.write("    return config_space\n")
    return config_file


def parse_args(args: argparse.Namespace) -> argparse.Namespace:
    """This method is used to post process the arguments."""
    args = _string_args_to_list_or_tuple(args)
    return args


def store_metrics(results: Dict[str, Any], metrics: nn.ModuleDict, prefix: str) -> None:
    """This method is used to store the metrics.

    Args:
        results (Dict[str, Any]): The dictionary where the metrics are stored.
        metrics (nn.ModuleDict): The metrics.
        prefix (str): The prefix to be used for the metrics.
    """
    results[prefix] = {}
    for metric_name, metric_value in metrics.items():
        nan_value = (
            NEGATIVE_INFINITY
            if METRIC_TENDENCY[parse_metric(metric_name)] == MAX_TENDENCY
            else POSITIVE_INFINITY
        )  # It is the worst possible value for the metric
        results[prefix][metric_name] = torch.nan_to_num(
            metric_value.compute(), nan=nan_value
        ).item()


def results_file(save_path: str) -> str:
    """This method is used to create the results file."""
    return os.path.join(save_path, "results.pickle")


def tuning_results_file(save_path: str) -> str:
    """This method is used to create the results file."""
    return os.path.join(save_path, "results.csv")


def model_file(save_path: str) -> str:
    """This method is used to create the model file."""
    return os.path.join(save_path, "model.pth")


def model_onnx_file(save_path: str) -> str:
    """This method is used to create the model file."""
    return os.path.join(save_path, "model.onnx")


def model_initial_file(save_path: str) -> str:
    """This method is used to create the randomly initialized model file."""
    return os.path.join(save_path, "model_initial.pth")


def model_quantized_file(save_path: str) -> str:
    """This method is used to create the quantized model file."""
    return os.path.join(save_path, "model_quantized.pth")


def model_best_on_validation_file(save_path: str) -> str:
    """This method is used to create the model which is the best on the validation set file."""
    return os.path.join(save_path, "model_best_on_val.pth")


def model_train_epoch_file(save_path: str, epoch: int) -> str:
    """This method is used to create the model file for a specific epoch."""
    return os.path.join(save_path, f"model_train_epoch_{epoch}.pth")


def method_file(save_path: str) -> str:
    """This method is used to create the method file."""
    return os.path.join(save_path, "method.pickle")


def args_file(save_path: str) -> str:
    """This method is used to create the args file."""
    return os.path.join(save_path, "args.pickle")


def args_dictionary_file(save_path: str) -> str:
    """This method is used to create the args dictionary file."""
    return os.path.join(save_path, "args_dictionary.pickle")


def plots_file(save_path: str, specific_name: str = "") -> str:
    """This method is used to create the plots file."""
    return os.path.join(save_path, f"predictions_{specific_name}.png")


def config_file(save_path: str) -> str:
    """This method is used to create the config file."""
    return os.path.join(save_path, "config.py")


def predictions_file(save_path: str, special_name: str = "") -> str:
    """This method is used to create the predictions file."""
    name = (
        f"predictions_{special_name}.pickle"
        if special_name != ""
        else "predictions.pickle"
    )
    return os.path.join(save_path, name)


def save_pickle(obj: Any, path: str) -> None:
    """This method is used to save a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> Any:
    """This method is used to load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model: nn.Module, path: str) -> None:
    """This method is used to save a model."""
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str) -> nn.Module:
    """This method is used to load a model."""
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    model_dict = model.state_dict()
    pretrained_dict = dict(state_dict.items())
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
    }
    # Perform check if everything is loaded properly
    for key, value in model_dict.items():
        if key not in pretrained_dict:
            raise ValueError(f"Missing key {key} in pretrained model")
        assert (
            value.shape == pretrained_dict[key].shape
        ), f"Shape mismatch for key {key}"
    # Check if there are any extra keys in the pretrained model
    for key, value in pretrained_dict.items():
        if key not in model_dict:
            raise ValueError(f"Extra key {key} in pretrained model")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def save_model_onnx(model: nn.Module, path: str) -> None:
    """This method is used to save a model."""
    export_onnx(model, model_onnx_file(path))


def save_method(method: LightningModule, path: str) -> None:
    """This method is used to save a method."""
    save_pickle(method.state_dict(), method_file(path))


def load_method(method: LightningModule, path: str) -> None:
    """This method is used to load a method."""
    state_dict = load_pickle(method_file(path))
    method.load_state_dict(state_dict)


def save_args(args: argparse.Namespace, path: str) -> None:
    """This method is used to save the arguments."""
    save_pickle(args, args_file(path))


def load_args(path: str) -> argparse.Namespace:
    """This method is used to load the arguments."""
    return load_pickle(args_file(path))


def save_args_dictionary(args: argparse.Namespace, path: str) -> None:
    """This method is used to save the arguments dictionary."""
    save_pickle(argparse_to_dictionary(args), args_dictionary_file(path))


def load_args_dictionary(path: str) -> Dict[str, Any]:
    """This method is used to load the arguments dictionary."""
    return load_pickle(args_dictionary_file(path))


def save_results(results: Dict[str, Any], path: str) -> None:
    """This method is used to save the results."""
    save_pickle(results, results_file(path))


def load_results(path: str) -> Dict[str, Any]:
    """This method is used to load the results."""
    return load_pickle(results_file(path))


def save_tuning_results(results: pd.DataFrame, path: str) -> None:
    """This method is used to save the results."""
    results.to_csv(tuning_results_file(path))


def load_tuning_results(path: str) -> pd.DataFrame:
    """This method is used to load the results."""
    return pd.read_csv(tuning_results_file(path))


def save_experiment(
    save_path: str,
    args: argparse.Namespace,
    method: LightningModule,
    results: Optional[Dict[str, Any]],
    overwrite: bool = False,
    overwrite_results: bool = False,
) -> None:
    """This method is used to save the experiment.

    Args:
        save_path (str): The experiment path folder.
        args (argparse.Namespace): The arguments.
        method (LightningModule): The method.
        model (nn.Module): The model.
        results (Optional[Dict[str, Any]]): The results dictionary.
        overwrite (bool, optional): Whether to overwrite the experiment. Defaults to False.
        overwrite_results (bool, optional): Whether to overwrite the results. Defaults to False.
    """
    # Save the arguments
    if not os.path.exists(args_file(save_path)) or overwrite:
        save_args(args, save_path)
        save_args_dictionary(args, save_path)

    # Save the method
    if not os.path.exists(method_file(save_path)) or overwrite:
        save_method(method, save_path)

    # Save the model
    if not os.path.exists(model_file(save_path)) or overwrite:
        save_model(method.model, model_file(save_path))

    # Save the quantized model
    if hasattr(method, QUANTIZED_MODEL_KEY) and (
        not os.path.exists(model_quantized_file(save_path)) or overwrite
    ):
        save_model(
            getattr(method, QUANTIZED_MODEL_KEY), model_quantized_file(save_path)
        )

    # Save the results
    if (
        (results is not None and not os.path.exists(results_file(save_path)))
        or overwrite
        or overwrite_results
    ):
        save_results(results, save_path)

    if args.onnx_export:
        save_model_onnx(method.model, save_path)


def remove_argparse_argument(parser: argparse.ArgumentParser, arg_name: str) -> None:
    """This method is used to remove an argument from the parser."""
    for action in parser._actions:
        if action.dest == arg_name:
            parser._actions.remove(action)
            break
