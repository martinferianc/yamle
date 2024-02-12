from typing import Dict, List, Tuple
import argparse
import numpy as np
import os
import natsort
import yamle.utils.file_utils as utils
from yamle.defaults import ARGS_CAN_BE_DIFFERENT

import logging

logging = logging.getLogger("pytorch_lightning")


def get_dictionary_path(dictionary: Dict[str, float], path: List[str] = []):
    """A helper function to get the path to the first value in a dictionary that is not a dictionary."""
    for key, value in dictionary.items():
        if type(value) is dict:
            return get_dictionary_path(dictionary[key], path + [key])
        return path + [key]
    return path


def get_dictionary_value(
    dictionary: Dict[str, float], path: List[str] = [], delete: bool = False
) -> float:
    """A helper function to get the value of a dictionary at a given path."""
    if len(path) == 1:
        val = dictionary[path[0]]
        if delete:
            dictionary.pop(path[0])
        return val
    else:
        return get_dictionary_value(dictionary[path[0]], path[1:], delete=delete)


def set_dictionary_value(
    dictionary: Dict[str, float], value: Tuple[float, float], path: List[str] = []
) -> None:
    """A helper function to set the value of a dictionary at a given path."""
    if len(path) == 1:
        dictionary[path[0]] = value
    else:
        if not path[0] in dictionary:
            dictionary[path[0]] = {}
        set_dictionary_value(dictionary[path[0]], value, path[1:])


def average(args: argparse.Namespace) -> None:
    """Averages the results of multiple experiments and saves them in a new folder or a set of new folders."""
    # Create experiment structure
    experiment_name = f"{utils.current_time()}-average"
    experiment_name += f"-{args.label}" if args.label is not None else ""

    # Create experiment directory
    save_path = os.path.join(args.save_path, experiment_name)
    save_path = utils.create_experiment_folder(save_path, "./src", cache_scripts=False)

    # Set the logger
    utils.config_logger(save_path)
    logging.info("Beginning averaging: %s", experiment_name)
    logging.info("Arguments: %s", args)
    logging.info("Command arguments to reproduce: %s", utils.argparse_to_command(args))

    # Check that only one of the arguments specifying the results paths is not `None`
    assert (
        sum(
            [
                args.separate_folder_paths is not None,
                args.results_paths is not None,
                args.single_folder_path is not None,
            ]
        )
        == 1
    ), f"Only one of the arguments specifying the results paths must be not `None` but got: {args.separate_folder_paths}, {args.results_paths}, {args.single_folder_path}"

    # If `args.folder_paths` is not `None` it means we are going to load in all the results
    # from each folder path and average them with respect to all 1 level down subfolders.

    # If `args.results_paths` is not `None` it means we are going to load in all the results
    # from each results path and average them with respect to all the results paths.

    # If `args.single_folder_path` is not `None` it means we are going to load in all the results
    # from that main folder path and average them with respect to all 1 level down subfolders.

    result_paths_tuples: List[Tuple[str, ...]] = []
    if args.separate_folder_paths is not None:
        for folder_path in args.separate_folder_paths:
            # Get all the subfolders in the folder path
            result_paths_single_folder = os.listdir(folder_path)
            # Make sure that the folder names are naturally sorted
            result_paths_single_folder = natsort.natsorted(result_paths_single_folder)
            # Get the full paths to the subfolders
            result_paths_single_folder = [
                os.path.join(folder_path, result_path)
                for result_path in result_paths_single_folder
            ]

            # Exclude the current save path
            result_paths_single_folder = [
                result_path
                for result_path in result_paths_single_folder
                if result_path != save_path
            ]

            # Add the subfolders to the list of results paths to average
            result_paths_tuples.append(list(result_paths_single_folder))

        # Zip the results paths into tuples to average them
        result_paths_tuples = list(zip(*result_paths_tuples))

    elif args.results_paths is not None:
        result_paths_tuples = list([args.results_paths])

    elif args.single_folder_path is not None:
        # Get all the subfolders in the folder path
        result_paths_single_folder = os.listdir(args.single_folder_path)
        # Make sure that the folder names are naturally sorted
        result_paths_single_folder = natsort.natsorted(result_paths_single_folder)
        # Get the full paths to the subfolders
        result_paths_single_folder = [
            os.path.join(args.single_folder_path, result_path)
            for result_path in result_paths_single_folder
        ]

        # Exclude the current save path
        result_paths_single_folder = [
            result_path
            for result_path in result_paths_single_folder
            if result_path != save_path
        ]

        # Add the subfolders to the list of results paths to average
        result_paths_tuples.append(list(result_paths_single_folder))

    logging.info("Results paths to average: %s", result_paths_tuples)
    logging.info("Arguments that can be different: %s", ARGS_CAN_BE_DIFFERENT)
    for i, results_paths_tuple in enumerate(result_paths_tuples):
        logging.info(f"Results paths to average: {results_paths_tuple}")
        results = []
        args_dictionaries = []
        for result_path in results_paths_tuple:
            result = utils.load_results(result_path)
            args_dictionary = utils.load_args_dictionary(result_path)
            results.append(result)
            for key in ARGS_CAN_BE_DIFFERENT + args.custom_args_to_ignore:
                if key in args_dictionary:
                    args_dictionary.pop(key)
            args_dictionaries.append(args_dictionary)
            logging.info(
                f"Loaded results from {result_path}: {result} with args: {args_dictionary}"
            )

        # Check that the arguments are the same for all the results to average
        if not all(
            args_dictionaries[0] == args_dictionary
            for args_dictionary in args_dictionaries
        ):
            # Compare the arguments of all the dictionaries
            for i, args_dictionary1 in enumerate(args_dictionaries):
                for j, args_dictionary2 in enumerate(args_dictionaries):
                    if i == j:
                        continue
                    for key1 in args_dictionary1.keys():
                        if key1 not in args_dictionary2:
                            logging.warning(
                                f"Argument {key1} of args dictionary {i} not in args dictionary {j}"
                            )
                        elif args_dictionary1[key1] != args_dictionary2[key1]:
                            logging.warning(
                                f"Argument {key1} of args dictionary {i} different from args dictionary {j}: {args_dictionary1[key1]} != {args_dictionary2[key1]}"
                            )
                    for key2 in args_dictionary2.keys():
                        if key2 not in args_dictionary1:
                            logging.warning(
                                f"Argument {key2} of args dictionary {j} not in args dictionary {i}"
                            )
                        elif args_dictionary2[key2] != args_dictionary1[key2]:
                            logging.warning(
                                f"Argument {key2} of args dictionary {j} different from args dictionary {i}: {args_dictionary2[key2]} != {args_dictionary1[key2]}"
                            )
            raise ValueError(
                "Arguments are not the same for all the results to average"
            )

        traversing_result = results[0]
        final_result = {}
        while len(get_dictionary_path(traversing_result)) != 0:
            path = get_dictionary_path(traversing_result)
            logging.info(f"Traversing path {path}")
            values = []
            for result in results:
                try:
                    value = get_dictionary_value(result, path, delete=True)
                    if not isinstance(value, dict):
                        values.append(value)
                except Exception as e:
                    value = None
                    logging.info(
                        f"Could not get value for path {path} in result {result} with error {e}"
                    )

            if (
                len(values) == 0
                or type(values[0]) == str
                or len(values) != len(results)
            ):
                # Check if traversing result path is just an empty dictionary
                logging.info(f"Could not get values for path {path}")
                continue

            values = np.array(values)
            mean = np.nanmean(values)
            standard_deviation = np.nanstd(values)
            logging.info(
                f"Average for path {path} is {mean} with standard deviation {standard_deviation}"
            )
            set_dictionary_value(final_result, (mean, standard_deviation), path)

        # Save the results
        # If only a single tuple of results is given, we save the results in the experiment folder
        # Otherwise, we save the results in a subfolder of the experiment folder
        current_average_path: str = ""
        if len(result_paths_tuples) == 1:
            current_average_path = save_path
        else:
            current_average_path = os.path.join(save_path, f"results_{i}")
            os.makedirs(current_average_path, exist_ok=True)

        utils.save_results(final_result, current_average_path)

        # Also save in the arguments of the experiment the arguments of the averaged results
        for k in range(len(results_paths_tuple)):
            os.makedirs(os.path.join(current_average_path, f"args_{k}"), exist_ok=True)
            utils.save_args(
                args_dictionaries[k], os.path.join(current_average_path, f"args_{k}")
            )

        logging.info(f"Final result: {final_result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("average_results")
    parser.add_argument(
        "--separate_folder_paths",
        nargs="+",
        type=str,
        default=None,
        help="The paths to separate folders to average. Each folder must contain the same number of subfolders.",
    )
    parser.add_argument(
        "--results_paths",
        nargs="+",
        type=str,
        default=None,
        help="The direct paths to folders where results are stored to average.",
    )
    parser.add_argument(
        "--single_folder_path",
        type=str,
        default=None,
        help="The path to a single folder containing results in subfolders to average.",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="./results",
        help="The directory in which to save the averaged results.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="An optional label to add to the name of the averaged results.",
    )
    parser.add_argument(
        "--custom_args_to_ignore",
        nargs="+",
        type=str,
        default=[],
        help="The arguments to ignore when checking if the arguments are the same for all the results to average.",
    )

    args = parser.parse_args()
    average(args)
