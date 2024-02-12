import logging
import os
import argparse
from yamle.cli.test import evaluate

import yamle.utils.file_utils as utils
import os
import json
import shutil

logging = logging.getLogger("pytorch_lightning")


def reevaluate(
    args: argparse.Namespace,
) -> None:
    """This function reruns an experiment with exactly the same arguments as the original experiment."""
    experiment_args = utils.load_args(args.load_experiment)
    if args.new_path is not None:
        experiment_name = None
        experiment_args.save_path = args.new_path
    else:
        experiment_name = os.path.basename(os.path.normpath(args.load_experiment))
        utils.config_logger(args.load_experiment)

    # Create a folder where to put the backup of the arguments and the results
    os.makedirs(os.path.join(experiment_args.save_path, "backup"), exist_ok=True)

    # Save the previous results
    shutil.copy(
        utils.results_file(args.load_experiment),
        utils.results_file(os.path.join(experiment_args.save_path, "backup")),
    )
    # Save the previous log file
    shutil.copy(
        utils.log_file(args.load_experiment),
        utils.log_file(os.path.join(experiment_args.save_path, "backup")),
    )

    logging.info("Rerunning experiment: %s", experiment_name)
    logging.info("Experiment arguments: %s", experiment_args)
    logging.info("Arguments: %s", args)
    logging.info("Command arguments to reproduce: %s", utils.argparse_to_command(args))

    # Update the arguments with the new arguments
    # Create a an args object from the new arguments
    new_args = argparse.Namespace(**args.new_args)
    new_args = utils.parse_args(new_args)
    for key, value in vars(new_args).items():
        if hasattr(experiment_args, key):
            logging.info("Updating argument %s to %s", key, value)
            setattr(experiment_args, key, value)
        else:
            raise ValueError(
                f"Argument {key} does not exist in the experiment arguments."
            )

    experiment_args.no_saving = False  # I have changed the name of the argument

    evaluate(experiment_args, experiment_name, overwrite=False, overwrite_results=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_experiment",
        type=str,
        default=None,
        help="The directory where the experiment results are stored and loaded from.",
    )
    parser.add_argument(
        "--new_args",
        type=json.loads,
        default={},
        help="The new arguments to be used for the experiment.",
    )
    parser.add_argument(
        "--new_path",
        type=str,
        default=None,
        help="The new path to store the experiment results.",
    )
    args = parser.parse_args()
    reevaluate(args)
