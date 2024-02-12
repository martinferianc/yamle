import logging
import os
from argparse import ArgumentParser

import yamle.utils.file_utils as utils
from yamle.utils.tuning_utils import (
    plot_different_runs_and_metrics,
    plot_different_runs_and_metric_config_combinations,
    plot_different_metrics_and_trial_id,
)

logging = logging.getLogger("pytorch_lightning")


def plot_analyze_tune(args: ArgumentParser) -> None:
    """This is a helper function which loads in the dataframe for a tuning experiment and plots the results
    and different metrics and statistics."""

    # Create experiment structure
    experiment_name = f"{utils.current_time()}-analyze-tune"
    experiment_name += f"-{args.label}" if args.label is not None else ""

    # Create experiment directory
    save_path = os.path.join(args.save_path, experiment_name)
    save_path = utils.create_experiment_folder(save_path, "./src", cache_scripts=False)

    # Set the logger
    utils.config_logger(save_path)
    logging.info("Beginning Analyze Tune: %s", experiment_name)
    logging.info("Arguments: %s", args)
    logging.info("Command arguments to reproduce: %s", utils.argparse_to_command(args))

    # Save the arguments
    utils.save_args(args, save_path)
    utils.save_args_dictionary(args, save_path)

    # Load in the dataframe
    df = utils.load_tuning_results(args.experiment)

    # Plot the different statistics and analysis
    plot_different_runs_and_metrics(df, save_path)
    plot_different_runs_and_metric_config_combinations(df, save_path)
    plot_different_metrics_and_trial_id(df, save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="An optional label to be added to the experiment name.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="experiments",
        help="The directory where the experiment results are stored.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="The name of the experiment to be analyzed.",
    )
    args = parser.parse_args()

    plot_analyze_tune(args)
