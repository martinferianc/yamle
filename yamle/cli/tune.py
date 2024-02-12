import json
import logging
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import tabulate
from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend import LocalBackend
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.baselines import MORandomScalarizationBayesOpt, baselines_dict

import yamle.utils.file_utils as utils
from yamle.evaluation.metrics.algorithmic import (
    METRIC_TENDENCY,
    MIN_TENDENCY,
    parse_metric,
)
from yamle.utils.tuning_utils import (
    best_config_to_command_arguments,
    plot_different_runs_and_metrics,
    plot_different_metrics_and_trial_id,
    plot_different_runs_and_metric_config_combinations,
    sample_initial_random_configs,
)
from yamle.utils.file_utils import tuning_results_file

logging = logging.getLogger("pytorch_lightning")


# This is a temporary fix to include Multi Objective Bayesian Optimization
# It is not included in the syne-tune package yet
baselines_dict["MOBO"] = MORandomScalarizationBayesOpt


def tune(args: ArgumentParser) -> None:
    # Set seed
    pl.seed_everything(args.seed, workers=True)
    # Create experiment structure
    optimizer_name = args.optimizer.replace(" ", "-")
    experiment_name = f"{utils.current_time()}-tune-{optimizer_name}"
    if args.label is not None:
        experiment_name += f"-{args.label.replace('_', '-')}"

    # Create experiment directory
    save_path = os.path.join(args.save_path, experiment_name)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    utils.config_logger(save_path)

    logging.info("Beginning tuning: %s", experiment_name)
    logging.info("Arguments: %s", args)
    logging.info("Command arguments to reproduce: %s", utils.argparse_to_command(args))

    # Copy the `args.config_file` to the experiment directory
    shutil.copy(args.config_file, utils.config_file(save_path))

    config_space = utils.import_config_function_from_file(args.config_file)()
    config_space = {**config_space, **args.additional_arguments}
    config_space["trainer_mode"] = "tune"
    config_space["label"] = "tune"

    logging.info("Config space: %s", config_space)

    # If optimization metric is a list with a single element, then convert it to a string
    # else keep it as a list, the same applies to modes
    modes = []
    optimization_metrics = []
    for optimization_metric in args.optimization_metric:
        modes.append(
            "min"
            if METRIC_TENDENCY[parse_metric(optimization_metric)] == MIN_TENDENCY
            else "max"
        )
        optimization_metrics.append(optimization_metric)
    if len(optimization_metrics) == 1:
        optimization_metrics = optimization_metrics[0]
        modes = modes[0]

    points_to_evaluate = None
    if args.initial_random_configs is not None:
        points_to_evaluate = sample_initial_random_configs(
            config_space, args.initial_random_configs
        )

    logging.info("Optimization metrics: %s", optimization_metrics)
    logging.info("Optimization tendencies: %s", modes)
    scheduler_kwargs = {
        "config_space": config_space,
        "metric": optimization_metrics,
        "max_t": config_space["trainer_epochs"],
        "random_seed": args.seed,
        "mode": modes,
        "points_to_evaluate": points_to_evaluate,
    }

    if args.optimizer == "ASHA":
        scheduler_kwargs["resource_attr"] = "epoch"
    if args.optimizer == "Grid Search":
        scheduler_kwargs["search_options"] = {}
        scheduler_kwargs["search_options"]["shuffle_config"] = False
        scheduler_kwargs["search_options"]["allow_duplicates"] = False
    # Add the additional arguments to the scheduler kwargs
    scheduler_kwargs = {**scheduler_kwargs, **args.additional_optimizer_arguments}
    logging.info("Scheduler kwargs: %s", scheduler_kwargs)

    scheduler = baselines_dict[args.optimizer](**scheduler_kwargs)
    tuner = Tuner(
        trial_backend=LocalBackend(entry_point="yamle/cli/train.py"),
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(
            max_wallclock_time=args.max_wallclock_time,
            max_num_trials_started=args.max_num_trials_started,
        ),
        max_failures=args.max_num_failed_trials,
        n_workers=args.n_workers,
        tuner_name=experiment_name,
    )

    if args.optimizer == "Grid Search":
        logging.info(
            f"Number of trials: {len(scheduler._searcher.hp_values_combinations)}, trials: {scheduler._searcher.hp_values_combinations} keys: {scheduler._searcher.hp_keys}"
        )

    logging.info("Tuner: %s", tuner)

    tuner.run()
    tuning_experiment = load_experiment(tuner.name)
    tuning_experiment.plot()
    plt.grid()
    plt.savefig(os.path.join(save_path, "tuning.png"), bbox_inches="tight")
    plt.close()
    plt.clf()
    best_config = tuning_experiment.best_config()
    logging.info("Best configuration: %s", best_config)
    logging.info(
        "Best configuration command: %s", best_config_to_command_arguments(best_config)
    )
    utils.save_pickle(best_config, os.path.join(save_path, "best_configuration.pkl"))
    # Write it also as a text file
    with open(os.path.join(save_path, "best_configuration.txt"), "w") as f:
        f.write(str(best_config))

    results_df = tuning_experiment.results
    results_df.to_csv(tuning_results_file(save_path))
    logging.info(
        "Results dataframe: %s",
        tabulate.tabulate(results_df, headers="keys", tablefmt="psql"),
    )
    plot_different_runs_and_metrics(results_df, save_path)
    plot_different_runs_and_metric_config_combinations(results_df, save_path)
    plot_different_metrics_and_trial_id(results_df, save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./experiments",
        help="The path to save the experiment.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="The config file to be used for hyperparameter tuning.",
    )
    parser.add_argument(
        "--optimization_metric",
        nargs="+",
        type=str,
        default="validation_accuracy",
        help="The metric to be used for hyperparameter optimization.",
    )
    parser.add_argument(
        "--additional_arguments",
        type=json.loads,
        default={},
        help="Additional arguments to be used for hyperparameter tuning.",
    )
    parser.add_argument(
        "--additional_optimizer_arguments",
        type=json.loads,
        default={},
        help="Additional arguments to be used for hyperparameter tuning and configuring the optimizer.",
    )
    parser.add_argument(
        "--initial_random_configs",
        type=int,
        default=None,
        help="The number of initial random configurations to be used for hyperparameter tuning.",
    )

    parser.add_argument_group("Experiment")
    parser.add_argument(
        "--seed", type=int, default=42, help="The seed to be used for training."
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="The label to be used for the experiment.",
    )

    parser.add_argument_group("Tuning")
    parser.add_argument(
        "--max_wallclock_time",
        type=int,
        default=86400,
        help="The maximum wallclock time to be used for hyperparameter tuning.",
    )
    parser.add_argument(
        "--max_num_trials_started",
        type=int,
        default=2400,
        help="The maximum number of trials to be started for hyperparameter tuning.",
    )
    parser.add_argument(
        "--max_num_failed_trials",
        type=int,
        default=100,
        help="The maximum number of failed trials to be used for hyperparameter tuning.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="The number of workers to be used for hyperparameter tuning.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=list(baselines_dict.keys()),
        default="Random Search",
        help="The optimizer to be used for hyperparameter tuning.",
    )

    args = parser.parse_known_args()[0]
    args = parser.parse_args()

    tune(args)
