import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import natsort

from yamle.evaluation.metrics.algorithmic import (
    METRIC_TENDENCY,
    MIN_TENDENCY,
    parse_metric,
)


def sample_initial_random_configs(
    config_space: Dict[str, Any], n_samples: int = 10
) -> List[Dict[str, Any]]:
    """Sample random initial configurations from the config space."""
    initial_configs = []
    for _ in range(n_samples):
        initial_config = {}
        for key, value in config_space.items():
            if hasattr(value, "sample"):
                initial_config[key] = value.sample()
            else:
                initial_config[key] = value

        initial_configs.append(initial_config)
    return initial_configs


def best_config_to_command_arguments(
    best_config: Dict[str, Any],
    omit: List[str] = ["label", "no_evaluation", "no_saving"],
) -> str:
    """Convert the best config to command arguments."""
    command = ""
    for key, value in best_config.items():
        if "config_" in key:
            key = key.replace("config_", "")
            if key in omit:
                continue
            if isinstance(value, list):
                for v in value:
                    command += f"--{key} {v} "
            else:
                command += f"--{key} {value} "

    return command


def plot_different_runs_and_metrics(results_df: pd.DataFrame, save_path: str) -> None:
    """Plot each different run separately on the same graph with respect to the epoch and all the logged metrics."""
    # Get unique runs
    # Color each run differently
    unique_runs = results_df["trial_id"].unique()
    # Get all the metrics from the results dataframe which where the columns can be found in the METRIC_TENDENCY dictionary's keys
    unique_metrics = [
        column
        for column in results_df.columns
        if parse_metric(column) in METRIC_TENDENCY
    ]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_runs)))
    # Create a folder to save the plots
    save_path = os.path.join(save_path, "individual_runs")
    # Get the maximum number of epochs, these are stored under `config_trainer_epochs` column
    epochs = max(results_df["config_trainer_epochs"].values)
    os.makedirs(save_path, exist_ok=True)
    for metric in unique_metrics:
        last_value = None
        best_id = None
        best_index = None
        full_training = 0  # Count the number of complete runs
        ymin = float("inf")
        ymax = float("-inf")
        for i, run in enumerate(unique_runs):
            run_df = results_df[results_df["trial_id"] == run]
            x = np.arange(0, len(run_df[metric].values))
            plt.plot(x, run_df[metric].values, color=colors[i])
            ymin = min(ymin, np.min(run_df[metric].values))
            ymax = max(ymax, np.max(run_df[metric].values))
            if last_value is None:
                last_value = run_df[metric].values[-1]
                best_id = run
                best_index = i
            else:
                if METRIC_TENDENCY[parse_metric(metric)] == MIN_TENDENCY:
                    if run_df[metric].values[-1] < last_value:
                        last_value = run_df[metric].values[-1]
                        best_id = run
                        best_index = i
                else:
                    if run_df[metric].values[-1] > last_value:
                        last_value = run_df[metric].values[-1]
                        best_id = run
                        best_index = i
            if len(x) == epochs:
                full_training += 1
        # Plot the best run with respect to black color
        best_run = results_df[results_df["trial_id"] == best_id]
        x = np.arange(0, len(best_run[metric].values))
        plt.plot(
            x,
            best_run[metric].values,
            color=colors[best_index],
            linestyle="--",
            linewidth=4,
        )
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.grid()
        plt.ylim(ymin - (ymax - ymin) * 0.1, ymax + (ymax - ymin) * 0.1)
        plt.title(
            f"Individual runs, Full training: {full_training}, All runs: {len(unique_runs)}, Best run id: {best_id}",
            fontsize=10,
        )
        plt.savefig(
            os.path.join(save_path, f"{metric}_individual_runs.pdf"),
            bbox_inches="tight",
        )
        plt.close()
        plt.clf()


def plot_different_runs_and_metric_config_combinations(
    results_df: pd.DataFrame, save_path: str
) -> None:
    """Plot a scatter plot for all the different runs and their end metric values with respect toall hyperparameter combinations."""
    # Get unique runs
    unique_runs = results_df["trial_id"].unique()
    # Get all the metrics from the results dataframe which where the columns can be found in the METRIC_TENDENCY dictionary's keys
    unique_metrics = [
        column
        for column in results_df.columns
        if parse_metric(column) in METRIC_TENDENCY
    ]
    unique_config_keys = [
        column for column in results_df.columns if "config_" in column
    ]

    # Create a folder to save the plots
    save_path = os.path.join(save_path, "metric_config_combinations")
    os.makedirs(save_path, exist_ok=True)

    for config_key in unique_config_keys:
        config_values = results_df[config_key].unique()
        # If there is just a single value this was a constant hyperparameter
        if len(config_values) == 1:
            continue

        is_float = all(isinstance(value, float) for value in config_values)

        config_path = os.path.join(save_path, config_key)
        # Create a folder to save the plots
        os.makedirs(config_path, exist_ok=True)
        # Iterate over all the metrics
        # For each metric, collect the end values for each run and the given hyperparameter
        for metric in unique_metrics:
            results = {}
            for run in unique_runs:
                run_df = results_df[results_df["trial_id"] == run]
                run_last_value = run_df[metric].values[-1]
                run_config_value = run_df[config_key].values[-1]
                if run_config_value not in results:
                    results[run_config_value] = []
                results[run_config_value].append(run_last_value)

            if not is_float:
                # Sort the keys in the results dictionary
                results = {
                    key: results[key] for key in natsort.natsorted(results.keys())
                }

                # Create an index mapping the config values to the x-axis
                # This is needed because the config values can be strings
                config_values_index = {
                    config_value: i for i, config_value in enumerate(results.keys())
                }
                index_config_values = {
                    i: config_value for i, config_value in enumerate(results.keys())
                }

            # Take the mean and standard deviation of the results
            # Next to the mean print how many runs were used for that specific hyperparameter value
            x = []
            y = []
            yerr = []
            counts = []
            nan_counts = []
            for config_value, config_results in results.items():
                if not is_float:
                    # Filter out the nan values
                    config_results = [
                        value for value in config_results if not np.isnan(value)
                    ]
                    y.append(np.mean(config_results))
                    yerr.append(
                        np.std(config_results) if len(config_results) > 1 else 0
                    )
                    counts.append(len(config_results))
                    nan_counts.append(len(results[config_value]) - len(config_results))
                    x.append(config_values_index[config_value])
                else:
                    config_results = [
                        value for value in config_results if not np.isnan(value)
                    ]
                    y.append(np.mean(config_results))
                    yerr.append(
                        np.std(config_results) if len(config_results) > 1 else 0
                    )
                    x.append(config_value)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(x, y, yerr=yerr, fmt="o")
            if not is_float:
                for i, count in enumerate(counts):
                    ax.annotate(f"{count}, NaN: {nan_counts[i]}", (x[i], y[i]))

            ax.set_xlabel(config_key)
            if not is_float:
                ax.set_xticks(x, [index_config_values[i] for i in x])
            ax.set_ylabel(metric)
            ax.set_title(f"{config_key} vs {metric}")
            ax.grid()
            plt.savefig(
                os.path.join(config_path, f"{metric}_vs_{config_key}.pdf"),
                bbox_inches="tight",
            )
            plt.close()
            plt.clf()


def plot_different_metrics_and_trial_id(
    results_df: pd.DataFrame, save_path: str
) -> None:
    """Plot a scatter plot where trial id is on the x-axis and the last value of the metric is on the y-axis."""
    # Get unique runs
    unique_runs = results_df["trial_id"].unique()
    # Get all the metrics from the results dataframe which where the columns can be found in the METRIC_TENDENCY dictionary's keys
    unique_metrics = [
        column
        for column in results_df.columns
        if parse_metric(column) in METRIC_TENDENCY
    ]
    # Create a folder to save the plots
    save_path = os.path.join(save_path, "metric_trial_id")
    os.makedirs(save_path, exist_ok=True)

    for metric in unique_metrics:
        results = {}
        for run in unique_runs:
            run_df = results_df[results_df["trial_id"] == run]
            run_last_value = run_df[metric].values[-1]
            results[run] = run_last_value

        # Plot the results
        x = []
        y = []
        for run, value in results.items():
            x.append(run)
            y.append(value)

        plt.scatter(x, y)
        plt.xlabel("Trial id")

        # Plot best fit line
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m * np.array(x) + b, color="red")

        plt.ylabel(metric)
        plt.grid()
        plt.savefig(
            os.path.join(save_path, f"{metric}_vs_trial_id.pdf"), bbox_inches="tight"
        )
        plt.close()
