from typing import Any, Dict, Tuple, List, Union
import numpy as np
import logging
from yamle.evaluation.metrics.algorithmic import METRIC_TENDENCY
from yamle.defaults import MAX_TENDENCY, POSITIVE_INFINITY, NEGATIVE_INFINITY
from paretoset import paretoset
from collections import namedtuple

logging = logging.getLogger("pytorch_lightning")


def average_augmentations(
    result: Dict[str, Any], metric: str
) -> Union[Tuple[float, float], bool]:
    """This function will average the results of the augmentations and return the mean and standard deviation of the results.

    It is performed across all levels of augmentation. For example, if we have 3 augmentations with 3 levels each,
    the function will return the mean and standard deviation of the 9 results.

    Args:
        result (Dict[str, Any]): The results dictionary
        metric (str): The metric to average
    """
    unique_augmentations = list(
        set(["_".join(key.split("_")[1:-1]) for key in result if "test_" in key])
    )

    # Get the number of levels of augmentation
    augmentation_levels = {
        key: len([value for value in result.keys() if value.startswith(f"test_{key}")])
        for key in unique_augmentations
    }
    # Check if all augmentations have the same number of levels
    assert (
        len(set(augmentation_levels.values())) == 1
    ), f"All augmentations must have the same number of levels. Got keys {augmentation_levels.keys()} with values {[augmentation_levels[key] for key in augmentation_levels.keys()]}."

    augmentation_results = np.zeros(
        (len(unique_augmentations), augmentation_levels[unique_augmentations[0]])
    )
    for j, augmentation in enumerate(unique_augmentations):
        for level in range(augmentation_levels[augmentation]):
            augmentation_name = f"test_{augmentation}_{level}"
            if metric not in result[augmentation_name]:
                logging.warning(
                    f"The metric {metric} is not in the results of the augmentation {augmentation_name}."
                )
                return False
            augmentation_results[j][level] = (
                result[augmentation_name][metric][0]
                if isinstance(result[augmentation_name][metric], tuple)
                else result[augmentation_name][metric]
            )
    return np.mean(augmentation_results), np.std(augmentation_results)


def fetch_value_and_std_from_result(
    result: Dict[str, Any], dataset: str, metric: str
) -> Tuple[float, float]:
    """This is a helper function that will fetch the value and error from a result dictionary."""
    if dataset == "augmentation":
        val, str = average_augmentations(result, metric)
    else:
        val, str = (
            result[dataset][metric]
            if isinstance(result[dataset][metric], tuple)
            else (result[dataset][metric], 0.0)
        )
    return val, str


def pareto_optimal(
    values: List[List[Tuple[float, float]]], metrics: List[str]
) -> Tuple[List[bool], List[namedtuple]]:
    """Selects the pareto optimal points from a list of points.

    Args:
        values (List[List[Tuple[float, float]]]): A list of value sets, where each value set contains a list of (mean, std dev) tuples.
        metrics (List[str]): A list of metric names, one for each value set.

    Returns:
        A tuple containing:
            - mask (List[bool]): A list indicating which points are pareto optimal
            - solutions (List[namedtuple]): The pareto optimal solutions
    """
    # Check if the length of the values and metrics are the same
    assert len(values) == len(
        metrics
    ), f"The length of the values and metrics must be the same. Got {len(values)} values and {len(metrics)} metrics."
    assert len(values) > 1, "There must be at least 2 values to compare."
    # Check if the values are of the same length
    assert (
        len(set([len(value) for value in values])) == 1
    ), "All values must be of the same length."

    metric_tendencies = [METRIC_TENDENCY[metric] == MAX_TENDENCY for metric in metrics]
    Solution = namedtuple("Solution", ["solution", "obj_value"])

    # Convert values into Solution objects
    solutions = []
    # Get the mean values of the metrics which is at index 0
    for i in range(len(values[0])):
        vals = []
        means = []
        problem = False
        for j in range(len(values)):
            # Check if the values are not in the range of infinity, those were NaNs, there is likely a problem
            # The 100s are for numerical stability
            if (
                values[j][i][0] > POSITIVE_INFINITY - 100
                or values[j][i][0] < NEGATIVE_INFINITY + 100
            ):
                logging.warning(
                    f"The experiment {i} has a NaN value in {[values[k][i] for k in range(len(values))]}, not including it in the pareto set."
                )
                problem = True
                break
            vals.append(values[j][i])
            means.append(values[j][i][0])
            
        # If there is a problem, replace all the values with the worst possible value based on the metric tendency
        if problem:
            vals = []
            means = []
            for metric in metrics:
                if METRIC_TENDENCY[metric] == MAX_TENDENCY:
                    vals.append((NEGATIVE_INFINITY, 0.0))
                    means.append(NEGATIVE_INFINITY)
                else:
                    vals.append((POSITIVE_INFINITY, 0.0))
                    means.append(POSITIVE_INFINITY)
            logging.warning(
                f"Replacing the values of experiment {i} with the worst possible values {[vals[k] for k in range(len(vals))]}"
            )
                    
        solutions.append(Solution(solution=tuple(vals), obj_value=np.array(means)))
    # Create an array of objective values and compute the non-dominated set
    objective_values_array = np.vstack([s.obj_value for s in solutions])
    sense = ["max" if tendency else "min" for tendency in metric_tendencies]

    mask = paretoset(objective_values_array, sense=sense)
    solutions = [solution for (solution, m) in zip(solutions, mask) if m]

    return mask, solutions
