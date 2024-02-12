from typing import Optional, List


import torchmetrics
import torch.nn as nn

from yamle.evaluation.metrics.algorithmic.classification import (
    BrierScore,
    NegativeLogLikelihood,
    CalibrationError,
    Accuracy,
    Precision,
    Recall,
    Perplexity,
    ClassificationDiversity,
    AUROC,
    F1Score,
    ClassConditionalCalibrationError
)

from yamle.evaluation.metrics.algorithmic.classification import PredictiveUncertainty as ClassificationPredictiveUncertainty
from yamle.evaluation.metrics.algorithmic.regression import PredictiveUncertainty as RegressionPredictiveUncertainty

from yamle.evaluation.metrics.algorithmic.classification import AleatoricUncertainty as ClassificationAleatoricUncertainty
from yamle.evaluation.metrics.algorithmic.regression import AleatoricUncertainty as RegressionAleatoricUncertainty

from yamle.evaluation.metrics.algorithmic.classification import EpistemicUncertainty as ClassificationEpistemicUncertainty
from yamle.evaluation.metrics.algorithmic.regression import EpistemicUncertainty as RegressionEpistemicUncertainty

from yamle.evaluation.metrics.algorithmic.regression import (
    NegativeLogLikelihood as RegressionNegativeLogLikelihood,
)
from yamle.evaluation.metrics.algorithmic.regression import (
    RootMeanSquaredError,
    MeanAbsoluteError,
    MeanSquaredError,
)
from yamle.evaluation.metrics.algorithmic.segmentation import IntersectionOverUnion
from yamle.defaults import (
    LOSS_KEY,
    LOSS_REGULARIZER_KEY,
    LOSS_KL_KEY,
    REGRESSION_KEY,
    CLASSIFICATION_KEY,
    SEGMENTATION_KEY,
    RECONSTRUCTION_KEY,
    TEXT_CLASSIFICATION_KEY,
    TRAIN_KEY,
    VALIDATION_KEY,
    TEST_KEY,
    MIN_TENDENCY,
    MAX_TENDENCY,
    DEPTH_ESTIMATION_KEY,
    PRE_TRAINING_KEY,
)

AVAILABLE_METRICS = [
    BrierScore,
    NegativeLogLikelihood,
    CalibrationError,
    ClassConditionalCalibrationError,
    ClassificationPredictiveUncertainty,
    RegressionPredictiveUncertainty,
    ClassificationAleatoricUncertainty,
    RegressionAleatoricUncertainty,
    ClassificationEpistemicUncertainty,
    RegressionEpistemicUncertainty,
    RegressionNegativeLogLikelihood,
    RootMeanSquaredError,
    MeanAbsoluteError,
    MeanSquaredError,
    IntersectionOverUnion,
    Accuracy,
    Precision,
    Recall,
    F1Score,
    Perplexity,
    ClassificationDiversity,
    AUROC,
]

INDIVIDUAL_PREDICTIONS_AND_MEAN_METRICS = (
    ClassificationEpistemicUncertainty,
    ClassificationAleatoricUncertainty,
    RegressionEpistemicUncertainty,
    RegressionAleatoricUncertainty,
    ClassificationDiversity,
)


PERMEMBER_METRIC_SPLITTING_KEY = "_permember-"
MEAN_METRIC_SPLITTING_KEY = "_mean"

METRICS_TO_DESCRIPTION = {
    LOSS_KEY: "Total Loss",
    LOSS_KL_KEY: "Kullback-Leibler parameter loss",
    LOSS_REGULARIZER_KEY: "Weight regularizer",
    "accuracy": "Accuracy [0-1]",
    "precision": "Precision [0-1]",
    "recall": "Recall [0-1]",
    "f1": "F1 [0-1]",
    "auroc": "AUROC [0-1]",
    "nll": "NLL [nats]",
    "nllregression": "NLL [nats]",
    "brierscore": "Brier Score [0-1]",
    "calibration": "ECE [0-1]",
    "classconditionalcalibration": "CC-ECE [0-1]",
    "predictiveuncertainty": "Predictive Uncertainty [nats]",
    "aleatoricuncertainty": "Aleatoric Uncertainty [nats]",
    "epistemicuncertainty": "Epistemic Uncertainty [nats]",
    "predictiveuncertaintyregression": "Predictive Uncertainty [nats]",
    "aleatoricuncertaintyregression": "Aleatoric Uncertainty [nats]",
    "epistemicuncertaintyregression": "Epistemic Uncertainty [nats]",
    "mse": "Mean Squared Error",
    "mae": "Mean Absolute Error",
    "rmse": "Root Mean Squared Error",
    "iou": "Intersection over Union [0-1]",
    "perplexity": "Perplexity [nats]",
    "diversity": "Diversity [0-1]",
    "flops": "Floating Point Operations",
    "params": "Parameters",
    "latency": "Latency [s]",
    "throughput": "Throughput [samples/s]",
    "energy": "Energy [J]",
}

METRICS_TO_RANGE = {
    "accuracy": [0, 1],
    "precision": [0, 1],
    "recall": [0, 1],
    "f1": [0, 1],
    "auroc": [0, 1],
    "nll": [0, float("inf")],
    "nllregression": [-float("inf"), float("inf")],
    "brierscore": [0, 1],
    "calibration": [0, 1],
    "classconditionalcalibration": [0, 1],
    "predictiveuncertainty": [0, float("inf")],
    "aleatoricuncertainty": [0, float("inf")],
    "epistemicuncertainty": [0, float("inf")],
    "predictiveuncertaintyregression": [0, float("inf")],
    "aleatoricuncertaintyregression": [0, float("inf")],
    "epistemicuncertaintyregression": [0, float("inf")],
    "mse": [0, float("inf")],
    "mae": [0, float("inf")],
    "rmse": [0, float("inf")],
    "iou": [0, 1],
    "perplexity": [0, float("inf")],
    "diversity": [0, 1],
    "flops": [0, float("inf")],
    "params": [0, float("inf")],
    "latency": [0, float("inf")],
    "throughput": [0, float("inf")],
    "energy": [0, float("inf")],
}

PER_MEMBER_METRICS = [
    "brierscore",
    "nll",
    "calibration",
    "classconditionalcalibration",
    "predictiveuncertainty",
    "predictiveuncertaintyregression",
    "nllregression",
    "rmse",
    "mae",
    "mse",
    "iou",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
]
# These are metrics which depend on predictions (e.g. flops and params do not depend on predictions).
PREDICTION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
    "nll",
    "nllregression",
    "brierscore",
    "calibration",
    "classconditionalcalibration",
    "predictiveuncertainty",
    "predictiveuncertaintyregression",
    "aleatoricuncertainty",
    "aleatoricuncertaintyregression",
    "epistemicuncertainty",
    "epistemicuncertaintyregression",
    "mse",
    "mae",
    "rmse",
    "iou",
    "perplexity",
    "diversity",
]

def parse_metric(metric_name: str) -> str:
    """Returns the metric name from the metric name."""
    if metric_name in METRICS_TO_DESCRIPTION or metric_name in METRIC_TENDENCY:
        return metric_name
    metric = metric_name.split("_")
    # If the length of the metric is 3 it is in {train, validation, test}_metric_{permember}
    if len(metric) == 3:
        return metric[1]
    # If the length of the metric is 2 it can be in {train, validation, test}_metric or metric_{permember}
    elif len(metric) == 2:
        if TRAIN_KEY in metric or VALIDATION_KEY in metric or TEST_KEY in metric:
            return metric[1]
        else:
            return metric[0]
    return metric_name


NAME_TO_METRIC = {
    "accuracy": Accuracy,
    "precision": Precision,
    "recall": Recall,
    "f1": F1Score,
    "auroc": AUROC,
    "nll": NegativeLogLikelihood,
    "nllregression": RegressionNegativeLogLikelihood,
    "brierscore": BrierScore,
    "calibration": CalibrationError,
    "classconditionalcalibration": ClassConditionalCalibrationError,
    "predictiveuncertainty": ClassificationPredictiveUncertainty,
    "predictiveuncertaintyregression": RegressionPredictiveUncertainty,
    "aleatoricuncertainty": ClassificationAleatoricUncertainty,
    "aleatoricuncertaintyregression": RegressionAleatoricUncertainty,
    "epistemicuncertainty": ClassificationEpistemicUncertainty,
    "epistemicuncertaintyregression": RegressionEpistemicUncertainty,
    "mse": MeanSquaredError,
    "mae": MeanAbsoluteError,
    "rmse": RootMeanSquaredError,
    "iou": IntersectionOverUnion,
    "perplexity": Perplexity,
    "diversity": ClassificationDiversity,
}

METRIC_TO_NAME = {v: k for k, v in NAME_TO_METRIC.items()}

METRIC_TENDENCY = {
    LOSS_KEY: MIN_TENDENCY,
    LOSS_KL_KEY: MIN_TENDENCY,
    LOSS_REGULARIZER_KEY: MIN_TENDENCY,
    "accuracy": MAX_TENDENCY,
    "precision": MAX_TENDENCY,
    "recall": MAX_TENDENCY,
    "f1": MAX_TENDENCY,
    "auroc": MAX_TENDENCY,
    "nll": MIN_TENDENCY,
    "nllregression": MIN_TENDENCY,
    "brierscore": MIN_TENDENCY,
    "calibration": MIN_TENDENCY,
    "classconditionalcalibration": MIN_TENDENCY,
    "predictiveuncertainty": MIN_TENDENCY,
    "predictiveuncertaintyregression": MIN_TENDENCY,
    "aleatoricuncertainty": MIN_TENDENCY,
    "aleatoricuncertaintyregression": MIN_TENDENCY,
    "epistemicuncertainty": MIN_TENDENCY,
    "epistemicuncertaintyregression": MIN_TENDENCY,
    "mse": MIN_TENDENCY,
    "mae": MIN_TENDENCY,
    "rmse": MIN_TENDENCY,
    "iou": MAX_TENDENCY,
    "perplexity": MIN_TENDENCY,
    "diversity": MAX_TENDENCY,
    "flops": MIN_TENDENCY,
    "params": MIN_TENDENCY,
    "latency": MIN_TENDENCY,
    "throughput": MAX_TENDENCY,
}

MAIN_METRIC = {
    REGRESSION_KEY: "mse",
    CLASSIFICATION_KEY: "accuracy",
    SEGMENTATION_KEY: "iou",
    TEXT_CLASSIFICATION_KEY: "nll",
    DEPTH_ESTIMATION_KEY: "rmse",
    PRE_TRAINING_KEY: LOSS_KEY,
    RECONSTRUCTION_KEY: "mse",
}


def metrics_factory(
    task: str,
    num_members: Optional[int] = None,
    outputs_dim: Optional[int] = None,
    ignore_indices: Optional[List[int]] = None,
    per_member: bool = False,
    metrics: Optional[List[str]] = None,
) -> nn.ModuleDict:
    """This method is used to create the metrics for the given task."""
    if metrics is not None:
        assert (
            MAIN_METRIC[task] in metrics
        ), f"The main metric {MAIN_METRIC[task]} is not in the metrics list {metrics}."
    metrics_dict = nn.ModuleDict(
        {
            LOSS_KEY: torchmetrics.MeanMetric(),
        }
    )
    metric_task: str = None
    if task == CLASSIFICATION_KEY:
        metric_task = "multiclass"
        metrics_dict.update(
            {
                "accuracy": NAME_TO_METRIC["accuracy"](
                    task=metric_task, average="micro", num_classes=outputs_dim
                ),
                "precision": NAME_TO_METRIC["precision"](
                    task=metric_task, average="macro", num_classes=outputs_dim
                ),
                "recall": NAME_TO_METRIC["recall"](
                    task=metric_task, average="macro", num_classes=outputs_dim
                ),
                "f1": NAME_TO_METRIC["f1"](
                    task=metric_task, average="macro", num_classes=outputs_dim
                ),
                "auroc": NAME_TO_METRIC["auroc"](
                    task=metric_task, average="macro", num_classes=outputs_dim
                ),
                "calibration": NAME_TO_METRIC["calibration"](
                    task=metric_task, num_classes=outputs_dim, norm="l1", n_bins=15
                ),
                "classconditionalcalibration": NAME_TO_METRIC[
                    "classconditionalcalibration"
                ](
                    task=metric_task, num_classes=outputs_dim, norm="l1", n_bins=15
                ),
                "brierscore": NAME_TO_METRIC["brierscore"](),
                "diversity": NAME_TO_METRIC["diversity"](num_members=num_members),
                "nll": NAME_TO_METRIC["nll"](),
                "predictiveuncertainty": NAME_TO_METRIC["predictiveuncertainty"](),
                "aleatoricuncertainty": NAME_TO_METRIC["aleatoricuncertainty"](),
                "epistemicuncertainty": NAME_TO_METRIC["epistemicuncertainty"](),
                "perplexity": NAME_TO_METRIC["perplexity"](),
            }
        )
    elif task == TEXT_CLASSIFICATION_KEY:
        metric_task = "multiclass"
        metrics_dict.update(
            {
                "accuracy": NAME_TO_METRIC["accuracy"](
                    task=metric_task,
                    average="micro",
                    flatten=True,
                    num_classes=outputs_dim,
                ),
                "calibration": NAME_TO_METRIC["calibration"](
                    flatten=True,
                    metric_task=metric_task,
                    num_classes=outputs_dim,
                    norm="l1",
                    n_bins=15,
                ),
                "classconditionalcalibration": NAME_TO_METRIC[
                    "classconditionalcalibration"
                ](
                    flatten=True,
                    metric_task=metric_task,
                    num_classes=outputs_dim,
                    norm="l1",
                    n_bins=15,
                ),
                "brierscore": NAME_TO_METRIC["brierscore"](flatten=True),
                "diversity": NAME_TO_METRIC["diversity"](
                    num_members=num_members, flatten=True
                ),
                "nll": NAME_TO_METRIC["nll"](flatten=True),
                "predictiveuncertainty": NAME_TO_METRIC["predictiveuncertainty"](
                    flatten=True
                ),
                "aleatoricuncertainty": NAME_TO_METRIC["aleatoricuncertainty"](
                    flatten=True
                ),
                "epistemicuncertainty": NAME_TO_METRIC["epistemicuncertainty"](
                    flatten=True
                ),
            }
        )
    elif task == REGRESSION_KEY:
        metrics_dict.update(
            {
                "mse": NAME_TO_METRIC["mse"](),
                "mae": NAME_TO_METRIC["mae"](),
                "rmse": NAME_TO_METRIC["rmse"](),
                "nllregression": NAME_TO_METRIC["nllregression"](),
                "predictiveuncertaintyregression": NAME_TO_METRIC[
                    "predictiveuncertaintyregression"
                ](),
                "aleatoricuncertaintyregression": NAME_TO_METRIC[
                    "aleatoricuncertaintyregression"
                ](),
                "epistemicuncertaintyregression": NAME_TO_METRIC[
                    "epistemicuncertaintyregression"
                ](),
            }
        )
    elif task == RECONSTRUCTION_KEY:
        metrics_dict.update(
            {
                "mse": NAME_TO_METRIC["mse"](flatten=True),
                "mae": NAME_TO_METRIC["mae"](flatten=True),
                "rmse": NAME_TO_METRIC["rmse"](flatten=True),
                "nllregression": NAME_TO_METRIC["nllregression"](flatten=True),
                "predictiveuncertaintyregression": NAME_TO_METRIC[
                    "predictiveuncertaintyregression"
                ](flatten=True),
                "aleatoricuncertaintyregression": NAME_TO_METRIC[
                    "aleatoricuncertaintyregression"
                ](flatten=True),
                "epistemicuncertaintyregression": NAME_TO_METRIC[
                    "epistemicuncertaintyregression"
                ](flatten=True),
            }
        )
    elif task == DEPTH_ESTIMATION_KEY:
        metrics_dict.update(
            {
                "mse": NAME_TO_METRIC["mse"](flatten=True),
                "mae": NAME_TO_METRIC["mae"](flatten=True),
                "rmse": NAME_TO_METRIC["rmse"](flatten=True),
                "nllregression": NAME_TO_METRIC["nllregression"](flatten=True),
                "predictiveuncertaintyregression": NAME_TO_METRIC[
                    "predictiveuncertaintyregression"
                ](flatten=True),
                "aleatoricuncertaintyregression": NAME_TO_METRIC[
                    "aleatoricuncertaintyregression"
                ](flatten=True),
                "epistemicuncertaintyregression": NAME_TO_METRIC[
                    "epistemicuncertaintyregression"
                ](flatten=True),
            }
        )
    elif task == SEGMENTATION_KEY:
        metric_task = "multiclass"
        metrics_dict.update(
            {
                "iou": NAME_TO_METRIC["iou"](
                    num_classes=outputs_dim,
                    flatten=True,
                    permute=True,
                    ignore_indices=ignore_indices,
                ),
                "calibration": NAME_TO_METRIC["calibration"](
                    flatten=True,
                    permute=True,
                    num_classes=outputs_dim,
                    norm="l1",
                    n_bins=15,
                ),
                "classconditionalcalibration": NAME_TO_METRIC[
                    "classconditionalcalibration"
                ](
                    flatten=True,
                    permute=True,
                    num_classes=outputs_dim,
                    norm="l1",
                    n_bins=15,
                ),
                "brierscore": NAME_TO_METRIC["brierscore"](flatten=True, permute=True),
                "diversity": NAME_TO_METRIC["diversity"](
                    num_members=num_members, flatten=True, permute=True
                ),
                "nll": NAME_TO_METRIC["nll"](flatten=True, permute=True),
                "predictiveuncertainty": NAME_TO_METRIC["predictiveuncertainty"](
                    flatten=True, permute=True
                ),
                "aleatoricuncertainty": NAME_TO_METRIC["aleatoricuncertainty"](
                    flatten=True, permute=True
                ),
                "epistemicuncertainty": NAME_TO_METRIC["epistemicuncertainty"](
                    flatten=True, permute=True
                ),
            }
        )
    elif task == PRE_TRAINING_KEY:
        # No metrics for pre-training.
        pass
    else:
        raise ValueError(f"Unknown task {task}.")

    if metrics is not None:
        # Remove the metrics that are not in the metrics list and they do not start with LOSS_KEY.
        metrics_dict = nn.ModuleDict(
            {
                k: v
                for k, v in metrics_dict.items()
                if k in metrics or k.startswith(LOSS_KEY)
            }
        )
        # Check that now the dictionary has only the metrics that are in the metrics list and the LOSS_KEY.
        for metric in metrics:
            assert (
                metric in metrics_dict
            ), f"Metric {metric} is an invalid metric for task {task}."

    # If the number of members is not None compute indvidual per member metrics.
    if num_members is not None and num_members > 1 and per_member:
        for metric in PER_MEMBER_METRICS:
            if metric in metrics_dict.keys():
                metric_kwargs = {}
                if task == SEGMENTATION_KEY:
                    metric_kwargs.update(
                        {
                            "flatten": True,
                        }
                    )
                if metric == "diversity":
                    metric_kwargs.update(
                        {
                            "num_members": num_members,
                        }
                    )
                if metric == "iou":
                    metric_kwargs.update(
                        {
                            "num_classes": outputs_dim,
                            "ignore_indices": ignore_indices,
                        }
                    )
                if metric in [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "auroc",
                ]:
                    metric_kwargs.update(
                        {
                            "num_classes": outputs_dim,
                            "task": metric_task,
                        }
                    )
                if metric in ["precision", "recall", "f1", "auroc"]:
                    metric_kwargs.update(
                        {
                            "average": "macro",
                        }
                    )
                if metric in ["accuracy"]:
                    metric_kwargs.update(
                        {
                            "average": "micro",
                        }
                    )
                if metric in ["calibration", "classconditionalcalibration"]:
                    metric_kwargs.update(
                        {
                            "norm": "l1",
                            "n_bins": 15,
                            "num_classes": outputs_dim,
                            "task": metric_task,
                        }
                    )
                if metrics is not None:
                    if f"{metric}" not in metrics:
                        continue
                for member in range(num_members):
                    metrics_dict[
                        f"{metric}{PERMEMBER_METRIC_SPLITTING_KEY}{member}"
                    ] = NAME_TO_METRIC[metric](**metric_kwargs)
                metrics_dict[f"{metric}{MEAN_METRIC_SPLITTING_KEY}"] = NAME_TO_METRIC[
                    metric
                ](**metric_kwargs)
    return metrics_dict
