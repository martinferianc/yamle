from typing import Type


from yamle.trainers.trainer import BaseTrainer
from yamle.trainers.ensemble import EnsembleTrainer, BaggingTrainer
from yamle.trainers.calibration import CalibrationTrainer

AVAILABLE_TRAINERS = {
    "base": BaseTrainer,
    "ensemble": EnsembleTrainer,
    "bagging": BaggingTrainer,
    "calibration": CalibrationTrainer,
}


def trainer_factory(trainer_type: str) -> Type[BaseTrainer]:
    """This function is used to create a trainer instance based on the trainer type.

    Args:
        trainer_type (str): The type of trainer to create.
    """
    if trainer_type not in AVAILABLE_TRAINERS:
        raise ValueError(f"Unknown trainer type {trainer_type}.")
    return AVAILABLE_TRAINERS[trainer_type]
