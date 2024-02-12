from typing import Optional, Type, Callable

from yamle.pruning.unstructured.magnitude import UnstructuredMagnitudePruner
from yamle.pruning.pruner import DummyPruner

AVAILABLE_PRUNERS = {
    "unstructured_magnitude": UnstructuredMagnitudePruner,
    None: DummyPruner,
    "dummy": DummyPruner,
    "none": DummyPruner,
}


def pruner_factory(pruner_type: Optional[str] = None) -> Type[Callable]:
    """This function is used to create a pruner instance based on the pruner type.

    Args:
        pruner_type (str): The type of pruner to create.
    """
    if pruner_type not in AVAILABLE_PRUNERS:
        raise ValueError(f"Unknown pruner type {pruner_type}.")
    return AVAILABLE_PRUNERS[pruner_type]
