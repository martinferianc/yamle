from typing import Type


from yamle.methods.method import BaseMethod
from yamle.methods.augmentation_classification import (
    CutOutImageClassificationMethod,
    CutMixImageClassificationMethod,
    MixUpImageClassificationMethod,
    RandomErasingImageClassificationMethod,
)
from yamle.methods.contrastive import SimCLRVisionMethod
from yamle.methods.mimo import (
    MIMOMethod,
    MixMoMethod,
    DataMUXMethod,
    UnMixMoMethod,
    MIMMOMethod,
    MixVitMethod,
)
from yamle.methods.pe import PEMethod
from yamle.methods.mcdropout import (
    MCDropoutMethod,
    MCDropConnectMethod,
    MCStandOutMethod,
    MCDropBlockMethod,
    MCStochasticDepthMethod,
)
from yamle.methods.ensemble import (
    EnsembleMethod,
    SnapsotEnsembleMethod,
    GradientBoostingEnsembleMethod,
)
from yamle.methods.moe import (
    MultiHeadEnsembleMethod,
    MixtureOfExpertsMethod,
)
from yamle.methods.dun import DUNMethod
from yamle.methods.early_exit import EarlyExitMethod
from yamle.methods.sngp import SNGPMethod
from yamle.methods.be import BEMethod
from yamle.methods.temperature_scaling import TemperatureMethod
from yamle.methods.rbnn import RBNNMethod
from yamle.methods.svi import (
    SVIRTMethod,
    SVILRTMethod,
    SVIFlipOutRTMethod,
    SVIFlipOutDropConnectMethod,
    SVILRTVDMethod,
)
from yamle.methods.delta_uq import DeltaUQMethod
from yamle.methods.gp import GPMethod
from yamle.methods.evidential_regression import (
    EvidentialRegressionMethod,
)
from yamle.methods.sgld import SGLDMethod
from yamle.methods.laplace import LaplaceMethod
from yamle.methods.swag import SWAGMethod

AVAILABLE_METHODS = {
    "base": BaseMethod,
    "simclrvision": SimCLRVisionMethod,
    "cutout": CutOutImageClassificationMethod,
    "cutmix": CutMixImageClassificationMethod,
    "mixup": MixUpImageClassificationMethod,
    "random_erasing": RandomErasingImageClassificationMethod,
    "mimo": MIMOMethod,
    "mimmo": MIMMOMethod,
    "mixmo": MixMoMethod,
    "mixvit": MixVitMethod,
    "unmixmo": UnMixMoMethod,
    "datamux": DataMUXMethod,
    "pe": PEMethod,
    "svirt": SVIRTMethod,
    "svilrt": SVILRTMethod,
    "svilrtvd": SVILRTVDMethod,
    "sviflipout_gaussian": SVIFlipOutRTMethod,
    "sviflipout_dropconnect": SVIFlipOutDropConnectMethod,
    "mcdropout": MCDropoutMethod,
    "mcdropconnect": MCDropConnectMethod,
    "mcstandout": MCStandOutMethod,
    "mcdropblock": MCDropBlockMethod,
    "mcstochasticdepth": MCStochasticDepthMethod,
    "ensemble": EnsembleMethod,
    "snapshot_ensemble": SnapsotEnsembleMethod,
    "multi_head_ensemble": MultiHeadEnsembleMethod,
    "mixture_of_experts": MixtureOfExpertsMethod,
    "gradient_boosting_ensemble": GradientBoostingEnsembleMethod,
    "sgld": SGLDMethod,
    "dun": DUNMethod,
    "swag": SWAGMethod,
    "early_exit": EarlyExitMethod,
    "sngp": SNGPMethod,
    "be": BEMethod,
    "temperature": TemperatureMethod,
    "rbnn": RBNNMethod,
    "delta_uq": DeltaUQMethod,
    "gp": GPMethod,
    "laplace": LaplaceMethod,
    "evidential_regression": EvidentialRegressionMethod,
}


def method_factory(method_type: str) -> Type[BaseMethod]:
    """This function is used to create a method instance based on the method type.

    Args:
        method_type (str): The type of method to create.
    """
    if method_type not in AVAILABLE_METHODS:
        raise ValueError(f"Unknown method type {method_type}.")
    return AVAILABLE_METHODS[method_type]
