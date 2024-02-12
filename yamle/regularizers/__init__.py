from typing import Type, Callable, Optional


from yamle.regularizers.regularizer import DummyRegularizer
from yamle.regularizers.feature import (
    L1FeatureRegularizer,
    L2FeatureRegularizer,
    InnerProductFeatureRegularizer,
    CorrelationFeatureRegularizer,
    CosineSimilarityFeatureRegularizer,
)
from yamle.regularizers.weight import (
    L1Regularizer,
    L2Regularizer,
    L1L2Regularizer,
    WeightDecayRegularizer,
)
from yamle.regularizers.gradient import GradientNoiseRegularizer
from yamle.regularizers.model import ShrinkAndPerturbRegularizer

AVAILABLE_REGULARIZERS = {
    "l1": L1Regularizer,
    "l2": L2Regularizer,
    "weight_decay": WeightDecayRegularizer,
    "l1l2": L1L2Regularizer,
    "l1_feature": L1FeatureRegularizer,
    "l2_feature": L2FeatureRegularizer,
    "inner_product_feature": InnerProductFeatureRegularizer,
    "correlation_feature": CorrelationFeatureRegularizer,
    "cosine_similarity_feature": CosineSimilarityFeatureRegularizer,
    "gradient_noise": GradientNoiseRegularizer,
    "shrink_and_perturb": ShrinkAndPerturbRegularizer,
    None: DummyRegularizer,
    "none": DummyRegularizer,
    "dummy": DummyRegularizer,
}


def regularizer_factory(regularizer_type: Optional[str] = None) -> Type[Callable]:
    """This function is used to create a regularizer instance based on the regularizer type.

    Args:
        regularizer_type (str): The type of regularizer to create.
    """
    if regularizer_type not in AVAILABLE_REGULARIZERS:
        raise ValueError(f"Unknown regularizer type {regularizer_type}.")
    return AVAILABLE_REGULARIZERS[regularizer_type]
