from typing import Type, Optional

import torch.nn as nn

from yamle.models.fc import FCModel, ResidualFCModel
from yamle.models.convnet import ConvNetModel, ResidualConvNetModel
from yamle.models.resnet import ResNetModel
from yamle.models.densenet import DenseNetModel
from yamle.models.unet import UNetModel
from yamle.models.transformer import TransformerModel
from yamle.models.visual_transformer import VisualTransformerModel
from yamle.models.rnn import RNNModel, RNNAutoEncoderModel
from yamle.models.mixer import MixerModel
from yamle.models.vgg import VGGModel

AVAILABLE_MODELS = {
    "fc": FCModel,
    "convnet": ConvNetModel,
    "residualconvnet": ResidualConvNetModel,
    "residualfc": ResidualFCModel,
    "resnet": ResNetModel,
    "densenet": DenseNetModel,
    "vgg": VGGModel,
    "unet": UNetModel,
    "transformer": TransformerModel,
    "visualtransformer": VisualTransformerModel,
    "mixer": MixerModel,
    "rnn": RNNModel,
    "rnnautoencoder": RNNAutoEncoderModel,
    None: nn.Identity,
}


def model_factory(model_type: Optional[str] = None) -> Type[nn.Module]:
    """This function is used to return a model instance based on the model type.

    Args:
        model_type (str): The type of model to create.
    """
    if model_type not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model type {model_type}.")
    return AVAILABLE_MODELS[model_type]
