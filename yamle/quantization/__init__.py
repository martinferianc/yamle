from typing import Optional, Type, Callable

from yamle.quantization.quantizer import DummyQuantizer
from yamle.quantization.static import StaticQuantizer
from yamle.quantization.qat import QATQuantizer

AVAILABLE_QUANTIZERS = {
    "static": StaticQuantizer,
    None: DummyQuantizer,
    "dummy": DummyQuantizer,
    "none": DummyQuantizer,
    "qat": QATQuantizer,
}


def quantizer_factory(quantizer_type: Optional[str] = None) -> Type[Callable]:
    """This function is used to create a quantizer instance based on the quantizer type.

    Args:
        quantizer_type (str): The type of pruner to create.
    """
    if quantizer_type not in AVAILABLE_QUANTIZERS:
        raise ValueError(f"Unknown pruner type {quantizer_type}.")
    return AVAILABLE_QUANTIZERS[quantizer_type]
