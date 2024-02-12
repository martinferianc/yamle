from typing import Any
import torch
import torch.nn.functional as F
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import QuantStub

from yamle.models.specific.mcdropout import Dropout1d, Dropout2d, Dropout3d

    
class QuantisedDropout1d(Dropout1d):
    """This is the dropout class but the probability is remebered in a `nn.Parameter`.

    Args:
        p (float): The probability of an element to be zeroed.
        inplace (bool): If set to `True`, will do this operation in-place.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.quant = FloatFunctional()
        self.quant_stub = QuantStub()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the dropout layer."""
        mask = F.dropout(torch.ones_like(x), p=self._p, training=True, inplace=self.inplace)
        mask = self.quant_stub(mask)
        return self.quant.mul(x, mask)

    
class QuantisedDropout2d(Dropout2d):
    """This is the dropout class but the probability is remebered in a `nn.Parameter`.

    Args:
        p (float): The probability of an element to be zeroed.
        inplace (bool): If set to `True`, will do this operation in-place.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.quant = FloatFunctional()
        self.quant_stub = QuantStub()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the dropout layer."""
        # Create a mask where filters will be completely zeroed out
        mask = F.dropout2d(torch.ones_like(x), p=self._p, training=True, inplace=self.inplace)
        mask = self.quant_stub(mask)
        return self.quant.mul(x, mask)
        
    
class QuantisedDropout3d(Dropout3d):
    """This is the dropout class but the probability is remebered in a `nn.Parameter`.

    Args:
        p (float): The probability of an element to be zeroed.
        inplace (bool): If set to `True`, will do this operation in-place.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.quant = FloatFunctional()
        self.quant_stub = QuantStub()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the dropout layer."""
        # Create a mask where filters will be completely zeroed out
        mask = F.dropout3d(torch.ones_like(x), p=self._p, training=True, inplace=self.inplace)
        mask = self.quant_stub(mask)
        return self.quant.mul(x, mask)
