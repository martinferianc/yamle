from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


class LaplaceLinear(nn.Module):
    """This class implements the Laplace linear layer.

    Once the Hessian is computed it uses the K-FAC approximation to compute the approximation to the Hessian.

    Args:
        U (torch.Tensor): The first matrix to compute the factorised Hessian.
        V (torch.Tensor): The second matrix to compute the factorised Hessian.
        weight (torch.Tensor): The weight matrix of the linear layer.
        bias (Optional[torch.Tensor]): The bias vector of the linear layer.
    """

    def __init__(
        self,
        U: torch.Tensor,
        V: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        # Initialize the buffers for the factorised Hessian
        self.register_buffer("_U", U)
        self.register_buffer("_V", V)
        # Initialize the weight and bias parameters, the weights are not trainable
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = (
            nn.Parameter(bias, requires_grad=False) if bias is not None else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This function computes the forward pass of the Laplace linear layer.

        Args:
            x (torch.Tensor): The input tensor.
        """
        # Compute the forward pass
        mean = F.linear(x, self.weight, self.bias)
        if self.training:
            return mean
        else:
            # Compute the variance prediction
            variance = (
                torch.mm(torch.mm(x, self._V), x.T).diag().reshape(-1, 1, 1) * self._U
            )
            distribution = MultivariateNormal(mean, variance)
            return distribution.sample()

    def extra_repr(self) -> str:
        """This function returns the extra representation of the layer."""
        return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, bias={self.bias is not None}"
