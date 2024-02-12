from typing import Union, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBE(nn.Linear):
    """This is a wrapper around a linear layer that implements BatchEnsemble.
    
    It implements the original rank 1 BatchEnsemble.
    
    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        num_members (int): The number of members in the ensemble.
        weight (nn.Parameter): The weight of the layer.
        bias (Optional[nn.Parameter]): The bias of the layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_members: int,
        weight: Optional[nn.Parameter] = None,
        bias: Optional[nn.Parameter] = None,
    ) -> None:
        super(LinearBE, self).__init__(in_features, out_features)
        if weight is not None:
            self.weight = weight
        self.bias = bias
        self._num_members = num_members
        self.r = torch.nn.Parameter(
            torch.ones(num_members, in_features), requires_grad=True
        )
        self.s = torch.nn.Parameter(
            torch.ones(num_members, out_features), requires_grad=True
        )

        self.r.data += torch.randn_like(self.r) * 0.01
        self.s.data += torch.randn_like(self.s) * 0.01
        
        if self.bias is not None:
            self.bias = torch.nn.Parameter(
                torch.ones(num_members, out_features), requires_grad=True
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the layer."""
        R = self.r.unsqueeze(0)
        S = self.s.unsqueeze(0)
        # Now the shape of R and S is (1, M, in_features) and (1, M, out_features)
        N = x.shape[0]
        M = self._num_members
        x = x.reshape(N // M, M, *x.shape[1:])
        
        # Unsqueeze R and S to match the shape of x in case the input has more than 2 dimensions
        while len(R.shape) < len(x.shape):
            R = R.unsqueeze(2)
        while len(S.shape) < len(x.shape):
            S = S.unsqueeze(2)
        # Now the shape of R and S is (1, M, 1, 1, ..., in_features) and (1, M, 1, 1, ..., out_features)
        x = x * R
        # The shape of x is (N // M, M, 1, 1, ..., in_features)
        output = F.linear(x, self.weight, None)
        # The shape of output is (N // M, M, 1, 1, ..., out_features)
        output = output * S
        # The shape of output is (N // M, M, 1, 1, ..., out_features)
        
        if self.bias is not None:
            bias = self.bias.unsqueeze(0)
            # The shape of bias is (1, M, out_features)
            while len(bias.shape) < len(output.shape):
                bias = bias.unsqueeze(2)
            # The shape of bias is (1, M, 1, 1, ..., out_features)
            output += bias

        return output.reshape(N, *output.shape[2:])

    def extra_repr(self) -> str:
        return super().extra_repr() + f", num_members={self._num_members}"


class Conv2dBE(nn.Conv2d):
    """This is a wrapper around a convolutional layer that implements BatchEnsemble.
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (Union[int, Tuple[int, int]]): The size of the convolutional kernel.
        stride (Union[int, Tuple[int, int]], optional): The stride of the convolution. Defaults to 1.
        padding (Union[int, Tuple[int, int]], optional): The padding of the convolution. Defaults to 0.
        dilation (Union[int, Tuple[int, int]], optional): The dilation of the convolution. Defaults to 1.
        groups (int, optional): The number of groups in the convolution. Defaults to 1.
        num_members (int): The number of members in the ensemble.
        weight (Optional[nn.Parameter]): The weight of the layer.
        bias (Optional[nn.Parameter]): The bias of the layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        num_members: int = 1,
        weight: Optional[nn.Parameter] = None,
        bias: Optional[nn.Parameter] = None,
    ) -> None:
        super(Conv2dBE, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if weight is not None:
            self.weight = weight
        self._num_members = num_members
        self.r = torch.nn.Parameter(
            torch.ones(num_members, in_channels, 1, 1), requires_grad=True
        )
        self.s = torch.nn.Parameter(
            torch.ones(num_members, out_channels, 1, 1), requires_grad=True
        )

        self.r.data += torch.randn_like(self.r) * 0.01
        self.s.data += torch.randn_like(self.s) * 0.01
        
        if self.bias is not None:
            self.bias = torch.nn.Parameter(
                torch.ones(num_members, out_channels), requires_grad=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the layer."""
        R = self.r.unsqueeze(0)
        S = self.s.unsqueeze(0)
        # The shape of R and S is (1, M, in_channels, 1, 1) and (1, M, out_channels, 1, 1)
        N = x.shape[0]
        M = self._num_members
        x = x.reshape(N // M, M, x.shape[1], x.shape[2], x.shape[3])
        # The shape of x is (N // M, M, in_channels, height, width)
        x = x * R
        # The shape of x is (N // M, M, in_channels, height, width)
        x = x.reshape(N, x.shape[2], x.shape[3], x.shape[4])
        # The shape of x is (N, in_channels, height, width) we need to do this because conv2d expects the input to be of shape (N, C, H, W)
        x = F.conv2d(
            x, self.weight, None, self.stride, self.padding, self.dilation, self.groups
        )
        # The shape of x is (N, out_channels, height, width)
        x = x.reshape(N // M, M, x.shape[1], x.shape[2], x.shape[3])
        # We need to reshape x to (N // M, M, out_channels, height, width)
        x = x * S
        if self.bias is not None:
            # The shape of bias is (1, M, out_channels)
            x += self.bias.unsqueeze(0)
            
        x = x.reshape(N, x.shape[2], x.shape[3], x.shape[4])

        return x

    def extra_repr(self) -> str:
        return super().extra_repr() + f", num_members={self._num_members}"