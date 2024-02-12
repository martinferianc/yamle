from typing import Any, Union, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from pytorch_lightning import LightningModule

from yamle.defaults import DISABLED_DROPOUT_KEY
from yamle.models.operations import ResidualLayer, LSTM, Multiply
from yamle.utils.tracing_utils import (
    get_sample_input_and_target,
    get_input_shape_from_model,
)


def _dropout(
    x: torch.Tensor, p: float, training: bool, inplace: bool = False
) -> torch.Tensor:
    """This is an implementation of dropout that can be used for any dimensionality.

    Similarly to PyTorch's dropout, it will zero out the elements of the input tensor
    but the kept elements will be scaled by `1 / (1 - p)` to keep the expected value
    of the output the same as the input.

    Args:
        x (torch.Tensor): The input tensor.
        p (float): The probability of an element to be zeroed.
        training (bool): If set to `True`, will perform the dropout.
        inplace (bool): If set to `True`, will do this operation in-place.
    """
    original_shape = x.shape
    mask = torch.ones_like(x).reshape(original_shape[0], -1)
    mask = F.dropout(mask, p=p, training=training, inplace=True)
    mask = mask.reshape(original_shape)
    return x * mask if not inplace else x.mul_(mask)


class Dropout1d(nn.Module):
    """This is the dropout class but the probability is remebered in a `nn.Parameter`.

    Args:
        p (float): The probability of an element to be zeroed.
        inplace (bool): If set to `True`, will do this operation in-place.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(Dropout1d, self).__init__()
        self.register_buffer("_p", torch.tensor(p))
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the dropout layer."""
        return _dropout(x, p=self._p, training=True, inplace=self.inplace)

    def extra_repr(self) -> str:
        return super().extra_repr() + f"p={self._p}, inplace={self.inplace}"


class Dropout2d(nn.Module):
    """This is the dropout class but the probability is remebered in a `nn.Parameter`.

    Args:
        p (float): The probability of an element to be zeroed.
        inplace (bool): If set to `True`, will do this operation in-place.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(Dropout2d, self).__init__()
        self.register_buffer("_p", torch.tensor(p))
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the dropout layer."""
        return F.dropout2d(x, p=self._p, training=True, inplace=self.inplace)

    def extra_repr(self) -> str:
        return super().extra_repr() + f"p={self._p}, inplace={self.inplace}"


class Dropout3d(nn.Module):
    """This is the dropout class but the probability is remebered in a `nn.Parameter`.

    Args:
        p (float): The probability of an element to be zeroed.
        inplace (bool): If set to `True`, will do this operation in-place.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(Dropout3d, self).__init__()
        self.register_buffer("_p", torch.tensor(p))
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the dropout layer."""
        return F.dropout3d(x, p=self._p, training=True, inplace=self.inplace)

    def extra_repr(self) -> str:
        return super().extra_repr() + f"p={self._p}, inplace={self.inplace}"


class DropoutLSTM(LSTM):
    """This is LSTM with Dropout implemented where the mask is shared across the time steps.

    A different dropout mask is applied to each gate but the masks are shared across the time steps.
    It is possible to set a different probability for each gate. Then `p` is ignored and
    needs to be set to `None`.

    Args:
        p (Optional[float]): The probability of an element to be zeroed. Defaults to 0.5.
        p_i (Optional[float]): The probability of an element to be zeroed in the input gate. Defaults to None.
        p_f (Optional[float]): The probability of an element to be zeroed in the forget gate. Defaults to None.
        p_o (Optional[float]): The probability of an element to be zeroed in the output gate. Defaults to None.
        p_g (Optional[float]): The probability of an element to be zeroed in the cell gate. Defaults to None.
    """

    def __init__(
        self,
        p: Optional[float] = 0.5,
        p_i: Optional[float] = None,
        p_f: Optional[float] = None,
        p_o: Optional[float] = None,
        p_g: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(DropoutLSTM, self).__init__(*args, **kwargs)
        assert p is not None or (
            p_i is not None and p_f is not None and p_o is not None and p_g is not None
        ), "Either p or p_i, p_f, p_o, and p_g should be specified."

        if p is not None:
            p_i = p
            p_f = p
            p_o = p
            p_g = p

        assert (
            0.0 <= p_i <= 1.0
        ), f"The probability of an element to be zeroed in the input gate should be between 0 and 1, but got {p_i}."
        assert (
            0.0 <= p_f <= 1.0
        ), f"The probability of an element to be zeroed in the forget gate should be between 0 and 1, but got {p_f}."
        assert (
            0.0 <= p_o <= 1.0
        ), f"The probability of an element to be zeroed in the output gate should be between 0 and 1, but got {p_o}."
        assert (
            0.0 <= p_g <= 1.0
        ), f"The probability of an element to be zeroed in the cell gate should be between 0 and 1, but got {p_g}."

        self.register_buffer("_p_i", torch.tensor(p_i))
        self.register_buffer("_p_f", torch.tensor(p_f))
        self.register_buffer("_p_o", torch.tensor(p_o))
        self.register_buffer("_p_g", torch.tensor(p_g))

        self._x_mask_multiply = Multiply()
        self._h_mask_multiply = Multiply()

        for m in self.modules():
            # We do not want to replace the linear layers in the LSTM
            disable_dropout_replacement(m)

    def _generate_mask(
        self, input_dim: int, p: float, device: torch.device
    ) -> torch.Tensor:
        """This method is used to generate the dropout mask.

        Args:
            input_dim (int): The input dimension.
            p (float): The probability of an element to be zeroed.
            device (torch.device): The device to put the mask on.
        """
        mask = torch.bernoulli(torch.ones(input_dim, device=device))
        if p != 0.0:
            mask = mask / (1 - p)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The forward function of the LSTM cell.

        Args:
            x (torch.Tensor): The input tensor of shape `(batch_size, T, input_size)`.
            h (torch.Tensor, optional): The hidden state of shape `(batch_size, hidden_size)`.
            c (torch.Tensor, optional): The cell state of shape `(batch_size, hidden_size)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The complete hidden state of shape `(batch_size, T, hidden_size)`
            and the last hidden and cell state of shape `(batch_size, hidden_size)`.
        """
        T = x.shape[1]

        if h is None:
            h = torch.zeros(x.size(0), self._hidden_size, device=x.device)

        if c is None:
            c = torch.zeros(x.size(0), self._hidden_size, device=x.device)

        h_t = []

        dropout_mask_x = [
            self._generate_mask(x.shape[2], self._p_i, x.device),
            self._generate_mask(x.shape[2], self._p_f, x.device),
            self._generate_mask(x.shape[2], self._p_o, x.device),
            self._generate_mask(x.shape[2], self._p_g, x.device),
        ]

        dropout_mask_h = [
            self._generate_mask(self._hidden_size, self._p_i, x.device),
            self._generate_mask(self._hidden_size, self._p_f, x.device),
            self._generate_mask(self._hidden_size, self._p_o, x.device),
            self._generate_mask(self._hidden_size, self._p_g, x.device),
        ]

        for t in range(T):
            x_i, x_f, x_o, x_g = [
                self._x_mask_multiply(x[:, t, :], dropout_mask_x[j])
                for j in range(len(dropout_mask_x))
            ]
            h_i, h_f, h_o, h_g = [
                self._h_mask_multiply(h, dropout_mask_h[j])
                for j in range(len(dropout_mask_h))
            ]

            i = self._input_gate_sigmoid(self._input_add(self._Wi(x_i), self._Ui(h_i)))
            f = self._forget_gate_sigmoid(
                self._forget_add(self._Wf(x_f), self._Uf(h_f))
            )
            o = self._output_gate_sigmoid(
                self._output_add(self._Wo(x_o), self._Uo(h_o))
            )
            g = self._g_gate_tanh(self._g_add(self._Wg(x_g), self._Ug(h_g)))
            c = self._cell_add(self._cell_multiply1(f, c), self._cell_multiply2(i, g))
            h = self._h_multiply(o, self._h_tanh(c))
            h_t.append(h)

        return torch.stack(h_t, dim=1), h, c


class StochasticDepth(nn.Module):
    """This is a wrapper around a layer that implements stochastic depth.

    It is used to be in combination with a `ResidualLayer` to implement stochastic depth.

    Args:
        p (float): The probability of dropping a residual layer.
    """

    def __init__(self, p: float = 0.5) -> None:
        super(StochasticDepth, self).__init__()
        self.register_buffer("_p", torch.tensor(p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the layer."""
        return ops.stochastic_depth(x, self._p, mode="batch", training=True)

    def extra_repr(self) -> str:
        return super().extra_repr() + f"p={self._p}"


class LinearDropConnect(nn.Linear):
    """This is a wrapper around a linear layer that implements dropconnect.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        p (float): The probability of dropping a weight.
        weight (Optional[nn.Parameter]): The weight of the layer.
        bias (Optional[nn.Parameter]): The bias of the layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        p: float = 0.5,
        weight: Optional[nn.Parameter] = None,
        bias: Optional[nn.Parameter] = None,
    ) -> None:
        super(LinearDropConnect, self).__init__(in_features, out_features)
        if weight is not None:
            self.weight = weight
        if bias is not None:
            self.bias = bias
        self.register_buffer("_p", torch.tensor(p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the layer."""
        weight = _dropout(self.weight, p=self._p, training=True, inplace=False) * (
            1 - self._p
        )
        if self.bias is not None:
            bias = _dropout(self.bias, p=self._p, training=True, inplace=False) * (
                1 - self._p
            )
        else:
            bias = None
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return super().extra_repr() + f"p={self._p}"


class Conv1dDropConnect(nn.Conv1d):
    """This is a wrapper around a convolutional layer that implements dropconnect.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int or tuple): The size of the kernel.
        stride (int or tuple, optional): The stride of the convolution. Default: 1.
        padding (int or tuple, optional): Implicit zero padding to be added on both sides. Default: 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        p (float): The probability of dropping a weight.
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
        p: float = 0.5,
        weight: Optional[nn.Parameter] = None,
        bias: Optional[nn.Parameter] = None,
    ) -> None:
        super(Conv1dDropConnect, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.register_buffer("_p", torch.tensor(p))
        if weight is not None:
            self.weight = weight
        if bias is not None:
            self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the layer."""
        weight = _dropout(self.weight, p=self._p, training=True, inplace=False) * (
            1 - self._p
        )
        if self.bias is not None:
            bias = _dropout(self.bias, p=self._p, training=True, inplace=False) * (
                1 - self._p
            )

        else:
            bias = None
        return F.conv1d(
            x,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class Conv2dDropConnect(nn.Conv2d):
    """This is a wrapper around a convolutional layer that implements dropconnect.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int or tuple): The size of the kernel.
        stride (int or tuple, optional): The stride of the convolution. Default: 1.
        padding (int or tuple, optional): Implicit zero padding to be added on both sides. Default: 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        p (float): The probability of dropping a weight.
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
        p: float = 0.5,
        weight: Optional[nn.Parameter] = None,
        bias: Optional[nn.Parameter] = None,
    ) -> None:
        super(Conv2dDropConnect, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.register_buffer("_p", torch.tensor(p))
        if weight is not None:
            self.weight = weight
        if bias is not None:
            self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the layer."""
        weight = _dropout(self.weight, p=self._p, training=True, inplace=False) * (
            1 - self._p
        )
        if self.bias is not None:
            bias = _dropout(self.bias, p=self._p, training=True, inplace=False) * (
                1 - self._p
            )
        else:
            bias = None
        return F.conv2d(
            x,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def extra_repr(self) -> str:
        return super().extra_repr() + f"p={self._p}"


class Conv3dDropConnect(nn.Conv3d):
    """This is a wrapper around a convolutional layer that implements dropconnect.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int or tuple): The size of the kernel.
        stride (int or tuple, optional): The stride of the convolution. Default: 1.
        padding (int or tuple, optional): Implicit zero padding to be added on both sides. Default: 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        p (float): The probability of dropping a weight.
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
        p: float = 0.5,
        weight: Optional[nn.Parameter] = None,
        bias: Optional[nn.Parameter] = None,
    ) -> None:
        super(Conv3dDropConnect, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.register_buffer("_p", torch.tensor(p))
        if weight is not None:
            self.weight = weight
        if bias is not None:
            self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the layer."""
        weight = _dropout(self.weight, p=self._p, training=True, inplace=False) * (
            1 - self._p
        )
        if self.bias is not None:
            bias = _dropout(self.bias, p=self._p, training=True, inplace=False) * (
                1 - self._p
            )
        else:
            bias = None
        return F.conv3d(
            x,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class StandOut(nn.Module):
    """This is a wrapper around a layer that implements stand-out.

    Args:
        alpha (float): The scaling parameter.
        beta (float): The bias parameter.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5) -> None:
        super(StandOut, self).__init__()
        self.register_buffer("_alpha", torch.tensor(alpha))
        self.register_buffer("_beta", torch.tensor(beta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the layer."""
        mask_probabilities = torch.sigmoid(self._alpha * x.detach() + self._beta)
        mask = torch.bernoulli(mask_probabilities)
        # The mask should be applied after the activation function
        return x * mask


class DropBlock(nn.Module):
    """DropBlock activation regularization module.

    As per: https://arxiv.org/pdf/1810.12890.pdf

    Args:
        block_size_percentage (float): The percentage of the input size to use as the block size.
        gamma (float): Probability of dropping activations.
        module (nn.Module): The `Conv2d` layer from where to get the input size.
    """

    def __init__(
        self, block_size_percentage: float, gamma: float, module: nn.Module
    ) -> None:
        super(DropBlock, self).__init__()
        self._block_size_percentage = block_size_percentage
        self._gamma = gamma

        def _get_module_shape():
            return get_input_shape_from_model(module)

        self._module_input_shape = _get_module_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the layer."""
        # At minimum, the block size should be 1.
        block_size = 1
        input_shape = self._module_input_shape()
        if input_shape is not None:
            block_size = max(
                int(input_shape[0][-1] * self._block_size_percentage), block_size
            )
        return ops.drop_block2d(
            x, p=self._gamma, block_size=block_size, inplace=False, training=True
        )

    def extra_repr(self) -> str:
        return (
            f"block_size_percentage={self._block_size_percentage}, gamma={self._gamma}"
        )


def disable_dropout_replacement(m: nn.Module) -> None:
    """This method is used to disable the dropout replacement for a given module.
    It will do it recursively for all the submodules.

    Args:
        m (nn.Module): The module to disable the dropout replacement for.
    """
    for module in m.modules():
        if isinstance(
            module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, ResidualLayer, LSTM)
        ):
            setattr(module, DISABLED_DROPOUT_KEY, True)


def enable_dropout_replacement(m: nn.Module) -> None:
    """This method is used to enable the dropout replacement for a given module.
    It will do it recursively for all the submodules.

    Args:
        m (nn.Module): The module to enable the dropout replacement for.
    """
    for module in m.modules():
        if isinstance(
            module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, ResidualLayer, LSTM)
        ):
            setattr(module, DISABLED_DROPOUT_KEY, False)


linear_conv_counter = 1


def linear_conv_counter_hook(
    module: nn.Module, input: torch.Tensor, output: torch.Tensor
) -> None:
    """This method is used to count the number of `nn.Linear`
    and `nn.Conv1d`, `nn.Conv2d`, or `nn.Conv3d` layers in a model.

    It is counting top-down where the counter is incremented as the model goes deeper.

    Args:
        module (nn.Module): The module to count the layers in.
        input (torch.Tensor): The input to the module.
        output (torch.Tensor): The output of the module.
    """
    global linear_conv_counter
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        module._counter = linear_conv_counter
        linear_conv_counter = linear_conv_counter + 1


lstm_counter = 1


def lstm_counter_hook(
    module: nn.Module, input: torch.Tensor, output: torch.Tensor
) -> None:
    """This method is used to count the number of `LSTM` layers in a model.

    It is counting top-down where the counter is incremented as the model goes deeper.

    Args:
        module (nn.Module): The module to count the layers in.
        input (torch.Tensor): The input to the module.
        output (torch.Tensor): The output of the module.
    """
    global lstm_counter
    if isinstance(module, LSTM):
        module._counter = lstm_counter
        lstm_counter = lstm_counter + 1


@torch.no_grad()
def count_lstm(method: LightningModule) -> int:
    """This method is used to count the number of `LSTM` layers in a model.

    It is counting top-down where the counter is incremented as the model goes deeper.

    It returns the maximum counter value.
    """
    global lstm_counter

    # We need to use the training mode because evaluation mode will turn on the Monte Carlo sampling
    method.train()
    hooks = []
    for m in method.model.modules():
        hooks.append(m.register_forward_hook(lstm_counter_hook))

    batch = get_sample_input_and_target(method)

    method.training_step(batch, batch_idx=0)
    for hook in hooks:
        hook.remove()

    # Copy the global counter value
    lstm_counter_copy = lstm_counter
    lstm_counter = 1

    return lstm_counter_copy


@torch.no_grad()
def count_linear_conv(method: LightningModule) -> int:
    """This method is used to count the number of `nn.Linear`, `nn.Conv1d`,
    `nn.Conv2d`, or `nn.Conv3d` layers in a model.

    It is counting top-down where the counter is incremented as the model goes deeper.

    It returns the maximum counter value.
    """
    global linear_conv_counter

    # We need to use the training mode because evaluation mode will turn on the Monte Carlo sampling
    method.train()
    hooks = []
    for m in method.model.modules():
        hooks.append(m.register_forward_hook(linear_conv_counter_hook))

    batch = get_sample_input_and_target(method)

    method.training_step(batch, batch_idx=0)
    for hook in hooks:
        hook.remove()

    # Copy the global counter value
    linear_conv_counter_copy = linear_conv_counter
    linear_conv_counter = 1

    return linear_conv_counter_copy


conv2d_counter = 1


def conv2d_counter_hook(
    module: nn.Module, input: torch.Tensor, output: torch.Tensor
) -> None:
    """This method is used to count the number of `nn.Conv2d` layers in a model.

    It is counting top-down where the counter is incremented as the model goes deeper.

    Args:
        module (nn.Module): The module to count the layers in.
        input (torch.Tensor): The input to the module.
        output (torch.Tensor): The output of the module.
    """
    global conv2d_counter
    if isinstance(module, nn.Conv2d):
        module._counter = conv2d_counter
        conv2d_counter = conv2d_counter + 1


@torch.no_grad()
def count_conv2d(method: LightningModule) -> int:
    """This method is used to count the number of `nn.Conv2d` layers in a model.

    It is counting top-down where the counter is incremented as the model goes deeper.

    It returns the maximum counter value.
    """
    global conv2d_counter

    # We need to use the training mode because evaluation mode will turn on the Monte Carlo sampling
    method.train()
    hooks = []
    for m in method.model.modules():
        hooks.append(m.register_forward_hook(conv2d_counter_hook))

    batch = get_sample_input_and_target(method)

    method.training_step(batch, batch_idx=0)
    for hook in hooks:
        hook.remove()

    # Copy the global counter value
    conv2d_counter_copy = conv2d_counter
    conv2d_counter = 1

    return conv2d_counter_copy


def replace_with_dropout(
    model: nn.Module,
    p: float,
    dropout_mapping: Dict[nn.Module, nn.Module],
    depth_start_end: Optional[Tuple[int, int]] = None,
    custom_indices: Optional[Tuple[int, ...]] = None,
) -> None:
    """This method is used to replace all the `nn.Linear`, `nn.Conv1d`, `nn.Conv2d`, or `nn.Conv3d` layers
       with a Sequential layer containing the original layer followed by a `Dropout` layer.

    Args:
        model (nn.Module): The model to replace the layers in.
        p (float): The probability in the `Dropout` layer.
        dropout_mapping (Dict[nn.Module, nn.Module]): The mapping from the original layer to the `Dropout` layer.
        depth_start_end (Optional[Tuple[int, int]]): The depth indices to start and end replacing the layers.
        The first index is the starting portion of the network where to start replacing the layers.
        The second index is the ending portion of the network where to end replacing the layers.
        For example, if the model has 10 `nn.Linear` layers and `depth_start_end=(2, 8)`, then
        the first 2 layers and the last 2 layers will not be replaced with `Dropout` layers.
        custom_indices (Optional[Tuple[int, ...]]): The indices of the layers to replace.
    """
    assert (
        depth_start_end is None or custom_indices is None
    ), "Either depth_start_end or custom_indices should be specified, but not both."
    for name, child in model.named_children():
        if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, LSTM)):
            if hasattr(child, DISABLED_DROPOUT_KEY) and getattr(
                child, DISABLED_DROPOUT_KEY
            ):
                continue
            if depth_start_end is not None:
                if (
                    hasattr(child, "_counter") and child._counter < depth_start_end[0]
                ) or (
                    hasattr(child, "_counter") and child._counter > depth_start_end[1]
                ):
                    continue
            if custom_indices is not None:
                if not hasattr(child, "_counter") or (
                    hasattr(child, "_counter") and child._counter not in custom_indices
                ):
                    continue

            if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                setattr(
                    model, name, nn.Sequential(dropout_mapping[type(child)](p=p), child)
                )
            elif isinstance(child, LSTM):
                setattr(
                    model,
                    name,
                    DropoutLSTM(
                        p=p,
                        input_size=child._input_size,
                        hidden_size=child._hidden_size,
                    ),
                )
        elif isinstance(child, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            setattr(model, name, dropout_mapping[type(child)](p=p))
        else:
            replace_with_dropout(
                child, p, dropout_mapping, depth_start_end, custom_indices
            )


def replace_with_dropconnect(
    model: nn.Module,
    p: float,
    depth_start_end: Optional[Tuple[int, int]] = None,
) -> None:
    """This method is used to replace all the `nn.Linear`, `nn.Conv1d`, `nn.Conv2d`, or `nn.Conv3d` layers
       with a `DropConnect` layer.

    Args:
        model (nn.Module): The model to replace the layers in.
        p (float): The probability in the `DropConnect` layer.
        dropout_mapping (Optional[Dict[nn.Module, nn.Module]]): The mapping from the original layer to the `DropConnect` layer.
        depth_start_end (Optional[Tuple[int, int]]): The depth indices to start and end replacing the layers.
        The first index is the starting portion of the network where to start replacing the layers.
        The second index is the ending portion of the network where to end replacing the layers.
        For example, if the model has 10 `nn.Linear` layers and `depth_start_end=(2, 8)`, then
        the first 2 layers and the last 2 layers will not be replaced with `DropConnect` layers.
    """
    for name, child in model.named_children():
        if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if hasattr(child, DISABLED_DROPOUT_KEY) and getattr(
                child, DISABLED_DROPOUT_KEY
            ):
                continue
            if depth_start_end is not None:
                if (
                    hasattr(child, "_counter") and child._counter < depth_start_end[0]
                ) or (
                    hasattr(child, "_counter") and child._counter > depth_start_end[1]
                ):
                    continue
        if isinstance(child, nn.Linear):
            setattr(
                model,
                name,
                LinearDropConnect(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    p=p,
                    bias=child.bias,
                    weight=child.weight,
                ),
            )
        elif isinstance(child, nn.Conv1d):
            setattr(
                model,
                name,
                Conv1dDropConnect(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias,
                    weight=child.weight,
                    p=p,
                ),
            )
        elif isinstance(child, nn.Conv2d):
            setattr(
                model,
                name,
                Conv2dDropConnect(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias,
                    weight=child.weight,
                    p=p,
                ),
            )
        elif isinstance(child, nn.Conv3d):
            setattr(
                model,
                name,
                Conv3dDropConnect(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias,
                    weight=child.weight,
                    p=p,
                ),
            )

        elif isinstance(child, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            # Disable any dropout layers that were in the original model
            setattr(model, name, nn.Identity())
        else:
            replace_with_dropconnect(child, p, depth_start_end)


def replace_with_standout(
    module: nn.Module,
    alpha: float = 0.5,
    beta: float = 0.5,
    depth_start_end: Optional[Tuple[int, int]] = None,
) -> None:
    """This method is used to replace all `nn.Linear`, `nn.Conv1d`, `nn.Conv2d`, or `nn.Conv3d` layers in a module with `StandOut` layers.

    Args:
        module (nn.Module): The module to replace the layers in.
        alpha (float): The scaling parameter.
        beta (float): The bias parameter.
        depth_start_end (Optional[Tuple[int, int]]): The depth indices to start and end replacing the layers.
        The first index is the starting portion of the network where to start replacing the layers.
        The second index is the ending portion of the network where to end replacing the layers.
        For example, if the model has 10 `nn.Linear` layers and `depth_start_end=(2, 8)`, then
        the first 2 layers and the last 2 layers will not be replaced with `StandOut` layers.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if hasattr(child, DISABLED_DROPOUT_KEY) and getattr(
                child, DISABLED_DROPOUT_KEY
            ):
                continue
            if depth_start_end is not None:
                if (
                    hasattr(child, "_counter") and child._counter < depth_start_end[0]
                ) or (
                    hasattr(child, "_counter") and child._counter > depth_start_end[1]
                ):
                    continue
            setattr(module, name, nn.Sequential(child, StandOut(alpha, beta)))
        elif isinstance(child, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            setattr(module, name, nn.Identity())
        else:
            replace_with_standout(child, alpha, beta, depth_start_end)


def replace_with_dropblock(
    module: nn.Module,
    block_size_percentage: float = 0.2,
    gamma: float = 0.2,
    depth_start_end: Optional[Tuple[int, int]] = None,
) -> None:
    """This method is used to replace all `nn.Conv2d` layers in a module with `DropBlock` layers.

    Args:
        module (nn.Module): The module to replace the layers in.
        block_size_percentage (float): Percentage of the input size to use as the block size.
        gamma (float): Probability of dropping activations.
        depth_start_end (Optional[Tuple[int, int]]): The depth indices to start and end replacing the layers.
        The first index is the starting portion of the network where to start replacing the layers.
        The second index is the ending portion of the network where to end replacing the layers.
        For example, if the model has 10 `nn.Conv2d` layers and `depth_start_end=(2, 8)`, then
        the first 2 layers and the last 2 layers will not be replaced with `DropBlock` layers.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            if hasattr(child, DISABLED_DROPOUT_KEY) and getattr(
                child, DISABLED_DROPOUT_KEY
            ):
                continue
            if depth_start_end is not None:
                if (
                    hasattr(child, "_counter") and child._counter < depth_start_end[0]
                ) or (
                    hasattr(child, "_counter") and child._counter > depth_start_end[1]
                ):
                    continue
            setattr(
                module,
                name,
                nn.Sequential(
                    child,
                    DropBlock(
                        block_size_percentage=block_size_percentage,
                        gamma=gamma,
                        module=child,
                    ),
                ),
            )
        elif isinstance(child, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            setattr(module, name, nn.Identity())
        else:
            replace_with_dropblock(child, block_size_percentage, gamma, depth_start_end)


residual_counter = 1


def residual_counter_hook(
    module: nn.Module, input: torch.Tensor, output: torch.Tensor
) -> None:
    """This method is used to count the number of `ResidualLayer` layers in a model.

    It is counting top-down where the counter is incremented as the model goes deeper.

    Args:
        module (nn.Module): The module to count the layers in.
        input (torch.Tensor): The input to the module.
        output (torch.Tensor): The output of the module.
    """
    global residual_counter
    if isinstance(module, ResidualLayer):
        module._counter = residual_counter
        residual_counter = residual_counter + 1


@torch.no_grad()
def count_residual(method: LightningModule) -> int:
    """This method is used to count the number of `ResidualLayer` layers in a model.

    It is counting top-down where the counter is incremented as the model goes deeper.

    It returns the maximum counter value.
    """
    global residual_counter

    # We need to use the training mode because evaluation mode will turn on the Monte Carlo sampling
    method.train()
    hooks = []
    for m in method.model.modules():
        hooks.append(m.register_forward_hook(residual_counter_hook))

    batch = get_sample_input_and_target(method)

    method.training_step(batch, batch_idx=0)
    for hook in hooks:
        hook.remove()

    # Copy the global counter value ande delete it
    residual_counter_copy = residual_counter
    residual_counter = 1

    return residual_counter_copy


def replace_with_stochastic_depth(module: nn.Module, L: int, p_L: float) -> None:
    """This method is used to replace add `StochasticDepth` to `ResidualLayer`.

    It adds `StochasticDepth` to `ResidualLayer`'s attribute `_layer` through
    `nn.Sequential` wrapper.

    Args:
        module (nn.Module): The module to replace the layers in.
        L (int): The number of ResidualLayer's in the model.
        p_L (float): Probability of dropping a residual branch.
    """
    for child in module.children():
        if isinstance(child, ResidualLayer):
            l = (
                child._counter
            )  # This is the ordered index of the ResidualLayer, depth-wise
            p_l = ((l + 1) / L) * p_L
            child._layer = nn.Sequential(child._layer, StochasticDepth(p=p_l))
        else:
            replace_with_stochastic_depth(child, L, p_L)
