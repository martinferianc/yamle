from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from yamle.defaults import TINY_EPSILON


class MixMoBlock(nn.Module):
    """This is a module which applies the MixMo regularisation to a model.

    As proposed in: "MixMo: Mixing Multiple Inputs for Multiple Outputs via Deep Subnetworks"

    Show that binary mixing in features - particularly with rectangular patches from CutMix -
    enhances results by making subnetworks stronger and more diverse.

    Args:
        num_members (int): The number of members in the ensemble.
        input (nn.Module): The input module that needs to be wrapped.
    """

    def __init__(self, num_members: int, input: nn.Module) -> None:
        super().__init__()
        assert (
            num_members > 0
        ), "The number of members in the ensemble must be positive."
        self._num_members = num_members
        self._input = input
        self.mask: torch.Tensor = None

    def forward(
        self, x: torch.Tensor, K: Optional[torch.Tensor] = None, p: float = 0.5
    ) -> torch.Tensor:
        """This method is used to perform a forward pass of the model.

        In validation it is done with respect to all models that have been trained.

        The input tensor `x` is assumed to havea shape `(batch_size, num_members, num_channels, height, width)`.
        The mixing coefficients `K` are assumed to have a shape `(batch_size, num_members)`.
        The outputs are assumed to have a shape `(batch_size, num_channels, height, width)`.

        Args:
            x (torch.Tensor): The input tensor.
            K (torch.Tensor): The mixing coefficients.
            p (float): The probability of binary mixing.
        """
        features = self._input(x)
        if self.training:
            # Whether we apply binary or linear mixing
            l_binary = torch.bernoulli(torch.ones([1]) * p).item()

            if l_binary:
                return self._cutmix_mixing(features, K)
            else:
                return self._linear_mixing(features, K)

        else:
            return (
                self._linear_mixing(
                    features, torch.ones(features.shape[:2]).to(features.device)
                )
                / self._num_members
            )

    def _linear_mixing(self, features: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """This method is used to perform linear mixing."""
        while K.dim() < features.dim():
            K = K.unsqueeze(-1)
        return torch.sum(features * K, dim=1) * self._num_members

    def _cutmix_mixing(self, features: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """This method is used to perform cutmix mixing."""
        self.reset_mask()
        B, _, C, H, W = features.shape
        device = features.device
        k = torch.randint(0, self._num_members, (B,), device=device)
        # Select the kth member from K
        mask = self._cutmix_mask((B, C, H, W), K[torch.arange(B, device=device), k])
        # Whether the first input is inside or outside the rectangle
        if torch.bernoulli(torch.ones([1]) * 0.5).item():
            mask = 1.0 - mask
            # Change K in place such that the weights are recomputed
            K *= -1.0
            K += 1.0
        K_k = K[torch.arange(B, device=device), k]
        K_without_k_mask = torch.ones(K.shape, device=device, dtype=torch.bool)
        K_without_k_mask[torch.arange(B, device=device), k] = False
        K_without_k = K[K_without_k_mask].reshape(B, self._num_members - 1)

        features_k = features[torch.arange(B, device=device), k]
        features_without_k = features[K_without_k_mask].reshape(
            B, self._num_members - 1, C, H, W
        )
        # Unsqueeze K_k and K_without_k to make them broadcastable
        while K_k.dim() < features_without_k.dim():
            K_k = K_k.unsqueeze(-1)
        while K_without_k.dim() < features_without_k.dim():
            K_without_k = K_without_k.unsqueeze(-1)
        new_features = features_k * mask
        new_features += (1.0 - mask) * torch.sum(
            (features_without_k * K_without_k) / (1.0 - K_k + TINY_EPSILON), dim=1
        )
        new_features *= self._num_members
        # Cache the mask to be optionally used for unmixing
        self.mask = mask
        return new_features

    def _cutmix_mask(
        self, x_shape: Tuple[int, int, int, int], K: torch.Tensor
    ) -> torch.Tensor:
        """This method is used to compute the cutmix mask."""
        B, C, H, W = x_shape
        cut_ratio = torch.sqrt(K)
        cut_width = (W * cut_ratio).long()
        cut_height = (H * cut_ratio).long()

        cut_x = torch.randint(0, W, (B,)).to(K.device)
        cut_y = torch.randint(0, H, (B,)).to(K.device)

        # Box corners naturally follow
        bbx1 = torch.clamp(cut_x - cut_width // 2, 0, W)
        bby1 = torch.clamp(cut_y - cut_height // 2, 0, H)
        bbx2 = torch.clamp(cut_x + cut_width // 2, 0, W)
        bby2 = torch.clamp(cut_y + cut_height // 2, 0, H)

        y_mask = torch.logical_or(
            torch.arange(H, device=K.device).unsqueeze(0) < bby1[:, None],
            torch.arange(H, device=K.device).unsqueeze(0) >= bby2[:, None],
        )
        x_mask = torch.logical_or(
            torch.arange(W, device=K.device).unsqueeze(0) < bbx1[:, None],
            torch.arange(W, device=K.device).unsqueeze(0) >= bbx2[:, None],
        )
        ones_mask = ~torch.logical_or(
            y_mask.unsqueeze(2), x_mask.unsqueeze(1)
        ).unsqueeze(1)
        mask = torch.zeros(x_shape, device=K.device)
        mask += ones_mask

        return mask

    def get_mask(self) -> torch.Tensor:
        """This method is used to get the mask that was cached during the last forward pass."""
        return self.mask

    def reset_mask(self) -> None:
        """This method is used to reset the mask that was cached during the last forward pass."""
        self.mask = None


class UnmixingBlock(nn.Module):
    """This class implements the unmixing block.

    It consists of SelectiveAdaptiveAvgPool2d that outputs two outputs, one for each version
    of the provided mask.
    This module is connected to the MixMo block from which it fetches the mask.
    There is also the `m` weight for creating the adapted mask for unmixing as in `Fadeout unmixing`.

    Args:
        mixmo_block (MixMoBlock): The MixMo block from which the mask is fetched.
        in_features (int): The number of input features for the linear layer.
        out_features (int): The number of output features for the linear layer.
        num_members (int): The number of members in the output.
        outputs_dim (int): The output dimension of the unmixing block.
        bias (bool): Whether to use a bias in the linear layer. Defaults to True.
    """

    def __init__(
        self,
        mixmo_block: MixMoBlock,
        in_features: int,
        out_features: int,
        num_members: int,
        outputs_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self._mixmo_block = mixmo_block
        assert num_members == 2, "Unmixing block is only implemented for 2 members"
        self._outputs_dim = outputs_dim
        self._output = nn.ModuleList(
            [
                nn.Linear(in_features, out_features, bias=bias)
                for _ in range(num_members)
            ]
        )
        self._selective_pool = SelectiveAdaptiveAvgPool2d(1)
        self._flatten = nn.Flatten()
        self.set_m(1.0)

    def set_m(self, m: float) -> None:
        """This method is used to set the `m` weight for creating the adapted mask for unmixing as in `Fadeout unmixing`."""
        assert 0.0 <= m <= 1.0, f"m must be in [0, 1], got {m}"
        self._m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        # If the input is not a 4D tensor, this cannot be used
        assert (
            len(input_shape) == 4
        ), f"Input must be a 4D tensor, got {len(input_shape)}D tensor."
        if not self.training:
            mask = torch.ones_like(x)
            x = self._selective_pool(x, [mask, mask])
        else:
            # Get the input shape of `x`
            # If the mask shape is different from the input shape, pooling has been previously performed
            # The mask is then reshaped to the input shape using torchvisision's resize function
            input_shape = x.shape
            mask = self._mixmo_block.get_mask()
            # Select any channel of the mask and repeat it to match the number of channels of the input
            mask = mask[:, 0:1, :, :].repeat(1, input_shape[1], 1, 1)
            mask_shape = mask.shape
            if mask_shape != input_shape:
                mask = TF.resize(
                    mask, input_shape[2:], interpolation=TF.InterpolationMode.NEAREST
                )

            # Create the adapted mask for unmixing
            ones_mask = mask
            zeros_mask = 1.0 - mask
            ones_mask = ones_mask + self._m * (1.0 - ones_mask)
            zeros_mask = zeros_mask + self._m * (1.0 - zeros_mask)

            # Perform the selective adaptive average pooling
            x = self._selective_pool(x, [ones_mask, zeros_mask])
        x = [self._flatten(x_) for x_ in x]
        x = [self._output[i](x[i]) for i in range(len(x))]
        x = torch.stack(x, dim=self._outputs_dim)
        self._mixmo_block.reset_mask()
        return x


class SelectiveAdaptiveAvgPool2d(nn.Module):
    """This class implements the selective adaptive average pooling layer.

    Given a list of masks, it performs adaptive average pooling on the input tensor
    for each mask and returns the output for each mask.

    Args:
        kernel_size (int): The kernel size.
    """

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self._average_pool = nn.AdaptiveAvgPool2d((kernel_size, kernel_size))

    def forward(self, x: torch.Tensor, masks: List[torch.Tensor]) -> List[torch.Tensor]:
        """This method is used to perform the forward pass.

        Args:
            x (torch.Tensor): The input tensor.
        """
        return [self._average_pool(x * mask) for mask in masks]
