from typing import Tuple, Any, Dict
import torch.nn as nn
import torch
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange
from yamle.models.operations import Add, OutputActivation
from yamle.models.visual_transformer import SpatialPositionalEmbedding
import matplotlib.pyplot as plt


class MixTokenEmbedding(nn.Module):
    """This class implements the MixToken augmentation followed by adding the spatial
    positional encoding.

    During training, at first, the images are converted to patches and then the patches are mixed.
    After mixing, the patches are encoded with a projection and a spatial positional encoding is added.
    During inference, the images are converted to patches and the patches are not mixed.
    The patches are encoded with a projection and a spatial positional encoding is added.
    Note that during inference a single image is selected from the `num_members=2` dimension.

    The input shape is `(batch_size, 2, channels, height, width)`.

    Args:
        inputs_dim (int): The dimension of the input.
        patch_size (int): The size of the patches.
        embedding_dim (int): The dimension of the embedding.
        dropout (float): The dropout rate.
    """

    def __init__(
        self,
        inputs_dim: Tuple[int, int, int],
        patch_size: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        C, H, W = inputs_dim
        self._inputs_dim = inputs_dim
        self._embedding_dim = embedding_dim
        assert (
            H % patch_size == 0 and W % patch_size == 0
        ), f"Image size ({H}x{W}) should be divisible by patch size ({patch_size}x{patch_size})."
        self._num_patches = (H // patch_size) * (W // patch_size)
        self._num_rows = H // patch_size
        self._num_cols = W // patch_size
        self._patch_size = patch_size
        patch_dim = C * patch_size * patch_size

        self._to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self._positional_encoding = nn.Parameter(
            torch.randn(1, self._num_patches, embedding_dim), requires_grad=True
        )
        self._dropout = nn.Dropout(dropout)
        self._add = Add()

        # We create a temporary variable to store the mask such that it can be retrieved
        # for the source attribution
        self._mask: torch.Tensor = None

    def _rand_bbox(self, lam: float) -> Tuple[int, int, int, int]:
        """A helper function to generate the random bounding box."""
        W, H = self._inputs_dim[1:]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Quantize the bounding box relative to the patch size.
        bbx1 = int(bbx1 // self._patch_size) * self._patch_size
        bby1 = int(bby1 // self._patch_size) * self._patch_size
        # The coordinates plus the patch size guarantees that
        # sometimes the corners of the image are included.
        bbx2 = int((bbx2 + self._patch_size) // self._patch_size) * self._patch_size
        bby2 = int((bby2 + self._patch_size) // self._patch_size) * self._patch_size

        return bbx1, bby1, bbx2, bby2

    def _update_mask(self, bbx1: int, bby1: int, bbx2: int, bby2: int) -> None:
        """A helper function to update the mask."""
        mask = torch.zeros(
            self._num_rows, self._num_cols, device=next(self.parameters()).device
        )
        # Convert the bounding box coordinates to the patch indices.
        bbx1 = int(bbx1 // self._patch_size)
        bby1 = int(bby1 // self._patch_size)
        bbx2 = int(bbx2 // self._patch_size)
        bby2 = int(bby2 // self._patch_size)
        mask[bbx1:bbx2, bby1:bby2] = 1
        # Use the rearrange function to convert the mask to a vector to make sure that
        # the mask has the same reshaping pattern as the patches
        self._mask = rearrange(mask, "h w -> (h w)")

    def reset_mask(self) -> None:
        """A helper function to reset the mask."""
        self._mask = None

    def _mix_token(self, x: torch.Tensor) -> torch.Tensor:
        """A helper function to mix the tokens."""
        x1 = x[:, 0, :, :, :]
        x2 = x[:, 1, :, :, :]
        lam = np.random.beta(1.0, 1.0)
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(lam)
        x1[:, :, bbx1:bbx2, bby1:bby2] = x2[:, :, bbx1:bbx2, bby1:bby2]
        self._update_mask(bbx1, bby1, bbx2, bby2)
        return x1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        During training, the images are mixed and converted to patches.
        During inference, the images are converted to patches and the patches are not mixed.
        """
        if self.training:
            x = self._mix_token(x)
            if self._debug:
                self._plot_sanity_check(x)
        else:
            x = x[:, 0, :, :, :]
        x = self._to_patch_embedding(x)
        x = self._add(x, self._positional_encoding)
        return self._dropout(x)

    def _plot_sanity_check(self, x: torch.Tensor) -> None:
        """A helper function to plot the sanity check."""
        if not hasattr(self, "_debug_counter"):
            self._debug_counter = 0
        # We are going to perform a sanity check to plot the input to make sure that it is correctly organised
        # In the first row we plot the mask in the second row we plot the image.
        fig, axs = plt.subplots(2, 10, figsize=(20, 6))
        for i in range(10):
            axs[0, i].imshow(x[i].permute(1, 2, 0).cpu())
            # Plot horizontal and vertical lines to visualize the patches.
            for j in range(0, self._inputs_dim[1], self._patch_size):
                axs[0, i].axvline(j, color="r")
            for j in range(0, self._inputs_dim[2], self._patch_size):
                axs[0, i].axhline(j, color="r")
            axs[1, i].imshow(self._mask.reshape(self._num_rows, self._num_cols).cpu())
        plt.savefig(f"{self._save_path}/{self._debug_counter}.png")
        self._debug_counter += 1


class Attribution(nn.Module):
    """This class implements the attribution module.

    It adds the source attribution to the diffrent patches depending on a mask which has a shape of
    `(num_patches,)`. Additionally a class token is added to the beginning of the sequence.

    Args:
        embedding_dim (int): The dimension of the embedding.
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._source_attribution = nn.Parameter(
            torch.randn(1, 2, embedding_dim), requires_grad=True
        )
        self._class_token = nn.Parameter(
            torch.randn(1, 1, embedding_dim), requires_grad=True
        )
        self._add = Add()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attribution = self._source_attribution.repeat(x.shape[0], 1, 1)
        mask = mask.reshape(1, -1, 1)
        attribution1 = attribution[:, 0, :].unsqueeze(1)
        attribution2 = attribution[:, 1, :].unsqueeze(1)
        # Check if the mask is 0 or 1.
        assert torch.all(
            (mask == 0) | (mask == 1)
        ), f"The mask must be 0 or 1. Got {mask}."
        x = self._add(mask * attribution1, x)
        x = self._add((1 - mask) * attribution2, x)
        class_token = self._class_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([class_token, x], dim=1)
        return x


class MixVitWrapper(nn.Module):
    """This class implements the MixVit model.

    Args:
        vit (nn.Module): The Vision Transformer model.
        depth (int): The depth at which to add the Attribution module.
    """

    def __init__(self, vit: nn.Module, depth: int) -> None:
        super().__init__()
        self._vit = vit
        self._depth = depth
        assert isinstance(
            self._vit._input, SpatialPositionalEmbedding
        ), f"The input must be a SpatialPositionalEmbedding. Got {type(self._vit._input)} but expected SpatialPositionalEmbedding."
        self._vit._input = MixTokenEmbedding(
            patch_size=vit._input._patch_size,
            inputs_dim=vit._input._inputs_dim,
            embedding_dim=vit._input._embedding_dim,
            dropout=vit._input._dropout,
        )
        self._vit._pooling = "cls"
        self._vit._layers[-1] = nn.Identity()

        self._vit._output_activation = OutputActivation(
            self._vit._task, self._vit._output_activation._dim + 1
        )
        self._vit._output = nn.Linear(
            self._vit._output.in_features, self._vit._output.out_features * 2
        )

        self._attribution = Attribution(vit._input._embedding_dim)

        assert depth < len(
            vit._layers
        ), f"The depth must be smaller than the number of layers. Got {depth} but expected a value smaller than {len(vit._layers)}."

    def add_method_specific_layers(self, method: str, **kwargs: Any) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def forward(
        self,
        x: torch.Tensor,
        staged_output: bool = False,
        input_kwargs: Dict[str, Any] = {},
        output_kwargs: Dict[str, Any] = {},
    ) -> torch.Tensor:
        B = x.shape[0]
        x = self._vit._input(x)
        mask = self._vit._input._mask
        cache = None
        if not self.training:
            mask = torch.ones(
                (1, self._vit._input._num_patches, 1),
                device=x.device,
                requires_grad=False,
            )
        for i, layer in enumerate(self._vit._layers):
            x = layer(x)
            if i == self._depth:
                cache = x
                x = self._attribution(x, mask)
        # Select the first token
        output = self._vit._output(x[:, 0])
        output = output.reshape(B, 2, -1)
        output = self._vit._output_activation(output)
        if not self.training:
            output1 = output[:, 0, :]
            x = cache
            mask = torch.zeros_like(mask, requires_grad=False)
            x = self._attribution(x, mask)
            for i, layer in enumerate(self._vit._layers[self._depth + 1 :]):
                x = layer(x)
            output2 = self._vit._output(x[:, 0])
            output2 = output2.reshape(B, 2, -1)
            output2 = self._vit._output_activation(output2)
            output2 = output2[:, 1, :]
            output = torch.stack([output1, output2], dim=1)
        else:
            self._vit._input.reset_mask()
        return output
