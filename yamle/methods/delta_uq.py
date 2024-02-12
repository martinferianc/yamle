from typing import Any, Dict, List
import torch
import argparse

from yamle.methods.uncertain_method import MCSamplingMethod


class DeltaUQMethod(MCSamplingMethod):
    """This class is the extension of the base method for delta-UQ method.

    The core of the method is in applying anchors during training and inference,
    the anchors are the samples themselves where the input of the network is:
    `[x-anchor, anchor]` where anchor is reshufled `x`.

    Note that, this method requires a special treatment in case of data augmentation.
    The anchors during test should be from a training set and not from the test set.
    Hence we will cache some of the training samples.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._replace_input_layer()
        self._anchors_cache: torch.Tensor = None
        self._anchors_cache_max_size = 1000

    def _replace_input_layer(self) -> None:
        """Replace the first layer with one where the input dimension is multiplied exactly by 2."""
        if isinstance(self.model._input, torch.nn.Linear):
            self.model._input = torch.nn.Linear(
                in_features=self.model._input.in_features * 2,
                out_features=self.model._input.out_features,
                bias=self.model._input.bias is not None,
            )
        elif isinstance(self.model._input, torch.nn.Conv2d):
            self.model._input = torch.nn.Conv2d(
                in_channels=self.model._input.in_channels * 2,
                out_channels=self.model._input.out_channels,
                kernel_size=self.model._input.kernel_size,
                stride=self.model._input.stride,
                padding=self.model._input.padding,
                dilation=self.model._input.dilation,
                groups=self.model._input.groups,
                bias=self.model._input.bias is not None,
            )
        else:
            raise ValueError(
                "The first layer of the model should be either a `torch.nn.Linear` or a "
                "`torch.nn.Conv2d`."
            )

    def state_dict(self) -> Dict[str, Any]:
        """This method is used to get the state of the method."""
        state_dict = super().state_dict()
        state_dict["anchors_cache"] = self._anchors_cache.cpu()
        state_dict["anchors_cache_max_size"] = self._anchors_cache_max_size
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """This method is used to load the state of the method."""
        super().load_state_dict(state_dict)
        self._anchors_cache = state_dict["anchors_cache"]
        self._anchors_cache_max_size = state_dict["anchors_cache_max_size"]

    def _predict(self, x: torch.Tensor, **forward_kwargs: Any) -> torch.Tensor:
        """This method is used to perform a forward pass of the model.

        It is done with respect to the number of samples specified in the constructor.
        It applies the reshuffling of the anchors and the subtraction of the anchors
        from the inputs.
        """
        outputs = []
        num_members = self.training_num_members if self.training else self._num_members
        for _ in range(num_members):
            anchors = (
                x[torch.randperm(x.size(0))]
                if self._anchors_cache is None or self.training
                else self._sample_anchors_from_cache(x.size(0)).to(x.device)
            )
            new_x = torch.cat([x - anchors, anchors], dim=1)
            outputs.append(
                super(MCSamplingMethod, self)._predict(new_x, **forward_kwargs)
            )
        return torch.cat(outputs, dim=1)

    def _sample_anchors_from_cache(self, N: int) -> torch.Tensor:
        """A helper function to sample from the cache, potentially with repeition."""
        if self._anchors_cache is None:
            raise ValueError(
                "The anchors cache is not initialized. Please run the training first."
            )
        return self._anchors_cache[
            torch.randint(low=0, high=len(self._anchors_cache), size=(N,))
        ]

    def _training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """This method is used to perform a training step.

        In this instance we will randomly cache some of the training samples.
        If the cache exists, in each training step randomly replace some indices.
        """
        x, _ = batch
        if self._anchors_cache is None:
            self._anchors_cache = x.clone().detach()
        elif (
            len(self._anchors_cache) < self._anchors_cache_max_size
            and self.current_epoch == 0
        ):
            self._anchors_cache = torch.cat([self._anchors_cache, x.clone().detach()])
        else:
            self._anchors_cache[
                torch.randperm(self._anchors_cache.size(0))[: x.size(0)]
            ] = x.clone().detach()
        return super()._training_step(batch, batch_idx)

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """This method is used to add the specific arguments for the class."""
        parser = super(DeltaUQMethod, DeltaUQMethod).add_specific_args(parent_parser)
        return parser
