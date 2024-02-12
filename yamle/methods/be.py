from typing import List, Dict, Any
import torch
import torch.nn as nn

from yamle.methods.uncertain_method import MemberMethod
from yamle.models.specific.be import LinearBE, Conv2dBE
from yamle.utils.operation_utils import average_predictions, repeat_inputs
from yamle.defaults import LOSS_KEY, TARGET_KEY, PREDICTION_KEY, MEAN_PREDICTION_KEY, TARGET_PER_MEMBER_KEY, PREDICTION_PER_MEMBER_KEY, AVERAGE_WEIGHTS_KEY


def replace_with_be(model: nn.Module, num_members: int) -> None:
    """This method is used to replace all the `nn.Linear`, `nn.Conv2d` layers
       with a `LinearBE`, `Conv2dBE` respectively.

    Args:
        model (nn.Module): The model to replace the layers in.
        num_members (int): The number of members in the ensemble.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(
                model,
                name,
                LinearBE(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias,
                    num_members=num_members,
                    weight=child.weight,
                ),
            )
        elif isinstance(child, nn.Conv2d):
            setattr(
                model,
                name,
                Conv2dBE(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    num_members=num_members,
                    weight=child.weight,
                    bias=child.bias,
                ),
            )
        else:
            replace_with_be(child, num_members)


class BEMethod(MemberMethod):
    """This class is the extension of the base method for BatchEnsemble models.

    The difference is in having to change the prediction to concatenate the `num_members` dimension.
    into the batch dimension during validation and testing.
    
    Note that only Linear and Conv2d layers are supported, not the batch norm layers.
    In practice this is not a problem https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/normalization.py#L111
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        replace_with_be(self.model, self._num_members)
        
    def _predict(
        self, x: torch.Tensor, unsqueeze: bool = True, **forward_kwargs: Any
    ) -> torch.Tensor:
        """This method is used to perform a forward pass of the model.
        
        If the model is in evaluation mode it replicates the inputs `num_members` times and
        concatenates them into the batch dimension.

        Args:
            x (torch.Tensor): The input to the model.
            **forward_kwargs (Any): The keyword arguments to be passed to the forward pass of the model.
        """
        if self.evaluation:
            x = repeat_inputs(x, self._num_members)
            
        output = self.model(x, **forward_kwargs)
        
        if self.evaluation:
            output = output.reshape(-1, self._num_members, *output.shape[1:])
        elif unsqueeze:
            output = output.unsqueeze(1)
        return output

    def _validation_test_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """This method is used to perform a single validation/test step.

        It assumes that the batch has a shape `(batch_size, num_features)`.
        It assumes that the output of the model has a shape `(batch_size, n_samples, num_classes)`.
        """
        x, y = batch
        y_hat_permember = self._predict(x)
        # Repeat the labels num_members times
        y_permember = torch.stack([y] * self._num_members, dim=1)
        loss = self._loss_per_member(y_hat_permember, y_permember)
        y_hat = average_predictions(y_hat_permember, self._task)
        output = {
            LOSS_KEY: loss,
            TARGET_KEY: y.detach(),
            PREDICTION_KEY: y_hat_permember.detach(),
            MEAN_PREDICTION_KEY: y_hat.detach(),
            TARGET_PER_MEMBER_KEY: y_permember.detach(),
            PREDICTION_PER_MEMBER_KEY: y_hat_permember.detach(),
            AVERAGE_WEIGHTS_KEY: None,
        }
        return output

    def _validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """This method is used to perform a single validation step."""
        return self._validation_test_step(batch, batch_idx)

    def _test_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """This method is used to perform a single test step."""
        return self._validation_test_step(batch, batch_idx)
