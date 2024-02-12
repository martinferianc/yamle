from abc import ABC, abstractmethod

from typing import Tuple, Union
import torch.nn as nn
import torch
import torch.nn.init as init
import math

from yamle.utils.regularizer_utils import disable_regularizer
from yamle.utils.optimization_utils import freeze_weights
from yamle.defaults import TINY_EPSILON


class VariationalWeight(nn.Module, ABC):
    """This class defines a wrapper for variational weights defined under some distribution.

    Args:
        shape (Tuple[int, ...]): The shape of the variational weight.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
    ) -> None:
        super().__init__()
        self._shape = shape

    @abstractmethod
    def kl_divergence(self) -> torch.Tensor:
        """This method is used to compute the KL divergence between the variational weight and its prior."""
        pass


class GaussianMeanField(VariationalWeight):
    """This class defines a wrapper for variational weights defined under a mean-field Gaussian distribution.

    Args:
        mean (Union[float, torch.Tensor]): The mean of the distribution. Defaults to 0.0.
        log_variance (float): The variance of the distribution. Defaults to -3.0.
        prior_mean (float): The mean of the prior distribution. Defaults to 0.0.
        prior_log_variance (float): The variance of the prior distribution. Defaults to -3.0.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        mean: Union[float, torch.Tensor] = 0.0,
        log_variance: float = -3.0,
        prior_mean: float = 0.0,
        prior_log_variance: float = -3.0,
    ) -> None:
        super().__init__(shape)
        self._mean = (
            nn.Parameter(torch.ones(shape) * mean)
            if isinstance(mean, float)
            else nn.Parameter(mean.detach(), requires_grad=True)
        )
        # Make all the weights slightly different from each other.
        self._mean.data += torch.randn_like(self._mean) * 0.01
        self._initial_log_variance = log_variance
        self._log_variance = nn.Parameter(
            torch.ones(shape) * log_variance, requires_grad=True
        )
        # Make all the weights slightly different from each other.
        self._log_variance.data += torch.randn_like(self._log_variance) * 0.01
        disable_regularizer([self._mean, self._log_variance])

        self._prior_mean = nn.Parameter(
            torch.ones(shape) * prior_mean, requires_grad=False
        )
        self._prior_log_variance = nn.Parameter(
            torch.ones(shape) * prior_log_variance, requires_grad=False
        )

        self._prior_mean_scalar = prior_mean
        self._prior_log_variance_scalar = prior_log_variance

    @property
    def mean(self) -> torch.Tensor:
        """This method returns the mean of the distribution."""
        return self._mean

    @property
    def weight(self) -> torch.Tensor:
        """This method returns the mean of the distribution."""
        return self._mean

    @property
    def prior_mean(self) -> torch.Tensor:
        """This method returns the mean of the prior distribution."""
        return self._prior_mean

    @property
    def variance(self) -> torch.Tensor:
        """This method returns the variance of the distribution."""
        return torch.clamp(torch.exp(self._log_variance), min=TINY_EPSILON)

    @property
    def std(self) -> torch.Tensor:
        """This method returns the standard deviation of the distribution."""
        return torch.sqrt(self.variance + TINY_EPSILON)

    @property
    def prior_variance(self) -> torch.Tensor:
        """This method returns the variance of the prior distribution."""
        return torch.clamp(torch.exp(self._prior_log_variance), min=TINY_EPSILON)

    @property
    def prior_std(self) -> torch.Tensor:
        """This method returns the standard deviation of the prior distribution."""
        return torch.sqrt(self.prior_variance + TINY_EPSILON)

    @property
    def log_variance(self) -> torch.Tensor:
        """This method returns the log variance of the distribution."""
        return torch.clamp(self._log_variance, min=-10, max=10)

    @property
    def log_std(self) -> torch.Tensor:
        """This method returns the log standard deviation of the distribution."""
        return 0.5 * self.log_variance

    @property
    def prior_log_variance(self) -> torch.Tensor:
        """This method returns the log variance of the prior distribution."""
        return torch.clamp(self._prior_log_variance, min=-10, max=10)

    @property
    def prior_log_std(self) -> torch.Tensor:
        """This method returns the log standard deviation of the prior distribution."""
        return 0.5 * self.prior_log_variance

    @property
    def shape(self) -> Tuple[int, ...]:
        """This method returns the shape of the variational weight."""
        return self._shape

    def sample(self) -> torch.Tensor:
        """This method samples from the distribution."""
        eta = torch.randn_like(self._mean).to(self._mean.device)
        eta = eta / torch.frobenius_norm(eta).to(self._mean.device)
        r = torch.randn(1).to(self._mean.device)
        return self._mean + torch.sqrt(self.variance + TINY_EPSILON) * eta * r

    def kl_divergence(self) -> torch.Tensor:
        """This method returns the KL divergence between the variational distribution and the prior."""
        return torch.sum(
            self.prior_log_std
            - self.log_std
            + (self.variance + (self.mean - self.prior_mean) ** 2)
            / (2 * self.prior_variance + TINY_EPSILON)
            - 0.5
        )

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f"prior_mean={self._prior_mean_scalar}, prior_log_variance={self._prior_log_variance_scalar}"
        )


class VariationalDropoutMeanField(GaussianMeanField):
    """This class defines a wrapper for variational dropout weights defined under a mean-field Gaussian distribution."""

    def _clip(self, x: torch.Tensor, val: float = 8.0) -> torch.Tensor:
        """This method clips the input tensor to the range `[-val, val]`."""
        return torch.clamp(x, min=-val, max=val)

    @property
    def log_alpha(self) -> torch.Tensor:
        """This method returns the log alpha of the distribution."""
        return self._clip(self._log_variance - torch.log(self.mean**2 + TINY_EPSILON))

    def kl_divergence(self) -> torch.Tensor:
        """This method returns the KL divergence between the variational distribution and the prior."""
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        mdkl = (
            k1 * torch.sigmoid(k2 + k3 * self.log_alpha)
            - 0.5 * torch.log1p(torch.exp(-self.log_alpha))
            + C
        )
        return -torch.sum(mdkl)

    @torch.no_grad()
    def sparsity(self, threshold: float = 3.0) -> float:
        """This method returns the sparsity of the variational dropout weights."""
        return (torch.sum(self.log_alpha > threshold) / self.log_alpha.numel()).item()

    @torch.no_grad()
    def reset_on_threshold(self, threshold: float = 3.0) -> None:
        """This method resets the variational dropout weights to their prior.

        The reset is performed with respect to weight indices where the log alpha is greater than the threshold.
        """
        log_alpha = self.log_alpha
        mask = log_alpha > threshold
        new_initialisation = torch.randn_like(self._mean.data)
        if new_initialisation.dim() > 1:
            init.kaiming_uniform_(new_initialisation, a=math.sqrt(5))
        else:
            init.uniform_(new_initialisation, -1, 1)

        self._mean.data[mask] = new_initialisation[mask]
        self._log_variance.data[mask] = self._initial_log_variance
        # Add some noise to the log variance to avoid the log variance to be the same
        self._log_variance.data[mask] += (
            torch.randn_like(self._log_variance.data[mask]) * 1e-2
        )

    def freeze_on_threshold(self, threshold: float = 3.0) -> None:
        """This function is used to freeze the weights that have a `log_alpha` lower than the threshold.

        It is done through computing the `log_alpha` values and assigning a 1D index tensor under `_frozen_mask`.
        for both the mean and the log variance parameters. The current values of the parameters are saved under
        `_frozen_data` for both the mean and the log variance parameters.
        """
        log_alpha = self.log_alpha
        mask = (log_alpha < threshold).view(-1)
        mask_indices = torch.arange(mask.numel(), device=mask.device)[mask]
        freeze_weights(
            [self._mean, self._log_variance],
            [mask_indices.clone(), mask_indices.clone()],
        )


class KulbackLeiblerParameterLoss:
    """This class defines a loss function that is used to minimize the
    KL divergence between the variational weights and their prior.

    Args:
        model (nn.Module): The model that contains the variational weights.
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model

    def __call__(self) -> torch.Tensor:
        """This method returns the KL divergence between the variational weights and their prior."""
        kl_divergence = torch.tensor(0.0).to(next(self._model.parameters()).device)
        for m in self._model.modules():
            if isinstance(m, VariationalWeight):
                kl_divergence += m.kl_divergence()
        return kl_divergence

    def __repr__(self) -> str:
        return f"KulbackLeiblerLoss(alpha={self._alpha})"
