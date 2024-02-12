from typing import Optional, Iterable, Callable
import torch
import torch.nn as nn
from torch.optim import Optimizer
import math
from yamle.defaults import TINY_EPSILON


class SGLD(Optimizer):
    """Stochastic Gradient Langevin Dynamics optimizer.

    Adopted from: https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Stochastic_Gradient_Langevin_Dynamics/optimizers.py

    We added the options for momentum and nestrov acceleration.

    Args:
        params (Iterable[nn.Parameter]): Parameters to optimize.
        lr (float, optional): Learning rate (default: 1e-2).
        momentum (float, optional): Momentum factor (default: 0.9).
        nestrov (bool, optional): Use nestrov acceleration (default: False).
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-2,
        momentum: float = 0.9,
        nestrov: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if momentum < 0.0 or not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(lr=lr, momentum=momentum, nestrov=nestrov)
        super(SGLD, self).__init__(params, defaults)

    def _step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            nestrov = group["nestrov"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        param_state["momentum_buffer"] = d_p.clone().detach()
                        buf = param_state["momentum_buffer"]
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1)

                    if nestrov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                # Generage noise according to eq. 4 in the paper
                noise = p.data.new(p.data.size()).normal_(
                    mean=0, std=math.sqrt(group["lr"] + TINY_EPSILON)
                )
                # Update the parameters according to eq. 4 in the paper
                p.data.add_(d_p, alpha=-group["lr"] * 0.5)
                p.data.add_(noise, alpha=-1)

        return loss


class pSGLD(Optimizer):
    """Preconditioned Stochastic Gradient Langevin Dynamics optimizer.

    Adopted from: https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Stochastic_Gradient_Langevin_Dynamics/optimizers.py

    Args:
        params (Iterable[nn.Parameter]): Parameters to optimize.
        lr (float, optional): Learning rate (default: 1e-2).
        alpha (float, optional): Alpha value for preconditioner (default: 0.99).
        centered (bool, optional): Use centered version of pSGLD (default: True).
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-2,
        alpha: float = 0.99,
        centered: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if alpha < 0.0 or not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")

        defaults = dict(
            lr=lr,
            alpha=alpha,
            centered=centered,
        )
        super(pSGLD, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            alpha = group["alpha"]
            centered = group["centered"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                state = self.state[p]
                if "square_avg" not in state:
                    state["square_avg"] = torch.zeros_like(p.data)
                square_avg = state["square_avg"]
                square_avg.mul_(alpha).addcmul_(d_p, d_p, value=1 - alpha)
                if centered:
                    if "grad_avg" not in state:
                        state["grad_avg"] = torch.zeros_like(p.data)
                    grad_avg = state["grad_avg"]
                    grad_avg.mul_(alpha).add_(d_p, alpha=1 - alpha)
                    avg = (
                        square_avg.addcmul(grad_avg, grad_avg, value=-1)
                        .add_(TINY_EPSILON)
                        .sqrt_()
                    )
                else:
                    avg = square_avg.sqrt().add_(TINY_EPSILON)

                noise = p.data.new(p.data.size()).normal_(mean=0, std=1.0) / math.sqrt(
                    group["lr"] + TINY_EPSILON
                )
                p.data.add_(
                    0.5 * d_p.div_(avg) + noise / torch.sqrt(avg), alpha=-group["lr"]
                )

        return loss
