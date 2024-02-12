from typing import List, Optional
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from yamle.defaults import TINY_EPSILON


@torch.no_grad()
def plot_input_layer_norm_bar(
    weights: List[torch.Tensor], save_path: str, iteration: Optional[int] = None
) -> torch.Tensor:
    """Plots the bar chart of the input layer norm weights for each member.

    It is assumed that the length of the weights is the number of members.
    Calculate the L1 norm of the weight for each member across input features.
    """
    fig = plt.figure(figsize=(30, 10))
    num_members = len(weights)
    norms = []
    for member_idx in range(num_members):
        member_weight = weights[member_idx]
        member_weight_norm = member_weight.norm(
            p=1, dim=(tuple(range(1, len(member_weight.shape))))
        )
        # Normalize the norm between 0 and 1
        member_weight_norm = member_weight_norm / (
            member_weight_norm.max() + TINY_EPSILON
        )
        norms.append(member_weight_norm)
        x = torch.arange(member_weight_norm.shape[0]).cpu().numpy()
        plt.bar(
            x,
            member_weight_norm.cpu().numpy(),
            label=f"Member {member_idx}",
            alpha=1 / num_members,
        )
    overlap = calculate_overlap_between_norms(norms)
    plt.legend()
    plt.grid()
    plt.xlabel("Output feature")
    plt.ylabel("Normalized L1 norm")
    plt.xlim(-0.5, member_weight_norm.shape[0] - 0.5)
    plt.ylim(0, 1)
    plt.title(f"Overlap: {overlap.item()}")
    plt.savefig(
        os.path.join(
            save_path,
            "input_layer_norm_bar.png"
            if iteration is None
            else f"input_layer_norm_bar_{iteration}.png",
        ),
        bbox_inches="tight",
    )
    plt.close(fig)
    plt.clf()
    return overlap


@torch.no_grad()
def plot_output_layer_norm_bar(
    weights: List[torch.Tensor], save_path: str, iteration: Optional[int] = None
) -> torch.Tensor:
    """Plots the bar chart of the output layer norm weights for each member.

    It is assumed that the length of the weights is the number of members.
    Calculate the L1 norm of the weight for each member across output features.
    """
    fig = plt.figure(figsize=(20, 10))
    norms = []
    num_members = len(weights)
    for member_idx in range(num_members):
        member_weight = weights[member_idx]
        member_weight_norm = member_weight.norm(
            p=1, dim=(0, *tuple(range(2, len(member_weight.shape))))
        )
        # Normalize the norm between 0 and 1
        member_weight_norm = member_weight_norm / (
            member_weight_norm.max() + TINY_EPSILON
        )
        norms.append(member_weight_norm)
        x = torch.arange(member_weight_norm.shape[0]).cpu().numpy()
        plt.bar(
            x,
            member_weight_norm.cpu().numpy(),
            label=f"Member {member_idx}",
            alpha=1 / num_members,
        )
    overlap = calculate_overlap_between_norms(norms)
    plt.legend()
    plt.title(f"Overlap: {overlap.item()}")
    plt.grid()
    plt.xlabel("Input feature")
    plt.ylabel("Normalized L1 norm")
    plt.xlim(-0.5, member_weight_norm.shape[0] - 0.5)
    plt.ylim(0, 1)
    plt.savefig(
        os.path.join(
            save_path,
            "output_layer_norm_bar.png"
            if iteration is None
            else f"output_layer_norm_bar_{iteration}.png",
        ),
        bbox_inches="tight",
    )
    plt.close(fig)
    plt.clf()
    return overlap


def calculate_overlap_between_norms(norms: List[torch.Tensor]) -> torch.Tensor:
    """Calculates the overlap between the norms of the members."""
    # Find the minimum and maximum for all members
    min_norms = torch.stack(norms).min(dim=0)[0]
    max_norms = torch.stack(norms).max(dim=0)[0]
    # Calculate the average between the minimum and maximum norm for each feature
    return torch.mean(min_norms / (max_norms + TINY_EPSILON))


@torch.no_grad()
def plot_overlap_between_members(
    overlap: torch.Tensor,
    save_path: str,
    input: bool = True,
    iteration: Optional[int] = None,
):
    """Plots the overlap between the members."""
    fig = plt.figure(figsize=(20, 10))
    x = torch.arange(overlap.shape[0]).cpu().numpy()
    plt.plot(x, overlap.cpu().numpy()[:, 0])
    plt.grid()
    plt.xlabel("Overlap")
    plt.ylabel("Iteration")
    plt.ylim(0, 1)
    name = (
        "overlap_between_members.png"
        if iteration is None
        else f"overlap_between_members_{iteration}.png"
    )
    name = "input_" + name if input else "output_" + name
    plt.savefig(os.path.join(save_path, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)
    plt.clf()


@torch.no_grad()
def plot_weight_trajectories(
    weight: torch.Tensor,
    save_path: str,
    iteration: Optional[int] = None,
    predicted: bool = True,
):
    """Given the depth weights plot the weight trajectories over iterations for each member separately."""
    if len(weight.shape) == 2:
        _, members = weight.shape
        depth = 1
        weight = weight.unsqueeze(1)
    elif len(weight.shape) == 3:
        _, depth, members = weight.shape
    else:
        raise ValueError(f"Weight shape {weight.shape} is not supported.")
    colors = plt.cm.rainbow(np.linspace(0, 1, depth))
    # Plot the weights for a member
    for i in range(members):
        fig = plt.figure(figsize=(6, 6))
        for j in range(depth):
            plt.plot(weight[:, j, i], color=colors[j], label=f"Depth {j}")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Weight")
        plt.grid()
        plt.title(f"Member: {i}")
        name = (
            f"member_{i}_weight_trajectory.png"
            if iteration is None
            else f"member_{i}_weight_trajectory_{iteration}.png"
        )
        name = "predicted_" + name if predicted else "oracle_" + name
        plt.savefig(os.path.join(save_path, name), bbox_inches="tight")
        plt.close(fig)
        plt.clf()


@torch.no_grad()
def plot_weight_histogram(
    weight: torch.Tensor,
    save_path: str,
    iteration: Optional[int] = None,
    predicted: bool = True,
):
    """Given the depth weights plot the weight histogram over iterations for each member separately.

    The weight has the shape `(N, depth, members)`. The histogram is plotted for each member.
    `N` is the number of samples for the histogram.

    For each member create a subplots for each depth. Try to make the number of subplots
    square in rows and columns.

    """
    if len(weight.shape) == 2:
        _, members = weight.shape
        depth = 1
        weight = weight.unsqueeze(1)
    elif len(weight.shape) == 3:
        _, depth, members = weight.shape
    else:
        raise ValueError(f"Weight shape {weight.shape} is not supported.")
    # Plot the weights for a member
    for i in range(members):
        fig, axes = plt.subplots(
            nrows=int(np.ceil(np.sqrt(depth))),
            ncols=int(np.ceil(np.sqrt(depth))),
            figsize=(20, 20),
        )
        for j in range(depth):
            row = j // int(np.ceil(np.sqrt(depth)))
            col = j % int(np.ceil(np.sqrt(depth)))
            axes[row, col].hist(
                weight[:, j, i].cpu().numpy(), bins=10, density=True, range=(0, 1)
            )
            axes[row, col].set_title(f"Depth {j}")
            axes[row, col].grid()

        plt.suptitle(f"Member: {i}")
        name = (
            f"member_{i}_weight_histogram.png"
            if iteration is None
            else f"member_{i}_weight_histogram_{iteration}.png"
        )
        name = "predicted_" + name if predicted else "oracle_" + name
        plt.savefig(os.path.join(save_path, name), bbox_inches="tight")
        plt.close(fig)
        plt.clf()
