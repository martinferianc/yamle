"""
This file is an adaptation to the original ImageNet-C corruptions but for
tabular data.
"""

import numpy as np


def additive_gaussian_noise(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """Adds gaussian noise with mean 0 and std deviation c.

    The standard deviation c is multiplied with respect to the features
    range to get the final standard deviation.
    
    Args:
        x (np.ndarray): The input data of shape (N, D).
        severity (int): The severity of the corruption. Must be in [1, 5].
    """
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    # Sample noise in the same shape as the input
    eta = np.random.normal(loc=0.0, scale=c, size=x.shape)
    eta = eta * x
    return x + eta


def multiplicative_gaussian_noise(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """Multiplies the input by a gaussian noise with mean 1 and std deviation c.

    The standard deviation c is multiplied with respect to the feature range
    """
    c = [0.05, 0.1, 0.15, 0.2, 0.3][severity - 1]
    return x * np.random.normal(loc=1.0, scale=c, size=x.shape)


def additive_uniform_noise(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """Adds uniform noise with range [-c, c].

    The range is multiplied with respect to the feature range to get the final range.
    """
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    # Sample noise in the same shape as the input
    eta = np.random.uniform(low=-c, high=c, size=x.shape)
    eta = eta * x
    return x + eta


def multiplicative_uniform_noise(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """Multiplies the input by a uniform noise with range [1-c, 1+c]."""
    c = [0.05, 0.1, 0.15, 0.2, 0.3][severity - 1]
    return x * np.random.uniform(low=1.0 - c, high=1.0 + c, size=x.shape)


def multiplicative_bernoulli_noise(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """Multiplies the input by a bernoulli noise which drops features with probability c."""
    c = [0.05, 0.1, 0.15, 0.2, 0.3][severity - 1]
    return x * np.random.binomial(n=1, p=1.0 - c, size=x.shape)
