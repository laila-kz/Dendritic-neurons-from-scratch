#This module holds generic utilities not specific to a particular neuron.

"""
Generic utility functions used across the project.

This module contains helpers for:

- Parameter initialization
- Dataset splitting
- Mini-batch iteration
- Small numerical utilities
"""

import random
import math
from typing import List, Tuple, Iterable, Any, Optional

# ---------- Parameter initialization ----------


def init_weights(
    shape: Tuple[int, ...],
    scale: float = 0.01,
    seed: Optional[int] = None,
) -> List[float]:
    """
    Initialize weights with small random values.

    Parameters
    ----------
    shape : tuple
        Shape of the weight array (e.g., (input_dim,) or (num_branches,)).
    scale : float
        Scaling factor for random values.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list of float
        Flattened list of initialized weights.
    """
    if seed is not None:
        random.seed(seed)

    if len(shape) != 1:
        raise ValueError("Only 1D weight initialization is supported")

    size = shape[0]
    return [random.uniform(-scale, scale) for _ in range(size)]


def init_bias(value: float = 0.0) -> float:
    """
    Initialize a bias value.

    Parameters
    ----------
    value : float
        Initial bias.

    Returns
    -------
    float
        Bias value.
    """
    return float(value)


# ---------- Dataset splitting ----------


def train_val_split(
    X: List[Any],
    y: List[Any],
    val_ratio: float = 0.2,
    seed: Optional[int] = None,
) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
    """
    Split dataset into training and validation sets.

    Parameters
    ----------
    X : list
        Input data.
    y : list
        Target data.
    val_ratio : float
        Fraction of data used for validation.
    seed : int, optional
        Random seed.

    Returns
    -------
    X_train, y_train, X_val, y_val
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    if seed is not None:
        random.seed(seed)

    indices = list(range(len(X)))
    random.shuffle(indices)

    split_idx = int(len(X) * (1.0 - val_ratio))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    return X_train, y_train, X_val, y_val


# ---------- Mini-batch iterator ----------


def batch_iterator(
    X: List[Any],
    y: List[Any],
    batch_size: int,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Iterable[Tuple[List[Any], List[Any]]]:
    """
    Yield mini-batches for one epoch.

    Parameters
    ----------
    X : list
        Input data.
    y : list
        Target data.
    batch_size : int
        Size of each mini-batch.
    shuffle : bool
        Whether to shuffle data before batching.
    seed : int, optional
        Random seed.

    Yields
    ------
    (X_batch, y_batch)
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    indices = list(range(len(X)))
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(indices)

    for start in range(0, len(X), batch_size):
        batch_indices = indices[start:start + batch_size]
        yield (
            [X[i] for i in batch_indices],
            [y[i] for i in batch_indices],
        )


# ---------- Small numerical helpers ----------


def clip_values(x: float, min_val: float, max_val: float) -> float:
    """
    Clip a value to a given range.

    Parameters
    ----------
    x : float
        Input value.
    min_val : float
        Minimum allowed value.
    max_val : float
        Maximum allowed value.

    Returns
    -------
    float
        Clipped value.
    """
    return max(min_val, min(x, max_val))


def sigmoid_safe(x: float, eps: float = 1e-8) -> float:
    """
    Numerically stable sigmoid function.

    Parameters
    ----------
    x : float
        Input value.
    eps : float
        Clipping value for numerical stability.

    Returns
    -------
    float
        Sigmoid output clipped to (eps, 1 - eps).
    """
    s = 1.0 / (1.0 + math.exp(-x))
    return clip_values(s, eps, 1.0 - eps)
