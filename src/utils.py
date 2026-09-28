"""Generic utility functions used across the project.

This module contains helpers for:

- Parameter initialization
- Dataset splitting
- Mini-batch iteration
- Small numerical utilities
"""

import random
from typing import Any, Iterable, List, Optional, Tuple

from src.activations import sigmoid

__all__ = [
    "init_weights",
    "init_bias",
    "train_val_split",
    "batch_iterator",
    "clip_values",
    "sigmoid_safe",
    "sigmoid",
]


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
        Random seed. Uses a private RNG so the global random state is left
        untouched.

    Returns
    -------
    list of float
        Flattened list of initialized weights.
    """
    if len(shape) != 1:
        raise ValueError("Only 1D weight initialization is supported")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool):
        raise ValueError("scale must be a number")
    if scale < 0:
        raise ValueError("scale must be non-negative")

    rng = random.Random(seed)
    size = shape[0]
    return [rng.uniform(-scale, scale) for _ in range(size)]


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
        Random seed. Uses a private RNG so the global random state is left
        untouched.

    Returns
    -------
    X_train, y_train, X_val, y_val
    """
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length, got {len(X)} and {len(y)}"
        )
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    rng = random.Random(seed)

    indices = list(range(len(X)))
    rng.shuffle(indices)

    n = len(X)
    # Always keep at least one example in each split, otherwise a small
    # dataset silently produces an empty training or validation set.
    n_val = int(round(n * val_ratio))
    n_val = max(1, min(n_val, n - 1)) if n > 1 else 0

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

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
        Random seed. Uses a private RNG so the global random state is left
        untouched.

    Yields
    ------
    (X_batch, y_batch)
    """
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length, got {len(X)} and {len(y)}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    indices = list(range(len(X)))
    if shuffle:
        random.Random(seed).shuffle(indices)

    for start in range(0, len(X), batch_size):
        batch_indices = indices[start : start + batch_size]
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
    if min_val > max_val:
        raise ValueError("min_val must be less than or equal to max_val")
    return max(min_val, min(x, max_val))


def sigmoid_safe(x: float, eps: float = 1e-8) -> float:
    """
    Numerically stable sigmoid function, clipped away from 0 and 1.

    Delegates to :func:`src.activations.sigmoid`, which splits on the sign of
    ``x`` so ``math.exp`` never receives a large positive argument. Computing
    ``1 / (1 + exp(-x))`` directly raises ``OverflowError`` for large negative
    ``x`` instead of returning a (correct) value very close to 0.

    Parameters
    ----------
    x : float
        Input value.
    eps : float
        Clipping value for numerical stability.

    Returns
    -------
    float
        Sigmoid output clipped to ``[eps, 1 - eps]``.
    """
    if eps <= 0 or eps >= 0.5:
        raise ValueError(f"eps must be in (0, 0.5), got {eps}")
    return clip_values(sigmoid(x), eps, 1.0 - eps)
