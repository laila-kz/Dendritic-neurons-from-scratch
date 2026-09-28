"""Loss functions and their derivatives for training.

All losses are *mean* reductions over the batch, and each ``*_derivative``
function returns the gradient of that mean loss with respect to the
predictions. Keeping that contract consistent is what lets
:meth:`src.point_neuron.PointNeuron.backward` and
:meth:`src.dendritic_neuron.DendriticNeuron.backward` sum the per-sample
contributions directly.
"""

import math
from typing import Callable, List, Tuple

__all__ = [
    "mse_loss",
    "mse_derivative",
    "binary_cross_entropy_loss",
    "binary_cross_entropy_derivative",
    "get_loss_function",
    "LOSS_NAMES",
]


def _check_pair(y_pred: List[float], y_true: List[float]) -> int:
    """Validate prediction/target arrays and return the batch size."""
    if len(y_pred) != len(y_true):
        raise ValueError(
            f"y_pred and y_true must have the same length, "
            f"got {len(y_pred)} and {len(y_true)}"
        )
    if len(y_pred) == 0:
        raise ValueError("y_pred and y_true must not be empty")
    return len(y_pred)


def mse_loss(y_pred: List[float], y_true: List[float]) -> float:
    """Mean Squared Error Loss."""
    n = _check_pair(y_pred, y_true)
    return sum((yp - yt) ** 2 for yp, yt in zip(y_pred, y_true)) / n


def mse_derivative(y_pred: List[float], y_true: List[float]) -> List[float]:
    """Derivative of the mean Squared Error Loss w.r.t. each prediction."""
    n = _check_pair(y_pred, y_true)
    return [2 * (yp - yt) / n for yp, yt in zip(y_pred, y_true)]


def binary_cross_entropy_loss(
    y_pred: List[float], y_true: List[float], eps: float = 1e-8
) -> float:
    """Binary Cross-Entropy Loss."""
    n = _check_pair(y_pred, y_true)

    loss = 0.0
    for yp, yt in zip(y_pred, y_true):
        # Clamp predictions away from the open interval boundaries so that
        # log(0) can never be reached.
        yp = min(max(yp, eps), 1.0 - eps)
        loss += -(yt * math.log(yp) + (1.0 - yt) * math.log(1.0 - yp))
    return loss / n


def binary_cross_entropy_derivative(
    y_pred: List[float], y_true: List[float], eps: float = 1e-8
) -> List[float]:
    """Derivative of the mean Binary Cross-Entropy Loss w.r.t. each prediction.

    For a single sample the derivative of
    ``-(y*log(p) + (1-y)*log(1-p))`` w.r.t. ``p`` is
    ``(p - y) / (p * (1 - p))``. The equivalent form used here is algebraically
    identical but avoids the intermediate division by ``p * (1 - p)``, which
    overflows to a division-by-zero for predictions that saturate near 0 or 1.
    """
    n = _check_pair(y_pred, y_true)

    derivatives: List[float] = []
    for yp, yt in zip(y_pred, y_true):
        # Clamp identically to the loss so the gradient matches the value
        # that was actually optimised.
        yp = min(max(yp, eps), 1.0 - eps)
        deriv = (yp - yt) / (yp * (1.0 - yp))
        derivatives.append(deriv / n)
    return derivatives


#: Maps a lowercase loss name to its ``(loss, derivative)`` pair.
_LOSSES = {
    "mse": (mse_loss, mse_derivative),
    "binary_cross_entropy": (binary_cross_entropy_loss, binary_cross_entropy_derivative),
    "bce": (binary_cross_entropy_loss, binary_cross_entropy_derivative),
}

#: All loss names accepted by :func:`get_loss_function`.
LOSS_NAMES = ("bce", "binary_cross_entropy", "mse")


def get_loss_function(
    name: str,
) -> Tuple[
    Callable[[List[float], List[float]], float],
    Callable[[List[float], List[float]], List[float]],
]:
    """Return the ``(loss, derivative)`` pair for the given name.

    Parameters
    ----------
    name : str
        One of :data:`LOSS_NAMES`. Case-insensitive.
    """
    if not isinstance(name, str):
        raise ValueError(f"Loss name must be a string, got {type(name).__name__}")

    key = name.strip().lower()
    if key not in _LOSSES:
        raise ValueError(
            f"Unsupported loss function: {name!r}. "
            f"Supported losses: {', '.join(LOSS_NAMES)}"
        )
    return _LOSSES[key]
