#This module defines loss functions and their derivatives for training.

import math
from typing import Callable, List, Tuple


def mse_loss(y_pred: List[float], y_true: List[float]) -> float:
    """Mean Squared Error Loss."""
    if len(y_pred) != len(y_true):
        raise ValueError("y_pred and y_true must have the same length")
    if len(y_pred) == 0:
        raise ValueError("y_pred and y_true must not be empty")
    return sum((yp - yt) ** 2 for yp, yt in zip(y_pred, y_true)) / len(y_true)


def mse_derivative(y_pred: List[float], y_true: List[float]) -> List[float]:
    """Derivative of Mean Squared Error Loss."""
    if len(y_pred) != len(y_true):
        raise ValueError("y_pred and y_true must have the same length")
    if len(y_pred) == 0:
        raise ValueError("y_pred and y_true must not be empty")
    return [2 * (yp - yt) / len(y_pred) for yp, yt in zip(y_pred, y_true)]


def binary_cross_entropy_loss(
    y_pred: List[float], y_true: List[float], eps: float = 1e-8
) -> float:
    """Binary Cross-Entropy Loss."""
    if len(y_pred) != len(y_true):
        raise ValueError("y_pred and y_true must have the same length")
    if len(y_pred) == 0:
        raise ValueError("y_pred and y_true must not be empty")

    loss = 0.0
    for yp, yt in zip(y_pred, y_true):
        # clamp predictions to avoid log(0)
        yp = min(max(yp, eps), 1 - eps)
        loss += -(yt * math.log(yp) + (1 - yt) * math.log(1 - yp))
    return loss / len(y_true)


def binary_cross_entropy_derivative(
    y_pred: List[float], y_true: List[float], eps: float = 1e-8
) -> List[float]:
    """Derivative of Binary Cross-Entropy Loss."""
    if len(y_pred) != len(y_true):
        raise ValueError("y_pred and y_true must have the same length")
    if len(y_pred) == 0:
        raise ValueError("y_pred and y_true must not be empty")

    derivatives: List[float] = []
    for yp, yt in zip(y_pred, y_true):
        # clamp predictions to avoid division by zero
        yp = min(max(yp, eps), 1 - eps)
        deriv = -(yt / yp) + (1 - yt) / (1 - yp)
        derivatives.append(deriv / len(y_true))
    return derivatives


def get_loss_function(
    name: str,
) -> Tuple[
    Callable[[List[float], List[float]], float],
    Callable[[List[float], List[float]], List[float]],
]:
    """Return the loss function corresponding to the given name."""
    name = name.lower()
    if name == "mse":
        return mse_loss, mse_derivative
    elif name in ("binary_cross_entropy", "bce"):
        return binary_cross_entropy_loss, binary_cross_entropy_derivative
    else:
        raise ValueError(f"Unsupported loss function: {name}")
