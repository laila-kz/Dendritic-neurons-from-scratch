"""Activation functions and their derivatives.

This module is the single source of truth for activations. Both
:class:`src.point_neuron.PointNeuron` and
:class:`src.dendritic_neuron.DendriticNeuron` resolve their activation
functions through here, so a neuron and the loss/gradient code can never
disagree about what ``"relu"`` means.

Every activation is a plain ``float -> float`` callable, which keeps the
neurons dependency-free and makes finite-difference gradient checks easy.
"""

import math
from typing import Callable, Dict, Tuple

__all__ = [
    "relu",
    "relu_derivative",
    "tanh",
    "tanh_derivative",
    "sigmoid",
    "sigmoid_derivative",
    "linear",
    "linear_derivative",
    "identity",
    "identity_derivative",
    "get_activation",
    "get_activation_derivative",
    "get_activation_pair",
    "ACTIVATION_NAMES",
]


def relu(x: float) -> float:
    """Rectified linear unit."""
    return max(0.0, x)


def relu_derivative(x: float) -> float:
    """Derivative of :func:`relu`.

    The subgradient at exactly ``0.0`` is defined as ``0.0``. This keeps the
    derivative well defined (and finite-difference checks stable) instead of
    returning the undefined left/right limits.
    """
    return 1.0 if x > 0.0 else 0.0


def tanh(x: float) -> float:
    """Hyperbolic tangent."""
    return math.tanh(x)


def tanh_derivative(x: float) -> float:
    """Derivative of :func:`tanh`."""
    return 1.0 - math.tanh(x) ** 2


def sigmoid(x: float) -> float:
    """Logistic sigmoid, evaluated in a numerically stable way.

    The naive ``1 / (1 + exp(-x))`` overflows for large negative ``x``
    (``math.exp`` raises ``OverflowError`` well before the true result
    underflows to 0). Splitting on the sign of ``x`` keeps every argument to
    ``exp`` non-positive, so the result is always finite.
    """
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def sigmoid_derivative(x: float) -> float:
    """Derivative of :func:`sigmoid`."""
    s = sigmoid(x)
    return s * (1.0 - s)


def linear(x: float) -> float:
    """Identity mapping, kept as a distinct name for readability."""
    return x


def linear_derivative(x: float) -> float:
    """Derivative of :func:`linear`."""
    return 1.0


def identity(x: float) -> float:
    """Alias of :func:`linear`."""
    return x


def identity_derivative(x: float) -> float:
    """Alias of :func:`linear_derivative`."""
    return 1.0


#: Maps a lowercase activation name to its ``(function, derivative)`` pair.
_ACTIVATIONS: Dict[str, Tuple[Callable[[float], float], Callable[[float], float]]] = {
    "relu": (relu, relu_derivative),
    "tanh": (tanh, tanh_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "linear": (linear, linear_derivative),
    "identity": (identity, identity_derivative),
}

#: All activation names accepted by :func:`get_activation`.
ACTIVATION_NAMES = tuple(sorted(_ACTIVATIONS))

# Backwards-compatible aliases for the original CamelCase spellings. New code
# should use the lowercase PEP 8 names defined above.
ReLU = relu
ReLU_derivative = relu_derivative
Tanh = tanh
Tanh_derivative = tanh_derivative
Sigmoid = sigmoid
Sigmoid_derivative = sigmoid_derivative
Linear = linear
Linear_derivative = linear_derivative


def get_activation(name: str) -> Callable[[float], float]:
    """Return the activation function registered under ``name``.

    Parameters
    ----------
    name : str
        One of :data:`ACTIVATION_NAMES`. Case-insensitive.

    Raises
    ------
    ValueError
        If ``name`` is not a supported activation, or is not a string.
    """
    if not isinstance(name, str):
        raise ValueError(
            f"Activation name must be a string, got {type(name).__name__}"
        )

    key = name.strip().lower()
    if key not in _ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation function: {name!r}. "
            f"Supported activations: {', '.join(ACTIVATION_NAMES)}"
        )
    return _ACTIVATIONS[key][0]


def get_activation_derivative(name: str) -> Callable[[float], float]:
    """Return the derivative of the activation registered under ``name``.

    Raises
    ------
    ValueError
        If ``name`` is not a supported activation, or is not a string.
    """
    if not isinstance(name, str):
        raise ValueError(
            f"Activation name must be a string, got {type(name).__name__}"
        )

    key = name.strip().lower()
    if key not in _ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation function: {name!r}. "
            f"Supported activations: {', '.join(ACTIVATION_NAMES)}"
        )
    return _ACTIVATIONS[key][1]


def get_activation_pair(
    name: str,
) -> Tuple[Callable[[float], float], Callable[[float], float]]:
    """Return ``(activation, derivative)`` for ``name`` in one lookup."""
    if not isinstance(name, str):
        raise ValueError(
            f"Activation name must be a string, got {type(name).__name__}"
        )

    key = name.strip().lower()
    if key not in _ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation function: {name!r}. "
            f"Supported activations: {', '.join(ACTIVATION_NAMES)}"
        )
    return _ACTIVATIONS[key]
