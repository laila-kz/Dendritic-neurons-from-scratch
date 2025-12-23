#This module groups activation functions and their derivatives so both neuron types can reuse them.

import math
from typing import Callable


def ReLU(x: float) -> float:
    return max(0.0, x)


def ReLU_derivative(x: float) -> float:
    return 1.0 if x > 0.0 else 0.0


def Tanh(x: float) -> float:
    return math.tanh(x)


def Tanh_derivative(x: float) -> float:
    return 1.0 - math.tanh(x) ** 2


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def Sigmoid_derivative(x: float) -> float:
    s = sigmoid(x)
    return s * (1.0 - s)


def Linear(x: float) -> float:
    return x


def Linear_derivative(x: float) -> float:
    return 1.0


def get_activation(name: str) -> Callable[[float], float]:
    """Return the activation function corresponding to the given name."""
    name = name.lower()
    if name == "relu":
        return ReLU
    elif name == "tanh":
        return Tanh
    elif name == "sigmoid":
        return sigmoid
    elif name == "linear":
        return Linear
    else:
        raise ValueError(f"Unsupported activation function: {name}")
