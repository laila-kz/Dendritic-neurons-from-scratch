"""
Core neural components package.

This package contains:

- Neuron models (PointNeuron, DendriticNeuron)
- Activation functions
- Loss functions
- Training utilities
- Generic helpers

It is designed to be used by:

- experiments/
- tests/

Importing from the package root is equivalent to importing from the
individual modules, e.g.::

    from src import DendriticNeuron, get_loss_function
"""

# ---- Neuron models ----

from src.activations import get_activation, get_activation_derivative, get_activation_pair
from src.dendritic_neuron import DendriticNeuron
from src.losses import (
    binary_cross_entropy_derivative,
    binary_cross_entropy_loss,
    get_loss_function,
    mse_derivative,
    mse_loss,
)
from src.point_neuron import PointNeuron
from src.training import evaluate_model, train_model, train_one_epoch
from src.utils import (
    batch_iterator,
    clip_values,
    init_bias,
    init_weights,
    sigmoid_safe,
    train_val_split,
)

__all__ = [
    # Neurons
    "PointNeuron",
    "DendriticNeuron",
    # Activations & losses
    "get_activation",
    "get_activation_derivative",
    "get_activation_pair",
    "get_loss_function",
    "mse_loss",
    "mse_derivative",
    "binary_cross_entropy_loss",
    "binary_cross_entropy_derivative",
    # Training
    "train_model",
    "train_one_epoch",
    "evaluate_model",
    # Utils
    "init_weights",
    "init_bias",
    "train_val_split",
    "batch_iterator",
    "clip_values",
    "sigmoid_safe",
]
