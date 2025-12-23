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
"""

# ---- Neuron models ----

from src.point_neuron import PointNeuron
from src.dendritic_neuron import DendriticNeuron

# ---- Math components ----

from src.activations import get_activation
from src.losses import get_loss_function

# ---- Training utilities ----

from src.training import train_model, evaluate_model

# ---- Generic utilities ----

from src.utils import (
    init_weights,
    init_bias,
    train_val_split,
    batch_iterator,
)

__all__ = [
    # Neurons
    "PointNeuron",
    "DendriticNeuron",
    # Activations & losses
    "get_activation",
    "get_loss_function",
    # Training
    "train_model",
    "evaluate_model",
    # Utils
    "init_weights",
    "init_bias",
    "train_val_split",
    "batch_iterator",
]
