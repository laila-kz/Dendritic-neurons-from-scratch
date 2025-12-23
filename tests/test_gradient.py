# This file ensures:
# - Your backward pass is mathematically correct
# - Gradients match numerical finite-difference estimates
# - Bugs are caught before training silently fails

import numpy as np

#add src to sys.path in tests (quick-and-dirty)
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
#This manually injects the project root into sys.path so src can be imported without touching environment variables.


from src.dendritic_neuron import DendriticNeuron
from src.point_neuron import PointNeuron
from src.losses import mse_loss


# Numerical gradient helper
def numerical_gradient(
    neuron,
    x,
    y_true,
    param_name,
    index,
    eps: float = 1e-5,
) -> float:
    """
    Compute numerical gradient for one parameter entry.

    param_name:
        "weights" or "bias" for PointNeuron
        "branch_weights" or "branch_biases" or "soma_weights" or "soma_bias" for DendriticNeuron
    index:
        - For vectors: int index
        - For branch_weights: (branch_idx, weight_idx)
    """
    params = neuron.get_parameters()

    # Helper to get/set element with flexible index
    def get_ref(p, idx):
        if isinstance(idx, tuple):
            b, i = idx
            return p[b][i]
        else:
            return p[idx]

    def set_ref(p, idx, value):
        if isinstance(idx, tuple):
            b, i = idx
            p[b][i] = value
        else:
            p[idx] = value

    original_value = get_ref(params[param_name], index)

    # f(theta + eps)
    set_ref(params[param_name], index, original_value + eps)
    neuron.set_parameters(params)
    y_pred = neuron.forward_single(x) if hasattr(neuron, "forward_single") else neuron.forward(x)
    loss_plus = mse_loss([y_pred], [y_true])

    # f(theta - eps)
    params = neuron.get_parameters()
    set_ref(params[param_name], index, original_value - eps)
    neuron.set_parameters(params)
    y_pred = neuron.forward_single(x) if hasattr(neuron, "forward_single") else neuron.forward(x)
    loss_minus = mse_loss([y_pred], [y_true])

    # restore
    params = neuron.get_parameters()
    set_ref(params[param_name], index, original_value)
    neuron.set_parameters(params)

    return (loss_plus - loss_minus) / (2 * eps)


# Point neuron gradient check
def test_point_neuron_gradient():
    neuron = PointNeuron(input_dim=3, activation="linear")

    x = np.array([1.0, -2.0, 3.0])
    y_true = 0.5

    # forward + backward
    y_pred = neuron.forward(x.tolist())

    # For MSE derivative wrt y: dL/dy = 2*(y_pred - y_true)/N; here N=1
    dL_dy = [2 * (y_pred - y_true)]

    grads = neuron.backward([x.tolist()], dL_dy)  # batch of size 1

    num_grad = numerical_gradient(
        neuron,
        x.tolist(),
        y_true,
        param_name="weights",
        index=0,
    )

    assert np.isclose(grads["weights"][0], num_grad, atol=1e-4), "PointNeuron gradient mismatch"


# Dendritic neuron gradient check (branch weight)
def test_dendritic_branch_gradient():
    neuron = DendriticNeuron(
        input_dim=2,
        num_branches=1,
        branch_input_map={0: [0, 1]},
        branch_activation="linear",
        soma_activation="linear",
    )

    x = np.array([1.0, 2.0])
    y_true = 1.0

    y_pred = neuron.forward_single(x.tolist())

    # dL/dy for single sample
    dL_dy = [2 * (y_pred - y_true)]

    grads = neuron.backward([x.tolist()], dL_dy)

    num_grad = numerical_gradient(
        neuron,
        x.tolist(),
        y_true,
        param_name="branch_weights",
        index=(0, 0),  # branch 0, weight 0
    )

    assert np.isclose(
        grads["branch_weights"][0][0],
        num_grad,
        atol=1e-4,
    ), "Dendritic branch gradient mismatch"
