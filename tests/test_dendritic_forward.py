# Tests: catch bugs early, especially in forward/gradient math.
# This file checks:
# - Shapes → are we returning what training expects?
# - Math correctness → does the dendritic computation match hand-computed math?
# - Branch isolation → do branches only "see" their assigned inputs?
# - Batch correctness → is batch logic consistent with single-sample logic?

import numpy as np
import pytest

import os
import sys

#add src to sys.path in tests (quick-and-dirty)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from src.dendritic_neuron import DendriticNeuron


# Test for shape and type
def test_forward_shapes():
    input_dim = 4
    branch_groups = [[0, 1], [2, 3]]

    neuron = DendriticNeuron(
        input_dim=input_dim,
        num_branches=len(branch_groups),
        branch_input_map={i: grp for i, grp in enumerate(branch_groups)},
        branch_activation="linear",
        soma_activation="linear",
    )

    # Single input
    x = [1.0, 2.0, 3.0, 4.0]
    y = neuron.forward_single(x)

    # Forward must return a scalar
    assert isinstance(y, (int, float)), f"Expected scalar, got {type(y)}"

    # Batch input
    X = [
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 0.5, 0.5, 0.5],
    ]
    y_batch = neuron.forward_batch(X)

    # forward_batch must return list with batch_size elements
    assert len(y_batch) == 2, "forward_batch() must return list with batch_size elements"


# Test for math correctness
def test_forward_known_values():
    # Verify dendritic computations: branches output and soma aggregation
    input_dim = 4
    branch_groups = [[0, 1], [2, 3]]

    neuron = DendriticNeuron(
        input_dim=input_dim,
        num_branches=len(branch_groups),
        branch_input_map={i: grp for i, grp in enumerate(branch_groups)},
        branch_activation="linear",
        soma_activation="linear",
    )

    # Manually set params
    neuron.branch_weights = [
        [1.0, 1.0],  # branch 0
        [1.0, 1.0],  # branch 1
    ]
    neuron.branch_biases = [0.0, 0.0]
    neuron.soma_weights = [1.0, 1.0]
    neuron.soma_bias = 0.0

    # The input
    x = [1.0, 2.0, 3.0, 4.0]

    # Branch 0: (1 + 2) = 3
    # Branch 1: (3 + 4) = 7
    # Soma: 3 + 7 = 10
    expected_output = 10.0
    y = neuron.forward_single(x)

    assert np.isclose(y, expected_output), f"Expected {expected_output}, got {y}"


# Test Branch isolation
# Changing an input index should only affect its assigned branch
def test_branch_isolation():
    # This catches a very common bug:
    # accidentally using the full input vector inside a branch
    input_dim = 3
    branch_groups = [[0], [1], [2]]

    neuron = DendriticNeuron(
        input_dim=input_dim,
        num_branches=len(branch_groups),
        branch_input_map={i: grp for i, grp in enumerate(branch_groups)},
        branch_activation="linear",
        soma_activation="linear",
    )

    # Manually input params
    neuron.branch_weights = [
        [1.0],
        [1.0],
        [1.0],
    ]
    neuron.branch_biases = [0.0, 0.0, 0.0]
    neuron.soma_weights = [1.0, 1.0, 1.0]
    neuron.soma_bias = 0.0

    x1 = [1.0, 0.0, 0.0]
    x2 = [2.0, 0.0, 0.0]

    y1 = neuron.forward_single(x1)
    y2 = neuron.forward_single(x2)

    # Only branch 0 should contribute to the change
    assert np.isclose(y2 - y1, 1.0), "Only branch 0 should change"


# Test Batch correctness/consistency
def test_batch_consistency():
    input_dim = 4
    branch_groups = [[0, 1], [2, 3]]

    neuron = DendriticNeuron(
        input_dim=input_dim,
        num_branches=len(branch_groups),
        branch_input_map={i: grp for i, grp in enumerate(branch_groups)},
        branch_activation="linear",
        soma_activation="linear",
    )

    X = np.random.randn(5, input_dim)

    outputs_loop = [neuron.forward_single(x.tolist()) for x in X]
    outputs_batch = neuron.forward_batch(X.tolist())

    assert np.allclose(outputs_loop, outputs_batch)
