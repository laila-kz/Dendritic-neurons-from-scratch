# This file ensures:
# - The backward pass is mathematically correct
# - Gradients match numerical finite-difference estimates
# - Batch gradients scale the same way for batch size 1 and N
# - Bugs are caught before training silently fails

import numpy as np
import pytest

from src.dendritic_neuron import DendriticNeuron
from src.losses import mse_derivative, mse_loss
from src.point_neuron import PointNeuron


def _get_ref(params, param_name, index):
    """Read a parameter entry, handling vectors, nested lists and scalars."""
    value = params[param_name]
    if index is None:  # scalar parameter, e.g. "bias" or "soma_bias"
        return value
    if isinstance(index, tuple):
        return value[index[0]][index[1]]
    return value[index]


def _set_ref(params, param_name, index, new_value):
    """Write a parameter entry, mirroring :func:`_get_ref`."""
    if index is None:
        params[param_name] = new_value
        return
    if isinstance(index, tuple):
        params[param_name][index[0]][index[1]] = new_value
    else:
        params[param_name][index] = new_value


def numerical_gradient(neuron, x, y_true, param_name, index=None, eps=1e-5):
    """
    Compute the numerical gradient of the loss w.r.t. one parameter entry.

    param_name:
        "weights" or "bias" for PointNeuron
        "branch_weights", "branch_biases", "soma_weights" or "soma_bias" for
        DendriticNeuron
    index:
        ``None`` for scalar parameters, an int for vectors, or
        ``(branch_idx, weight_idx)`` for branch weights.
    """

    def loss_at(value):
        params = neuron.get_parameters()
        _set_ref(params, param_name, index, value)
        neuron.set_parameters(params)
        y_pred = _forward_one(neuron, x)
        return mse_loss([y_pred], [y_true])

    original_value = _get_ref(neuron.get_parameters(), param_name, index)

    loss_plus = loss_at(original_value + eps)
    loss_minus = loss_at(original_value - eps)

    # restore the original parameters
    restored = neuron.get_parameters()
    _set_ref(restored, param_name, index, original_value)
    neuron.set_parameters(restored)

    return (loss_plus - loss_minus) / (2 * eps)


def _forward_one(neuron, x):
    if hasattr(neuron, "forward_single"):
        return neuron.forward_single(x)
    return neuron.forward(x)


def batch_numerical_gradient(neuron, X, y, param_name, index=None, eps=1e-5):
    """Numerical gradient of the *mean* batch loss w.r.t. one parameter."""
    original = _get_ref(neuron.get_parameters(), param_name, index)

    params = neuron.get_parameters()
    _set_ref(params, param_name, index, original + eps)
    neuron.set_parameters(params)
    loss_plus = mse_loss(neuron.forward_batch(X), y)

    params = neuron.get_parameters()
    _set_ref(params, param_name, index, original - eps)
    neuron.set_parameters(params)
    loss_minus = mse_loss(neuron.forward_batch(X), y)

    restored = neuron.get_parameters()
    _set_ref(restored, param_name, index, original)
    neuron.set_parameters(restored)

    return (loss_plus - loss_minus) / (2 * eps)


# ---------- PointNeuron ----------


def test_point_neuron_gradient():
    neuron = PointNeuron(input_dim=3, activation="linear", seed=1)

    x = [1.0, -2.0, 3.0]
    y_true = 0.5

    y_pred = neuron.forward(x)
    dL_dy = mse_derivative([y_pred], [y_true])

    grads = neuron.backward([x], dL_dy)

    num_grad = numerical_gradient(neuron, x, y_true, "weights", 0)
    assert np.isclose(grads["weights"][0], num_grad, atol=1e-6)


def test_point_neuron_bias_gradient():
    neuron = PointNeuron(input_dim=3, activation="linear", seed=1)

    x = [1.0, -2.0, 3.0]
    y_true = 0.5

    y_pred = neuron.forward(x)
    grads = neuron.backward([x], mse_derivative([y_pred], [y_true]))

    num_grad = numerical_gradient(neuron, x, y_true, "bias")
    assert np.isclose(grads["bias"], num_grad, atol=1e-6)


def test_point_neuron_all_weight_gradients():
    neuron = PointNeuron(input_dim=4, activation="linear", seed=5)
    x = [0.5, -1.0, 2.0, 0.25]
    y_true = 1.0

    grads = neuron.backward([x], mse_derivative([neuron.forward(x)], [y_true]))
    for i in range(4):
        num_grad = numerical_gradient(neuron, x, y_true, "weights", i)
        assert np.isclose(grads["weights"][i], num_grad, atol=1e-6), f"w{i} mismatch"


def test_point_neuron_gradient_with_nonlinear_activation():
    """Gradients must also be correct through a nonlinearity."""
    neuron = PointNeuron(input_dim=2, activation="tanh", seed=3)
    x = [1.0, -2.0]
    y_true = 0.5

    grads = neuron.backward([x], mse_derivative([neuron.forward(x)], [y_true]))
    for i in range(2):
        num_grad = numerical_gradient(neuron, x, y_true, "weights", i)
        assert np.isclose(grads["weights"][i], num_grad, atol=1e-6), f"w{i} mismatch"


# ---------- DendriticNeuron ----------


def test_dendritic_branch_gradient():
    neuron = DendriticNeuron(
        input_dim=2,
        num_branches=1,
        branch_input_map={0: [0, 1]},
        branch_activation="linear",
        soma_activation="linear",
        seed=1,
    )

    x = [1.0, 2.0]
    y_true = 1.0

    grads = neuron.backward([x], mse_derivative([neuron.forward_single(x)], [y_true]))
    num_grad = numerical_gradient(neuron, x, y_true, "branch_weights", (0, 0))

    assert np.isclose(grads["branch_weights"][0][0], num_grad, atol=1e-6)


def test_dendritic_all_parameters_gradient():
    """Every parameter of a multi-branch neuron must match finite differences."""
    neuron = DendriticNeuron(
        input_dim=4,
        num_branches=2,
        branch_input_map={0: [0, 1], 1: [2, 3]},
        branch_activation="tanh",
        soma_activation="linear",
        seed=7,
    )

    X = [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 0.25], [2.0, 0.0, -2.0, 1.0]]
    y = [0.0, 1.0, 0.5]
    grads = neuron.backward(X, mse_derivative(neuron.forward_batch(X), y))

    for b in range(2):
        for j in range(2):
            num_grad = batch_numerical_gradient(neuron, X, y, "branch_weights", (b, j))
            assert np.isclose(
                grads["branch_weights"][b][j], num_grad, atol=1e-5
            ), f"branch_weights[{b}][{j}] mismatch"

        num_grad = batch_numerical_gradient(neuron, X, y, "branch_biases", b)
        assert np.isclose(
            grads["branch_biases"][b], num_grad, atol=1e-5
        ), f"branch_biases[{b}] mismatch"

    for b in range(2):
        num_grad = batch_numerical_gradient(neuron, X, y, "soma_weights", b)
        assert np.isclose(
            grads["soma_weights"][b], num_grad, atol=1e-5
        ), f"soma_weights[{b}] mismatch"

    num_grad = batch_numerical_gradient(neuron, X, y, "soma_bias")
    assert np.isclose(grads["soma_bias"], num_grad, atol=1e-5), "soma_bias mismatch"


def test_dendritic_gradient_with_sigmoid_soma():
    """The non-linear soma path must produce correct gradients too."""
    neuron = DendriticNeuron(
        input_dim=2,
        num_branches=2,
        branch_input_map={0: [0], 1: [1]},
        branch_activation="linear",
        soma_activation="sigmoid",
        seed=11,
    )

    X = [[1.0, 2.0], [0.5, -1.0], [3.0, 0.25]]
    y = [0.0, 1.0, 0.5]
    grads = neuron.backward(X, mse_derivative(neuron.forward_batch(X), y))

    for b in range(2):
        num_grad = batch_numerical_gradient(neuron, X, y, "branch_weights", (b, 0))
        assert np.isclose(
            grads["branch_weights"][b][0], num_grad, atol=1e-5
        ), f"branch_weights[{b}][0] mismatch"


# ---------- Batch scaling regression test ----------


def test_batch_gradient_matches_mean_loss_gradient():
    """Regression test: gradients must not be divided by the batch size twice.

    `src.losses` returns the derivative of the *mean* loss (already divided by
    the batch size), so `backward` must sum the per-sample contributions rather
    than averaging them again. The old implementation divided by `batch_size`
    a second time, which scaled every gradient by `1/batch_size` and silently
    weakened the effective learning rate.
    """
    X = [[1.0, 2.0], [0.5, -1.0], [2.0, 0.0], [-1.0, 1.0], [3.0, 0.5]]
    y = [0.0, 1.0, 0.0, 1.0, 0.5]

    neuron = PointNeuron(input_dim=2, activation="linear", seed=2)
    grads = neuron.backward(X, mse_derivative(neuron.forward_batch(X), y))

    for i in range(2):
        num_grad = batch_numerical_gradient(neuron, X, y, "weights", i)
        assert np.isclose(
            grads["weights"][i], num_grad, atol=1e-5
        ), f"batch gradient for w{i} is scaled incorrectly"


def test_gradient_magnitude_is_independent_of_batch_size():
    """Duplicating a batch must double the summed gradient.

    The upstream gradient is held fixed at ``[1.0, 1.0]`` on purpose. If it
    came from ``mse_derivative`` the mean loss would halve each entry, which
    would mask whether ``backward`` sums or averages. Holding ``dL_dy`` fixed
    isolates the behaviour under test: ``backward`` must sum the per-sample
    contributions, so the total scales with the number of samples.
    """
    x = [1.0, 2.0]
    fixed_dL_dy = [1.0]

    neuron = PointNeuron(input_dim=2, activation="linear", seed=4)
    g_single = neuron.backward([x], fixed_dL_dy)["weights"][0]

    g_double = neuron.backward([x, x], [1.0, 1.0])["weights"][0]

    assert np.isclose(g_single, 1.0 * 1.0)  # dL_dy * activation' * x[0]
    assert np.isclose(g_double, 2 * g_single, rtol=1e-9)


def test_dendritic_batch_gradient_scales_with_batch_size():
    """Same scaling check for the dendritic neuron, with a fixed upstream gradient."""
    x = [1.0, 2.0, 3.0, 4.0]

    neuron = DendriticNeuron(
        input_dim=4,
        num_branches=2,
        branch_input_map={0: [0, 1], 1: [2, 3]},
        branch_activation="linear",
        soma_activation="linear",
        seed=9,
    )

    g_single = neuron.backward([x], [1.0])
    g_double = neuron.backward([x, x], [1.0, 1.0])

    assert np.isclose(
        g_double["soma_bias"], 2 * g_single["soma_bias"], rtol=1e-9
    ), "soma_bias must scale with the number of samples"

    for b in range(2):
        for j in range(2):
            assert np.isclose(
                g_double["branch_weights"][b][j],
                2 * g_single["branch_weights"][b][j],
                rtol=1e-9,
            ), f"branch_weights[{b}][{j}] scaling mismatch"


# ---------- Input validation on backward ----------


def test_backward_rejects_empty_batch():
    neuron = PointNeuron(input_dim=2, activation="linear")
    with pytest.raises(ValueError, match="empty batch"):
        neuron.backward([], [])


def test_backward_rejects_mismatched_gradient_length():
    neuron = PointNeuron(input_dim=2, activation="linear")
    with pytest.raises(ValueError, match="same length"):
        neuron.backward([[1.0, 2.0], [1.0, 2.0]], [1.0])


def test_dendritic_backward_rejects_empty_batch():
    neuron = DendriticNeuron(
        input_dim=2, num_branches=1, branch_input_map={0: [0, 1]}
    )
    with pytest.raises(ValueError, match="empty batch"):
        neuron.backward([], [])


def test_dendritic_backward_rejects_mismatched_gradient_length():
    neuron = DendriticNeuron(
        input_dim=2, num_branches=1, branch_input_map={0: [0, 1]}
    )
    with pytest.raises(ValueError, match="same length"):
        neuron.backward([[1.0, 2.0]], [1.0, 2.0])


# ---------- Gradient descent actually reduces the loss ----------


def test_apply_gradients_moves_towards_lower_loss():
    neuron = PointNeuron(input_dim=2, activation="linear", seed=6)
    X = [[1.0, 2.0], [0.5, -1.0], [3.0, 0.5]]
    y = [0.0, 1.0, 1.0]

    before = mse_loss(neuron.forward_batch(X), y)
    grads = neuron.backward(X, mse_derivative(neuron.forward_batch(X), y))
    neuron.apply_gradients(grads, learning_rate=0.1)
    after = mse_loss(neuron.forward_batch(X), y)

    assert after < before, "A single gradient step must reduce the loss"


def test_dendritic_apply_gradients_moves_towards_lower_loss():
    neuron = DendriticNeuron(
        input_dim=4,
        num_branches=2,
        branch_input_map={0: [0, 1], 1: [2, 3]},
        branch_activation="tanh",
        soma_activation="sigmoid",
        seed=8,
    )
    X = [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 0.25], [3.0, 0.5, -1.0, 2.0]]
    y = [0.0, 1.0, 1.0]

    before = mse_loss(neuron.forward_batch(X), y)
    grads = neuron.backward(X, mse_derivative(neuron.forward_batch(X), y))
    neuron.apply_gradients(grads, learning_rate=0.1)
    after = mse_loss(neuron.forward_batch(X), y)

    assert after < before, "A single gradient step must reduce the loss"
