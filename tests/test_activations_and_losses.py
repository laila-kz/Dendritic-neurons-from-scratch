# Tests for activation functions and loss functions.
#
# Activations matter because both neurons resolve their nonlinearity through
# this module; losses matter because their derivatives drive every weight
# update, so a wrong derivative silently produces a model that does not train.

import math

import numpy as np
import pytest

from src.activations import (
    ACTIVATION_NAMES,
    get_activation,
    get_activation_derivative,
    get_activation_pair,
    sigmoid,
)
from src.losses import (
    binary_cross_entropy_derivative,
    binary_cross_entropy_loss,
    get_loss_function,
    mse_derivative,
    mse_loss,
)


# ---------- Activations ----------


@pytest.mark.parametrize("name", ACTIVATION_NAMES)
def test_every_activation_has_a_derivative(name):
    assert callable(get_activation(name))
    assert callable(get_activation_derivative(name))


@pytest.mark.parametrize("name", ACTIVATION_NAMES)
def test_activation_derivatives_match_finite_differences(name):
    """d/dx f(x) must agree with a central finite difference."""
    f = get_activation(name)
    df = get_activation_derivative(name)

    # Avoid the non-differentiable point of ReLU.
    for x in (-1.7, -0.4, 0.6, 2.3):
        eps = 1e-6
        numeric = (f(x + eps) - f(x - eps)) / (2 * eps)
        assert np.isclose(df(x), numeric, atol=1e-5), f"{name} at x={x}"


def test_sigmoid_is_numerically_stable():
    """The naive formula overflows for large negative inputs."""
    assert np.isfinite(sigmoid(-1000.0))
    assert np.isfinite(sigmoid(1000.0))
    assert np.isclose(sigmoid(-1000.0), 0.0)
    assert np.isclose(sigmoid(1000.0), 1.0)
    assert np.isclose(sigmoid(0.0), 0.5)


def test_relu_subgradient_at_zero_is_zero():
    assert get_activation_derivative("relu")(0.0) == 0.0


def test_activation_lookup_is_case_insensitive():
    assert get_activation("ReLU")(1.0) == get_activation("relu")(1.0)
    assert get_activation_derivative("  TANH ")(0.5) == get_activation_derivative(
        "tanh"
    )(0.5)


def test_unknown_activation_raises_with_helpful_message():
    with pytest.raises(ValueError, match="Unsupported activation"):
        get_activation("nope")

    with pytest.raises(ValueError, match="must be a string"):
        get_activation(3)


def test_get_activation_pair_returns_matching_functions():
    f, df = get_activation_pair("sigmoid")
    assert np.isclose(f(0.0), 0.5)
    assert np.isclose(df(0.0), 0.25)


def test_identity_and_linear_are_equivalent():
    assert np.isclose(get_activation("identity")(3.0), get_activation("linear")(3.0))
    assert np.isclose(
        get_activation_derivative("identity")(3.0),
        get_activation_derivative("linear")(3.0),
    )


# ---------- Losses ----------


def test_mse_loss_known_value():
    assert np.isclose(mse_loss([1.0, 2.0], [0.0, 0.0]), 2.5)


def test_mse_derivative_matches_finite_differences():
    y_pred = [0.3, 0.8, -0.4]
    y_true = [0.5, 0.1, 0.0]

    analytic = mse_derivative(y_pred, y_true)
    eps = 1e-6
    for i in range(len(y_pred)):
        plus = list(y_pred)
        plus[i] += eps
        minus = list(y_pred)
        minus[i] -= eps
        numeric = (mse_loss(plus, y_true) - mse_loss(minus, y_true)) / (2 * eps)
        assert np.isclose(analytic[i], numeric, atol=1e-6)


def test_mse_derivative_matches_the_conventional_form():
    """2*(y_pred - y_true)/N for a single sample."""
    assert np.isclose(mse_derivative([2.0], [1.0])[0], 2.0)


def test_bce_loss_known_value():
    # -(1*log(0.5) + 0*log(0.5)) = log(2)
    assert np.isclose(binary_cross_entropy_loss([0.5], [1.0]), math.log(2))


def test_bce_loss_matches_formula():
    y_pred = [0.2, 0.8]
    y_true = [1.0, 0.0]
    expected = -(
        1.0 * math.log(0.2) + 0.0 * math.log(0.8)
        + 0.0 * math.log(0.8) + 1.0 * math.log(0.2)
    ) / 2
    assert np.isclose(binary_cross_entropy_loss(y_pred, y_true), expected)


def test_bce_derivative_matches_finite_differences():
    y_pred = [0.3, 0.7, 0.45]
    y_true = [1.0, 0.0, 1.0]

    analytic = binary_cross_entropy_derivative(y_pred, y_true)
    eps = 1e-6
    for i in range(len(y_pred)):
        plus = list(y_pred)
        plus[i] += eps
        minus = list(y_pred)
        minus[i] -= eps
        numeric = (
            binary_cross_entropy_loss(plus, y_true)
            - binary_cross_entropy_loss(minus, y_true)
        ) / (2 * eps)
        assert np.isclose(analytic[i], numeric, atol=1e-5), f"index {i}"


def test_bce_derivative_does_not_divide_by_zero():
    """Saturated predictions must not produce inf or nan."""
    result = binary_cross_entropy_derivative([1e-12, 1 - 1e-12], [1.0, 0.0])
    assert all(np.isfinite(v) for v in result)


def test_losses_reject_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        mse_loss([1.0, 2.0], [1.0])

    with pytest.raises(ValueError, match="same length"):
        binary_cross_entropy_loss([1.0, 0.5], [1.0])


def test_losses_reject_empty_input():
    with pytest.raises(ValueError, match="must not be empty"):
        mse_loss([], [])

    with pytest.raises(ValueError, match="must not be empty"):
        binary_cross_entropy_loss([], [])


def test_get_loss_function_returns_pair():
    loss, deriv = get_loss_function("mse")
    assert loss is mse_loss and deriv is mse_derivative

    loss, deriv = get_loss_function("bce")
    assert loss is binary_cross_entropy_loss
    assert deriv is binary_cross_entropy_derivative


def test_get_loss_function_accepts_aliases_and_case():
    assert get_loss_function("BCE")[0] is binary_cross_entropy_loss
    assert get_loss_function("Binary_Cross_Entropy")[0] is binary_cross_entropy_loss


def test_get_loss_function_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported loss function"):
        get_loss_function("hinge")
