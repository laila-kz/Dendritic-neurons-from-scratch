# Tests for the PointNeuron baseline: shapes, dimension mismatches,
# parameter round-tripping and the documented activation behaviour.

import numpy as np
import pytest

from src.point_neuron import PointNeuron


# ---------- Forward shapes ----------


def test_forward_returns_float():
    neuron = PointNeuron(input_dim=3, activation="linear")
    y = neuron.forward([1.0, 2.0, 3.0])
    assert isinstance(y, float)


def test_forward_known_value():
    neuron = PointNeuron(input_dim=2, activation="linear")
    neuron.weights = [2.0, 3.0]
    neuron.bias = 1.0

    # 2*1 + 3*2 + 1 = 9
    assert np.isclose(neuron.forward([1.0, 2.0]), 9.0)


def test_forward_batch_consistency():
    neuron = PointNeuron(input_dim=3, activation="tanh", seed=3)
    X = np.random.randn(6, 3)

    loop = [neuron.forward(x.tolist()) for x in X]
    batch = neuron.forward_batch(X.tolist())
    assert np.allclose(loop, batch)


def test_forward_batch_empty_returns_empty():
    neuron = PointNeuron(input_dim=3)
    assert neuron.forward_batch([]) == []


# ---------- Input dimension mismatches ----------


def test_forward_rejects_wrong_dimension():
    neuron = PointNeuron(input_dim=3, activation="linear")

    with pytest.raises(ValueError, match="dimension 3"):
        neuron.forward([1.0, 2.0])

    with pytest.raises(ValueError, match="dimension 3"):
        neuron.forward([1.0, 2.0, 3.0, 4.0])


def test_forward_batch_rejects_wrong_dimension():
    neuron = PointNeuron(input_dim=3, activation="linear")

    with pytest.raises(ValueError, match="dimension 3"):
        neuron.forward_batch([[1.0, 2.0]])

    with pytest.raises(ValueError, match="dimension 3"):
        neuron.forward_batch([[1.0, 2.0, 3.0, 4.0]])


# ---------- Constructor validation ----------


def test_rejects_non_positive_input_dim():
    with pytest.raises(ValueError, match="input_dim must be positive"):
        PointNeuron(input_dim=0)


def test_rejects_unknown_activation():
    with pytest.raises(ValueError, match="Unsupported activation"):
        PointNeuron(input_dim=3, activation="banana")


def test_rejects_non_string_activation():
    with pytest.raises(ValueError, match="must be a string"):
        PointNeuron(input_dim=3, activation=42)


def test_rejects_negative_init_scale():
    with pytest.raises(ValueError, match="init_scale must be non-negative"):
        PointNeuron(input_dim=3, init_scale=-1.0)


def test_accepts_identity_activation():
    """`identity` must be supported here too, matching the dendritic neuron."""
    neuron = PointNeuron(input_dim=2, activation="identity")
    neuron.weights = [1.0, 1.0]
    neuron.bias = 0.0
    assert np.isclose(neuron.forward([2.0, 3.0]), 5.0)


def test_accepts_mixed_case_activation():
    neuron = PointNeuron(input_dim=2, activation="SIGMOID")
    assert neuron.activation == "sigmoid"
    assert 0.0 <= neuron.forward([1.0, 1.0]) <= 1.0


# ---------- Parameters ----------


def test_get_parameters_returns_copies():
    neuron = PointNeuron(input_dim=3, seed=1)
    params = neuron.get_parameters()
    params["weights"][0] = 999.0
    assert neuron.weights[0] != 999.0


def test_set_parameters_rejects_wrong_dimension():
    neuron = PointNeuron(input_dim=3, activation="linear")

    with pytest.raises(ValueError, match="incorrect dimension"):
        neuron.set_parameters({"weights": [1.0, 2.0], "bias": 0.0})


def test_set_parameters_rejects_missing_keys():
    neuron = PointNeuron(input_dim=3, activation="linear")
    with pytest.raises(ValueError, match="must contain"):
        neuron.set_parameters({"weights": [1.0, 2.0, 3.0]})


def test_set_parameters_accepts_numpy_arrays():
    neuron = PointNeuron(input_dim=3, activation="linear")
    neuron.set_parameters(
        {"weights": np.array([1.0, 2.0, 3.0]), "bias": np.float64(0.5)}
    )
    assert neuron.weights == [1.0, 2.0, 3.0]
    assert isinstance(neuron.bias, float)
    assert np.isclose(neuron.forward([1.0, 1.0, 1.0]), 6.5)


def test_seed_is_reproducible_and_isolated():
    a = PointNeuron(input_dim=4, seed=99)
    b = PointNeuron(input_dim=4, seed=99)
    c = PointNeuron(input_dim=4, seed=100)

    assert a.weights == b.weights
    assert a.bias == b.bias
    assert a.weights != c.weights


def test_summary_mentions_configuration():
    summary = PointNeuron(input_dim=3, activation="tanh").summary()
    assert "PointNeuron" in summary
    assert "tanh" in summary
