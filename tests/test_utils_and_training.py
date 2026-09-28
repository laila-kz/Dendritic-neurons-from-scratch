# Tests for the generic helpers and the training loop.
#
# These are the pieces the experiments rely on, so a bug here changes every
# reported number without necessarily raising.

import math
import random

import numpy as np
import pytest

from src.dendritic_neuron import DendriticNeuron
from src.losses import mse_derivative, mse_loss
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


# ---------- init_weights / init_bias ----------


def test_init_weights_shape_and_range():
    weights = init_weights((5,), scale=0.1, seed=1)
    assert len(weights) == 5
    assert all(-0.1 <= w <= 0.1 for w in weights)


def test_init_weights_is_reproducible():
    assert init_weights((4,), seed=7) == init_weights((4,), seed=7)


def test_init_weights_does_not_touch_global_random():
    """Seeding here must not disturb the caller's RNG stream."""
    random.seed(0)
    expected = [random.random() for _ in range(3)]

    random.seed(0)
    init_weights((3,), seed=123)
    actual = [random.random() for _ in range(3)]

    assert actual == expected


def test_init_weights_rejects_multidimensional_shape():
    with pytest.raises(ValueError, match="1D"):
        init_weights((2, 3))


def test_init_weights_rejects_negative_scale():
    with pytest.raises(ValueError, match="non-negative"):
        init_weights((3,), scale=-1.0)


def test_init_bias_returns_float():
    assert init_bias(0.5) == 0.5
    assert isinstance(init_bias(), float)


# ---------- train_val_split ----------


def test_train_val_split_sizes():
    X = [[float(i)] for i in range(100)]
    y = [i % 2 for i in range(100)]

    X_train, y_train, X_val, y_val = train_val_split(X, y, val_ratio=0.2, seed=1)

    assert len(X_train) == 80
    assert len(X_val) == 20
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)


def test_train_val_split_is_disjoint_and_complete():
    X = [[float(i)] for i in range(50)]
    y = [i % 2 for i in range(50)]

    X_train, y_train, X_val, y_val = train_val_split(X, y, seed=3)

    train_rows = {tuple(x) for x in X_train}
    val_rows = {tuple(x) for x in X_val}
    assert not (train_rows & val_rows)
    assert train_rows | val_rows == {tuple(x) for x in X}


def test_train_val_split_is_reproducible():
    X = [[float(i)] for i in range(30)]
    y = [i % 2 for i in range(30)]

    first = train_val_split(X, y, seed=5)
    second = train_val_split(X, y, seed=5)
    assert first == second


def test_train_val_split_keeps_both_splits_non_empty_for_tiny_data():
    """A small dataset must not produce an empty train or validation split."""
    X = [[1.0], [2.0], [3.0]]
    y = [0, 1, 0]

    X_train, y_train, X_val, y_val = train_val_split(X, y, val_ratio=0.2, seed=1)

    assert len(X_train) > 0
    assert len(X_val) > 0


def test_train_val_split_rejects_bad_inputs():
    X = [[1.0], [2.0]]
    y = [0, 1]

    with pytest.raises(ValueError, match="same length"):
        train_val_split(X, [0])

    with pytest.raises(ValueError, match="val_ratio"):
        train_val_split(X, y, val_ratio=1.5)

    with pytest.raises(ValueError, match="val_ratio"):
        train_val_split(X, y, val_ratio=0.0)


# ---------- batch_iterator ----------


def test_batch_iterator_covers_all_data():
    X = [[float(i)] for i in range(10)]
    y = list(range(10))

    batches = list(batch_iterator(X, y, batch_size=3, shuffle=False))
    assert [len(bx) for bx, _ in batches] == [3, 3, 3, 1]
    assert sum(len(bx) for bx, _ in batches) == 10


def test_batch_iterator_pairs_are_aligned():
    X = [[float(i)] for i in range(10)]
    y = list(range(10))

    for bx, by in batch_iterator(X, y, batch_size=4, shuffle=False):
        assert [row[0] for row in bx] == list(by)


def test_batch_iterator_is_reproducible():
    X = [[float(i)] for i in range(20)]
    y = list(range(20))

    first = [b for b, _ in batch_iterator(X, y, 5, seed=11)]
    second = [b for b, _ in batch_iterator(X, y, 5, seed=11)]
    assert first == second


def test_batch_iterator_rejects_bad_inputs():
    X = [[1.0], [2.0]]
    y = [0, 1]

    with pytest.raises(ValueError, match="same length"):
        list(batch_iterator(X, [0], 2))

    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(batch_iterator(X, y, 0))


# ---------- numerical helpers ----------


def test_clip_values():
    assert clip_values(5.0, 0.0, 1.0) == 1.0
    assert clip_values(-5.0, 0.0, 1.0) == 0.0
    assert clip_values(0.5, 0.0, 1.0) == 0.5


def test_clip_values_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="min_val"):
        clip_values(0.5, 1.0, 0.0)


def test_sigmoid_safe_clips_away_from_boundaries():
    assert sigmoid_safe(-1000.0) == pytest.approx(1e-8)
    assert sigmoid_safe(1000.0) == pytest.approx(1 - 1e-8)
    assert np.isfinite(sigmoid_safe(-800.0))


def test_sigmoid_safe_rejects_bad_eps():
    with pytest.raises(ValueError, match="eps"):
        sigmoid_safe(1.0, eps=0.0)


# ---------- Training loop ----------


def _tiny_problem():
    X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.5], [0.5, 2.0], [1.5, 1.5]]
    y = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    return X, y


def test_train_one_epoch_reduces_loss():
    X, y = _tiny_problem()
    neuron = PointNeuron(input_dim=2, activation="linear", seed=1)

    before = mse_loss(neuron.forward_batch(X), y)
    train_one_epoch(
        neuron,
        X,
        y,
        mse_loss,
        mse_derivative,
        learning_rate=0.1,
        batch_size=3,
    )
    after = mse_loss(neuron.forward_batch(X), y)
    assert after < before


def test_train_one_epoch_averages_over_examples_not_batches():
    """A smaller trailing batch must not be over-weighted in the reported loss."""
    X = [[1.0], [1.0], [1.0], [1.0], [1.0]]  # 5 examples
    y = [0.0] * 5

    neuron = PointNeuron(input_dim=1, activation="linear", seed=2)
    # batch_size=3 -> batches of 3 and 2. Both batches see identical data, so
    # the example-weighted mean must equal the plain mean of predictions.
    reported = train_one_epoch(
        neuron,
        X,
        y,
        mse_loss,
        mse_derivative,
        learning_rate=0.0,  # no updates, so the loss is a pure function of X
        batch_size=3,
    )

    expected = mse_loss(neuron.forward_batch(X), y)
    assert math.isclose(reported, expected, rel_tol=1e-12)


def test_train_one_epoch_rejects_bad_inputs():
    X, y = _tiny_problem()
    neuron = PointNeuron(input_dim=2, activation="linear")

    with pytest.raises(ValueError, match="same length"):
        train_one_epoch(neuron, X, y[:2], mse_loss, mse_derivative, 0.1, 2)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        train_one_epoch(neuron, X, y, mse_loss, mse_derivative, 0.1, 0)

    with pytest.raises(ValueError, match="must not be empty"):
        train_one_epoch(neuron, [], [], mse_loss, mse_derivative, 0.1, 2)


def test_train_model_records_history():
    X, y = _tiny_problem()
    neuron = PointNeuron(input_dim=2, activation="linear", seed=3)

    history, trained = train_model(
        neuron=neuron,
        X_train=X,
        y_train=y,
        X_val=X,
        y_val=y,
        loss_fn=mse_loss,
        loss_deriv_fn=mse_derivative,
        learning_rate=0.1,
        batch_size=3,
        num_epochs=4,
        verbose=False,
    )

    assert trained is neuron
    assert len(history["train_loss"]) == 4
    assert len(history["val_loss"]) == 4
    assert len(history["val_accuracy"]) == 4
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_train_model_rejects_bad_inputs():
    X, y = _tiny_problem()
    neuron = PointNeuron(input_dim=2, activation="linear")

    with pytest.raises(ValueError, match="num_epochs must be positive"):
        train_model(neuron, X, y, X, y, mse_loss, mse_derivative, 0.1, 2, 0, verbose=False)

    with pytest.raises(ValueError, match="must not be empty"):
        train_model(neuron, X, y, [], [], mse_loss, mse_derivative, 0.1, 2, 1, verbose=False)


def _separating_neuron(input_dim=2):
    """A neuron that predicts 1 when *both* features are on, 0 otherwise.

    This matches the label pattern of :func:`_tiny_problem`, which is not
    linearly separable by a single hyperplane but is separable by "both
    features present".
    """
    neuron = PointNeuron(input_dim=input_dim, activation="sigmoid")
    neuron.weights = [50.0] * input_dim
    neuron.bias = -75.0
    return neuron


def test_evaluate_model_reports_binary_accuracy():
    X, y = _tiny_problem()
    metrics = evaluate_model(_separating_neuron(), X, y, mse_loss)
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 1.0


def test_evaluate_model_handles_numpy_targets():
    """`if y` used to raise for numpy arrays; accuracy must still be reported."""
    X, y = _tiny_problem()
    metrics = evaluate_model(_separating_neuron(), X, np.asarray(y), mse_loss)
    assert metrics["accuracy"] == 1.0


def test_evaluate_model_omits_accuracy_for_continuous_targets():
    X = [[1.0], [2.0]]
    y = [0.25, 0.75]
    neuron = PointNeuron(input_dim=1, activation="linear")

    metrics = evaluate_model(neuron, X, y, mse_loss)
    assert "accuracy" not in metrics


def test_evaluate_model_rejects_bad_inputs():
    neuron = PointNeuron(input_dim=1, activation="linear")

    with pytest.raises(ValueError, match="same length"):
        evaluate_model(neuron, [[1.0], [2.0]], [0.0], mse_loss)

    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_model(neuron, [], [], mse_loss)


# ---------- End-to-end smoke test ----------


def test_dendritic_neuron_can_learn_a_simple_task():
    """A short training run must actually reduce the loss."""
    X, y = _tiny_problem()
    neuron = DendriticNeuron(
        input_dim=2,
        num_branches=2,
        branch_input_map={0: [0], 1: [1]},
        branch_activation="tanh",
        soma_activation="linear",
        seed=6,
    )

    before = mse_loss(neuron.forward_batch(X), y)
    for _ in range(200):
        grads = neuron.backward(X, mse_derivative(neuron.forward_batch(X), y))
        neuron.apply_gradients(grads, learning_rate=0.05)
    after = mse_loss(neuron.forward_batch(X), y)

    assert after < before
