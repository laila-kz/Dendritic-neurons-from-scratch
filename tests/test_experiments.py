"""Tests for the experiment layer.

These guard the properties the comparison depends on: both models must be
trained on the same split, and evaluation must rebuild each neuron with the
architecture it was actually trained with.
"""

import json
import os

import numpy as np
import pytest

from experiments.common import (
    BRANCH_ACTIVATION,
    INPUT_DIM,
    SOMA_ACTIVATION,
    build_branch_input_map,
    build_dataset,
    ensure_dirs,
    load_params,
    save_json,
    train_val_split,
)
from experiments.evaluate_neurons import (
    build_dendritic_neuron,
    build_point_neuron,
    evaluate,
)
from experiments.synthetic_dataset import (
    get_branch_index_groups,
    make_synthetic_data,
)
from experiments.train_dendritic import save_neuron_parameters
from src.losses import get_loss_function


# ---------- Dataset ----------


def test_branch_index_groups_default():
    assert get_branch_index_groups(6) == [[0, 1], [2, 3], [4, 5]]


def test_branch_index_groups_scales():
    assert get_branch_index_groups(4) == [[0, 1], [2, 3]]
    assert len(get_branch_index_groups(8)) == 4


def test_branch_index_groups_rejects_odd_input_dim():
    with pytest.raises(ValueError, match="multiple of 2"):
        get_branch_index_groups(5)


def test_make_synthetic_data_shape():
    X, y = make_synthetic_data(num_samples=50, seed=1)

    assert len(X) == 50
    assert len(y) == 50
    assert all(len(row) == INPUT_DIM for row in X)
    assert set(y) == {0, 1}
    assert sum(1 for v in y if v == 0) == sum(1 for v in y if v == 1)


def test_make_synthetic_data_is_reproducible():
    X1, y1 = make_synthetic_data(num_samples=30, seed=42)
    X2, y2 = make_synthetic_data(num_samples=30, seed=42)
    assert X1 == X2 and y1 == y2


def test_make_synthetic_data_differs_by_seed():
    X1, _ = make_synthetic_data(num_samples=30, seed=1)
    X2, _ = make_synthetic_data(num_samples=30, seed=2)
    assert X1 != X2


def test_make_synthetic_data_supports_wider_inputs():
    X, y = make_synthetic_data(num_samples=20, input_dim=8, seed=1)
    assert all(len(row) == 8 for row in X)


def test_make_synthetic_data_rejects_too_few_samples():
    with pytest.raises(ValueError, match="at least 2"):
        make_synthetic_data(num_samples=1)


# ---------- Shared split ----------


def test_split_is_stratified():
    X, y = make_synthetic_data(num_samples=200, seed=1)
    _, y_train, _, y_val = train_val_split(X, y, val_ratio=0.2, seed=1)

    total = y.count(0)
    assert y_train.count(0) == pytest.approx(total * 0.8, abs=2)
    assert y_val.count(0) == total - y_train.count(0)
    assert set(y_train) == {0, 1}
    assert set(y_val) == {0, 1}


def test_split_is_deterministic():
    X, y = make_synthetic_data(num_samples=100, seed=1)
    first = train_val_split(X, y, seed=1)
    second = train_val_split(X, y, seed=1)
    assert first == second


def test_split_covers_every_sample_exactly_once():
    X, y = make_synthetic_data(num_samples=60, seed=1)
    X_train, y_train, X_val, y_val = train_val_split(X, y, seed=1)

    assert len(X_train) + len(X_val) == len(X)
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)


def test_both_models_get_the_same_split():
    """The point of experiments/common.py: a single source of truth for data."""
    X, y = make_synthetic_data(num_samples=80, seed=1)
    split_a = train_val_split(X, y, seed=1)
    split_b = train_val_split(X, y, seed=1)
    assert split_a == split_b


# ---------- Architecture parity between training and evaluation ----------


def test_evaluation_dendritic_neuron_matches_training_activations():
    """Evaluation must not silently change the architecture.

    The original evaluate script rebuilt the neuron with branch_activation
    "sigmoid" while training used "tanh", so the saved weights were evaluated
    under a different model.
    """
    from experiments.train_dendritic import (
        BRANCH_ACTIVATION as TRAIN_BRANCH,
        SOMA_ACTIVATION as TRAIN_SOMA,
    )

    neuron = build_dendritic_neuron()
    assert neuron.branch_activation == TRAIN_BRANCH == BRANCH_ACTIVATION
    assert neuron.soma_activation == TRAIN_SOMA == SOMA_ACTIVATION


def test_evaluation_point_neuron_matches_training_activation():
    neuron = build_point_neuron()
    assert neuron.activation == SOMA_ACTIVATION


def test_build_branch_input_map_covers_all_features():
    branch_input_map = build_branch_input_map()
    covered = sorted(i for group in branch_input_map.values() for i in group)
    assert covered == list(range(INPUT_DIM))


# ---------- Parameter persistence ----------


def test_dendritic_parameters_round_trip_through_npz(tmp_path):
    from src.dendritic_neuron import DendriticNeuron

    neuron = DendriticNeuron(
        input_dim=INPUT_DIM,
        num_branches=3,
        branch_input_map=build_branch_input_map(),
        branch_activation=BRANCH_ACTIVATION,
        soma_activation=SOMA_ACTIVATION,
        seed=1,
    )

    path = os.path.join(str(tmp_path), "params.npz")
    save_neuron_parameters(neuron, path)

    restored = build_dendritic_neuron()
    restored.set_parameters(load_params(path))

    assert np.allclose(restored.branch_weights, neuron.branch_weights)
    assert np.allclose(restored.branch_biases, neuron.branch_biases)
    assert np.allclose(restored.soma_weights, neuron.soma_weights)
    assert np.isclose(restored.soma_bias, neuron.soma_bias)


def test_save_neuron_parameters_creates_the_parents_of_the_given_path(
    tmp_path, monkeypatch
):
    """Regression test: saving must not touch the repo's own results directory.

    ``save_neuron_parameters`` used to call ``ensure_dirs(MODEL_DIR)``,
    ignoring the path it was handed. That created stray directories in the
    working tree on every call, and would have raised FileNotFoundError for any
    destination whose parent directory did not already exist.
    """
    from src.dendritic_neuron import DendriticNeuron

    scratch_cwd = tmp_path / "cwd"
    scratch_cwd.mkdir()
    monkeypatch.chdir(scratch_cwd)

    neuron = DendriticNeuron(
        input_dim=INPUT_DIM,
        num_branches=3,
        branch_input_map=build_branch_input_map(),
        branch_activation=BRANCH_ACTIVATION,
        soma_activation=SOMA_ACTIVATION,
        seed=1,
    )

    nested = tmp_path / "does" / "not" / "exist" / "params.npz"
    save_neuron_parameters(neuron, str(nested))

    assert nested.is_file()
    assert list(scratch_cwd.iterdir()) == []


def test_saved_params_need_no_pickle(tmp_path):
    """The parameter file must be loadable with allow_pickle disabled.

    ``numpy.load(..., allow_pickle=True)`` executes arbitrary code embedded in
    the file, so the format deliberately avoids object arrays.
    """
    from src.dendritic_neuron import DendriticNeuron

    neuron = DendriticNeuron(
        input_dim=INPUT_DIM,
        num_branches=3,
        branch_input_map=build_branch_input_map(),
        branch_activation=BRANCH_ACTIVATION,
        soma_activation=SOMA_ACTIVATION,
        seed=1,
    )

    path = os.path.join(str(tmp_path), "params.npz")
    save_neuron_parameters(neuron, path)

    with np.load(path, allow_pickle=False) as data:
        assert "branch_weight_0" in data.files
        assert "branch_weights" not in data.files

    restored = build_dendritic_neuron()
    restored.set_parameters(load_params(path))
    assert np.allclose(restored.branch_weights, neuron.branch_weights)


def test_load_params_gives_a_helpful_error_when_missing(tmp_path):
    missing = os.path.join(str(tmp_path), "nope.npz")
    with pytest.raises(FileNotFoundError, match="train_dendritic"):
        load_params(missing)


def test_save_and_load_json(tmp_path):
    path = os.path.join(str(tmp_path), "nested", "out.json")
    save_json(path, {"a": 1, "b": [1, 2, 3]})

    with open(path, encoding="utf-8") as handle:
        assert json.load(handle) == {"a": 1, "b": [1, 2, 3]}


def test_ensure_dirs_is_idempotent(tmp_path):
    target = os.path.join(str(tmp_path), "a", "b")
    ensure_dirs(target)
    ensure_dirs(target)
    assert os.path.isdir(target)


# ---------- Evaluation helper ----------


def test_evaluate_reports_loss_and_accuracy():
    neuron = build_point_neuron()
    neuron.set_parameters(
        {"weights": [50.0] * INPUT_DIM, "bias": -75.0}
    )

    loss_fn, _ = get_loss_function("binary_cross_entropy")
    X = [[1.0] * INPUT_DIM, [0.0] * INPUT_DIM]

    metrics = evaluate(neuron, X, [1, 1], loss_fn)
    assert "loss" in metrics and "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
