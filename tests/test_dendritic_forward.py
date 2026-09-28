# Tests: catch bugs early, especially in forward/gradient math.
# This file checks:
# - Shapes -> are we returning what training expects?
# - Math correctness -> does the dendritic computation match hand-computed math?
# - Branch isolation -> do branches only "see" their assigned inputs?
# - Batch correctness -> is batch logic consistent with single-sample logic?
# - Input dimension mismatches -> do we fail loudly instead of silently?

import numpy as np
import pytest

from src.dendritic_neuron import DendriticNeuron


def make_neuron(
    branch_groups,
    input_dim,
    branch_activation="linear",
    soma_activation="linear",
    **kwargs,
):
    """Build a neuron from a list of branch groups."""
    return DendriticNeuron(
        input_dim=input_dim,
        num_branches=len(branch_groups),
        branch_input_map={i: grp for i, grp in enumerate(branch_groups)},
        branch_activation=branch_activation,
        soma_activation=soma_activation,
        **kwargs,
    )


# ---------- Shape and type checks ----------


def test_forward_shapes():
    neuron = make_neuron([[0, 1], [2, 3]], input_dim=4)

    # Single input: the forward pass must return a scalar.
    y = neuron.forward_single([1.0, 2.0, 3.0, 4.0])
    assert isinstance(y, float), f"Expected float, got {type(y)}"

    # Batch input: one output per row.
    y_batch = neuron.forward_batch([[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5]])
    assert len(y_batch) == 2, "forward_batch() must return one output per row"
    assert all(isinstance(v, float) for v in y_batch)


def test_forward_batch_empty_returns_empty():
    neuron = make_neuron([[0, 1], [2, 3]], input_dim=4)
    assert neuron.forward_batch([]) == []


# ---------- Math correctness ----------


def test_forward_known_values():
    neuron = make_neuron([[0, 1], [2, 3]], input_dim=4)

    # Manually set params
    neuron.branch_weights = [
        [1.0, 1.0],  # branch 0
        [1.0, 1.0],  # branch 1
    ]
    neuron.branch_biases = [0.0, 0.0]
    neuron.soma_weights = [1.0, 1.0]
    neuron.soma_bias = 0.0

    # Branch 0: (1 + 2) = 3
    # Branch 1: (3 + 4) = 7
    # Soma: 3 + 7 = 10
    y = neuron.forward_single([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(y, 10.0), f"Expected 10.0, got {y}"


def test_forward_known_values_with_biases():
    """Biases on branches and on the soma must all be applied."""
    neuron = make_neuron([[0, 1], [2, 3]], input_dim=4)

    neuron.branch_weights = [[2.0, 0.0], [0.0, 3.0]]
    neuron.branch_biases = [1.0, -1.0]
    neuron.soma_weights = [1.0, -1.0]
    neuron.soma_bias = 0.5

    # branch0 = 2*1 + 0*2 + 1 = 3 ; branch1 = 0*3 + 3*4 - 1 = 11
    # soma    = 3 - 11 + 0.5   = -7.5
    y = neuron.forward_single([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(y, -7.5), f"Expected -7.5, got {y}"


def test_branch_outputs_and_soma_input_are_cached():
    neuron = make_neuron([[0, 1], [2, 3]], input_dim=4)
    neuron.branch_weights = [[1.0, 0.0], [0.0, 1.0]]
    neuron.branch_biases = [0.0, 0.0]
    neuron.soma_weights = [1.0, 1.0]
    neuron.soma_bias = 0.0

    neuron.forward_single([1.0, 2.0, 3.0, 4.0])

    assert np.allclose(neuron.get_branch_outputs(), [1.0, 4.0])
    assert np.isclose(neuron.get_soma_input(), 5.0)


# ---------- Branch isolation ----------


def test_branch_isolation():
    # This catches a very common bug: accidentally using the full input vector
    # inside a branch.
    neuron = make_neuron([[0], [1], [2]], input_dim=3)

    neuron.branch_weights = [[1.0], [1.0], [1.0]]
    neuron.branch_biases = [0.0, 0.0, 0.0]
    neuron.soma_weights = [1.0, 1.0, 1.0]
    neuron.soma_bias = 0.0

    y1 = neuron.forward_single([1.0, 0.0, 0.0])
    y2 = neuron.forward_single([2.0, 0.0, 0.0])

    # Only branch 0 should contribute to the change.
    assert np.isclose(y2 - y1, 1.0), "Only branch 0 should change"


def test_branch_ignores_features_assigned_to_other_branches():
    """A branch must not be influenced by inputs outside its own group."""
    neuron = make_neuron([[0], [1]], input_dim=2)
    neuron.branch_weights = [[1.0], [1.0]]
    neuron.branch_biases = [0.0, 0.0]
    neuron.soma_weights = [1.0, 1.0]
    neuron.soma_bias = 0.0

    a = neuron.forward_single([1.0, 1.0])
    # Change only feature 0, which belongs to branch 0.
    b = neuron.forward_single([5.0, 1.0])
    assert np.isclose(b - a, 4.0)

    # Change only feature 1, which belongs to branch 1.
    c = neuron.forward_single([1.0, 7.0])
    assert np.isclose(c - a, 6.0)


# ---------- Batch correctness ----------


def test_batch_consistency():
    neuron = make_neuron([[0, 1], [2, 3]], input_dim=4)
    X = np.random.randn(5, 4)

    outputs_loop = [neuron.forward_single(x.tolist()) for x in X]
    outputs_batch = neuron.forward_batch(X.tolist())

    assert np.allclose(outputs_loop, outputs_batch)


# ---------- Input dimension mismatches ----------


def test_forward_single_rejects_wrong_dimension():
    neuron = make_neuron([[0, 1], [2, 3]], input_dim=4)

    with pytest.raises(ValueError, match="dimension 4"):
        neuron.forward_single([1.0, 2.0, 3.0])  # too short

    with pytest.raises(ValueError, match="dimension 4"):
        neuron.forward_single([1.0, 2.0, 3.0, 4.0, 5.0])  # too long


def test_forward_batch_rejects_wrong_dimension():
    neuron = make_neuron([[0, 1], [2, 3]], input_dim=4)

    with pytest.raises(ValueError, match="dimension 4"):
        neuron.forward_batch([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError, match="dimension 4"):
        neuron.forward_batch([[1.0, 2.0, 3.0, 4.0, 5.0]])


def test_forward_batch_rejects_ragged_rows():
    """A row whose length differs from the first row must not pass silently."""
    neuron = make_neuron([[0, 1], [2, 3]], input_dim=4)

    with pytest.raises(ValueError, match="dimension 4"):
        neuron.forward_batch([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0]])


# ---------- Edge-case branch configurations ----------


def test_single_branch_covering_all_features():
    """A one-branch neuron is structurally a point neuron."""
    neuron = make_neuron([[0, 1, 2]], input_dim=3)
    neuron.branch_weights = [[1.0, 1.0, 1.0]]
    neuron.branch_biases = [0.0]
    neuron.soma_weights = [2.0]
    neuron.soma_bias = 0.0

    assert np.isclose(neuron.forward_single([1.0, 2.0, 3.0]), 12.0)


def test_single_branch_single_feature():
    neuron = make_neuron([[0]], input_dim=1)
    neuron.branch_weights = [[3.0]]
    neuron.branch_biases = [1.0]
    neuron.soma_weights = [1.0]
    neuron.soma_bias = 0.0

    assert np.isclose(neuron.forward_single([2.0]), 7.0)


def test_many_single_feature_branches():
    """One branch per feature is the most fragmented configuration."""
    groups = [[0], [1], [2], [3], [4]]
    neuron = make_neuron(groups, input_dim=5)
    neuron.branch_weights = [[1.0]] * 5
    neuron.branch_biases = [0.0] * 5
    neuron.soma_weights = [1.0] * 5
    neuron.soma_bias = 0.0

    assert np.isclose(neuron.forward_single([1.0, 2.0, 3.0, 4.0, 5.0]), 15.0)


def test_branches_can_share_input_features():
    """Overlapping branch groups are legal: each branch still has its own weights."""
    neuron = make_neuron([[0, 1], [1, 2]], input_dim=3)
    neuron.branch_weights = [[1.0, 1.0], [10.0, 10.0]]
    neuron.branch_biases = [0.0, 0.0]
    neuron.soma_weights = [1.0, 1.0]
    neuron.soma_bias = 0.0

    # branch0 = 1*1 + 1*2 = 3 ; branch1 = 10*2 + 10*3 = 50
    assert np.isclose(neuron.forward_single([1.0, 2.0, 3.0]), 53.0)


def test_unused_features_do_not_affect_output():
    """Features not assigned to any branch must not influence the result."""
    neuron = make_neuron([[0]], input_dim=3)
    neuron.branch_weights = [[1.0]]
    neuron.branch_biases = [0.0]
    neuron.soma_weights = [1.0]
    neuron.soma_bias = 0.0

    a = neuron.forward_single([1.0, 0.0, 0.0])
    b = neuron.forward_single([1.0, 99.0, -99.0])
    assert np.isclose(a, b)


def test_branch_input_map_rejects_wrong_keys():
    """Keys must be exactly 0..num_branches-1.

    The original implementation only checked the number of entries, so a map
    like {0: [0, 1], 7: [2, 3]} passed construction and then raised a bare
    KeyError during the forward pass.
    """
    with pytest.raises(ValueError, match="keys must be exactly"):
        DendriticNeuron(
            input_dim=4,
            num_branches=2,
            branch_input_map={0: [0, 1], 7: [2, 3]},
        )

    with pytest.raises(ValueError, match="keys must be exactly"):
        DendriticNeuron(
            input_dim=4,
            num_branches=2,
            branch_input_map={1: [0, 1], 2: [2, 3]},
        )


def test_branch_input_map_rejects_out_of_range_indices():
    with pytest.raises(ValueError, match="out of range"):
        DendriticNeuron(
            input_dim=3,
            num_branches=1,
            branch_input_map={0: [0, 5]},
        )


def test_branch_input_map_rejects_negative_indices():
    """Negative indices would silently wrap around in plain Python indexing."""
    with pytest.raises(ValueError, match="out of range"):
        DendriticNeuron(
            input_dim=3,
            num_branches=1,
            branch_input_map={0: [-1]},
        )


def test_branch_input_map_rejects_empty_branch():
    with pytest.raises(ValueError, match="non-empty"):
        DendriticNeuron(
            input_dim=3,
            num_branches=2,
            branch_input_map={0: [0, 1], 1: []},
        )


def test_branch_input_map_rejects_wrong_entry_count():
    with pytest.raises(ValueError, match="exactly 2 entries"):
        DendriticNeuron(
            input_dim=4,
            num_branches=2,
            branch_input_map={0: [0, 1]},
        )


def test_branch_input_map_rejects_non_list_values():
    with pytest.raises(ValueError):
        DendriticNeuron(
            input_dim=3,
            num_branches=1,
            branch_input_map={0: (0, 1)},
        )


def test_constructor_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="input_dim must be positive"):
        DendriticNeuron(
            input_dim=0,
            num_branches=1,
            branch_input_map={0: [0]},
        )

    with pytest.raises(ValueError, match="num_branches must be positive"):
        DendriticNeuron(
            input_dim=3,
            num_branches=0,
            branch_input_map={},
        )


def test_constructor_rejects_unknown_activation():
    with pytest.raises(ValueError, match="Unsupported activation"):
        DendriticNeuron(
            input_dim=3,
            num_branches=1,
            branch_input_map={0: [0, 1]},
            soma_activation="not_an_activation",
        )


def test_constructor_accepts_mixed_case_activations():
    neuron = make_neuron([[0, 1]], input_dim=2, soma_activation="ReLU")
    assert neuron.soma_activation == "relu"


# ---------- Parameter round-tripping ----------


def test_get_parameters_returns_copies():
    neuron = make_neuron([[0, 1]], input_dim=2)
    params = neuron.get_parameters()
    params["branch_weights"][0][0] = 999.0
    assert neuron.branch_weights[0][0] != 999.0, "get_parameters must deep-copy"


def test_set_parameters_round_trip():
    neuron = make_neuron([[0, 1], [2, 3]], input_dim=4)
    before = neuron.get_parameters()
    neuron.set_parameters(before)
    after = neuron.get_parameters()

    for b in range(2):
        assert np.allclose(before["branch_weights"][b], after["branch_weights"][b])
    assert np.isclose(before["soma_bias"], after["soma_bias"])


def test_set_parameters_rejects_mismatched_shapes():
    neuron = make_neuron([[0, 1], [2, 3]], input_dim=4)

    with pytest.raises(ValueError):
        neuron.set_parameters(
            {
                "branch_weights": [[1.0, 1.0], [1.0]],  # second branch too short
                "branch_biases": [0.0, 0.0],
                "soma_weights": [1.0, 1.0],
                "soma_bias": 0.0,
            }
        )

    with pytest.raises(ValueError, match="missing required keys"):
        neuron.set_parameters({"branch_weights": [[1.0, 1.0], [1.0, 1.0]]})


def test_seed_is_reproducible_and_isolated():
    a = make_neuron([[0, 1], [2, 3]], input_dim=4, seed=123)
    b = make_neuron([[0, 1], [2, 3]], input_dim=4, seed=123)
    c = make_neuron([[0, 1], [2, 3]], input_dim=4, seed=124)

    assert np.allclose(a.branch_weights[0], b.branch_weights[0])
    assert np.allclose(a.soma_weights, b.soma_weights)
    assert not np.allclose(a.branch_weights[0], c.branch_weights[0])


def test_summary_mentions_structure():
    neuron = make_neuron([[0, 1], [2, 3]], input_dim=4)
    summary = neuron.summary()
    assert "DendriticNeuron" in summary
    assert "Branch 0" in summary
