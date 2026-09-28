"""Shared configuration and helpers for the experiment scripts.

The dendritic and point-neuron experiments must be directly comparable, so
every hyperparameter, the dataset, and the train/validation split are defined
exactly once here and imported by both training scripts.

The original code built the dendritic split with
``sklearn.model_selection.train_test_split(stratify=y)`` while the point run
used ``src.utils.train_val_split`` (unstratified, a different shuffle). The two
models were therefore trained on different data, which made the headline
comparison in ``evaluate_neurons.py`` meaningless.
"""

import json
import os
import random
from typing import Any, Dict, List, Tuple

from experiments.synthetic_dataset import get_branch_index_groups, make_synthetic_data

# ---------- Experiment hyperparameters ----------

INPUT_DIM = 6
NUM_SAMPLES = 2000
VAL_RATIO = 0.2
LEARNING_RATE = 0.01
BATCH_SIZE = 32
NUM_EPOCHS = 50
RANDOM_SEED = 42

BRANCH_ACTIVATION = "tanh"
SOMA_ACTIVATION = "sigmoid"
ACTIVATION = SOMA_ACTIVATION
LOSS_NAME = "binary_cross_entropy"

# ---------- Output locations ----------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summaries")

DENDRITIC_PARAMS_PATH = os.path.join(MODEL_DIR, "dendritic_params.npz")
POINT_PARAMS_PATH = os.path.join(MODEL_DIR, "point_neuron_params.npz")
DENDRITIC_HISTORY_PATH = os.path.join(LOG_DIR, "dendritic_history.json")
POINT_HISTORY_PATH = os.path.join(LOG_DIR, "point_neuron_history.json")
COMPARISON_PATH = os.path.join(SUMMARY_DIR, "comparison.json")

# ---------- Evaluation settings ----------

#: Seed for the held-out test set. Distinct from RANDOM_SEED so the test data
#: is not the data the models were trained on.
TEST_SEED = 999
#: Standard deviation of the Gaussian noise added for the robustness check.
NOISE_STD = 0.5
#: Seed for that noise, so the reported noisy accuracy is reproducible.
NOISE_SEED = 1234


# ---------- Data / split helpers ----------


def build_branch_input_map() -> Dict[int, List[int]]:
    """Branch index -> input feature indices for the default dataset layout.

    Defined here so that training and evaluation cannot disagree about the
    branch structure.
    """
    return {
        index: list(group)
        for index, group in enumerate(get_branch_index_groups(INPUT_DIM))
    }


def build_dataset(
    num_samples: int = NUM_SAMPLES,
    input_dim: int = INPUT_DIM,
    noise_std: float = 0.4,
    seed: int = RANDOM_SEED,
) -> Tuple[List[List[float]], List[int]]:
    """Generate the synthetic dataset used by every experiment."""
    return make_synthetic_data(
        num_samples=num_samples,
        input_dim=input_dim,
        noise_std=noise_std,
        seed=seed,
    )


def train_val_split(
    X: List[List[float]],
    y: List[int],
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[List[List[float]], List[int], List[List[float]], List[int]]:
    """Deterministic, stratified train/validation split.

    Splitting per class keeps the class balance of the full dataset in both
    splits, which matters for the small-sample runs where an unstratified
    shuffle can leave a class with no validation examples. Deterministic so
    repeated runs are reproducible.
    """
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length, got {len(X)} and {len(y)}"
        )
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    rng = make_rng(seed)

    train_idx: List[int] = []
    val_idx: List[int] = []

    # Group indices by label and split each group independently.
    by_label: Dict[Any, List[int]] = {}
    for idx, label in enumerate(y):
        by_label.setdefault(label, []).append(idx)

    for label in sorted(by_label, key=repr):
        indices = list(by_label[label])
        rng.shuffle(indices)

        n_label = len(indices)
        n_val = int(round(n_label * val_ratio))
        # Give every non-trivial class at least one validation example, while
        # still leaving at least one training example when possible.
        if n_label >= 2:
            n_val = max(1, min(n_val, n_label - 1))

        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return (
        [X[i] for i in train_idx],
        [y[i] for i in train_idx],
        [X[i] for i in val_idx],
        [y[i] for i in val_idx],
    )


# ---------- IO helpers ----------


def ensure_dirs(*paths: str) -> None:
    """Create the given directories if they do not exist yet."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` as JSON, creating parent directories."""
    ensure_dirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_params(path: str) -> Dict[str, Any]:
    """Load a saved parameter file.

    The dendritic parameter file stores one numeric array per branch
    (``branch_weight_0``, ``branch_weight_1``, ...) so that no pickled object
    array is needed; ``allow_pickle`` stays off, which keeps loading safe and
    portable. The per-branch keys are reassembled here into the
    ``branch_weights`` list-of-lists that ``set_parameters`` expects. Point
    neuron files contain only plain numeric keys and pass through unchanged.
    """
    import numpy as np

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing model parameters at '{path}'. "
            f"Run the training scripts first, e.g. "
            f"'python -m experiments.train_dendritic' and "
            f"'python -m experiments.train_point'."
        )

    with np.load(path) as data:
        params = {key: data[key] for key in data.files}

    branch_keys = [key for key in params if key.startswith("branch_weight_")]
    if branch_keys:
        branch_keys.sort(key=lambda key: int(key.rsplit("_", 1)[1]))
        params["branch_weights"] = [
            np.asarray(params.pop(key)).tolist() for key in branch_keys
        ]
    params.pop("num_branches", None)
    return params


def make_rng(seed: int = RANDOM_SEED) -> random.Random:
    """Return a seeded RNG for reproducible shuffling."""
    return random.Random(seed)
