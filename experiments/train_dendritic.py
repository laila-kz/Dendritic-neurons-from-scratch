"""Train a DendriticNeuron on the branch-structured synthetic dataset.

Creates the data, splits it, builds the neuron, trains it, and reports
performance. Parameters are written to ``results/models/dendritic_params.npz``
and the training history to ``results/logs/dendritic_history.json``.

Run with::

    python -m experiments.train_dendritic
"""

import os

import numpy as np

from experiments.common import (
    BATCH_SIZE,
    BRANCH_ACTIVATION,
    DENDRITIC_HISTORY_PATH,
    DENDRITIC_PARAMS_PATH,
    INPUT_DIM,
    LEARNING_RATE,
    LOG_DIR,
    LOSS_NAME,
    NUM_EPOCHS,
    NUM_SAMPLES,
    RANDOM_SEED,
    SOMA_ACTIVATION,
    build_branch_input_map,
    build_dataset,
    ensure_dirs,
    make_rng,
    save_json,
    train_val_split,
)
from src.dendritic_neuron import DendriticNeuron
from src.losses import get_loss_function
from src.training import train_model


def save_neuron_parameters(neuron: DendriticNeuron, path: str) -> None:
    """Persist the trained parameters to a ``.npz`` file.

    ``branch_weights`` is a ragged list-of-lists, which numpy cannot store as a
    rectangular array. Rather than fall back to an object array (which would
    force ``allow_pickle=True`` on load, and loading a pickled ``.npz`` can
    execute arbitrary code), each branch is stored under its own numeric key
    (``branch_weight_0``, ``branch_weight_1``, ...). ``load_params`` reassembles
    them into the list-of-lists that ``set_parameters`` expects.
    """
    ensure_dirs(os.path.dirname(os.path.abspath(path)))
    params = neuron.get_parameters()
    payload = {
        "num_branches": np.array(len(params["branch_weights"]), dtype=int),
        "branch_biases": np.array(params["branch_biases"], dtype=float),
        "soma_weights": np.array(params["soma_weights"], dtype=float),
        "soma_bias": np.array(params["soma_bias"], dtype=float),
    }
    for index, branch in enumerate(params["branch_weights"]):
        payload[f"branch_weight_{index}"] = np.array(branch, dtype=float)
    np.savez(path, **payload)


def main() -> None:
    print("=== Dendritic Neuron Training Experiment ===")

    X, y = build_dataset()

    print(f"Generated dataset with {len(X)} samples.")
    print(f"Class 0 count: {sum(1 for v in y if v == 0)}")
    print(f"Class 1 count: {sum(1 for v in y if v == 1)}")

    X_train, y_train, X_val, y_val = train_val_split(X, y)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Loss function and its derivative
    loss_fn, loss_deriv_fn = get_loss_function(LOSS_NAME)

    # Branch structure
    branch_input_map = build_branch_input_map()
    print("Branch structure:")
    for i, group in branch_input_map.items():
        print(f" Branch {i}: features {group}")

    neuron = DendriticNeuron(
        input_dim=INPUT_DIM,
        num_branches=len(branch_input_map),
        branch_input_map=branch_input_map,
        branch_activation=BRANCH_ACTIVATION,
        soma_activation=SOMA_ACTIVATION,
        seed=RANDOM_SEED,
    )
    print(neuron.summary())
    print("Initialized Dendritic Neuron model.")

    history, trained_neuron = train_model(
        neuron=neuron,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        loss_fn=loss_fn,
        loss_deriv_fn=loss_deriv_fn,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        rng=make_rng(RANDOM_SEED),
    )

    print("\n=== Final Results ===")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    print(f"Final val acc: {history['val_accuracy'][-1]:.4f}")

    save_neuron_parameters(trained_neuron, DENDRITIC_PARAMS_PATH)
    print(f"\nSaved dendritic params to {DENDRITIC_PARAMS_PATH}")

    ensure_dirs(LOG_DIR)
    save_json(DENDRITIC_HISTORY_PATH, history)
    print(f"Saved training history to {DENDRITIC_HISTORY_PATH}")


if __name__ == "__main__":
    main()
    print("\nDone.")
