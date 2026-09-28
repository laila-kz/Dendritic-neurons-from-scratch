"""Train a PointNeuron baseline on exactly the same data and hyperparameters.

Sharing the configuration in :mod:`experiments.common` guarantees the
comparison against the dendritic neuron is apples-to-apples: same dataset,
same split, same optimizer settings.

Run with::

    python -m experiments.train_point
"""

import numpy as np

from experiments.common import (
    ACTIVATION,
    BATCH_SIZE,
    INPUT_DIM,
    LEARNING_RATE,
    LOSS_NAME,
    MODEL_DIR,
    NUM_EPOCHS,
    NUM_SAMPLES,
    POINT_HISTORY_PATH,
    POINT_PARAMS_PATH,
    RANDOM_SEED,
    build_dataset,
    ensure_dirs,
    make_rng,
    save_json,
    train_val_split,
)
from src.losses import get_loss_function
from src.point_neuron import PointNeuron
from src.training import train_model


def main() -> None:
    print("=== Training Point Neuron ===")

    X, y = build_dataset()

    y_counts = np.bincount(np.asarray(y, dtype=int))
    print(f"Generated dataset with {len(X)} samples.")
    print(f"Class balance: {y_counts}")

    # Identical split to the dendritic run.
    X_train, y_train, X_val, y_val = train_val_split(X, y)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    loss_fn, loss_deriv_fn = get_loss_function(LOSS_NAME)

    neuron = PointNeuron(
        input_dim=INPUT_DIM,
        activation=ACTIVATION,
        seed=RANDOM_SEED,
    )
    print(neuron.summary())

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

    print("\n=== Final Point Neuron Metrics ===")
    print(f"Train loss: {history['train_loss'][-1]:.6f}")
    print(f"Val loss: {history['val_loss'][-1]:.6f}")
    print(f"Val acc: {history['val_accuracy'][-1]:.4f}")

    ensure_dirs(MODEL_DIR)
    np.savez(POINT_PARAMS_PATH, **trained_neuron.get_parameters())
    print(f"\nSaved model params to {POINT_PARAMS_PATH}")

    save_json(POINT_HISTORY_PATH, history)
    print(f"Saved training history to {POINT_HISTORY_PATH}")


if __name__ == "__main__":
    main()
