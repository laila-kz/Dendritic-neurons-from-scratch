"""Evaluate the trained dendritic and point neurons on a fresh test set.

Also measures robustness to additive Gaussian noise on the inputs.

Run with::

    python -m experiments.evaluate_neurons
"""

import numpy as np

from experiments.common import (
    BRANCH_ACTIVATION,
    COMPARISON_PATH,
    DENDRITIC_PARAMS_PATH,
    INPUT_DIM,
    LOSS_NAME,
    NOISE_SEED,
    NOISE_STD,
    NUM_SAMPLES,
    POINT_PARAMS_PATH,
    RANDOM_SEED,
    SOMA_ACTIVATION,
    SUMMARY_DIR,
    TEST_SEED,
    build_branch_input_map,
    build_dataset,
    ensure_dirs,
    load_params,
    save_json,
)
from src.dendritic_neuron import DendriticNeuron
from src.losses import get_loss_function
from src.point_neuron import PointNeuron


# ---------- Builders ----------


def build_dendritic_neuron() -> DendriticNeuron:
    """Rebuild the dendritic neuron with the same architecture used for training.

    The activations must match ``experiments.train_dendritic`` exactly. The
    previous version rebuilt the neuron with ``branch_activation="sigmoid"``
    while training used ``"tanh"``, so the saved weights were being evaluated
    under a different architecture than they were trained with, which produced
    meaningless accuracy numbers.
    """
    branch_input_map = build_branch_input_map()
    return DendriticNeuron(
        input_dim=INPUT_DIM,
        num_branches=len(branch_input_map),
        branch_input_map=branch_input_map,
        branch_activation=BRANCH_ACTIVATION,
        soma_activation=SOMA_ACTIVATION,
        seed=RANDOM_SEED,
    )


def build_point_neuron() -> PointNeuron:
    """Rebuild the point neuron with the same configuration used for training."""
    return PointNeuron(
        input_dim=INPUT_DIM,
        activation=SOMA_ACTIVATION,
        seed=RANDOM_SEED,
    )


def load_parameters(neuron, path: str):
    """Load saved parameters into a neuron instance."""
    neuron.set_parameters(load_params(path))
    return neuron


# ---------- Evaluation ----------


def evaluate(neuron, X, y, loss_fn) -> dict:
    """Compute loss and accuracy for a neuron on a dataset."""
    y_pred = neuron.forward_batch(X)
    loss = loss_fn(y_pred, y)

    y_pred_np = np.asarray(y_pred, dtype=float)
    preds = (y_pred_np >= 0.5).astype(int)
    accuracy = float(np.mean(preds == np.asarray(y)))

    return {
        "loss": float(loss),
        "accuracy": accuracy,
    }


def main() -> None:
    print("=== Evaluating Trained Neurons ===")

    X_test, y_test = build_dataset(num_samples=NUM_SAMPLES, seed=TEST_SEED)
    print(f"Generated test set with {len(X_test)} samples.")

    loss_fn, _ = get_loss_function(LOSS_NAME)

    dendritic = load_parameters(build_dendritic_neuron(), DENDRITIC_PARAMS_PATH)
    dendritic_metrics = evaluate(dendritic, X_test, y_test, loss_fn)

    point = load_parameters(build_point_neuron(), POINT_PARAMS_PATH)
    point_metrics = evaluate(point, X_test, y_test, loss_fn)

    # Noise robustness, with a fixed seed so the numbers are reproducible.
    X_test_np = np.asarray(X_test, dtype=float)
    noise = np.random.default_rng(NOISE_SEED).normal(
        0.0, NOISE_STD, size=X_test_np.shape
    )
    X_noisy = (X_test_np + noise).tolist()

    dendritic_noisy = evaluate(dendritic, X_noisy, y_test, loss_fn)
    point_noisy = evaluate(point, X_noisy, y_test, loss_fn)

    for metrics, noisy in (
        (dendritic_metrics, dendritic_noisy),
        (point_metrics, point_noisy),
    ):
        metrics["loss_noisy"] = noisy["loss"]
        metrics["accuracy_noisy"] = noisy["accuracy"]

    print("\n=== Test Results ===")
    print(
        f"Dendritic neuron: loss={dendritic_metrics['loss']:.6f}, "
        f"acc={dendritic_metrics['accuracy']:.4f}, "
        f"acc_noisy={dendritic_metrics['accuracy_noisy']:.4f}"
    )
    print(
        f"Point neuron: loss={point_metrics['loss']:.6f}, "
        f"acc={point_metrics['accuracy']:.4f}, "
        f"acc_noisy={point_metrics['accuracy_noisy']:.4f}"
    )

    ensure_dirs(SUMMARY_DIR)
    summary = {
        "dendritic": dendritic_metrics,
        "point": point_metrics,
    }
    save_json(COMPARISON_PATH, summary)
    print(f"\nSaved comparison to {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
