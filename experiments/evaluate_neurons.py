# Evaluate trained dendritic and point neurons on a fresh test set

import os
import json
import numpy as np

from experiments.synthetic_dataset import make_synthetic_data, get_branch_index_grps
from src.dendritic_neuron import DendriticNeuron
from src.point_neuron import PointNeuron
from src.losses import get_loss_function

input_dim = 6
num_samples = 2000
activation = "sigmoid"
loss_fct = "binary_cross_entropy"

DENDRITIC_PARAMS_PATH = "results/models/dendritic_params.npz"
POINT_PARAMS_PATH = "results/models/point_neuron_params.npz"
SUMMARY_DIR = "results/summaries"

TEST_SEED = 999
NOISE_STD = 0.5


# ---------- Helpers ----------
def build_dendritic_neuron():
    branch_groups = get_branch_index_grps(input_dim)
    branch_input_map = {i: grp for i, grp in enumerate(branch_groups)}

    neuron = DendriticNeuron(
        input_dim=input_dim,
        num_branches=len(branch_groups),
        branch_input_map=branch_input_map,
        branch_activation=activation,
        soma_activation=activation,
    )
    return neuron


def build_point_neuron():
    neuron = PointNeuron(
        input_dim=input_dim,
        activation=activation,
    )
    return neuron


def load_parameters(neuron, path):
    params = dict(np.load(path, allow_pickle=True))
    neuron.set_parameters(params)
    return neuron


def evaluate(neuron, X, y, loss_fn):
    y_pred = neuron.forward_batch(X)
    loss = loss_fn(y_pred, y)

    y_pred_np = np.array(y_pred)
    preds = (y_pred_np >= 0.5).astype(int)
    accuracy = np.mean(preds == y)

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
    }


# ---------- Main ----------
def main():
    print("=== Evaluating Trained Neurons ===")

    # ----- Test dataset -----
    X_test, y_test = make_synthetic_data(
        num_samples=num_samples,
        input_dim=input_dim,
        seed=TEST_SEED,
    )

    loss_fn, _ = get_loss_function(loss_fct)

    # ----- Dendritic neuron -----
    dendritic = build_dendritic_neuron()
    dendritic = load_parameters(dendritic, DENDRITIC_PARAMS_PATH)
    dendritic_metrics = evaluate(dendritic, X_test, y_test, loss_fn)

    # ----- Point neuron -----
    point = build_point_neuron()
    point = load_parameters(point, POINT_PARAMS_PATH)
    point_metrics = evaluate(point, X_test, y_test, loss_fn)

    # ----- Noise robustness -----
    X_test_np = np.array(X_test)
    noise = np.random.normal(0.0, NOISE_STD, size=X_test_np.shape)
    X_noisy = X_test_np + noise
    X_noisy_list = X_noisy.tolist()

    dendritic_noisy = evaluate(dendritic, X_noisy_list, y_test, loss_fn)
    point_noisy = evaluate(point, X_noisy_list, y_test, loss_fn)

    dendritic_metrics["accuracy_noisy"] = dendritic_noisy["accuracy"]
    point_metrics["accuracy_noisy"] = point_noisy["accuracy"]

    # ----- Print results -----
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

    # ----- Save summary -----
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    summary = {
        "dendritic": dendritic_metrics,
        "point": point_metrics,
    }

    summary_path = os.path.join(SUMMARY_DIR, "comparison.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved comparison to {summary_path}")


# ---------- Entry point ----------
if __name__ == "__main__":
    main()
