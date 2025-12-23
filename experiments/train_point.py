# Train a PointNeuron on exactly the same dataset and with the same training hyperparameters
# as the dendritic neuron, so results are comparable

import os
import json
import numpy as np

from experiments.synthetic_dataset import make_synthetic_data
from src.point_neuron import PointNeuron
from src.losses import get_loss_function
from src.training import train_model
from src.utils import train_val_split

input_dim = 6
num_samples = 2000
val_ratio = 0.2
learning_rate = 0.01
batch_size = 32
num_epochs = 50

activation = "sigmoid"
loss_fct = "binary_cross_entropy"
random_seed = 42

model_dir = "results/models"
log_dir = "results/logs"


def main():
    print("=== Training Point Neuron ===")

    X, y = make_synthetic_data(
        num_samples=num_samples,
        input_dim=input_dim,
        seed=random_seed,
    )

    X = np.asarray(X)
    y = np.asarray(y, dtype=int)

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class balance: {np.bincount(y)}")

    # Training and validation split
    X_train, y_train, X_val, y_val = train_val_split(
        X.tolist(), y.tolist(), val_ratio=val_ratio, seed=random_seed
    )

    # Loss function
    loss_fn, loss_deriv_fn = get_loss_function(loss_fct)

    # Neuron instantiation
    neuron = PointNeuron(
        input_dim=input_dim,
        activation=activation,
        seed=random_seed,
    )

    print(f"PointNeuron initialized (input_dim={input_dim}, activation={activation})")

    # Train
    history, trained_neuron = train_model(
        neuron=neuron,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        loss_fn=loss_fn,
        loss_deriv_fn=loss_deriv_fn,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    # ----- Final metrics -----
    final_train_loss = history["train_loss"][-1]
    final_val_loss = history["val_loss"][-1]
    final_val_acc = history["val_accuracy"][-1]

    print("\n=== Final Point Neuron Metrics ===")
    print(f"Train loss: {final_train_loss:.6f}")
    print(f"Val loss: {final_val_loss:.6f}")
    print(f"Val acc: {final_val_acc:.4f}")

    # Save results
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    params_path = os.path.join(model_dir, "point_neuron_params.npz")
    history_path = os.path.join(log_dir, "point_neuron_history.json")

    np.savez(params_path, **trained_neuron.get_parameters())

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved model params to {params_path}")
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
