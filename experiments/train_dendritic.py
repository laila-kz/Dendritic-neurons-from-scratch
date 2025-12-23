#orchestrate everything: create data, split it, build the neuron, train it, and report performance.

import numpy as np
from experiments.synthetic_dataset import make_synthetic_data, get_branch_index_grps
from src.dendritic_neuron import DendriticNeuron
from src.activations import get_activation
from src.losses import get_loss_function
from src.training import train_model
from sklearn.model_selection import train_test_split

input_dim = 6
num_samples = 2000
val_ratio = 0.2
learning_rate = 0.01
batch_size = 32
num_epochs = 50

branch_activation = "tanh"
soma_activation = "sigmoid"
loss_fct = "binary_cross_entropy"
random_seed = 42


def main():
    print("===Dendritic Neuron Training Experiment===")

    X, y = make_synthetic_data(
        num_samples=num_samples,
        input_dim=input_dim,
        noise_std=0.4,
        seed=random_seed,
    )

    print(f"Generated dataset with {len(X)} samples.")
    print(f"Class 0 count: {sum(1 for v in y if v == 0)}")
    print(f"Class 1 count: {sum(1 for v in y if v == 1)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # FIX: Just get the activation functions, don't unpack derivatives
    # Since get_activation() returns only the function, not a tuple
    branch_activation_fct = get_activation(branch_activation)
    soma_activation_fct = get_activation(soma_activation)
    _ = branch_activation_fct
    _ = soma_activation_fct

    # Get loss function and derivative
    loss_fct_impl, loss_deri_impl = get_loss_function(loss_fct)

    # Branch structure
    branch_index_grps = get_branch_index_grps(input_dim=input_dim)
    branch_input_map = {i: grp for i, grp in enumerate(branch_index_grps)}

    print("Branch structure:")
    for i, grp in enumerate(branch_index_grps):
        print(f" Branch {i}: features {grp}")

    # Create neuron with correct parameters
    neuron = DendriticNeuron(
        input_dim=input_dim,
        num_branches=len(branch_index_grps),
        branch_input_map=branch_input_map,
        branch_activation=branch_activation,
        soma_activation=soma_activation,
        seed=random_seed,
    )

    print("Initialized Dendritic Neuron model.")

    # Training
    history, trained_neuron = train_model(
        neuron=neuron,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        loss_fn=loss_fct_impl,
        loss_deriv_fn=loss_deri_impl,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    # Final metrics
    final_train_loss = history["train_loss"][-1]
    final_val_loss = history["val_loss"][-1]
    final_val_acc = history["val_accuracy"][-1]

    print("\n=== Final Results ===")
    print(f"Final train loss: {final_train_loss:.6f}")
    print(f"Final val loss: {final_val_loss:.6f}")
    print(f"Final val acc: {final_val_acc:.4f}")

        # Save trained dendritic neuron parameters
    import os
    import numpy as np

    model_dir = "results/models"
    os.makedirs(model_dir, exist_ok=True)

    params_path = os.path.join(model_dir, "dendritic_params.npz")
    np.savez(params_path, **trained_neuron.get_parameters())

    print(f"\nSaved dendritic params to {params_path}")



if __name__ == "__main__":
    main()

    print("\nDone.")