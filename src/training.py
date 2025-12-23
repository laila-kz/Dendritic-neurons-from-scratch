#This module implements generic training loops that can work with both DendriticNeuron and PointNeuron.

import random
from typing import Callable, Dict, List, Tuple, Any

"""
Generic training utilities for PointNeuron and DendriticNeuron.

This module is neuron-agnostic: it assumes neurons expose a forward
interface and handle their own backpropagation logic.
"""

# ---------- Utility: batch generator ----------


def _make_batches(X: List[Any], y: List[Any], batch_size: int):
    """Yield mini-batches from X and y."""
    for i in range(0, len(X), batch_size):
        yield X[i : i + batch_size], y[i : i + batch_size]


# ---------- Single-epoch training ----------


def train_one_epoch(
    neuron: Any,
    X_train: List[Any],
    y_train: List[float],
    loss_fn: Callable[[List[float], List[float]], float],
    loss_deriv_fn: Callable[[List[float], List[float]], List[float]],
    learning_rate: float,
    batch_size: int,
) -> float:
    """
    Train a neuron for one epoch.

    Parameters
    ----------
    neuron : object
        PointNeuron or DendriticNeuron instance.
    X_train : list
        Training inputs, shape (N, input_dim).
    y_train : list
        Training targets, shape (N,).
    loss_fn : callable
        Loss function from losses.py.
    loss_deriv_fn : callable
        Derivative of the loss function.
    learning_rate : float
        Gradient descent learning rate.
    batch_size : int
        Mini-batch size.

    Returns
    -------
    float
        Average training loss for the epoch.
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length")

    # Shuffle data
    indices = list(range(len(X_train)))
    random.shuffle(indices)

    X_shuffled = [X_train[i] for i in indices]
    y_shuffled = [y_train[i] for i in indices]

    total_loss = 0.0
    num_batches = 0

    for X_batch, y_batch in _make_batches(X_shuffled, y_shuffled, batch_size):
        # ----- Forward pass -----
        if hasattr(neuron, "forward_batch"):
            y_pred = neuron.forward_batch(X_batch)
        else:
            y_pred = [neuron.forward(x) for x in X_batch]

        # ----- Loss computation -----
        batch_loss = loss_fn(y_pred, y_batch)
        total_loss += batch_loss
        num_batches += 1

        # ----- Gradient of loss w.r.t outputs -----
        dL_dy = loss_deriv_fn(y_pred, y_batch)

        # ----- Backpropagation -----
        grads = neuron.backward(X_batch, dL_dy)

        # ----- Parameter update -----
        neuron.apply_gradients(grads, learning_rate)

    return total_loss / max(num_batches, 1)


# ---------- Full training loop ----------


def train_model(
    neuron: Any,
    X_train: List[Any],
    y_train: List[float],
    X_val: List[Any],
    y_val: List[float],
    loss_fn: Callable[[List[float], List[float]], float],
    loss_deriv_fn: Callable[[List[float], List[float]], List[float]],
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
) -> Tuple[Dict[str, List[float]], Any]:
    """
    Train a neuron for multiple epochs.

    Returns
    -------
    history : dict
        Contains training and validation metrics.
    neuron : object
        Trained neuron.
    """
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            neuron,
            X_train,
            y_train,
            loss_fn,
            loss_deriv_fn,
            learning_rate,
            batch_size,
        )

        val_metrics = evaluate_model(neuron, X_val, y_val, loss_fn)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics.get("accuracy", 0.0))

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Val Acc: {val_metrics.get('accuracy')}"
        )

    return history, neuron


# ---------- Evaluation ----------


def evaluate_model(
    neuron: Any,
    X: List[Any],
    y: List[float],
    loss_fn: Callable[[List[float], List[float]], float],
) -> Dict[str, float]:
    """
    Evaluate a trained neuron.

    Returns
    -------
    dict
        Metrics such as loss and accuracy.
    """
    if hasattr(neuron, "forward_batch"):
        y_pred = neuron.forward_batch(X)
    else:
        y_pred = [neuron.forward(x) for x in X]

    loss = loss_fn(y_pred, y)
    metrics: Dict[str, float] = {"loss": float(loss)}

    # Optional accuracy for binary classification
    if y and all(target in (0, 1) for target in y):
        correct = 0
        for yp, yt in zip(y_pred, y):
            pred_label = 1 if yp >= 0.5 else 0
            if pred_label == yt:
                correct += 1
        metrics["accuracy"] = correct / len(y)

    return metrics
