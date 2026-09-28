"""Generic training utilities for PointNeuron and DendriticNeuron.

This module is neuron-agnostic: it assumes neurons expose a forward
interface and handle their own backpropagation logic.

A neuron must provide:

* ``forward_batch(X)`` (or ``forward(x)`` for a single sample), and
* ``backward(X_batch, dL_dy_batch)`` returning a gradient dict, plus
* ``apply_gradients(grads, learning_rate)``.
"""

import random
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

__all__ = [
    "train_one_epoch",
    "train_model",
    "evaluate_model",
]


def _neuron_forward(neuron: Any, X: List[Any]) -> List[float]:
    """Run a forward pass over ``X`` using whichever interface the neuron has."""
    if hasattr(neuron, "forward_batch"):
        return neuron.forward_batch(X)
    return [neuron.forward(x) for x in X]


def _make_batches(
    X: List[Any], y: List[Any], batch_size: int
) -> Iterator[Tuple[List[Any], List[Any]]]:
    """Yield consecutive mini-batches from X and y."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

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
    rng: Optional[random.Random] = None,
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
        Loss function from :mod:`src.losses`.
    loss_deriv_fn : callable
        Derivative of the loss function.
    learning_rate : float
        Gradient descent learning rate.
    batch_size : int
        Mini-batch size.
    rng : random.Random, optional
        RNG used to shuffle the data. Pass a seeded instance to make the
        shuffling reproducible without touching the global random state.

    Returns
    -------
    float
        Average training loss for the epoch.
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train and y_train must have the same length, "
            f"got {len(X_train)} and {len(y_train)}"
        )
    if len(X_train) == 0:
        raise ValueError("X_train and y_train must not be empty")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    shuffle_rng = rng if rng is not None else random.Random()

    # Shuffle data
    indices = list(range(len(X_train)))
    shuffle_rng.shuffle(indices)

    X_shuffled = [X_train[i] for i in indices]
    y_shuffled = [y_train[i] for i in indices]

    total_loss = 0.0
    num_examples = 0

    for X_batch, y_batch in _make_batches(X_shuffled, y_shuffled, batch_size):
        # ----- Forward pass -----
        y_pred = _neuron_forward(neuron, X_batch)

        # ----- Loss computation -----
        batch_loss = loss_fn(y_pred, y_batch)
        batch_n = len(y_batch)
        total_loss += batch_loss * batch_n
        num_examples += batch_n

        # ----- Gradient of loss w.r.t outputs -----
        dL_dy = loss_deriv_fn(y_pred, y_batch)

        # ----- Backpropagation -----
        grads = neuron.backward(X_batch, dL_dy)

        # ----- Parameter update -----
        neuron.apply_gradients(grads, learning_rate)

    # Average over examples, not over batches: batches have different sizes, so
    # averaging the per-batch means would over-weight the final partial batch.
    return total_loss / num_examples if num_examples else 0.0


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
    rng: Optional[random.Random] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, List[float]], Any]:
    """
    Train a neuron for multiple epochs.

    Returns
    -------
    history : dict
        Contains training and validation metrics per epoch.
    neuron : object
        The trained neuron (mutated in place and returned for convenience).
    """
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {num_epochs}")
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train and y_train must have the same length, "
            f"got {len(X_train)} and {len(y_train)}"
        )
    if len(X_val) != len(y_val):
        raise ValueError(
            f"X_val and y_val must have the same length, "
            f"got {len(X_val)} and {len(y_val)}"
        )
    if len(X_val) == 0:
        raise ValueError("X_val and y_val must not be empty")

    shuffle_rng = rng if rng is not None else random.Random()

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
            rng=shuffle_rng,
        )

        val_metrics = evaluate_model(neuron, X_val, y_val, loss_fn)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics.get("accuracy", 0.0))

        if verbose:
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
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length, got {len(X)} and {len(y)}"
        )
    if len(X) == 0:
        raise ValueError("X and y must not be empty")

    y_pred = _neuron_forward(neuron, X)

    loss = loss_fn(y_pred, y)
    metrics: Dict[str, float] = {"loss": float(loss)}

    # Optional accuracy for binary classification. Compare with `in (0, 1)`
    # rather than truth-testing the container: `if y` raises for numpy arrays
    # and panics on arrays with more than one element.
    is_binary = all(target in (0, 1) for target in y)
    if is_binary:
        correct = sum(1 for yp, yt in zip(y_pred, y) if (yp >= 0.5) == bool(yt))
        metrics["accuracy"] = correct / len(y)

    return metrics
