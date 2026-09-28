"""A point neuron: the standard single weighted sum plus one nonlinearity."""

import random
from typing import Callable, Dict, List, Optional

from src.activations import ACTIVATION_NAMES, get_activation_pair


class PointNeuron:
    """
    Standard neuron where all inputs are combined in one weighted sum plus a bias,
    then passed through a single activation.

    This is the baseline the dendritic neuron is compared against.
    """

    def __init__(
        self,
        input_dim: int,
        activation: str = "relu",
        init_scale: float = 0.01,
        seed: Optional[int] = None,
    ):
        """
        Initialize the point neuron.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        activation : str
            Activation function name, e.g. 'relu', 'tanh', 'sigmoid',
            'linear' or 'identity'.
        init_scale : float
            Scale of random weight initialization.
        seed : optional int
            Random seed for this neuron only. Uses a private RNG so that
            constructing a neuron never disturbs the global ``random`` state.
        """
        if not isinstance(input_dim, int) or isinstance(input_dim, bool):
            raise ValueError("input_dim must be an integer")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not isinstance(activation, str):
            raise ValueError(
                f"Activation name must be a string, got {type(activation).__name__}"
            )
        if not isinstance(init_scale, (int, float)) or isinstance(init_scale, bool):
            raise ValueError("init_scale must be a number")
        if init_scale < 0:
            raise ValueError("init_scale must be non-negative")

        self.input_dim = input_dim
        self.init_scale = float(init_scale)

        normalized = activation.strip().lower()
        if normalized not in ACTIVATION_NAMES:
            raise ValueError(
                f"Unsupported activation: {activation!r}. "
                f"Supported activations: {', '.join(ACTIVATION_NAMES)}"
            )
        self.activation = normalized
        self.activation_fct, self.activation_deriv = get_activation_pair(normalized)

        # Private RNG: seeding must not clobber the global random state.
        self._rng = random.Random(seed)

        # Initialize weights and bias
        self.weights: List[float] = [
            self._rng.uniform(-self.init_scale, self.init_scale)
            for _ in range(input_dim)
        ]
        self.bias: float = self._rng.uniform(-self.init_scale, self.init_scale)

        # Cache for backpropagation
        self.last_input: Optional[List[float]] = None
        self.last_output: Optional[float] = None
        self.last_pre_activation: Optional[float] = None

    # ---------- Forward ----------

    def forward(self, input_vector: List[float]) -> float:
        """Compute the output of the neuron for a single input vector."""
        if len(input_vector) != self.input_dim:
            raise ValueError(
                f"Expected input vector of dimension {self.input_dim}, got {len(input_vector)}"
            )

        # Pre-activation: weighted sum plus bias
        z = self.bias
        for xi, wi in zip(input_vector, self.weights):
            z += xi * wi

        # Activation
        y = self.activation_fct(z)

        # Store for backpropagation
        self.last_input = list(input_vector)
        self.last_pre_activation = z
        self.last_output = y

        return y

    def forward_batch(self, input_matrix: List[List[float]]) -> List[float]:
        """Compute outputs for a batch of input vectors."""
        if input_matrix is None or len(input_matrix) == 0:
            return []

        if len(input_matrix[0]) != self.input_dim:
            raise ValueError(
                f"Expected input vectors of dimension {self.input_dim}, got {len(input_matrix[0])}"
            )

        outputs: List[float] = []
        for input_vector in input_matrix:
            output = self.forward(input_vector)
            outputs.append(output)
        return outputs

    # ---------- Backward ----------

    def backward(
        self,
        X_batch: List[List[float]],
        dL_dy_batch: List[float],
    ) -> Dict[str, object]:
        """
        Compute gradients for a batch of samples.

        Parameters
        ----------
        X_batch : List[List[float]]
            Batch of input vectors
        dL_dy_batch : List[float]
            Gradients of loss w.r.t. outputs for each sample in batch

        Returns
        -------
        Dict containing accumulated gradients.

        Notes
        -----
        ``dL_dy_batch`` is expected to already be the gradient of the *mean*
        batch loss with respect to the outputs, which is what the helpers in
        :mod:`src.losses` return (they divide by the batch size). The total
        parameter gradient is therefore the plain **sum** of the per-sample
        contributions. Dividing by ``batch_size`` a second time here would
        scale every gradient by ``1/batch_size`` and silently shrink the
        effective learning rate.
        """
        batch_size = len(X_batch)
        if batch_size == 0:
            raise ValueError("Cannot compute gradients for an empty batch")
        if len(dL_dy_batch) != batch_size:
            raise ValueError(
                f"dL_dy_batch must have the same length as X_batch "
                f"({batch_size}), got {len(dL_dy_batch)}"
            )

        # Initialize gradient accumulator
        grad_weights = [0.0 for _ in self.weights]
        grad_bias = 0.0

        # Compute gradients for each sample in batch
        for x, dL_dy in zip(X_batch, dL_dy_batch):
            # Forward pass to populate cache
            self.forward(x)

            # Backprop through activation
            dL_dz = dL_dy * self.activation_deriv(self.last_pre_activation)

            # Gradient w.r.t bias
            grad_bias += dL_dz

            # Gradients w.r.t weights
            for i in range(self.input_dim):
                grad_weights[i] += dL_dz * x[i]

        # Return gradients (sum over batch; see the note in the docstring)
        return {
            "weights": grad_weights,
            "bias": grad_bias,
        }

    def apply_gradients(self, gradients: Dict[str, object], learning_rate: float) -> None:
        """Apply gradients to update parameters (in-place gradient descent)."""
        if "weights" not in gradients or "bias" not in gradients:
            raise ValueError("gradients must contain 'weights' and 'bias'")

        grad_w = gradients["weights"]
        for i in range(self.input_dim):
            self.weights[i] -= learning_rate * grad_w[i]

        self.bias -= learning_rate * gradients["bias"]

    # ---------- Parameters ----------

    def get_parameters(self) -> Dict[str, object]:
        """Return the neuron's parameters (weights and bias)."""
        return {
            "weights": list(self.weights),
            "bias": self.bias,
        }

    def set_parameters(self, params: Dict[str, object]) -> None:
        """
        Set the neuron's parameters.

        Parameters
        ----------
        params : dict
            Must contain 'weights' and 'bias'.
        """
        if not isinstance(params, dict):
            raise ValueError("params must be a dict")
        if "weights" not in params or "bias" not in params:
            raise ValueError("params must contain 'weights' and 'bias'")

        weights = params["weights"]

        # Handle numpy arrays or lists
        if hasattr(weights, "tolist"):
            weights = weights.tolist()
        else:
            weights = list(weights)

        if len(weights) != self.input_dim:
            raise ValueError(
                f"Weight vector has incorrect dimension: expected {self.input_dim}, "
                f"got {len(weights)}"
            )

        self.weights = [float(w) for w in weights]
        self.bias = float(params["bias"])

    # ---------- Introspection ----------

    def summary(self) -> str:
        """Return a human-readable summary of the neuron."""
        lines: List[str] = []
        lines.append("PointNeuron Summary")
        lines.append("-------------------")
        lines.append(f"Input dimension: {self.input_dim}")
        lines.append(f"Activation: {self.activation}")
        lines.append(f"Weights shape: ({len(self.weights)},)")
        lines.append("Bias: scalar")
        return "\n".join(lines)

    # ---------- Backwards-compatible activation resolvers ----------

    def _resolve_activation(self, name: str) -> Callable[[float], float]:
        """Map activation name to function."""
        if not isinstance(name, str):
            raise ValueError(
                f"Activation name must be a string, got {type(name).__name__}"
            )
        key = name.strip().lower()
        if key not in ACTIVATION_NAMES:
            raise ValueError(
                f"Unsupported activation: {name!r}. "
                f"Supported activations: {', '.join(ACTIVATION_NAMES)}"
            )
        return get_activation_pair(key)[0]

    def _resolve_activation_derivative(self, name: str) -> Callable[[float], float]:
        """Map activation name to its derivative."""
        if not isinstance(name, str):
            raise ValueError(
                f"Activation name must be a string, got {type(name).__name__}"
            )
        key = name.strip().lower()
        if key not in ACTIVATION_NAMES:
            raise ValueError(
                f"Unsupported activation: {name!r}. "
                f"Supported activations: {', '.join(ACTIVATION_NAMES)}"
            )
        return get_activation_pair(key)[1]
