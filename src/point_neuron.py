from typing import Dict, List, Optional, Callable
import random
import math


class PointNeuron:
    """
    Standard neuron where all inputs are combined in one weighted sum plus a bias,
    then passed through a single activation.
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
            Activation function name ('relu', 'tanh', 'sigmoid', 'linear').
        init_scale : float
            Scale of random weight initialization.
        seed : optional int
            Random seed for reproducibility.
        """
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        self.input_dim = input_dim
        self.activation = activation.lower()
        self.activation_fct = self._resolve_activation(self.activation)
        self.activation_deriv = self._resolve_activation_derivative(self.activation)
        self.init_scale = init_scale

        if seed is not None:
            random.seed(seed)

        # Initialize weights and bias
        self.weights: List[float] = [
            random.uniform(-init_scale, init_scale) for _ in range(input_dim)
        ]
        self.bias: float = random.uniform(-init_scale, init_scale)

        # Cache for backpropagation
        self.last_input: Optional[List[float]] = None
        self.last_output: Optional[float] = None
        self.last_pre_activation: Optional[float] = None

    def forward(self, input_vector: List[float]) -> float:
        """Compute the output of the neuron for a single input vector."""
        if len(input_vector) != self.input_dim:
            raise ValueError(
                f"Expected input vector of dimension {self.input_dim}, got {len(input_vector)}"
            )

        # Pre-activation: weighted sum plus bias
        z = 0.0
        for xi, wi in zip(input_vector, self.weights):
            z += xi * wi
        z += self.bias

        # Activation
        y = self.activation_fct(z)

        # Store for backpropagation
        self.last_input = input_vector
        self.last_pre_activation = z
        self.last_output = y

        return y

    def forward_batch(self, input_matrix: List[List[float]]) -> List[float]:
        """Compute outputs for a batch of input vectors."""
        if not input_matrix:
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
        Dict containing accumulated gradients
        """
        batch_size = len(X_batch)

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

        # Return gradients (averaged over batch)
        return {
            "weights": [g / batch_size for g in grad_weights],
            "bias": grad_bias / batch_size,
        }

    def apply_gradients(self, gradients: Dict[str, object], learning_rate: float) -> None:
        """Apply gradients to update parameters."""
        # Update weights
        grad_w = gradients["weights"]
        for i in range(self.input_dim):
            self.weights[i] -= learning_rate * grad_w[i]

        # Update bias
        self.bias -= learning_rate * gradients["bias"]

    def get_parameters(self) -> Dict[str, object]:
        """Return the neuron's parameters (weights and bias)."""
        return {
            "weights": self.weights.copy(),
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
        if "weights" not in params or "bias" not in params:
            raise ValueError("params must contain 'weights' and 'bias'")

        weights = params["weights"]

        # Handle numpy arrays or lists
        if hasattr(weights, "tolist"):
            weights = weights.tolist()

        if len(weights) != self.input_dim:
            raise ValueError("Weight vector has incorrect dimension")

        self.weights = list(weights)
        self.bias = float(params["bias"])

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

    def _resolve_activation(self, name: str) -> Callable[[float], float]:
        """Map activation name to function."""
        if name == "relu":
            return lambda x: max(0.0, x)
        if name == "tanh":
            return math.tanh
        if name == "sigmoid":
            return lambda x: 1.0 / (1.0 + math.exp(-x))
        if name == "linear":
            return lambda x: x
        raise ValueError(f"Unsupported activation: {name}")

    def _resolve_activation_derivative(self, name: str) -> Callable[[float], float]:
        """Map activation name to its derivative."""
        if name == "relu":
            return lambda x: 1.0 if x > 0.0 else 0.0
        if name == "tanh":
            return lambda x: 1.0 - math.tanh(x) ** 2
        if name == "sigmoid":
            s = lambda x: 1.0 / (1.0 + math.exp(-x))
            return lambda x: s(x) * (1.0 - s(x))
        if name == "linear":
            return lambda x: 1.0
        raise ValueError(f"Unsupported activation: {name}")
