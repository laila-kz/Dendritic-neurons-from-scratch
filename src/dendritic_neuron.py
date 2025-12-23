from typing import Optional, List, Dict, Callable
import math
import random


class DendriticNeuron:
    """
    A dendritic neuron with multiple branches.

    Branches are dendrites; each sees part of the input.
    Soma is the cell body that combines branch outputs into one final value.
    """

    def __init__(
        self,
            input_dim: int,
            num_branches: int,
            branch_input_map: Dict[int, List[int]],
            branch_activation: str = "relu",
            soma_activation: str = "relu",
            branch_weights: Optional[List[List[float]]] = None,
            branch_biases: Optional[List[float]] = None,
            soma_weights: Optional[List[float]] = None,
            soma_bias: Optional[float] = None,
            seed: Optional[int] = None,
    ):
        """
        Initialize the dendritic neuron.

        Parameters
        ----------
        input_dim : int
            Total number of input features.
        num_branches : int
            Number of dendritic branches.
        branch_input_map : dict
            Maps branch index -> list of input indices.
        branch_activation : str
            Activation function name for branches ('relu', 'tanh', 'sigmoid', 'linear', 'identity').
        soma_activation : str
            Activation function name for soma ('relu', 'tanh', 'sigmoid', 'linear', 'identity').
        branch_weights : optional
            List of weight vectors, one per branch.
        branch_biases : optional
            List of biases, one per branch.
        soma_weights : optional
            Weights applied to branch outputs.
        soma_bias : optional
            Bias term for the soma.
        seed : optional
            Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)

        # Basic validation
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_branches <= 0:
            raise ValueError("num_branches must be positive")
        if len(branch_input_map) != num_branches:
            raise ValueError("branch_input_map must have entries for all branches")
        if not all(isinstance(v, list) for v in branch_input_map.values()):
            raise ValueError("branch_input_map values must be lists")
        if not all(isinstance(k, int) for k in branch_input_map.keys()):
            raise ValueError("branch_input_map keys must be integers")
        if not all(all(isinstance(i, int) for i in v) for v in branch_input_map.values()):
            raise ValueError("branch_input_map values must be lists of integers")
        if not all(v for v in branch_input_map.values()):
            raise ValueError("branch_input_map values must be non-empty lists")

        if not isinstance(branch_activation, str):
            raise ValueError("branch_activation must be a string")
        if branch_activation not in ["relu", "tanh", "sigmoid", "linear", "identity"]:
            raise ValueError(
                "branch_activation must be one of ['relu', 'tanh', 'sigmoid', 'linear', 'identity']"
            )

        if not isinstance(soma_activation, str):
            raise ValueError("soma_activation must be a string")
        if soma_activation not in ["relu", "tanh", "sigmoid", "linear", "identity"]:
            raise ValueError(
                "soma_activation must be one of ['relu', 'tanh', 'sigmoid', 'linear', 'identity']"
            )

        self.input_dim = input_dim
        self.num_branches = num_branches
        self.branch_input_map = branch_input_map

        # Activation functions
        self.branch_activation = branch_activation.lower()
        self.soma_activation = soma_activation.lower()

        self.branch_activation_fct = self.resolve_activation(self.branch_activation)
        self.soma_activation_fct = self.resolve_activation(self.soma_activation)

        self.branch_activation_deriv = self.resolve_activation_derivative(
            self.branch_activation
        )
        self.soma_activation_deriv = self.resolve_activation_derivative(
            self.soma_activation
        )

        # Initialize weights and biases
        self.branch_weights: List[List[float]] = []
        self.branch_biases: List[float] = []

        for b in range(num_branches):
            inputs_for_branch = branch_input_map[b]
            if not inputs_for_branch:
                raise ValueError(f"Branch {b} has no input features assigned")

            if branch_weights is None:
                w = [random.uniform(-0.5, 0.5) for _ in inputs_for_branch]
            else:
                w = branch_weights[b]

            if branch_biases is None:
                bias = 0.0
            else:
                bias = branch_biases[b]

            self.branch_weights.append(w)
            self.branch_biases.append(bias)

        # Initialize soma weights and bias
        if soma_weights is None:
            self.soma_weights = [random.uniform(-0.5, 0.5) for _ in range(num_branches)]
        else:
            self.soma_weights = soma_weights

        if soma_bias is None:
            self.soma_bias = random.uniform(-0.5, 0.5)
        else:
            self.soma_bias = soma_bias

        # Store last forward pass values for backpropagation
        self.last_input: Optional[List[float]] = None
        self.last_branch_pre_activation: Optional[List[float]] = None
        self.last_branch_output: Optional[List[float]] = None
        self.last_soma_pre_activation: Optional[float] = None
        self.last_soma_output: Optional[float] = None

    def forward_single(self, input_vector: List[float]) -> float:
        """Compute output for a single input vector."""
        if len(input_vector) != self.input_dim:
            raise ValueError(
                f"Expected input vector of dimension {self.input_dim}, got {len(input_vector)}"
            )

        # Store input for backpropagation
        self.last_input = input_vector

        # Branch computations
        branch_pre_activations: List[float] = []
        branch_outputs: List[float] = []

        for b in range(self.num_branches):
            indices = self.branch_input_map[b]
            weights = self.branch_weights[b]
            bias = self.branch_biases[b]

            # Weighted sum
            local_sum = 0.0
            for i, w in zip(indices, weights):
                local_sum += input_vector[i] * w
            local_sum += bias

            branch_pre_activations.append(local_sum)
            branch_output = self.branch_activation_fct(local_sum)
            branch_outputs.append(branch_output)

        # Soma integration
        soma_pre_activation = 0.0
        for bo, sw in zip(branch_outputs, self.soma_weights):
            soma_pre_activation += bo * sw
        soma_pre_activation += self.soma_bias

        soma_output = self.soma_activation_fct(soma_pre_activation)

        # Store for backpropagation
        self.last_branch_pre_activation = branch_pre_activations
        self.last_branch_output = branch_outputs
        self.last_soma_pre_activation = soma_pre_activation
        self.last_soma_output = soma_output

        return soma_output

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
            output = self.forward_single(input_vector)
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
        grad_branch_weights = [[0.0 for _ in w] for w in self.branch_weights]
        grad_branch_biases = [0.0 for _ in self.branch_biases]
        grad_soma_weights = [0.0 for _ in self.soma_weights]
        grad_soma_bias = 0.0

        # Compute gradients for each sample in batch
        for x, dL_dy in zip(X_batch, dL_dy_batch):
            # Forward pass to populate cache
            self.forward_single(x)

            # Backprop through soma activation
            dL_dz_soma = dL_dy * self.soma_activation_deriv(
                self.last_soma_pre_activation
            )

            # Gradient w.r.t soma bias
            grad_soma_bias += dL_dz_soma

            # Gradients w.r.t soma weights and branch outputs
            for b in range(self.num_branches):
                grad_soma_weights[b] += dL_dz_soma * self.last_branch_output[b]

            # Backprop through branches
            for b in range(self.num_branches):
                # Gradient flowing into branch output
                dL_d_branch_output = dL_dz_soma * self.soma_weights[b]

                # Backprop through branch activation
                dL_dz_branch = (
                    dL_d_branch_output
                    * self.branch_activation_deriv(self.last_branch_pre_activation[b])
                )

                # Gradient w.r.t branch bias
                grad_branch_biases[b] += dL_dz_branch

                # Gradients w.r.t branch weights
                indices = self.branch_input_map[b]
                for idx, input_idx in enumerate(indices):
                    grad_branch_weights[b][idx] += dL_dz_branch * x[input_idx]

        # Return gradients (averaged over batch)
        return {
            "branch_weights": [
                [g / batch_size for g in branch] for branch in grad_branch_weights
            ],
            "branch_biases": [g / batch_size for g in grad_branch_biases],
            "soma_weights": [g / batch_size for g in grad_soma_weights],
            "soma_bias": grad_soma_bias / batch_size,
        }

    def apply_gradients(self, gradients: Dict[str, object], learning_rate: float) -> None:
        """Apply gradients to update parameters."""
        # Update branch weights
        for b in range(self.num_branches):
            for i in range(len(self.branch_weights[b])):
                self.branch_weights[b][i] -= (
                    learning_rate * gradients["branch_weights"][b][i]
                )

        # Update branch biases
        for b in range(self.num_branches):
            self.branch_biases[b] -= learning_rate * gradients["branch_biases"][b]

        # Update soma weights
        for b in range(self.num_branches):
            self.soma_weights[b] -= learning_rate * gradients["soma_weights"][b]

        # Update soma bias
        self.soma_bias -= learning_rate * gradients["soma_bias"]

    def get_parameters(self) -> Dict[str, object]:
        """Return all parameters."""
        return {
            "branch_weights": [w.copy() for w in self.branch_weights],
            "branch_biases": self.branch_biases.copy(),
            "soma_weights": self.soma_weights.copy(),
            "soma_bias": self.soma_bias,
        }

    def set_parameters(self, params: Dict[str, object]) -> None:
        """Set all parameters."""
        self.branch_weights = [
            w.copy() if isinstance(w, list) else list(w)
            for w in params["branch_weights"]
        ]
        self.branch_biases = list(params["branch_biases"])
        self.soma_weights = list(params["soma_weights"])
        self.soma_bias = float(params["soma_bias"])

    def get_branch_outputs(self) -> Optional[List[float]]:
        """Return last branch outputs."""
        return self.last_branch_output

    def get_soma_input(self) -> Optional[float]:
        """Returns the soma pre-activation value from the last forward pass."""
        return self.last_soma_pre_activation

    def summary(self) -> str:
        """Returns a human-readable summary of the neuron configuration."""
        lines: List[str] = []
        lines.append("DendriticNeuron Summary")
        lines.append("----------------------")
        lines.append(f"Input dimension: {self.input_dim}")
        lines.append(f"Number of branches: {self.num_branches}")
        lines.append(f"Branch activation: {self.branch_activation}")
        lines.append(f"Soma activation: {self.soma_activation}")
        lines.append("Branch input mapping:")
        for b in range(self.num_branches):
            lines.append(f" Branch {b}: inputs {self.branch_input_map[b]}")
        return "\n".join(lines)

    def resolve_activation(self, name: str) -> Callable[[float], float]:
        """Maps activation function name to actual function."""
        if name == "relu":
            return lambda x: max(0.0, x)
        elif name == "tanh":
            return lambda x: math.tanh(x)
        elif name == "sigmoid":
            return lambda x: 1 / (1 + math.exp(-x))
        elif name in ("linear", "identity"):
            return lambda x: x
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def resolve_activation_derivative(self, name: str) -> Callable[[float], float]:
        """Maps activation function name to its derivative."""
        if name == "relu":
            return lambda x: 1.0 if x > 0.0 else 0.0
        elif name == "tanh":
            return lambda x: 1.0 - math.tanh(x) ** 2
        elif name == "sigmoid":
            s = lambda x: 1 / (1 + math.exp(-x))
            return lambda x: s(x) * (1.0 - s(x))
        elif name in ("linear", "identity"):
            return lambda x: 1.0
        else:
            raise ValueError(f"Unsupported activation function: {name}")
