"""A dendritic neuron: several branches feeding a single soma.

Each branch is a small "dendrite" that sees only a subset of the input
features, applies its own affine transform and nonlinearity, and hands its
output to the soma. The soma combines the branch outputs with a second
affine transform and nonlinearity to produce the neuron's output.
"""

import random
from typing import Callable, Dict, List, Optional

from src.activations import (
    ACTIVATION_NAMES,
    get_activation_derivative,
    get_activation_pair,
)


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
            Maps branch index -> list of input indices. Must contain exactly the
            keys ``0 .. num_branches - 1``.
        branch_activation : str
            Activation function name for branches.
        soma_activation : str
            Activation function name for soma.
        branch_weights : optional
            List of weight vectors, one per branch.
        branch_biases : optional
            List of biases, one per branch.
        soma_weights : optional
            Weights applied to branch outputs.
        soma_bias : optional
            Bias term for the soma.
        seed : optional
            Random seed for this neuron only. Uses a private RNG so that
            constructing a neuron never disturbs the global ``random`` state
            (and therefore never perturbs dataset generation or shuffling).
        """
        # Basic validation
        if not isinstance(input_dim, int) or isinstance(input_dim, bool):
            raise ValueError("input_dim must be an integer")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not isinstance(num_branches, int) or isinstance(num_branches, bool):
            raise ValueError("num_branches must be an integer")
        if num_branches <= 0:
            raise ValueError("num_branches must be positive")

        self._validate_branch_input_map(branch_input_map, num_branches, input_dim)

        self.input_dim = input_dim
        self.num_branches = num_branches
        self.branch_input_map = {
            b: list(branch_input_map[b]) for b in range(num_branches)
        }

        # Activation functions
        self.branch_activation = self._normalize_activation_name(branch_activation)
        self.soma_activation = self._normalize_activation_name(soma_activation)

        self.branch_activation_fct, self.branch_activation_deriv = get_activation_pair(
            self.branch_activation
        )
        self.soma_activation_fct, self.soma_activation_deriv = get_activation_pair(
            self.soma_activation
        )

        # Private RNG: seeding must not clobber the global random state.
        self._rng = random.Random(seed)

        # Initialize weights and biases
        self.branch_weights: List[List[float]] = []
        self.branch_biases: List[float] = []

        for b in range(num_branches):
            inputs_for_branch = self.branch_input_map[b]

            if branch_weights is None:
                w = [self._rng.uniform(-0.5, 0.5) for _ in inputs_for_branch]
            else:
                w = [float(v) for v in branch_weights[b]]

            if branch_biases is None:
                bias = 0.0
            else:
                bias = float(branch_biases[b])

            self.branch_weights.append(w)
            self.branch_biases.append(bias)

        self._validate_branch_parameter_shapes(
            self.branch_weights, self.branch_biases
        )

        # Initialize soma weights and bias
        if soma_weights is None:
            self.soma_weights = [
                self._rng.uniform(-0.5, 0.5) for _ in range(num_branches)
            ]
        else:
            self.soma_weights = [float(v) for v in soma_weights]

        if soma_bias is None:
            self.soma_bias = self._rng.uniform(-0.5, 0.5)
        else:
            self.soma_bias = float(soma_bias)

        if len(self.soma_weights) != self.num_branches:
            raise ValueError(
                f"soma_weights must have {self.num_branches} entries, "
                f"got {len(self.soma_weights)}"
            )

        # Store last forward pass values for backpropagation
        self.last_input: Optional[List[float]] = None
        self.last_branch_pre_activation: Optional[List[float]] = None
        self.last_branch_output: Optional[List[float]] = None
        self.last_soma_pre_activation: Optional[float] = None
        self.last_soma_output: Optional[float] = None

    # ---------- Validation helpers ----------

    @staticmethod
    def _normalize_activation_name(name: str) -> str:
        """Validate and normalize an activation name (case/whitespace)."""
        if not isinstance(name, str):
            raise ValueError(
                f"Activation name must be a string, got {type(name).__name__}"
            )
        normalized = name.strip().lower()
        if normalized not in ACTIVATION_NAMES:
            raise ValueError(
                f"Unsupported activation function: {name!r}. "
                f"Supported activations: {', '.join(ACTIVATION_NAMES)}"
            )
        return normalized

    @staticmethod
    def _validate_branch_input_map(
        branch_input_map: Dict[int, List[int]],
        num_branches: int,
        input_dim: int,
    ) -> None:
        """Validate the branch -> input-index mapping.

        The original implementation only checked ``len(branch_input_map)``, so a
        mapping such as ``{0: [0, 1], 7: [2, 3]}`` passed construction and then
        raised a bare ``KeyError`` later during the forward pass. Out-of-range
        indices were not checked at all, and negative indices silently wrapped
        around in Python.
        """
        if not isinstance(branch_input_map, dict):
            raise ValueError("branch_input_map must be a dict")

        if len(branch_input_map) != num_branches:
            raise ValueError(
                f"branch_input_map must have exactly {num_branches} entries, "
                f"got {len(branch_input_map)}"
            )

        expected_keys = set(range(num_branches))
        actual_keys = set(branch_input_map.keys())
        if actual_keys != expected_keys:
            raise ValueError(
                f"branch_input_map keys must be exactly {sorted(expected_keys)}, "
                f"got {sorted(actual_keys)}"
            )

        for branch_idx, indices in branch_input_map.items():
            if not isinstance(indices, list):
                raise ValueError(
                    f"branch_input_map[{branch_idx}] must be a list, "
                    f"got {type(indices).__name__}"
                )
            if not indices:
                raise ValueError(
                    f"branch_input_map[{branch_idx}] must be a non-empty list"
                )
            for idx in indices:
                if not isinstance(idx, int) or isinstance(idx, bool):
                    raise ValueError(
                        f"branch_input_map[{branch_idx}] must contain integers, "
                        f"got {idx!r}"
                    )
                if not 0 <= idx < input_dim:
                    raise ValueError(
                        f"branch_input_map[{branch_idx}] index {idx} is out of "
                        f"range for input_dim={input_dim}"
                    )

    def _validate_branch_parameter_shapes(
        self,
        branch_weights: List[List[float]],
        branch_biases: List[float],
    ) -> None:
        """Check that parameter containers match the configured structure."""
        if len(branch_weights) != self.num_branches:
            raise ValueError(
                f"branch_weights must have {self.num_branches} entries, "
                f"got {len(branch_weights)}"
            )
        if len(branch_biases) != self.num_branches:
            raise ValueError(
                f"branch_biases must have {self.num_branches} entries, "
                f"got {len(branch_biases)}"
            )

        for b, weights in enumerate(branch_weights):
            expected = len(self.branch_input_map[b])
            if len(weights) != expected:
                raise ValueError(
                    f"branch_weights[{b}] must have {expected} entries to match "
                    f"branch_input_map[{b}], got {len(weights)}"
                )

    # ---------- Forward ----------

    def forward_single(self, input_vector: List[float]) -> float:
        """Compute output for a single input vector."""
        if len(input_vector) != self.input_dim:
            raise ValueError(
                f"Expected input vector of dimension {self.input_dim}, got {len(input_vector)}"
            )

        # Store input for backpropagation
        self.last_input = list(input_vector)

        # Branch computations
        branch_pre_activations: List[float] = []
        branch_outputs: List[float] = []

        for b in range(self.num_branches):
            indices = self.branch_input_map[b]
            weights = self.branch_weights[b]
            bias = self.branch_biases[b]

            # Weighted sum
            local_sum = bias
            for i, w in zip(indices, weights):
                local_sum += input_vector[i] * w

            branch_pre_activations.append(local_sum)
            branch_output = self.branch_activation_fct(local_sum)
            branch_outputs.append(branch_output)

        # Soma integration
        soma_pre_activation = self.soma_bias
        for bo, sw in zip(branch_outputs, self.soma_weights):
            soma_pre_activation += bo * sw

        soma_output = self.soma_activation_fct(soma_pre_activation)

        # Store for backpropagation
        self.last_branch_pre_activation = branch_pre_activations
        self.last_branch_output = branch_outputs
        self.last_soma_pre_activation = soma_pre_activation
        self.last_soma_output = soma_output

        return soma_output

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
            output = self.forward_single(input_vector)
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

        # Return gradients (sum over batch; see the note in the docstring)
        return {
            "branch_weights": grad_branch_weights,
            "branch_biases": grad_branch_biases,
            "soma_weights": grad_soma_weights,
            "soma_bias": grad_soma_bias,
        }

    def apply_gradients(self, gradients: Dict[str, object], learning_rate: float) -> None:
        """Apply gradients to update parameters (in-place gradient descent)."""
        grad_branch_weights = gradients["branch_weights"]
        grad_branch_biases = gradients["branch_biases"]
        grad_soma_weights = gradients["soma_weights"]

        for b in range(self.num_branches):
            for i in range(len(self.branch_weights[b])):
                self.branch_weights[b][i] -= (
                    learning_rate * grad_branch_weights[b][i]
                )
            self.branch_biases[b] -= learning_rate * grad_branch_biases[b]
            self.soma_weights[b] -= learning_rate * grad_soma_weights[b]

        self.soma_bias -= learning_rate * gradients["soma_bias"]

    # ---------- Parameters ----------

    def get_parameters(self) -> Dict[str, object]:
        """Return all parameters."""
        return {
            "branch_weights": [w.copy() for w in self.branch_weights],
            "branch_biases": list(self.branch_biases),
            "soma_weights": list(self.soma_weights),
            "soma_bias": self.soma_bias,
        }

    def set_parameters(self, params: Dict[str, object]) -> None:
        """Set all parameters, validating shapes against this configuration."""
        if not isinstance(params, dict):
            raise ValueError("params must be a dict")

        required = {"branch_weights", "branch_biases", "soma_weights", "soma_bias"}
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"params is missing required keys: {sorted(missing)}")

        # Accept numpy arrays as well as nested lists.
        branch_weights = [
            list(w) if not hasattr(w, "tolist") else w.tolist()
            for w in params["branch_weights"]
        ]
        branch_biases = list(params["branch_biases"])
        soma_weights = list(params["soma_weights"])

        self._validate_branch_parameter_shapes(branch_weights, branch_biases)
        if len(soma_weights) != self.num_branches:
            raise ValueError(
                f"soma_weights must have {self.num_branches} entries, "
                f"got {len(soma_weights)}"
            )

        self.branch_weights = [[float(v) for v in w] for w in branch_weights]
        self.branch_biases = [float(v) for v in branch_biases]
        self.soma_weights = [float(v) for v in soma_weights]
        self.soma_bias = float(params["soma_bias"])

    # ---------- Introspection ----------

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

    # ---------- Backwards-compatible activation resolvers ----------

    def resolve_activation(self, name: str) -> Callable[[float], float]:
        """Maps activation function name to actual function."""
        return get_activation_pair(self._normalize_activation_name(name))[0]

    def resolve_activation_derivative(self, name: str) -> Callable[[float], float]:
        """Maps activation function name to its derivative."""
        return get_activation_derivative(self._normalize_activation_name(name))
