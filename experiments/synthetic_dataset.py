"""Synthetic dataset used by the dendritic-vs-point experiments.

Generates a toy binary-classification problem with 6 input features grouped
into three 2D blocks. Each class is built from Gaussian clusters centred on
different points in that 6D space, so the classes are not linearly separable
and structure inside the neuron has a chance to matter.
"""

import random
from typing import List, Optional, Tuple

# Number of features per branch in the default grouping.
FEATURES_PER_BRANCH = 2


def get_branch_index_groups(input_dim: int = 6) -> List[List[int]]:
    """Return the default mapping from branch index to input feature indices.

    For the default ``input_dim=6`` this yields three branches of two features
    each: ``[[0, 1], [2, 3], [4, 5]]``.

    Raises
    ------
    ValueError
        If ``input_dim`` is not a multiple of :data:`FEATURES_PER_BRANCH`.
    """
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if input_dim % FEATURES_PER_BRANCH != 0:
        raise ValueError(
            f"input_dim must be a multiple of {FEATURES_PER_BRANCH}, got {input_dim}"
        )

    return [
        list(range(start, start + FEATURES_PER_BRANCH))
        for start in range(0, input_dim, FEATURES_PER_BRANCH)
    ]


# Backwards-compatible alias for the original (misspelled) function name.
get_branch_index_grps = get_branch_index_groups


def make_synthetic_data(
    num_samples: int,
    input_dim: int = 6,
    noise_std: float = 0.4,
    seed: Optional[int] = None,
) -> Tuple[List[List[float]], List[int]]:
    """Generate a branch-structured synthetic binary classification dataset.

    Parameters
    ----------
    num_samples : int
        Total number of samples to generate, split evenly between the classes.
    input_dim : int
        Number of input features. Must be even; each consecutive pair of
        features forms one branch-sized block.
    noise_std : float
        Standard deviation of the Gaussian clusters.
    seed : int, optional
        Random seed. Uses a private RNG so the global random state is left
        untouched.

    Returns
    -------
    (X, y)
        ``X`` is a list of ``num_samples`` feature vectors, ``y`` the matching
        integer labels in ``{0, 1}``.
    """
    if not isinstance(num_samples, int) or isinstance(num_samples, bool):
        raise ValueError("num_samples must be an integer")
    if num_samples < 2:
        raise ValueError(
            f"num_samples must be at least 2 to have both classes, got {num_samples}"
        )
    if noise_std < 0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")

    branch_groups = get_branch_index_groups(input_dim)
    num_branches = len(branch_groups)

    rng = random.Random(seed)

    # samples per class
    n_class0 = num_samples // 2
    n_class1 = num_samples - n_class0

    # Class centres per branch: class 0 and class 1 use mirrored centres so
    # that no single linear boundary separates them.
    class_centers = {
        0: [(-2.0, -2.0), (2.0, 2.0), (0.0, 0.0)],
        1: [(2.0, 2.0), (-2.0, -2.0), (0.0, 0.0)],
    }
    # Only the first three branch blocks have a defined centre pattern; for
    # wider inputs the remaining blocks are centred on the origin.
    centers_by_class = {
        label: (centers * ((num_branches // len(centers)) + 1))[:num_branches]
        for label, centers in class_centers.items()
    }

    X: List[List[float]] = []
    y: List[int] = []

    def sample_gaussian_2d(center: Tuple[float, float]) -> List[float]:
        return [
            rng.gauss(center[0], noise_std),
            rng.gauss(center[1], noise_std),
        ]

    for label, n_samples in ((0, n_class0), (1, n_class1)):
        centers = centers_by_class[label]
        for _ in range(n_samples):
            x: List[float] = []
            for center in centers:
                x.extend(sample_gaussian_2d(center))
            X.append(x)
            y.append(label)

    # shuffle the dataset
    indices = list(range(num_samples))
    rng.shuffle(indices)

    X = [X[i] for i in indices]
    y = [y[i] for i in indices]

    return X, y


def plot_2d_projection(
    X: List[List[float]],
    y: List[int],
    dim1: int = 0,
    dim2: int = 1,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot a 2D projection of the dataset for inspection.

    ``matplotlib`` is imported lazily so that importing this module (or running
    the training scripts) does not require a plotting backend.

    Parameters
    ----------
    X : list of samples
    y : list of labels
    dim1 : int
        First dimension to plot.
    dim2 : int
        Second dimension to plot.
    save_path : str, optional
        If given, the figure is written to this path.
    show : bool
        Whether to call ``plt.show()``. Set to ``False`` in headless contexts.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "matplotlib is required for plotting. Install it with "
            "`pip install matplotlib`."
        ) from exc

    input_dim = len(X[0]) if X else 0
    for dim in (dim1, dim2):
        if not 0 <= dim < input_dim:
            raise ValueError(
                f"dimension {dim} is out of range for data with {input_dim} features"
            )

    xs_0 = [x[dim1] for x, label in zip(X, y) if label == 0]
    ys_0 = [x[dim2] for x, label in zip(X, y) if label == 0]
    xs_1 = [x[dim1] for x, label in zip(X, y) if label == 1]
    ys_1 = [x[dim2] for x, label in zip(X, y) if label == 1]

    plt.scatter(xs_0, ys_0, label="Class 0", alpha=0.7)
    plt.scatter(xs_1, ys_1, label="Class 1", alpha=0.7)
    plt.xlabel(f"Feature {dim1}")
    plt.ylabel(f"Feature {dim2}")
    plt.legend()
    plt.title("2D projection of synthetic dataset")

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
