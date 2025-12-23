#provides fct that creates toy inputs for testing purposes

import numpy as np
from typing import List, Optional, Tuple
import random
import matplotlib.pyplot as plt


def get_branch_index_grps(input_dim: int = 6) -> List[List[int]]:
    # input_dim = 6
    # #three branches, each with two features
    # num_branches = 3
    # num_features_per_branch = 2
    if input_dim != 6:
        raise NotImplementedError("Only input_dim=6 is implemented")
    return [[0, 1], [2, 3], [4, 5]]


# data generateion function
def make_synthetic_data(
    num_samples: int,
    input_dim: int = 6,
    noise_std: float = 0.4,
    seed: Optional[int] = None,
) -> Tuple[List[List[float]], List[int]]:
    if seed is not None:
        random.seed(seed)

    if input_dim != 6:
        raise NotImplementedError("Only input_dim=6 is implemented")

    #samples per class
    n_class0 = num_samples // 2
    n_class1 = num_samples - n_class0

    #class centers per branch
    class_centers = {
        0: [(-2, -2), (2, 2), (0, 0)],
        1: [(2, 2), (-2, -2), (0, 0)],
    }

    X: List[List[float]] = []
    y: List[int] = []

    #helper fct to generate samples for a given class
    def sample_gaussian_2d(center):
        return [
            random.gauss(center[0], noise_std),
            random.gauss(center[1], noise_std),
        ]

    #generate class 0 samples
    for _ in range(n_class0):
        x: List[float] = []
        for branch_center in class_centers[0]:
            x.extend(sample_gaussian_2d(branch_center))
        X.append(x)
        y.append(0)

    #geenerate class 1 samples
    for _ in range(n_class1):
        x = []
        for branch_center in class_centers[1]:
            x.extend(sample_gaussian_2d(branch_center))
        X.append(x)
        y.append(1)

    #shuffle the dataset
    indix = list(range(num_samples))
    random.shuffle(indix)

    X = [X[i] for i in indix]
    y = [y[i] for i in indix]

    return X, y


# ---------- Optional visualization ----------
def plot_2d_projection(
    X: List[List[float]],
    y: List[int],
    dim1: int = 0,
    dim2: int = 1,
):
    """
    Plot a 2D projection of the dataset for inspection.

    Parameters
    ----------
    X : list of samples
    y : list of labels
    dim1 : int
        First dimension to plot.
    dim2 : int
        Second dimension to plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

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
    plt.show()
