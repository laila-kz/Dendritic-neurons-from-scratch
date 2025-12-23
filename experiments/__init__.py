"""
Experiments package.

This package contains:

- Synthetic dataset generation
- Training scripts for dendritic and point neurons
- Evaluation and comparison utilities
"""

from experiments.synthetic_dataset import make_synthetic_data, get_branch_index_grps

__all__ = [
    "make_synthetic_data",
    "get_branch_index_grps",
]
