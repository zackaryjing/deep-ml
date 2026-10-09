# Implement K-Fold Cross-Validation

from typing import List, Tuple

import torch

import numpy as np


def k_fold_cross_validation(
    n_samples: int, k: int = 5, shuffle: bool = True
) -> List[Tuple[List[int], List[int]]]:
    """
    Return train/test index splits for k-fold cross-validation.

    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds
        shuffle: Whether to shuffle indices before splitting

    Returns:
        List of (train_indices, test_indices) tuples, where each is a list of ints
    """
    # Your implementation here
    size_fold = n_samples // k
    raw_dataset = [i for i in range(n_samples)]
    if shuffle:
        np.random.shuffle(raw_dataset)
    dataset = raw_dataset
    size_first_fold = size_fold + n_samples % size_fold

    return [(dataset[size_first_fold:], dataset[:size_first_fold])] + [
        (
            dataset[: size_first_fold + size_fold * i]
            + dataset[size_first_fold + size_fold * (i + 1) :],
            dataset[
                size_first_fold + size_fold * i : size_first_fold + size_fold * (i + 1)
            ],
        )
        for i in range(k - 1)
    ]


def main():
    np.random.seed(42)
    print(k_fold_cross_validation(n_samples=10, k=3, shuffle=False))


if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-18 18:47:34
#
