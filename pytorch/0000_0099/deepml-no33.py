# Generate Random Subsets of a Dataset

import numpy as np
from typing import List, Tuple
from itertools import combinations_with_replacement,combinations
import torch
from torch.utils.data import TensorDataset

def get_random_subsets(X, y, n_subsets, replacements=True) -> list:
    """
    Generate n_subsets random subsets from the dataset (X, y).
    Each subset is a tuple (X_subset, y_subset), where both are lists.
    
    Args:
        X: 2D array of shape (n_samples, n_features)
        y: 1D array of shape (n_samples,)
        n_subsets: Number of subsets to generate
        replacements: If True, sample with replacement
                      If False, sample without replacement
    
    Returns:
        List of (X_subset, y_subset) tuples
    """
    
    np.random.seed(42)
    n_samples, n_features = X.shape
    res = []
    for _ in range(n_subsets):
        subset = []
        label = []
        count = n_samples if replacements else n_samples // 2
        index = np.random.choice(n_samples,count,replacements)
        content_x = X[list(index)]
        content_y = y[list(index)]
        subset.append(content_x)
        label.append(content_y)
        res.append((subset,label))
    return res

def main():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    n_subsets = 3
    replacements = False
    print(get_random_subsets(X, y, n_subsets, replacements))
    
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-26 13:13:58
#
