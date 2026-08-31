# Calculate Correlation Matrix

import torch
from typing import Optional, Union
import numpy as np


def calculate_correlation_matrix(
    X: Union[torch.Tensor, list, "np.ndarray"],
    Y: Optional[Union[torch.Tensor, list, "np.ndarray"]] = None,
) -> torch.Tensor:
    """
    Compute the correlation matrix of X (and optionally Y) using PyTorch.
    If Y is None, returns the correlation matrix of X with itself.
    """
    X = torch.tensor(X, dtype=torch.float)
    if Y is not None:
        Y = torch.tensor(Y, dtype=torch.float)
    else:
        Y = X
    n_samples, n_features = X.shape
    return (
        1 / (n_samples - 1) * (X - X.mean(dim=0)).T @ (Y - Y.mean(dim=0)) / (torch.outer(X.std(dim=0) ,Y.std(dim=0).T))
    )


def main():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    output = calculate_correlation_matrix(X)
    print(output)
    
    X = np.array([[1,2,3],[7,15,6],[7,8,9]])
    print(torch.round(calculate_correlation_matrix(X) * 1e8) / 1e8)


if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-26 14:29:35
#
