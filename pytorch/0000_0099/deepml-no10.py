# Calculate Covariance Matrix

import torch


def calculate_covariance_matrix(vectors) -> torch.Tensor:
    """
    Calculate the covariance matrix for given feature vectors using PyTorch.
    Input: 2D array-like of shape (n_features, n_observations).
    Returns a tensor of shape (n_features, n_features).
    """
    v_t = torch.as_tensor(vectors, dtype=torch.float)
    # Your implementation here
    n_features, n_observations = v_t.shape
    avg = v_t.sum(dim=1, keepdim=True).expand_as(v_t) / n_observations
    v_center = v_t - avg
    return 1 / (n_observations - 1) * v_center @ v_center.T


def main():
    print(calculate_covariance_matrix([[1, 2, 3], [4, 5, 6]]))


if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-14 21:29:39
#
