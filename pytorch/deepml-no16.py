# Feature Scaling Implementation

import torch
import numpy as np


def feature_scaling(data) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standardize and Min-Max normalize input data using PyTorch.
    Input: Tensor or convertible of shape (m,n).
    Returns (standardized_data, normalized_data), both rounded to 4 decimals.
    """
    data_t = torch.as_tensor(data, dtype=torch.float)
    mx, _ = data_t.max(dim=0)
    mn, _ = data_t.min(dim=0)
    min_max_norm = (data_t - mn) / (mx - mn)
    mean = data_t.mean(dim=0)
    std = data_t.std(dim=0, correction=0)
    standardize_norm = (data_t - mean) / std
    return (standardize_norm.round(decimals=4), min_max_norm.round(decimals=4))


def main():
    print(feature_scaling(np.array([[1, 2], [3, 4], [5, 6]])))


if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-17 20:03:51
#
