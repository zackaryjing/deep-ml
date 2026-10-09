# Linear Kernel Function

import torch
import numpy as np


def kernel_function(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Computes the linear kernel between two input vectors.
    The linear kernel is defined as the dot product (inner product) of two vectors.

    Args:
        x1: First input tensor (1D vector)
        x2: Second input tensor (1D vector)

    Returns:
        Scalar tensor representing the linear kernel (dot product)
    """
    # Your implementation here
    return x1 @ x2


def main():
    x1 = torch.tensor(np.array([1, 2, 3]))
    x2 = torch.tensor(np.array([4, 5, 6]))
    print(kernel_function(x1, x2))


if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-26 10:51:36
#
