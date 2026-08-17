# problem: Calculate Eigenvalues of a Matrix


from typing import *
import torch
import math


def calculate_eigenvalues(matrix: torch.Tensor | List) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    # Your implementation here
    a, b = matrix[0][0], matrix[0][1]
    c, d = matrix[1][0], matrix[1][1]
    lambda1 = math.sqrt(c * b - a * d + (a + d) ** 2 / 4) + (a + d) / 2
    lambda2 = -math.sqrt(c * b - a * d + (a + d) ** 2 / 4) + (a + d) / 2
    return torch.tensor([lambda1, lambda2])


def main():
    test = [[2, 1], [1, 2]]
    print(calculate_eigenvalues(test))


if __name__ == "__main__":
    main()

#
# Created By ASUS At 2026-04-09 18:43:23
#
