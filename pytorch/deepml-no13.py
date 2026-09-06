# problem: Determinant of a 4x4 Matrix using Laplace's Expansion (hard)

import torch


def determinant_4x4(matrix) -> float:
    """
    Compute the determinant of a 4×4 matrix using PyTorch.
    Input can be a Python list, NumPy array, or torch Tensor of shape (4,4).
    Returns a Python float.
    """
    # Convert to tensor
    m = torch.as_tensor(matrix, dtype=torch.float)

    # Your implementation here
    def compute(matrix: torch.Tensor) -> torch.Tensor:
        m, n = matrix.shape
        res = 0
        if n == 2:
            return matrix[0,0] * matrix[1,1] - matrix[0,1] * matrix[1,0]
        for j in range(n):
            row_index = torch.arange(1, m)
            col_index = torch.cat([torch.arange(0, j), torch.arange(j + 1, n)])
            res += (-1) ** j * matrix[0, j] * compute(matrix[row_index][:, col_index])
        return res
    return compute(m).item()


def main():
    a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    print(determinant_4x4(a))
    print(determinant_4x4(torch.eye(4)))


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-06 12:46:41
#
