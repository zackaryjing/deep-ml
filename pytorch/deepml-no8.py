# Calculate 2x2 Matrix Inverse

import torch


def inverse_2x2(matrix) -> torch.Tensor | None:
    """
    Compute the inverse of a 2x2 matrix using PyTorch.

    Args:
        matrix: A 2x2 matrix (can be list, numpy array, or torch.Tensor)

    Returns:
        A 2x2 tensor containing the inverse, or None if the matrix is singular
    """
    m = torch.as_tensor(matrix, dtype=torch.float)
    # Your code here
    a, b = m[0]
    c, d = m[1]
    if a * d - b * c == 0:
        return None
    else:
        return m.inverse()


def main():
    print(inverse_2x2([[4, 7], [2, 6]]))


if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-15 14:23:32
#
