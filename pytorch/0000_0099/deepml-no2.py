# problem: Transpose of a Matrix

import torch


def transpose_matrix(a) -> torch.Tensor:
    """
    Transpose a 2D matrix using PyTorch.

    Args:
        a: A 2D matrix (can be list, numpy array, or torch.Tensor)

    Returns:
        A transposed torch.Tensor
    """
    a_t = torch.as_tensor(a)
    # Your code here
    return a_t.transpose(0, 1)


def main():
    test = [[1], [2]]
    print(transpose_matrix(test))


if __name__ == "__main__":
    main()

#
# Created By ASUS At 2026-04-09 16:04:11
#
