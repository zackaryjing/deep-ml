# problem: Reshape Matrix

import torch


def reshape_matrix(a, new_shape) -> torch.Tensor:
    """
    Reshape a 2D matrix `a` to shape `new_shape` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a tensor of shape `new_shape`, or an empty tensor on mismatch.
    """
    # Dimension check
    if len(a) * len(a[0]) != new_shape[0] * new_shape[1]:
        return torch.tensor([])
    # Convert to tensor and reshape
    a_t = torch.as_tensor(a, dtype=torch.float)
    # Your implementation here
    return a_t.reshape(new_shape)


def main():
    test = [[1, 2, 3, 4], [5, 6, 7, 8]]
    print(reshape_matrix(test, (4, 2)))


if __name__ == "__main__":
    main()

#
# Created By ASUS At 2026-04-09 16:13:42
#
