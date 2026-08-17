# Scalar Multiplication of a Matrix
import torch

def scalar_multiply(matrix, scalar) -> torch.Tensor:
    """
    Multiply each element of a 2D matrix by a scalar using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of the same shape.
    """
    # Convert input to tensor
    m_t = torch.as_tensor(matrix, dtype=torch.float)
    # Your implementation here

    m_t *= torch.tensor(scalar)
    return m_t


def main():
    print(scalar_multiply([[1, 2], [3, 4]], 2))


if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-16 20:19:47
#
