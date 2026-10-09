# problem: Implement Orthogonal Projection of a Vector onto a Line

import torch


def orthogonal_projection(v: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Compute the orthogonal projection of vector v onto line L using PyTorch.

    :param v: The vector to be projected (torch.Tensor)
    :param L: The line vector defining the direction of projection (torch.Tensor)
    :return: torch.Tensor representing the projection of v onto L, rounded to 3 decimal places
    """
    return (v @ L) / (L @ L) * L


def main():
    v = torch.tensor([3, 4])
    L = torch.tensor([1, 0])
    print(orthogonal_projection(v, L))


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-06 14:47:56
#
