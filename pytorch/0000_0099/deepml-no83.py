# problem: Dot Product Calculator

import torch


def calculate_dot_product(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the dot product of two vectors.
    Args:
        vec1 (torch.Tensor): 1D tensor representing the first vector.
        vec2 (torch.Tensor): 1D tensor representing the second vector.
    Returns:
        torch.Tensor: The dot product of the two vectors as a scalar tensor.
    """
    # Your code here
    return vec1 @ vec2


def main():
    vec1 = calculate_dot_product(torch.tensor([1, 2, 3]), vec2 = torch.tensor([4, 5, 6]))
    print(vec1)


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-10 13:28:02
#
