# problem: Implement RMSNorm (Root Mean Square Layer Normalization)

import torch


def rmsnorm(x: torch.Tensor, g: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Apply RMSNorm to the input tensor.

    Parameters:
        x   : torch.Tensor of shape (batch_size, features)
        g   : torch.Tensor of shape (features,) - gain parameter
        eps : float - small constant for numerical stability

    Returns:
        torch.Tensor of same shape as x
    """
    pass


def main():


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-10-18 23:26:43
#
