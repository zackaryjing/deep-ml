# problem: Implement RMSNorm (Root Mean Square Layer Normalization)


import numpy as np
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
    n = x.shape[-1]
    rms = torch.sqrt(torch.sum(x ** 2,dim=-1) / n + eps ) # [features]
    # print(torch.sum(x ** 2,dim=0))
    x_norm = x / rms[:,None] * g
    return x_norm




def main():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    g = torch.tensor([1.0, 1.0, 1.0])
    result = rmsnorm(x, g)
    print(np.round(result.numpy(), 4).tolist())

    x = torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
    g = torch.tensor([1.0, 1.0, 1.0, 1.0])
    result = rmsnorm(x, g)
    print(np.round(result.numpy(), 4).tolist())


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-10-18 23:26:43
#
