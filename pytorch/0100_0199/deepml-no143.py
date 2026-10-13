# Instance Normalization (IN) Implementation

import torch
import numpy as np

def instance_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Instance Normalization over a 4D tensor X of shape (B, C, H, W).
    gamma: scale parameter of shape (C,)
    beta: shift parameter of shape (C,)
    epsilon: small value for numerical stability
    Returns: normalized tensor of same shape as X
    """
    mean = X.mean(dim=(2,3),keepdim=True)
    var = X.var(dim=(2,3),keepdim=True,unbiased=False)
    X_norm = (X - mean) / (torch.sqrt(var + epsilon)) * gamma + beta
    return X_norm

def main():
    B, C, H, W = 2, 2, 2, 2
    np.random.seed(42)
    X = torch.from_numpy(np.random.randn(B, C, H, W))
    gamma = torch.ones(C, dtype=torch.float64)
    beta = torch.zeros(C, dtype=torch.float64)
    out = instance_normalization(X, gamma, beta)
    print(np.round(out[1,1].numpy(), 4).tolist())
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-10 15:55:38
#
