# Implement Group Normalization

import torch
import numpy as np

def group_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_groups: int, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Group Normalization on a 4D input tensor.
    
    Args:
        X: torch tensor of shape (B, C, H, W), input data
        gamma: torch tensor of shape (1, C, 1, 1), scale parameter
        beta: torch tensor of shape (1, C, 1, 1), shift parameter
        num_groups: number of groups for normalization
        epsilon: small constant to avoid division by zero
    
    Returns:
        norm_X: torch tensor of shape (B, C, H, W), normalized output
    """
    B,C,H,W = X.shape
    G = num_groups
    X = X.reshape(B,G,C // G ,H,W)
    print(X.shape)
    mean = X.mean(dim=(2,3,4),keepdim=True)
    var = X.var(dim=(2,3,4),unbiased=False,keepdim=True)
    X_norm =  ((X - mean) / torch.sqrt(var + epsilon))
    print(X_norm.shape)
    X_norm = X_norm.reshape(B,C,H,W)
    X_norm = X_norm * gamma + beta
    return X_norm
    

def main():
    np.random.seed(42)
    B, C, H, W = 2, 2, 2, 2
    X = torch.from_numpy(np.random.randn(B, C, H, W))
    gamma = torch.ones(C, dtype=torch.float64).reshape(1, C, 1, 1)
    beta = torch.zeros(C, dtype=torch.float64).reshape(1, C, 1, 1)
    num_groups = 2
    output = group_normalization(X, gamma, beta, num_groups)
    print([[[[round(v, 4) for v in r] for r in c] for c in b] for b in output.numpy().tolist()])
    
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-10 15:34:15
#
