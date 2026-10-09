# Implement Batch Normalization for BCHW Input

import numpy as np
import torch

def batch_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """Perform Batch Normalization on a 4D tensor in BCHW format."""
    mean = X.mean(dim=(0,2,3),keepdim=True) 
    var = X.var(dim=(0,2,3),keepdim=True,unbiased=False)
    return ((X - mean) / torch.sqrt(var + epsilon)) * gamma+ beta

def main():
    X = torch.randn(2, 2, 2, 2)
    gamma = torch.ones((1, 2, 1, 1))
    beta = torch.zeros((1, 2, 1, 1))
    print(batch_normalization(X,gamma,beta,0.001))
    
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-09 16:00:04
#

