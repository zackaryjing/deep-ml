# Implement Ridge Regression Loss Function

import torch
from torch.nn.functional import mse_loss

def ridge_loss(X: torch.Tensor, w: torch.Tensor, y_true: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Implements the Ridge Regression Loss Function using PyTorch.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        w: Weight vector of shape (n_features,)
        y_true: True target values of shape (n_samples,)
        alpha: Regularization parameter (lambda)
    
    Returns:
        The Ridge loss value as a scalar tensor
    """
    y_true = y_true.to(dtype=torch.float,copy=False)
    X = X.to(dtype=torch.float,copy=False)
    
    return mse_loss(y_true,X @ w) + alpha * (w @ w.T)
    
    


def main():
        
    X = torch.tensor([[1, 1], [2, 1], [3, 1], [4, 1]])
    w = torch.tensor([0.2, 2])
    y_true = torch.tensor([2, 3, 4, 5])
    alpha = 0.1

    loss = ridge_loss(X, w, y_true, alpha)
    print(loss)
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-26 12:05:34
#
