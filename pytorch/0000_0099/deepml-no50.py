# Implement Lasso Regression using ISTA

import torch

def soft_threshold(w: torch.Tensor, threshold: float) -> torch.Tensor:
    """Apply soft-thresholding operator element-wise.
    
    S(w, λ) = sign(w) * max(|w| - λ, 0)
    
    Args:
        w: Input tensor
        threshold: Threshold value λ
    
    Returns:
        Soft-thresholded tensor where:
        - Values with |w| > λ are shrunk toward zero by λ
        - Values with |w| ≤ λ become exactly zero
    """
    # Your code here
    return torch.sign(w) * torch.clamp(w.abs() - threshold,min=0)
    

def l1_regularization_gradient_descent(X: torch.Tensor, y: torch.Tensor, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
    """
    Implement Lasso Regression using ISTA (Iterative Shrinkage-Thresholding Algorithm).
    
    ISTA alternates between:
    1. Gradient step on MSE loss: w_temp = w - lr * gradient_mse
    2. Proximal step (soft-thresholding): w_new = soft_threshold(w_temp, lr * alpha)
    
    Args:
        X: Feature matrix tensor of shape (n_samples, n_features)
        y: Target vector tensor of shape (n_samples,)
        alpha: L1 regularization strength
        learning_rate: Step size for gradient descent
        max_iter: Maximum iterations
        tol: Convergence tolerance on weight change
    
    Returns:
        tuple: (weights, bias) as torch.Tensors
    
    Note: The bias term is NOT regularized.
    """
    
    n_samples, n_features = X.shape
    weights = torch.zeros(n_features, dtype=torch.float32)
    bias = torch.tensor(0.0, dtype=torch.float32)
    for t in range(max_iter):
        residual = X @ weights + bias - y
        grad_w = (2.0 / n_samples) * X.T @ residual
        grad_b = (2.0 / n_samples) * residual.sum()
        weights -= learning_rate * grad_w
        weights = soft_threshold(weights,learning_rate * alpha)
        bias -= learning_rate * grad_b
    return weights,bias


def main():
    # Test that Lasso produces sparse weights (some exactly zero)
    X = torch.tensor([[1, 0.01], [2, 0.02], [3, 0.03], [4, 0.04], [5, 0.05]], dtype=torch.float32)
    y = torch.tensor([2, 4, 6, 8, 10], dtype=torch.float32)

    weights, bias = l1_regularization_gradient_descent(X, y, alpha=0.5, learning_rate=0.01, max_iter=1000)
    print(f"Sparse weight is zero: {weights[1].item() == 0.0}")
    
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-09-03 15:47:15
#
