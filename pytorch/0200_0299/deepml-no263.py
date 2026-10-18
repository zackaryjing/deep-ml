# Implement Binary Cross-Entropy Loss

import torch

def binary_cross_entropy(y_true: torch.Tensor, y_pred: torch.Tensor, epsilon: float = 1e-15) -> float:
    """
    Compute binary cross-entropy loss.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities (between 0 and 1)
        epsilon: Small value for numerical stability
    
    Returns:
        Mean binary cross-entropy loss
    """
    # Your code here
    bce = -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    return bce.item()

def main():
    result = binary_cross_entropy(torch.tensor([1.0, 0.0, 1.0, 0.0]), torch.tensor([0.9, 0.1, 0.8, 0.2]))
    print(round(result, 4))
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-14 20:23:18
#
