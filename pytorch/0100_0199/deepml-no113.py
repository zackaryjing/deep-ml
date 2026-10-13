# Implement a Simple Residual Block with Shortcut Connection

import torch

def residual_block(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """
    Implement a simple residual block with shortcut connection.
    
    Args:
        x: 1D input tensor
        w1: First weight matrix
        w2: Second weight matrix
    
    Returns:
        Output tensor after residual block processing
    """
    x_ = w1 @ x
    x_ = torch.relu(x_)
    x_ = w2 @ x_
    return torch.relu(x_ + x)

def main():
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-11 23:27:30
#
