# Implement Gradient Clipping by Value

import torch

def clip_gradients_by_value(gradients: torch.Tensor, clip_value: float) -> torch.Tensor:
    """
    Clip gradient values to be within [-clip_value, clip_value].
    
    Args:
        gradients: A torch.Tensor representing gradients (any shape)
        clip_value: The maximum absolute value for any gradient element (non-negative)
    
    Returns:
        Clipped gradients with same shape as input
    """
    gradients = gradients.clamp(-clip_value,clip_value)
    return gradients
    

def main():
    gradients = torch.tensor([1.0, -2.0, 3.0, -0.5, 0.2], dtype=torch.float64)
    result = clip_gradients_by_value(gradients, 1.5)
    print(result.numpy().tolist())
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-17 23:18:05
#
