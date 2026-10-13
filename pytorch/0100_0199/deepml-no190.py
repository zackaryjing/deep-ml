# Overlapping Max Pooling 

import torch
import torch.nn.functional as F

def overlapping_max_pool2d(x: torch.Tensor, kernel_size: int = 3, stride: int = 2) -> torch.Tensor:
    """
    Apply overlapping max pooling using PyTorch.
    Must match the ceil mode behavior described in the problem.
    """
    res = F.max_pool2d(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        ceil_mode=True
    )
    # Your code here
    return res 

def main():
    x = torch.arange(1, 17, dtype=torch.float32).view(1, 1, 4, 4)
    print(overlapping_max_pool2d(x, 3, 2))
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-13 12:51:47
#
