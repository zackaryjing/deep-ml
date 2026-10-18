# Implement Xavier/Glorot Weight Initialization

import torch
import math
import torch.nn as nn


def xavier_initialization(shape: tuple, mode: str = 'uniform', seed: int = None) -> torch.Tensor:
    """
    Implement Xavier/Glorot weight initialization using PyTorch.
    
    Args:
        shape: Tuple of (fan_in, fan_out) representing weight matrix dimensions
        mode: 'uniform' or 'normal' initialization
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Initialized weight matrix as torch.Tensor
    """
    # Your code here
    torch.manual_seed(seed)
    parameter = torch.empty(shape,dtype=torch.float32)
    fan_in,fan_out = shape
    rng = math.sqrt(6 / (fan_in + fan_out)) 
    std = math.sqrt(2 / (fan_in + fan_out))
    if mode == 'uniform':
        parameter.uniform_(-rng,rng)
    elif mode=='normal':
        parameter.normal_(0,std)
    else:
        raise ValueError("ModeError")
    return parameter

    


def main():
    result = xavier_initialization((3, 3), 'uniform', seed=42)
    print(list(result.shape))  

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-16 11:14:25
#
