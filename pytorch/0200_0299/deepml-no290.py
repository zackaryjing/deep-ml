# Implement He Weight Initialization

import torch
from torch.nn import parameter
import math

def he_initialization(n_in: int, n_out: int, mode: str = 'fan_in', distribution: str = 'normal', seed: int = None) -> torch.Tensor:
    """
    Implement He (Kaiming) weight initialization using PyTorch.
    
    Parameters:
    n_in: number of input units
    n_out: number of output units
    mode: 'fan_in' or 'fan_out'
    distribution: 'normal' or 'uniform'
    seed: random seed for reproducibility
    
    Returns:
    torch.Tensor of shape (n_in, n_out) with He-initialized weights
    """
    torch.manual_seed(seed)
    fan = n_out
    if mode == 'fan_in':
        fan = n_in
        
    parameter = torch.empty((n_in,n_out))
    if distribution == 'normal':
        std = math.sqrt(2 / fan)
        parameter.normal_(0, std)
    elif distribution == 'uniform':
        rng = math.sqrt(6 / fan)
        parameter.uniform_(-rng,rng)
    else:
        raise ValueError("ModeError")
    return parameter
        
        

    
    
    
def main():
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-16 19:07:12
#
