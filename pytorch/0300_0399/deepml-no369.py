# problem:

import math
import torch
import torch.nn as nn


def xavier_init(fan_in: int, fan_out: int, mode: str = 'uniform', seed: int = 42) -> dict:
    """
    Perform Xavier/Glorot weight initialization using PyTorch built-ins.

    Args:
        fan_in (int): Number of input units.
        fan_out (int): Number of output units.
        mode (str): 'uniform' or 'normal'.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Contains 'weights' (nested list), 'shape' (list), and 'param' (float).
    """
    # Your code here
    torch.manual_seed(seed)
    parameter = torch.empty((fan_in, fan_out))
    a = 0
    if mode == 'uniform':
        a = math.sqrt(6 / (fan_in + fan_out))
        parameter.uniform_(-a, a)
    elif mode == 'normal':
        a = math.sqrt(2 / (fan_in + fan_out))
        parameter.normal_(0, a)
    else:
        raise ValueError("mode must be 'uniform' or 'normal'")
    return {"weights": parameter.tolist(),
            "shape": [fan_in, fan_out],
            "param": round(a, 4)}


def main():
    result = xavier_init(5, 3, 'uniform', seed=1)
    print(result['shape'])


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-10-18 17:01:55
#
