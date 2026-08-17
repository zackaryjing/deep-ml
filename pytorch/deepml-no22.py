# Sigmoid Activation Function Understanding

import torch

def sigmoid(z: float) -> float:
    """
    Compute the sigmoid activation function.
    Input:
      - z: float or torch scalar tensor
    Returns:
      - sigmoid(z) as Python float rounded to 4 decimals.
    """
    # Your implementation here
    return torch.sigmoid(torch.tensor(z)).round(decimals=4).item()

def main():
    print(sigmoid(0))

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-17 21:45:06
#
