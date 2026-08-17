# Softmax Activation Function Implementation

import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    # Your implementation here
    s_t = torch.tensor(scores,dtype=torch.float)
    return torch.softmax(s_t,dim=0).round(decimals=4).tolist()
    

def main():
    print([round(x, 4) for x in softmax([1, 2, 3])])
    

if __name__ == "__main__":
    main()

#
# Created By jing At 2026-08-17 22:37:46
#
