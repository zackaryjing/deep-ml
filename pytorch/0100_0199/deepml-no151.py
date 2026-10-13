# Dropout Layer

import torch

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer.
        
        Attributes to set:
            self.p: the dropout rate
            self.mask: stores the dropout mask (initially None)
        """
        # Your code here
        self.p = p
        self.mask = None

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Forward pass of the dropout layer.
        
        Generate a new mask on each training forward pass and store it in self.mask.
        """
        # Your code here
        if training:
            self.mask = (torch.rand_like(x) > self.p).float()
            self.mask /= (1 - self.p)
        else:
            self.mask = torch.ones_like(x)
        return x * self.mask

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        """Backward pass of the dropout layer.
        Use the stored self.mask from the most recent forward pass.
        """
        # Your code here
        assert self.mask != None
        return grad * self.mask

def main():
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-11 23:36:48
#
