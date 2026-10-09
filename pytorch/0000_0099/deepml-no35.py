# Convert Vector to Diagonal Matrix

import torch
from typing import Union
import numpy as np

def make_diagonal(x: Union[torch.Tensor, list, "np.ndarray"]) -> torch.Tensor:
    """Return a diagonal matrix whose diagonal elements are the 1-D values in `x`.
    If `x` is not a torch tensor it will be converted automatically.
    
    Hint: `torch.diag_embed` makes this very short!
    """
    x = torch.tensor(x,dtype=torch.float)
    return torch.diagflat(x)


def main():
    print(make_diagonal(torch.tensor(np.array([1, 2, 3]))))
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-26 14:10:44
#
