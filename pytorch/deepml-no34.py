# One-Hot Encoding of Nominal Values

import torch
from typing import Optional
import numpy as np
from torch.nn.functional import one_hot

def to_categorical(x: torch.Tensor, n_col: Optional[int] = None) -> torch.Tensor:
    """
    Perform one-hot encoding on a 1D integer tensor `x`. If `n_col` is not provided, infer it from the max value in `x`.
    """
    # Hint: You can use torch.nn.functional.one_hot
    return one_hot(x,num_classes=n_col if n_col else -1).to(dtype=torch.float,copy=False)


def main():
    x = torch.tensor(np.array([0, 1, 2, 1, 0]))
    output = to_categorical(x,4)
    print(output)
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-26 09:33:52
#
