# Calculate Accuracy Score

import torch
from typing import Union
import numpy as np

def accuracy_score(y_true: Union[torch.Tensor, list, "np.ndarray"],
                   y_pred: Union[torch.Tensor, list, "np.ndarray"]) -> float:
    """
    Compute the accuracy: fraction of matching elements in y_true and y_pred.
    Both inputs may be torch.Tensor, list, or numpy.ndarray.
    """
    return np.count_nonzero([y_true == y_pred]) / y_true.shape[0]


def main():
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 0, 1])
    output = accuracy_score(y_true, y_pred)
    print(output)
    print()
    
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-26 14:25:03
#
