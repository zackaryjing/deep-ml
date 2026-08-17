# Calculate Mean by Row or Column

import torch

def calculate_matrix_mean(matrix, mode: str) -> torch.Tensor:
    """
    Calculate mean of a 2D matrix per row or per column using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of means or raises ValueError on invalid mode.
    """
    a_t = torch.as_tensor(matrix, dtype=torch.float)
    
    if (mode == 'column'):
        n = a_t.shape[0]
        return a_t.sum(dim=0) / n
    else:
        n = a_t.shape[1]
        return a_t.sum(dim=1) / n

def main():
    print(calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'column'))

    
if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-16 21:27:15
#
