# Matrix Transformation 

import torch

def transform_matrix(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform T⁻¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2×2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)
    # Your implementation here
    if (T_t.shape[0] == torch.linalg.matrix_rank(T_t) and S_t.shape[0] == torch.linalg.matrix_rank(S_t)): 
        return T_t.inverse() @ A_t @ S_t
    return torch.tensor(-1)

def main():
    print(transform_matrix([[1, 2], [3, 4]], [[2, 0], [0, 2]], [[1, 1], [0, 1]]))
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-16 20:19:47
#
