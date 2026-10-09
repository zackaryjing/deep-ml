# https://www.deep-ml.com/problems/9

import torch

def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    m1,n1 = a.shape
    m2,n2 = b.shape
    if (n1 != m2): return torch.tensor(-1);
    return a @ b

def main():
    print(matrixmul([[1,2,3],[2,3,4],[5,6,7]],[[3,2,1],[4,3,2],[5,4,3]]))
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-15 16:13:34
#
