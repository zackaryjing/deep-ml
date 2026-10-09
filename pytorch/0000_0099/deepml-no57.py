# problem: Gauss-Seidel Method for Solving Linear Systems

import torch


def gauss_seidel(A: torch.Tensor, b: torch.Tensor, n: int, x_ini=None) -> torch.Tensor:
    """
    Implements the Gauss-Seidel iterative method for solving linear systems Ax = b.

    Args:
        A: Square coefficient matrix (torch.Tensor)
        b: Right-hand side vector (torch.Tensor)
        n: Number of iterations
        x_ini: Optional initial guess tensor (if None, zeros are used)

    Returns:
        Approximated solution vector x after n iterations
    """
    m = b.shape[0]
    if x_ini:
        theta = x_ini
    else:
        theta = torch.zeros((m,),dtype=torch.float64)
    for _ in range(n):
        for i in range(m):
            theta[i] = (b[i] - A[i] @ theta + A[i, i] * theta[i]) / A[i, i]
    return theta


def main():

    A = torch.tensor([[4, 1, 2], [3, 5, 1], [1, 1, 3]], dtype=torch.float64)
    b = torch.tensor([4, 7, 3], dtype=torch.float64)
    n = 5
    result = gauss_seidel(A, b, n)
    print([round(v, 8) for v in result.numpy().tolist()])


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-04 15:59:53
#
