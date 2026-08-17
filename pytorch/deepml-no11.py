# Solve Linear Equations using Jacobi Method

import torch
import numpy as np


def solve_jacobi(A, b, n) -> torch.Tensor:
    """
    Solve Ax = b using the Jacobi iterative method for n iterations.
    A: (m,m) tensor; b: (m,) tensor; n: number of iterations.
    Returns a 1-D tensor of length m, rounded to 4 decimals.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)
    # Your implementation here
    #  ? ?
    m = b_t.shape[0]
    theta = torch.zeros((m,))
    
    for _ in range(n):
        theta_new = torch.zeros(m)
        for i in range(m):
            theta_new[i] = (b_t[i] - A_t[i] @ theta + A_t[i,i] * theta[i]) / A_t[i,i]
        theta = theta_new
    
    return theta


def main():
    print(
        solve_jacobi(
            np.array([[5, -2, 3], [-3, 9, 1], [2, -1, -7]]), np.array([-1, 2, 3]), 2
        )
    )
    print(
        solve_jacobi(
            np.array([[5, -2, 3], [-3, 9, 1], [2, -1, -7]]), np.array([-1, 2, 3]), 50
        )
    )


if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-15 20:23:59
#
