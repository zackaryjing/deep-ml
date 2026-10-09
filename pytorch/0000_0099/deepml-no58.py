# problem: Gaussian Elimination for Solving Linear Systems

import numpy as np


def gaussian_elimination(A: np.ndarray[float], b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.

    :param A: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """

    m, n = A.shape
    mat = np.concatenate([A, b.reshape(-1, 1)], axis=1).astype(float)
    for j in range(0,n):
        if abs(mat[j][j]) < 1e-10:
            for i in range(j,m):
                if abs(mat[i][j]) > 1e-10:
                    mat[j] += mat[i]
                    break
        for i in range(j + 1, m):
            ratio = mat[i][j] / mat[j][j]
            mat[i] -= ratio * mat[j]
    x = np.zeros_like(b)
    print(mat)
    for i in range(m - 1, -1, -1):
        x[i] = ((np.concatenate([-x, [1]]) @ mat[i]) / mat[i][i])
    return x


def main():
    A = np.array([[2, 8, 4], [2, 5, 1], [4, 10, -1]], dtype=float)
    b = np.array([2, 5, 1], dtype=float)

    print(gaussian_elimination(A, b))

    A = np.array([
        [0, 2, 1, 0, 0, 0, 0],
        [2, 6, 2, 1, 0, 0, 0],
        [1, 2, 7, 2, 1, 0, 0],
        [0, 1, 2, 8, 2, 1, 0],
        [0, 0, 1, 2, 9, 2, 1],
        [0, 0, 0, 1, 2, 10, 2],
        [0, 0, 0, 0, 1, 2, 11]
    ], dtype=float)
    b = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)
    print(gaussian_elimination(A, b))


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-04 16:30:39
#
