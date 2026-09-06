# problem: Find the column space of a matrix


import numpy as np

def rref(matrix: np.ndarray):
    matrix = matrix.astype(dtype=float)
    m, n = matrix.shape
    column = 0
    handled = []
    for i in range(m):
        row = 0
        found = False
        while not found:
            for k in range(m):
                if k not in handled and matrix[k][column] != 0:
                    row = k
                    found = True
                    break
            if not found:
                column += 1
            if column >= n:
                handled.extend(list(set(range(m)) - set(handled)))
                break
        if found:
            handled.append(row)
        else:
            break
        matrix[row, :] *= 1 / matrix[row][column]
        for j in range(row):
            if matrix[j][column] != 0:
                matrix[j, :] -= matrix[j][column] * matrix[row, :]
        for j in range(row + 1, m):
            if matrix[j][column] != 0:
                matrix[j, :] -= matrix[j][column] * matrix[row, :]
    matrix[matrix == 0] = 0
    matrix = matrix[handled]
    return matrix

def matrix_image(A:np.ndarray):
    # Write your code here
    B = rref(A)
    n,m = A.shape
    res = []
    for i in range(n):
        for j in range(m):
            if B[i,j] == 1 :
                res.append( j )
    return A[:,res]

def main():
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    print(matrix_image(matrix))

if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-06 15:29:28
#
