# Implement Reduced Row Echelon Form (RREF) Function

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


def main():
    matrix = np.array([[1, 2, -1, -4], [2, 3, -1, -11], [-2, 0, -3, 22]])

    output = rref(matrix)
    print(output)

    matrix = np.array([
        [0, 2, -1, -4],
        [2, 0, -1, -11],
        [-2, 0, 0, 22]
    ])

    output = rref(matrix)
    print(output)
    
    matrix = np.array([
            [1, 2, -1],
            [2, 4, -1],
            [-2, -4, -3]])

    output = rref(matrix)
    print(output)
        
if __name__ == "__main__":
    main()


#
# Created By jing At 2026-09-01 17:28:19
#
