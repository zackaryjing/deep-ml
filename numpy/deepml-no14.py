# Linear Regression Using Normal Equation

import numpy as np


def linear_regression_normal_equation(
    X: list[list[float]], y: list[float]
) -> list[float]:
    # Your code hereg, make sure to roun
    theta = np.linalg.pinv(X) @ y
    return list(theta)


def main():
    print(linear_regression_normal_equation([[1, 1], [1, 2], [1, 3]], [1, 2, 3]))


if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-14 20:17:57
#
