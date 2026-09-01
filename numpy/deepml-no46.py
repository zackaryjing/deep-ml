# Implement Precision Metric

import numpy as np


def precision(y_true, y_pred):
    # Your code here
    dividsion = np.count_nonzero(y_pred)
    return (
        0.0
        if dividsion == 0
        else 1 - np.count_nonzero((y_true ^ y_pred) & y_pred) / dividsion
    )


# y_true 0100101
# y_pred 0010101
#        0110000


def main():
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0])
    # y_pred = np.array([1, 0, 1, 0, 0, 1])

    result = precision(y_true, y_pred)
    print(result)


if __name__ == "__main__":
    main()


#
# Created By jing At 2026-09-01 15:51:17
#
