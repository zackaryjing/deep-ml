# problem: Implement Gini Impurity Calculation for a Set of Classes
from collections import Counter

import torch

def gini_impurity(y: torch.Tensor) -> float:
    """
    Calculate Gini Impurity for a tensor of class labels.

    :param y: 1D Tensor of class labels (integer type)
    :return: Gini Impurity rounded to three decimal places
    """
    n = y.shape[0]
    counts = Counter(y.tolist())
    temp = 0
    for _,cnt in counts.items():
        temp += (cnt / n) ** 2
    return 1 - temp


# 4 9 / 25
# 13 / 25

def main():
    y = torch.tensor([0, 1, 1, 1, 0])
    print(gini_impurity(y))


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-06 14:01:00
#
