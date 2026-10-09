# problem: Implement F-Score Calculation for Binary Classification

import torch


def f_score(y_true: torch.Tensor, y_pred: torch.Tensor, beta: float) -> float:
    """
    Calculate F-Score for a binary classification task.

    :param y_true: torch.Tensor of true labels (binary)
    :param y_pred: torch.Tensor of predicted labels (binary)
    :param beta: The weight of precision in the harmonic mean
    :return: F-Score rounded to three decimal places
    """
    precision = torch.count_nonzero( y_true & y_pred ) / torch.count_nonzero(y_pred)
    recall = torch.count_nonzero(y_true & y_pred) / torch.count_nonzero(y_true)

    return ((1 + beta ** 2 ) * precision * recall / ((beta ** 2 * precision) + recall)).item()


def main():
    y_true = torch.tensor([1, 0, 1, 1, 0, 1])
    y_pred = torch.tensor([1, 0, 1, 0, 0, 1])
    beta = 1

    print(f_score(y_true, y_pred, beta))



if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-05 15:35:07
#
