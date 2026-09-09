# problem: Generate a Confusion Matrix for Binary Classification

import torch


def confusion_matrix(data: list) -> torch.Tensor:
    """
    Generate a 2x2 confusion matrix for binary classification.

    Args:
        data: A list of [y_true, y_pred] pairs for binary labels (0 or 1)

    Returns:
        A 2x2 torch.Tensor confusion matrix arranged as [[TP, FN], [FP, TN]]
    """
    y_true,y_pred = zip(*data)
    y_true = torch.tensor(y_true,dtype=torch.bool)
    y_pred = torch.tensor(y_pred,dtype=torch.bool)
    return torch.tensor([[torch.count_nonzero(y_true & ~y_pred), torch.count_nonzero(~y_pred & y_true)],
                        [torch.count_nonzero(y_pred & ~y_true),torch.count_nonzero(y_true & y_pred)]])



def main():
    data = [[1, 1], [1, 0], [0, 1], [0, 0], [0, 1]]
    print(confusion_matrix(data))


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-08 10:17:06
#
