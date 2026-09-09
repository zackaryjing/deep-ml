# Calculate Dice Score for Classificatio

import torch


def dice_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the Dice Score (Sørensen-Dice coefficient) for binary classification.

    Args:
        y_true: Binary tensor of true labels.
        y_pred: Binary tensor of predicted labels.

    Returns:
        Dice Score as a float rounded to 3 decimal places.
    """
    # Write your code here
    if torch.count_nonzero(y_pred) == torch.count_nonzero(y_true) == 0:
        return 0.0
    elif (torch.count_nonzero(y_pred & y_true)) == 0:
        return 0.0
    return round(
        (
            2
            * (torch.count_nonzero(y_pred & y_true))
            / (torch.count_nonzero(y_true) + torch.count_nonzero(y_pred))
        ).item(),
        3,
    )


def main():
    y_true = torch.tensor([1, 1, 0, 1, 0, 1])
    y_pred = torch.tensor([1, 1, 0, 0, 0, 1])
    print(dice_score(y_true, y_pred))

    y_true = torch.tensor([1, 1, 0, 1, 0, 1])
    y_pred = torch.tensor([0, 0, 0, 0, 0, 0])
    print(dice_score(y_true, y_pred))

    y_true = torch.tensor([0, 0, 0, 0, 0, 0])
    y_pred = torch.tensor([0, 0, 0, 0, 0, 0])
    print(dice_score(y_true, y_pred))


if __name__ == "__main__":
    main()


#
# Created By jing At 2026-09-08 09:15:31
#
