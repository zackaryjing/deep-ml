# Calculate Jaccard Index for Binary Classification

import torch

def jaccard_index(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the Jaccard Index for binary classification.

    Args:
        y_true: Binary tensor of true labels.
        y_pred: Binary tensor of predicted labels.

    Returns:
        Jaccard Index as a float rounded to 3 decimal places.
    """
    # Write your code here
    numerator = torch.count_nonzero(y_true & y_pred)
    denominator = torch.count_nonzero(y_true | y_pred)
    if (denominator == 0): 
        return 0
    return round((numerator / denominator).item(),3)


def main():
    y_true = torch.tensor([1, 0, 1, 1, 0, 1])
    y_pred = torch.tensor([1, 0, 1, 0, 0, 1])
    print(jaccard_index(y_true, y_pred))
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-09-08 08:56:07
#
