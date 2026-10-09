# Calculate Root Mean Square Error (RMSE)

import torch
import torch.nn.functional as F

def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate Root Mean Square Error (RMSE) between actual and predicted values.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.

    Returns:
        RMSE value rounded to three decimal places.
    """
    # Write your code here
    residuals = (y_true - y_pred)
    n = y_true.shape[0]
    res = torch.sqrt(1 / n *  (residuals @ residuals))
    return round(res.item(),3)

def main():
    y_true = torch.tensor([3, -0.5, 2, 7])
    y_pred = torch.tensor([2.5, 0.0, 2, 8])
    print(rmse(y_true, y_pred))
        

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-09-08 08:49:09
#
