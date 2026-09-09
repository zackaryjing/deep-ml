# problem: Calculate R-squared for Regression Analysis

import torch

def r_squared(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the R-squared (R²) coefficient of determination using PyTorch.

    Args:
        y_true (torch.Tensor): Tensor of true values
        y_pred (torch.Tensor): Tensor of predicted values

    Returns:
        float: R-squared value rounded to 3 decimal places
    """
    residuals = y_pred - y_true
    SSR = residuals @ residuals
    mean = torch.mean(y_true)
    deviation = y_true - mean
    SST = deviation @ deviation
    return (1 - SSR / SST).item()
    


def main():
    y_true = torch.tensor([1, 2, 3, 4, 5],dtype=torch.float)
    y_pred = torch.tensor([1.1, 2.1, 2.9, 4.2, 4.8],dtype=torch.float)
    print(r_squared(y_true, y_pred))


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-07 18:08:01
#
