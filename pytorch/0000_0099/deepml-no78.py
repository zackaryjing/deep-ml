# Descriptive Statistics Calculator

import torch

def descriptive_statistics(data: torch.Tensor) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset using PyTorch.
    
    Args:
        data: List, torch.Tensor, or array-like of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    mean = data.mean()
    meidan = data.quantile(0.50)
    mode = data.mode()
    variance = data.var(unbiased=False)
    standard_deviation = data.std(unbiased=False)
    q25 = data.quantile(0.25)
    q50 = data.quantile(0.50)
    q75 = data.quantile(0.75)
    iqr = q75 - q25
    return {"mean": mean.item(), "median": meidan.item(), "mode": int(mode.values.item()), "variance": variance.item(),
            "standard_deviation": standard_deviation.item(), "25th_percentile": q25.item(), "50th_percentile": q50.item(),
            "75th_percentile": q75.item(), "interquartile_range": iqr.item()}


def main():
    t = torch.tensor([1, 2, 2, 3, 4, 4, 4, 5],dtype=torch.float)
    print(descriptive_statistics(t))


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-09 10:58:41
#
