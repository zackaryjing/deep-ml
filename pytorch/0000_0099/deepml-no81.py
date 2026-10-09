# Poisson Distribution Probability Calculator

from math import factorial

import torch

def poisson_probability(k: int, lam: float) -> float:
    """
    Calculate the probability of observing exactly k events in a fixed interval,
    given the mean rate of events lam, using the Poisson distribution formula.
    :param k: Number of events (non-negative integer)
    :param lam: The average rate (mean) of occurrences in a fixed interval
    :return: Probability of k events occurring, rounded to 5 decimal places
    """
    k_tensor = torch.tensor(k, dtype=torch.float32)
    lam_tensor = torch.tensor(lam, dtype=torch.float32)
    return (lam_tensor ** k_tensor * torch.exp(-lam_tensor)).item() / factorial(k)
    # Your code here using torch.exp(), torch.lgamma(), torch.pow(), etc.
    

def main():
    print(torch.tensor(3),torch.tensor(5))
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-09-09 19:24:07
#
