# KL Divergence Between Two Normal Distributions

import torch
import math

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q) -> torch.Tensor:
    """
    Compute the KL divergence between two normal distributions P and Q.
    
    Args:
        mu_p: Mean of distribution P
        sigma_p: Standard deviation of distribution P
        mu_q: Mean of distribution Q
        sigma_q: Standard deviation of distribution Q
    
    Returns:
        torch.Tensor: KL divergence KL(P || Q)
    """
    return torch.log(torch.tensor(sigma_q / sigma_p)) + (sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2) - 1 / 2

def main():
    mu_p = 0.0
    sigma_p = 1.0
    mu_q = 1.0
    sigma_q = 1.0

    print(kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q).item())
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-09-04 14:00:28
#
