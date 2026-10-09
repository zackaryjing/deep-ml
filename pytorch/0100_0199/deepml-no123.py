# Calculate Computational Efficiency of MoE

import torch

def compute_efficiency(n_experts: int, k_active: int, d_in: int, d_out: int) -> torch.Tensor:
    """
    Calculate computational savings of MoE vs. dense layer.

    Args:
        n_experts: Total number of experts
        k_active: Number of active experts (sparsity)
        d_in: Input dimension
        d_out: Output dimension

    Returns:
        Percentage savings in FLOPs as a torch.Tensor
    """
    return torch.Tensor([(n_experts -  k_active) / n_experts * 100])
    

def main():
    print(compute_efficiency(1000, 2, 512, 512))
    result = compute_efficiency(1000, 2, 512, 512)
    print(round(result.item(), 1))
if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-09 16:19:36
#
