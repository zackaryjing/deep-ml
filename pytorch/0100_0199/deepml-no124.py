# Implement the Noisy Top-K Gating Function

import torch
import torch.nn.functional as F

def noisy_topk_gating(
    X: torch.Tensor,
    W_g: torch.Tensor,
    W_noise: torch.Tensor,
    N: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Args:
        X: Input data, shape (batch_size, features)
        W_g: Gating weight matrix, shape (features, num_experts)
        W_noise: Noise weight matrix, shape (features, num_experts)
        N: Noise samples, shape (batch_size, num_experts)
        k: Number of experts to keep per example
    Returns:
        Gating probabilities, shape (batch_size, num_experts)
    """
    # Your code here
    H_base = X @ W_g
    H_noise = X @ W_noise
    H = H_base + N * F.softplus(H_noise)
    
    values, indices = torch.topk(H, k, dim=-1)
    
    H_topk = torch.full_like(H,float('-inf'))
    H_topk.scatter_(dim=-1,index=indices,src=values)
    return F.softmax(H_topk,dim=-1)
    

def main():
    X = torch.tensor([[1.0, 2.0]])
    W_g = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    W_noise = torch.zeros((2,2))
    N = torch.zeros((1,2))
    result = noisy_topk_gating(X, W_g, W_noise, N, k=1)
    print(result.numpy())
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-09 20:26:55
#
