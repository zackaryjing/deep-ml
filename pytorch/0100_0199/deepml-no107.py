# Implement Masked Self-Attention

import torch
import math
import numpy as np

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)

def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked self-attention.
    """
    d_model = Q.shape[1] 
    print(Q.shape)
    score = Q @ K.T / math.sqrt(d_model)
    masked_score = score + mask
    print(masked_score)
    softmax_score = torch.softmax(masked_score,dim=1)
    return softmax_score @ V
    

def main():
    np.random.seed(42)
    X = np.arange(48).reshape(6,8)
    X = np.random.permutation(X.flatten()).reshape(6, 8)
    mask = np.triu(np.ones((6, 6))*(-np.inf), k=1)
    W_q = np.random.randint(0,4,size=(8,8))
    W_k = np.random.randint(0,5,size=(8,8))
    W_v = np.random.randint(0,6,size=(8,8))
    X = torch.tensor(X, dtype=torch.float64)
    W_q = torch.tensor(W_q, dtype=torch.float64)
    W_k = torch.tensor(W_k, dtype=torch.float64)
    W_v = torch.tensor(W_v, dtype=torch.float64)
    mask = torch.tensor(mask, dtype=torch.float64)
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    result = masked_attention(Q, K, V, mask)
    print(result.numpy())
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-09 15:07:54
#
