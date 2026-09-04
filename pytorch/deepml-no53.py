# Implement Self-Attention Mechanism 

import torch

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = torch.matmul(X, W_q)
    K = torch.matmul(X, W_k)
    V = torch.matmul(X, W_v)
    return Q, K, V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)

    Returns:
        Attention output of shape (seq_len, d_v)
    """
    seq_len,d_k = Q.shape
    return torch.softmax((Q @ K.T) / torch.sqrt(torch.tensor(d_k)),dim=-1) @ V


def main():
    Q = torch.tensor([[1, 0], [0, 1]],dtype=torch.float)
    K = torch.tensor([[1, 0], [0, 1]],dtype=torch.float)
    V = torch.tensor([[1, 2], [3, 4]],dtype=torch.float)

    output = self_attention(Q, K, V)
    print(output)
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-09-04 14:07:55
#
