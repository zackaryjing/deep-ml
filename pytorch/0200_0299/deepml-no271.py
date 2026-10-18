# Implement Gated Attention

import torch
import math

def gated_attention(
    X: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_g: torch.Tensor
) -> torch.Tensor:
    """
    Compute Gated Attention output.
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_q: Query projection of shape (d_model, d_k)
        W_k: Key projection of shape (d_model, d_k)
        W_v: Value projection of shape (d_model, d_v)
        W_g: Gate projection of shape (d_model, d_v)
    
    Returns:
        Gated attention output of shape (seq_len, d_v), rounded to 4 decimal places
    
    Hint: First compute standard scaled dot-product attention, then apply
    a sigmoid gate to modulate the output.
    """
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    G = torch.sigmoid(X @ W_g)
    d_k = W_q.shape[-1]
    Y = torch.softmax((Q @ K.T) / math.sqrt(d_k),dim=-1) @ V
    Y_gated = G * Y
    return Y_gated


def main():
    X = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    W_q = torch.eye(2)
    W_k = torch.eye(2)
    W_v = torch.eye(2)
    W_g = torch.zeros((2, 2))
    result = gated_attention(X, W_q, W_k, W_v, W_g).numpy().tolist()
    print([[round(v, 4) for v in row] for row in result])

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-18 14:59:16
#
