# problem: Implement Multiquery Attention (MQA)

import torch
import math


def multiquery_attention(X: torch.Tensor, W_queries: list, W_key: torch.Tensor, W_value: torch.Tensor,
                         W_out: torch.Tensor) -> torch.Tensor:
    """
    Compute Multi-Query Attention.

    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_queries: List of query weight tensors, each (d_model, d_k), one per head
        W_key: Shared key weight tensor of shape (d_model, d_k)
        W_value: Shared value weight tensor of shape (d_model, d_v)
        W_out: Output projection tensor of shape (num_heads * d_v, d_model)

    Returns:
        Output tensor of shape (seq_len, d_model), rounded to 4 decimal places
    """
    K_q = X @ W_key # [seq, d_k]
    seq_len = X.shape[0]
    V_q = X @ W_value
    d_mode, d_k = W_key.shape
    num_heads = len(W_queries)
    print(num_heads)
    W_q = torch.stack(W_queries, dim=0) # [H,d_model,d_k]
    Q = X @ W_q # [seq_len,d_model] @ [H,d_model,d_k] -> [H,seq_len,d_k]
    K = torch.stack([K_q] * num_heads, dim=0) # [H,seq_len,d_k]
    V = torch.stack([V_q] * num_heads, dim=0) # [H,seq_len,d_k]

    score = Q @ K.mT / math.sqrt(d_k) # [H,seq_len,seq_len]
    attn = torch.softmax(score, dim=-1) # [H,seq_len,seq_len]
    out = attn @ V # [H, seq_len, d_k]
    out = out.transpose(0,1).reshape(seq_len, -1)
    # -> [seq_len,H,d_k] -> [seq_len, H * d_k]
    return (out @ W_out).round(decimals=4)






def main():
    X = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    W_queries = [torch.eye(2, dtype=torch.float64)]
    W_key = torch.eye(2, dtype=torch.float64)
    W_value = torch.eye(2, dtype=torch.float64)
    W_out = torch.eye(2, dtype=torch.float64)
    result = multiquery_attention(X, W_queries, W_key, W_value, W_out)
    print(result.numpy().tolist())

if __name__ == "__main__":
    main()

#
# Created By jing At 2026-10-19 21:00:31
#
