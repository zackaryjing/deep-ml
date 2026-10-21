# problem: Implement Grouped Query Attention (GQA)

import numpy as np
import torch
import math
import torch.nn.functional as F


def grouped_query_attention2(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, num_heads: int, num_kv_heads: int
) -> torch.Tensor:
    """
    Compute Grouped Query Attention.

    Args:
        Q: Query tensor, shape (batch_size, seq_len, num_heads * head_dim)
        K: Key tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        V: Value tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads

    Returns:
        Output tensor, shape (batch_size, seq_len, num_heads * head_dim)
    """
    model_dim = V.shape[-1]
    head_dim = model_dim // num_kv_heads
    batch_size, seq_len, _ = Q.shape
    Q_h = Q.reshape(batch_size, seq_len, num_heads, head_dim)
    K_h = K.reshape(batch_size, seq_len, num_kv_heads, head_dim)
    V_h = V.reshape(batch_size, seq_len, num_kv_heads, head_dim)
    group = num_heads // num_kv_heads
    K_h = K_h.repeat_interleave(group, dim=2)
    V_h = V_h.repeat_interleave(group, dim=2)
    score = torch.einsum("bsnh,btnh->bnst", Q_h, K_h) / math.sqrt(head_dim)
    score = torch.softmax(score, dim=-1)  # [B,H,seq_len,seq_len]
    V_h = V_h.permute(0, 2, 1, 3)  # [B,H,seq_len,head_dim]
    atten = score @ V_h  # [B,H,seq_len,head_dim]
    atten = atten.permute(0, 2, 1, 3)
    atten = atten.reshape([batch_size, seq_len, -1])
    return atten


def grouped_query_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, num_heads: int, num_kv_heads: int
) -> torch.Tensor:
    """
    Compute Grouped Query Attention.

    Args:
        Q: Query tensor, shape (batch_size, seq_len, num_heads * head_dim)
        K: Key tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        V: Value tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads

    Returns:
        Output tensor, shape (batch_size, seq_len, num_heads * head_dim)
    """
    model_dim = V.shape[-1]
    head_dim = model_dim // num_kv_heads
    batch_size, seq_len, _ = Q.shape
    Q_h = Q.reshape(batch_size, seq_len, num_heads, head_dim)
    K_h = K.reshape(batch_size, seq_len, num_kv_heads, head_dim)
    V_h = V.reshape(batch_size, seq_len, num_kv_heads, head_dim)
    group = num_heads // num_kv_heads
    Q_h = Q_h.reshape([batch_size, seq_len, num_kv_heads,group , head_dim])
    score = torch.einsum("bsngh,btnh->bngst", Q_h, K_h) / math.sqrt(head_dim)
    score = torch.softmax(score, dim=-1)  # [B,G,H,seq_len,seq_len]
    V_h = V_h.permute(0, 2, 1, 3)  # [B,H,seq_len,head_dim]
    V_h = V_h.unsqueeze(2)
    atten = score @ V_h  # [B,H,G,seq_len,head_dim]
    atten = atten.permute(0, 3, 1, 2, 4)
    atten = atten.reshape([batch_size, seq_len, -1])
    return atten



# Q:  A1 A2 A3 A4 
# KV: B1 B2
# 
#  NO.1 : B1 -> A1 A2   B2 -> A3 A4
#  NO.2 : B1,B2 -> A1,A2  B1,B2 -> A3,A4


def main():
    Q = torch.tensor([[[1.0, 0.0, 0.5, 0.5]]])
    K = torch.tensor([[[1.0, 1.0]]])
    V = torch.tensor([[[2.0, 3.0]]])
    result = grouped_query_attention(Q, K, V, num_heads=2, num_kv_heads=1)
    print(torch.round(result, decimals=4).numpy().tolist())


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-10-19 23:23:55
#
