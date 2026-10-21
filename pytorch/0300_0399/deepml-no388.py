# problem:

import numpy as np
import math

import torch


def sliding_window_attention2(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Compute sliding window attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        window_size: Number of positions to the left and right each query can attend to

    Returns:
        Output matrix of shape (seq_len, d_v), rounded to 4 decimal places.
    """
    seq_len, d_k = K.shape
    d_v = V.shape[-1]
    res = torch.empty(0, d_v)
    for i in range(seq_len):
        start = max(i - window_size, 0)
        end = min(i + window_size, seq_len - 1) + 1
        Q_sub = Q[i]  # [w,d_k]
        K_sub = K[start:end]  # [w,d_k]
        V_sub = V[start:end]  # [w,d_v]
        score = torch.softmax(Q_sub @ K_sub.T / math.sqrt(d_k), dim=0)
        attention = score @ V_sub
        res = torch.cat([res, attention.unsqueeze(0)], dim=0)
    return res


def sliding_window_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Compute sliding window attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        window_size: Number of positions to the left and right each query can attend to

    Returns:
        Output matrix of shape (seq_len, d_v), rounded to 4 decimal places.
    """
    seq_len, d_k = K.shape

    offsets = torch.arange(-window_size, window_size + 1)
    indices = torch.arange(seq_len)[:, None] + offsets[None, :]
    valid = (indices >= 0) & (indices < seq_len)
    indices = indices.clamp(0, seq_len - 1)

    K_w = K[indices]
    V_w = V[indices]

    score = torch.einsum('nd,nwd->nw', Q, K_w) / math.sqrt(d_k)
    score = score.masked_fill(~valid, float('-inf'))
    score = torch.softmax(score, dim=-1)
    return torch.einsum('nw,nwd->nd', score, V_w)


def main():
    Q = torch.tensor([[1.0], [2.0], [3.0]])
    K = torch.tensor([[1.0], [2.0], [3.0]])
    V = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = sliding_window_attention(Q, K, V, window_size=1)
    print((torch.round(result.double() * 10000) / 10000).tolist())


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-10-19 16:46:11
#
