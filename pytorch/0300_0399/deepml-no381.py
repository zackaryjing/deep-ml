# problem: Rotary Positional Embeddings (RoPE)

import torch


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.

    Args:
        x: Input embeddings of shape (seq_len, d), d must be even
        positions: Position indices of shape (seq_len,)
        base: Base for frequency computation (default: 10000.0)

    Returns:
        Embeddings with rotary positional encoding applied, shape (seq_len, d)
    """
    # Your code here
    seq_len, d = x.shape
    theta = 1 / (base ** (torch.arange(0, d, 2) / d))

    #        [seq_len,1] * [1, d / 2]
    angles = positions[:, None] * theta
    c = torch.cos(angles)  # [seq_len,d / 2]
    s = torch.sin(angles)  # [seq_len,d / 2]
    x_even = x[:, torch.arange(0, d // 2) * 2]  # [seq_len, d / 2]
    x_odd = x[:, torch.arange(0, d // 2) * 2 + 1]  # [seq_len, d / 2]
    out_even = c * x_even - s * x_odd
    out_odd = s * x_even + c * x_odd
    x_stacked = torch.stack([out_even, out_odd], dim=-1)  # [seq_len, d / 2, 2]
    x_res = x_stacked.reshape(seq_len, d)
    return x_res


def main():
    def fmt(x):
        if isinstance(x, list):
            return [fmt(i) for i in x]
        return round(x, 4)
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    positions = torch.tensor([1])
    result = apply_rope(x, positions)
    print(fmt(result.numpy().tolist()))


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-10-18 17:32:29
#
