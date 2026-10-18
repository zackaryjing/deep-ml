# problem: Implement GRU Cell

import torch
import numpy as np


def gru_cell(x: torch.Tensor, h_prev: torch.Tensor,
             W_z: torch.Tensor, U_z: torch.Tensor, b_z: torch.Tensor,
             W_r: torch.Tensor, U_r: torch.Tensor, b_r: torch.Tensor,
             W_h: torch.Tensor, U_h: torch.Tensor, b_h: torch.Tensor) -> torch.Tensor:
    """
    Implements a single GRU cell forward pass.

    Args:
        x: Input vector of shape (input_size,)
        h_prev: Previous hidden state of shape (hidden_size,)
        W_z, W_r, W_h: Weight matrices for input of shape (hidden_size, input_size)
        U_z, U_r, U_h: Weight matrices for hidden state of shape (hidden_size, hidden_size)
        b_z, b_r, b_h: Bias vectors of shape (hidden_size,)

    Returns:
        h_next: New hidden state of shape (hidden_size,)
    """
    # Your code here
    h_t = h_prev
    x_t=  x
    z_t = torch.sigmoid(W_z @ x_t + U_z @ h_t + b_z)
    r_t = torch.sigmoid(W_r @ x_t + U_r @ h_t + b_r)
    h_t_t = torch.tanh(W_h @ x_t + U_h @ (r_t * h_t) + b_h)
    h_t = (1 - z_t) * h_t + z_t * h_t_t
    return h_t



def main():
    x = torch.tensor([1.0, 0.5], dtype=torch.float64)
    h_prev = torch.zeros(3, dtype=torch.float64)
    W_z = torch.ones((3, 2), dtype=torch.float64) * 0.1
    U_z = torch.ones((3, 3), dtype=torch.float64) * 0.1
    b_z = torch.zeros(3, dtype=torch.float64)
    W_r = torch.ones((3, 2), dtype=torch.float64) * 0.1
    U_r = torch.ones((3, 3), dtype=torch.float64) * 0.1
    b_r = torch.zeros(3, dtype=torch.float64)
    W_h = torch.ones((3, 2), dtype=torch.float64) * 0.2
    U_h = torch.ones((3, 3), dtype=torch.float64) * 0.2
    b_h = torch.zeros(3, dtype=torch.float64)
    result = gru_cell(x, h_prev, W_z, U_z, b_z, W_r, U_r, b_r, W_h, U_h, b_h)
    print(np.round(result.numpy(), 4).tolist())

if __name__ == "__main__":
    main()

#
# Created By jing At 2026-10-15 18:56:06
#
