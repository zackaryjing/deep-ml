# problem: Spectral Normalization

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def spectral_normalization(
        W: torch.Tensor,
        num_iters: int = 10,
        u_init: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, float]:
    """
    Apply spectral normalization to a weight matrix using power iteration.

    Args:
        W: Weight matrix of shape (m, n) as a torch.Tensor
        num_iters: Number of power iteration steps
        u_init: Optional initial vector of shape (m,) as a torch.Tensor

    Returns:
        Tuple of (W_sn, sigma):
            W_sn: Spectrally normalized weight matrix as a torch.Tensor
            sigma: Estimated spectral norm (largest singular value) as a float
    """
    m,n = W.shape
    u = torch.rand(m)
    u /=torch.norm(u)
    v = torch.rand(n) # dummy
    if u_init != None:
        u = u_init

    for i in range(num_iters):
        v_tiled = W.T @ u # [n,1]
        v = v_tiled / torch.norm(v_tiled)
        u_tiled = W @ v # [m, 1]
        u = u_tiled / torch.norm(u_tiled)
    sigma = (u @ W @ v).item()
    # [1,m] [m,n] [n,1]
    W_sn = W / sigma
    return W_sn, sigma




def main():
    W = torch.tensor([[4.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
    u_init = torch.tensor([1.0, 0.0], dtype=torch.float64)
    W_sn, sigma = spectral_normalization(W, num_iters=1, u_init=u_init)
    print(round(sigma, 4))
    print([[round(x, 4) for x in row] for row in W_sn.numpy().tolist()])


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-10-19 15:42:26
#
