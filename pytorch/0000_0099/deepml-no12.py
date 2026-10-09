# problem: Singular Value Decomposition (SVD) of 2x2 Matrix

import torch


def svd_2x2_singular_values(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.

    Args:
        A: A 2x2 torch tensor

    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
    """
    # Your code here
    ATA = A.T @ A
    alpha = ATA[0, 0]
    beta = ATA[1, 1]
    gamma = ATA[1, 0]
    if torch.abs(gamma) > 1e-12:
        zeta = (beta - alpha) / (2 * gamma)
        sign_zeta = torch.where(zeta >= 0 ,torch.tensor(1.0),torch.tensor(-1.0))
        t = sign_zeta / (torch.abs(zeta) + torch.sqrt(1 + zeta ** 2))
        c_theta = 1 / (torch.sqrt(1 + t ** 2))
    else:
        c_theta = torch.tensor(1.0)
        t = torch.tensor(0.0)
    s_theta = t * c_theta
    V = torch.tensor([[c_theta, -s_theta], [s_theta, c_theta]])
    A_b = A @ V
    sigma1 = torch.norm(A_b[:, 0])
    sigma2 = torch.norm(A_b[:, 1])
    u1 = A_b[:, 0] / sigma1
    u2 = A_b[:, 1] / sigma2
    U = torch.stack([u1, u2], dim=1)
    Sigma = torch.tensor([[sigma1, 0], [0, sigma2]])
    S = torch.stack([sigma1,sigma2])
    return U,S, V.T


def main():
    A = torch.tensor([[2, 1], [1, 2]],dtype=torch.float)
    print(svd_2x2_singular_values(A))

if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-05 16:41:17
#
