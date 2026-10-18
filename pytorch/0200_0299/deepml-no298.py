# problem: Implement mHC Forward Pass

import torch


def mhc_forward(
        x: torch.Tensor,
        H_pre_raw: torch.Tensor,
        H_post_raw: torch.Tensor,
        H_res_raw: torch.Tensor,
        layer_output: torch.Tensor,
        sinkhorn_iters: int = 5
) -> torch.Tensor:
    """
    Compute mHC forward pass with manifold-constrained mappings.

    The mHC formula is:
        x_out = H_res @ x + H_post.T @ layer_output

    Where:
        - H_pre (not used here, already applied to get layer_output)
        - H_post = 2 * sigmoid(H_post_raw)  (non-negative constraint)
        - H_res = Sinkhorn(exp(H_res_raw))  (doubly stochastic constraint)

    Args:
        x: Hidden states, shape (n, C) where n is num streams
        H_pre_raw: Raw pre-mapping coefficients, shape (1, n) - not used in this step
        H_post_raw: Raw post-mapping coefficients, shape (1, n)
        H_res_raw: Raw residual mapping coefficients, shape (n, n)
        layer_output: Output from layer F, shape (1, C)
        sinkhorn_iters: Number of Sinkhorn iterations

    Returns:
        x_out: Updated hidden states, shape (n, C)
    """
    def sinkhorn(H,iters=5):
        A = torch.exp(H)
        for _ in range(iters):
            A = A / A.sum(dim=-1,keepdim=True)
            A = A / A.sum(dim=-2,keepdim=True)
        return A
    H_post = 2 * torch.sigmoid(H_post_raw)
    H_res = sinkhorn(H_res_raw,sinkhorn_iters)
    x_plus = H_res @ x + H_post.T @ layer_output
    #        [n,n]@[n,C]->[n,C]     [n,C]
    return x_plus




def main():
    x = torch.randn(4, 8)
    H_pre_raw = torch.zeros(1, 4)
    H_post_raw = torch.zeros(1, 4)
    H_res_raw = torch.zeros(4, 4)
    layer_out = torch.randn(1, 8)
    out = mhc_forward(x, H_pre_raw, H_post_raw, H_res_raw, layer_out)
    print(list(out.shape))


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-10-18 15:26:09
#
