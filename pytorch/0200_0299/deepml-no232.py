# PTX Loss for Catastrophic Forgetting Prevention (RLHF)

import torch
import torch.nn.functional as F


def compute_ptx_loss_pytorch(
    rl_loss: torch.Tensor,
    pretrain_logits: torch.Tensor,
    pretrain_labels: torch.Tensor,
    beta_ptx: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PTX Loss using PyTorch.

    Args:
        rl_loss: RL loss tensor (scalar)
        pretrain_logits: Shape (batch_size, vocab_size)
        pretrain_labels: Shape (batch_size,)
        beta_ptx: Weight coefficient

    Returns:
        (total_loss, ce_loss, weighted_ce_loss) as tensors

    Hints:
        - Use F.cross_entropy() for numerical stability
        - cross_entropy automatically applies softmax
        - reduction='mean' for batch averaging
    """
    # Your code here
    ce_loss = F.cross_entropy(pretrain_logits, pretrain_labels, reduction="mean")
    weighted_ce_loss = beta_ptx * ce_loss
    L_total = rl_loss + weighted_ce_loss
    return L_total, ce_loss, weighted_ce_loss


def main():
    logits = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    labels = torch.tensor([0, 2], dtype=torch.long)
    rl_loss = torch.tensor(0.3)
    total, ce, weighted = compute_ptx_loss_pytorch(rl_loss, logits, labels, 0.2)
    print(f"{total.item():.6f}")


if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-13 17:19:18
#
