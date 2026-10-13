# Knowledge Distillation Loss

import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute knowledge distillation loss using PyTorch.

    Args:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher model
        temperature: Softmax temperature

    Returns:
        Distillation loss
    """
    # Your code here
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="sum")
    return loss * temperature**2


def main():
    student = torch.tensor([1.0, 2.0, 3.0])
    teacher = torch.tensor([1.0, 2.0, 3.0])
    result = distillation_loss(student, teacher, temperature=1.0)
    print(round(result.item(), 4))


if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-13 14:31:55
#
