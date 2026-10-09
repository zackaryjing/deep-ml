# problem: 2D Translation Matrix Implementation

import torch


def translate_object(points, tx, ty) -> torch.Tensor:
    """
    Apply a 2D translation matrix to a set of points.

    Args:
        points: list of [x, y] coordinates or torch.Tensor of shape (N, 2)
        tx: translation distance in x direction
        ty: translation distance in y direction

    Returns:
        torch.Tensor of translated points with shape (N, 2)
    """
    points = torch.tensor(points,dtype=torch.float)
    N = points.shape[0]
    ones = torch.ones(N, 1, dtype=points.dtype, device=points.device)
    points_h = torch.cat([points, ones], dim=1)  # (N, 3)
    translation = torch.eye(3)
    translation[:,2] += torch.tensor([tx,ty,0])
    res = translation @ points_h.T
    return res[:-1].T

def main():
    points = [[0, 0], [1, 0], [0.5, 1]]
    tx, ty = 2, 3
    print(translate_object(points, tx, ty))


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-04 15:08:57
#
