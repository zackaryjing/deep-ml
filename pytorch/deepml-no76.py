# Calculate Cosine Similarity Between Vectors

import torch

def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    Calculate the cosine similarity of two vectors using PyTorch.
    Args:
        v1 (torch.Tensor): 1D tensor representing the first vector.
        v2 (torch.Tensor): 1D tensor representing the second vector.
    Returns:
        float: The cosine similarity of the two vectors.
    """
    v1 = v1.to(torch.float)
    v2 = v2.to(torch.float)
    return (v1 @ v2 / (torch.norm(v1) * torch.norm(v2))).item()

def main():
    v1 = torch.tensor([1, 2, 3])
    v2 = torch.tensor([2, 4, 6])
    print(round(cosine_similarity(v1, v2), 3))
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-09-09 10:43:34
#
