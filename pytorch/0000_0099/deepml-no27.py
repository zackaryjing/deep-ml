# Transformation Matrix from Basis B to C

import torch
from typing import List


def transform_basis(B: List[List[float]], C: List[List[float]]) -> List[List[float]]:
    """Return the change-of-basis matrix **P = C⁻¹ B**.

    - *B*, *C* may be 2×2 or 3×3 nested lists.
    - Result is rounded to 4 decimals and returned as a nested list.
    """
    # Your implementation here
    B_t = torch.tensor(B,dtype=float)
    C_t = torch.tensor(C,dtype=float)
    return (B_t @ C_t.inverse()).tolist()
    pass

def main():
    B = [[1, 0, 0.0], 
             [0, 1, 0], 
             [0, 0, 1]]
    C = [[1, 2.3, 3], 
            [4.4, 25, 6], 
            [7.4, 8, 9]]
    print(transform_basis(B,C))
    
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-22 23:55:43
#
