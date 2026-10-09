# problem: Create Composite Hypervector for a Dataset Row

import torch
import numpy as np


def deterministic_hash(s):
    '''Converts a string to a deterministic integer.'''
    h = 0
    for c in str(s):
        h = (h * 31 + ord(c)) % (2 ** 31)
    return h


def create_hv(dim: int, seed: int) -> torch.Tensor:
    '''Creates a bipolar hypervector of given dimension using the seed.
    Returns a torch.Tensor of shape (dim,) with values in {-1, 1}.
    '''
    np.random.seed(seed % (2 ** 32 - 1))
    hv = np.random.choice([-1, 1], dim)
    return torch.tensor(hv, dtype=torch.float32)


def create_row_hv(row: dict, dim: int, random_seeds: dict) -> torch.Tensor:
    '''Create composite hypervector for a dataset row using PyTorch operations.

    For each feature:
    1. Create a bipolar hypervector for the feature name
    2. Create a bipolar hypervector for the feature value
    3. Bind them via element-wise multiplication (torch.mul)

    Bundle all bound hypervectors via torch.sum, then normalize with torch.where.

    Hint: For each feature, the value seed should combine the base seed
    with the hashed value using modular arithmetic.

    Returns:
        torch.Tensor of shape (dim,) with bipolar values (-1 or 1).
    '''
    
    bound_hvs = []
    for key,item in row.items():
        base_seed = random_seeds[key]
        name_hv = create_hv(dim,base_seed)
        value_seed = (base_seed + deterministic_hash(item) % (2 ** 31))
        value_hv = create_hv(dim,value_seed)
        
        bound = torch.mul(name_hv,value_hv)
        bound_hvs.append(bound)
    
    stacked = torch.stack(bound_hvs)
    bundled = torch.sum(stacked,dim=0)
    result = torch.where(bundled >= 0,torch.tensor(1.0),torch.tensor(-1.0))
    return result


def main():
    row = {'FeatureA': 'value1', 'FeatureB': 'value2'}
    dim = 5
    random_seeds = {'FeatureA': 42, 'FeatureB': 7}
    print(create_row_hv(row, dim, random_seeds))


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-09 20:57:39
#
