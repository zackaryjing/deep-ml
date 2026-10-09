# Generate Sorted Polynomial Features

import torch
from itertools import combinations_with_replacement
import numpy as np

def polynomial_features(X, degree):
    """
    Given a 2D tensor X and integer degree, return a new tensor of all polynomial feature combinations
    (with constant term), sorted for each sample from smallest to largest.
    """
    # Hint: Use combinations_with_replacement and torch.prod.
    X = torch.tensor(X)
    n_samples, n_feature = X.shape
    res = []
    for sample in X: 
        features = [torch.tensor([1.0])]
        for i in range(1,degree + 1):
            for comb in combinations_with_replacement(range(n_feature),i):
                poly_val = sample[list(comb)].prod()
                features.append(poly_val.unsqueeze(0))
        torch.cat(features)
        res.append(features)
    return torch.tensor(res)
        

def main():
    X = torch.tensor([[2, 3],
              [3, 4],
              [5, 6]])
    degree = 2
    output = polynomial_features(X, degree)
    out = polynomial_features(np.array([[2, 3], [3, 4], [5, 6]]), 2)
    print(output)
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-26 12:46:38
#
