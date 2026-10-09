# Principal Component Analysis (PCA) Implementation

import torch
import numpy as np

def pca(data, k) -> torch.Tensor:
    """
    Perform PCA on `data`, returning the top `k` principal components as a tensor.
    Input: Tensor or convertible of shape (n_samples, n_features).
    Returns: a torch.Tensor of shape (n_features, k), with floats rounded to 4 decimals.
    Note: If an eigenvector's first non-zero value is negative, flip its sign.
    """
    # Your implementation here
    
    data_t = torch.tensor(data,dtype=torch.float)
    n_samples,n_features = data_t.shape
    mu = data_t.mean(dim=0)
    std = data_t.std(dim=0)
    data_c = (data_t - mu) / std
    C = (1 / (n_samples)) * ( data_c.T @ data_c)
    eigenvalues,eigenvectors = torch.linalg.eigh(C)
    idx = torch.argsort(eigenvalues,descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    
    mask = eigenvectors[0,:] < 0
    eigenvectors[:,mask] *= -1
    
    return eigenvectors[:,:k].round(decimals=4)
    

def main():
    print(pca(np.array([[1, 2], [3, 4], [5, 6]]), k=1).tolist())
    res = pca([[1.0, 6.0], [2.0, 4.0], [3.0, 2.0]], 1)
    print([[round(val, 4) for val in row] for row in res.tolist()])
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-18 21:37:28
#
