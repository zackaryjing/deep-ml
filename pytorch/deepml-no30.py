# Batch Iterator for Dataset

import torch
from torch.utils.data import TensorDataset, DataLoader import numpy as np


def batch_iterator(X, y=None, batch_size=64):
    """
    Batch iterator for a dataset represented as PyTorch tensors.

    Args:
        X: Tensor of shape (n_samples, n_features)
        y: Optional Tensor of shape (n_samples,)
        batch_size: Number of samples per batch

    Returns:
        List of Tensor batches (or [X_batch, y_batch] pairs if y is provided)
    """
    # Hint: Use DataLoader with shuffle=False to preserve sample order.
    if y is not None:
        dataset = TensorDataset(X, y)
    else:
        dataset = X
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle = False)
    return dataloader


def main():
    # x = torch.tensor(np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]))
    # y = torch.tensor(np.array([1, 2, 3, 4, 5]))
    # for idx,(batch_x,batch_y) in enumerate(batch_iterator(x, y=y, batch_size=2)):
    #     print(idx,(batch_x,batch_y))
        
    X = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]])
    result = batch_iterator(X, batch_size=3)
    output = [batch.numpy().tolist() for batch in result]
    print(output)       


if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-26 08:21:33
#
