# Implement Gradient Descent Variants with MSE Loss

import torch
from torch.utils.data import TensorDataset, DataLoader


def gradient_descent(
    X: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
    learning_rate: float,
    n_epochs: int,
    batch_size: int = 1,
    method: str = "batch",
) -> torch.Tensor:
    """
    Implements three variants of gradient descent: Batch, Stochastic, and Mini-Batch.
    Uses Mean Squared Error (MSE) as the loss function.

    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')

    Returns:
        Optimized weights as a tensor
    """

    m, n = X.shape

    if method == "batch":
        for _ in range(n_epochs):
            weights -= learning_rate * 2 / m  * (X.T @ ((X @ weights) - y))
    elif method == "stochastic":
        for _ in range(n_epochs):
            for sample, truth in zip(X, y):
                weights -= learning_rate * 2 * (sample * ((sample @ weights) - truth))
    else:
        for _ in range(n_epochs):
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for samples, ground_truth in dataloader:
                weights -=  (
                    learning_rate * 2 / batch_size * (samples.T @ (samples @ weights - ground_truth))
                )
    return weights


def main():
    X = torch.tensor([[1, 1], [2, 1], [3, 1], [4, 1]], dtype=torch.float)
    y = torch.tensor([2, 3, 4, 5], dtype=torch.float)

    learning_rate = 0.01
    n_epochs = 1000
    batch_size = 2

    weights = torch.zeros(X.shape[1])
    print(f"weights: {weights}")

    final_weights = gradient_descent(
        X, y, weights, learning_rate, n_epochs, method="batch"
    )
    print(final_weights)
    final_weights = gradient_descent(
        X, y, weights, learning_rate, n_epochs, method="stochastic"
    )
    print(final_weights)
    final_weights = gradient_descent(
        X, y, weights, learning_rate, n_epochs, batch_size, method="mini_batch"
    )
    print(final_weights)
    
    
    X = torch.tensor([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]])
    y = torch.tensor([2.0, 3.0, 4.0, 5.0])
    weights = torch.zeros(X.shape[1])
    learning_rate = 0.01
    n_epochs = 100

# Test Batch Gradient Descent
    output = gradient_descent(X, y, weights, learning_rate, n_epochs, method='batch')
    print([round(float(x), 8) for x in output])


    X = torch.tensor([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]])
    y = torch.tensor([2.0, 3.0, 4.0, 5.0])
    weights = torch.zeros(X.shape[1])
    learning_rate = 0.01
    n_epochs = 100
    batch_size = 2

# Test Mini-Batch Gradient Descent
    output = gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size, method='mini_batch')
    print([round(float(x), 8) for x in output])

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-09-01 16:02:46
#
