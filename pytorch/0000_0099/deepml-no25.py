# problem: Single Neuron with Backpropagation

import torch
from torch.nn.functional import sigmoid


def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float,
                 learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # Your code here

    weights = initial_weights.clone()
    bias = initial_bias
    n_samples, n_features = features.shape
    mse_values = []
    for i in range(epochs):
        z = features @ weights + bias
        preds = sigmoid(z)

        mse = torch.mean((preds - labels) ** 2).item()
        mse_values.append(round(mse,4))

        error = (preds - labels)
        grad_z = error * preds * (1 - preds) * ( 2 / n_samples)
        weights -= learning_rate * (features.T @ grad_z)
        bias -= learning_rate * grad_z.sum()
    return ([round(i,4) for i in weights.tolist()], bias.item(), mse_values)


def main():
    features = torch.tensor([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]])
    labels = torch.tensor([1, 0, 0])
    initial_weights = torch.tensor([0.1, -0.2])
    initial_bias = 0.0
    learning_rate = 0.1
    epochs = 2
    print(train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs))


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-05 10:18:16
#
