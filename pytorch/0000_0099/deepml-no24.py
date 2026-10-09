# Single Neuron

import torch
import torch.nn.functional as F


def single_neuron_model(
    features: list[list[float]], labels: list[int], weights: list[float], bias: float
) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.

    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term

    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    # Your code here using PyTorch built-ins:
    # - torch.matmul() for linear combination
    # - torch.sigmoid() for activation
    # - torch.nn.functional.mse_loss() for MSE
    features_t = torch.tensor(features, dtype=torch.float)
    labels_t = torch.tensor(labels, dtype=torch.float)
    weights_t = torch.tensor(weights, dtype=torch.float)
    bias_t = torch.tensor(bias, dtype=torch.float)
    labels_hat = torch.sigmoid(features_t @ weights_t + bias_t)
    l = F.mse_loss(labels_t, labels_hat)
    return (labels_hat.round(decimals=4).tolist(), l.round(decimals=4).item())

def main():
    print(
        single_neuron_model(
            [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], [0, 1, 0], [0.7, -0.4], -0.1
        )
    )


if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-17 23:27:43
#
