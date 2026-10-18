# Calculate Number of Parameters in Neural Network

from typing import Tuple
import math

import torch
import torch.nn as nn


def count_parameters(layers: list) -> int:
    """
    Count the total number of trainable parameters in a neural network
    using PyTorch's built-in nn modules.

    Args:
        layers: A list of dictionaries describing each layer.
                Each dict contains 'type' and layer-specific parameters.

    Returns:
        Total number of trainable parameters as an integer.
    """
    # Your code here
    total_parameters = 0
    for layer in layers:
        if layer["type"] == "dense":
            total_parameters += layer["input_size"] * layer["output_size"]
            if layer["use_bias"]:
                total_parameters += layer["output_size"]
        elif layer["type"] == "conv2d":
            kernel_size = layer["kernel_size"]
            out_channels = layer["out_channels"]
            in_channels = layer["in_channels"]
            channels = in_channels * out_channels
            # if isinstance(kernel_size,(tuple,list)): 
            if type(kernel_size) is tuple:
                total_parameters += channels * kernel_size[0] * kernel_size[1]
            else:
                total_parameters += kernel_size * kernel_size * channels
            if layer["use_bias"]:
                total_parameters += out_channels
        elif layer["type"] == "embedding":
            total_parameters += (layer["num_embeddings"] + 1) * layer["embedding_dim"]
    return total_parameters


def main():
    layers = [
        {
            "type": "conv2d",
            "in_channels": 3,
            "out_channels": 16,
            "kernel_size": 3,
            "use_bias": True,
        },
        {
            "type": "conv2d",
            "in_channels": 16,
            "out_channels": 32,
            "kernel_size": 3,
            "use_bias": True,
        },
    ]
    result = count_parameters(layers)
    print(result)

    layers = [{"type": "conv2d", "in_channels": 1, "out_channels": 8, "kernel_size": (5, 5), "use_bias": True}]
    result = count_parameters(layers)
    print(result)

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-16 22:56:06
#
