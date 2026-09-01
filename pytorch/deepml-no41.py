# Simple Convolutional 2D Layer

import torch
import torch.nn.functional as F


def simple_conv2d(
    input_matrix: torch.Tensor, kernel: torch.Tensor, padding: int, stride: int
) -> torch.Tensor:
    """
    Perform a 2D convolution on a single-channel input using PyTorch's built-in conv2d.
    input_matrix: 2D tensor (H, W)
    kernel: 2D tensor (kH, kW)
    padding: int, zero-padding on all sides
    stride: int, stride of the convolution
    """
    # Hint: conv2d expects input of shape (N, C, H, W) and weight of shape (out_channels, in_channels, kH, kW)
    input_matrix = input_matrix.unsqueeze(0).unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    res = F.conv2d(input_matrix, kernel, stride=stride, padding=padding)
    res = res.squeeze(0).squeeze(0)
    return res


def main():
    input_matrix = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0, 25.0],
        ]
    )
    kernel = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, -1.0],
        ]
    )
    output = simple_conv2d(input_matrix, kernel, padding=0, stride=1)
    print(output)


if __name__ == "__main__":
    main()


#
# Created By jing At 2026-09-01 14:23:34
#


'''
# [!Note] oj bug hack


# Simple Convolutional 2D Layer

import os
import contextlib
import torch
import torch.nn.functional as F


@contextlib.contextmanager
def suppress_stderr():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)


def simple_conv2d(
    input_matrix: torch.Tensor, kernel: torch.Tensor, padding: int, stride: int
) -> torch.Tensor:
    """
    Perform a 2D convolution on a single-channel input using PyTorch's built-in conv2d.
    input_matrix: 2D tensor (H, W)
    kernel: 2D tensor (kH, kW)
    padding: int, zero-padding on all sides
    stride: int, stride of the convolution
    """
    input_matrix = input_matrix.unsqueeze(0).unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    with suppress_stderr():
        res = F.conv2d(input_matrix, kernel, stride=stride, padding=padding)

    res = res.squeeze(0).squeeze(0)
    return res


def main():
    input_matrix = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0, 25.0],
        ]
    )
    kernel = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, -1.0],
        ]
    )
    output = simple_conv2d(input_matrix, kernel, padding=0, stride=1)
    print(output)


if __name__ == "__main__":
    main()
'''
