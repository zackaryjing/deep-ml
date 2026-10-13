# 3D CNN Forward Pass Implementation

import torch

def conv3d_forward_pass_pytorch(
    input_volume: torch.Tensor,
    kernel_weight: torch.Tensor,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0),
) -> torch.Tensor:
    """
    Perform 3D convolution forward pass using PyTorch.

    Use PyTorch's nn.Conv3d layer to perform 3D convolution.
    Note: PyTorch expects input shape (N, C, D, H, W) where N is batch size.

    Args:
            input_volume: Shape (C, D, H, W)
            kernel_weight: Shape (1, C, kD, kH, kW)
              First dim is output channels (1 for simplified version)
            stride: (stride_d, stride_h, stride_w)
            padding: (pad_d, pad_h, pad_w)

    Returns:
            Output tensor: Shape (1, D_out, H_out, W_out)

    Hints:
            - Add batch dimension to input: input.unsqueeze(0)
            - Create Conv3d layer with kernel size from kernel_weight
            - Set conv.weight parameter to kernel_weight
            - Set conv.bias to None or zeros
            - Apply convolution and remove batch dimension
    """
    # Your code here
    conv = torch.nn.Conv3d(
        in_channels=input_volume.shape[0],
        out_channels=kernel_weight.shape[0],
        kernel_size=kernel_weight.shape[-3:],
        stride=stride,
        padding=padding,
        bias=False
    )
    with torch.no_grad():
        conv.weight.copy_(kernel_weight)
    return conv(input_volume.unsqueeze(0)).squeeze(0)


def main():
    input_vol = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
    kernel = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]])
    result = conv3d_forward_pass_pytorch(input_vol, kernel)
    print(result.tolist())


if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-13 16:30:41
#
