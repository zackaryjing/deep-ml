# Implement a Dense Block with 2D Convolutions

import torch
import torch.nn.functional as F
import numpy as np

torch.backends.nnpack.set_flags(False)

def dense_net_block(input_data: torch.Tensor, num_layers: int, growth_rate: int, kernels: list, kernel_size: tuple = (3, 3)) -> torch.Tensor:
    """
    Perform forward pass of a DenseNet dense block.
    
    Args:
        input_data: Input tensor of shape (N, H, W, C0) in NHWC format
        num_layers: Number of layers in the dense block
        growth_rate: Number of output channels per layer
        kernels: List of kernel tensors, each of shape (kh, kw, in_channels, out_channels)
        kernel_size: Tuple (kh, kw) for kernel dimensions
    
    Returns:
        Output tensor of shape (N, H, W, C0 + num_layers * growth_rate)
    """
    X = input_data.permute(0,3,1,2)
    kh,kw = kernel_size
    pad_H = kh // 2
    pad_W = kw // 2
    res = torch.empty_like(input_data)
    
    for layer in range(0,num_layers):
        X_relu = F.relu(X)
        X_relu = F.pad(X_relu,(pad_W,pad_W,pad_H,pad_H)) 
        kernel=kernels[layer].permute(3,2,0,1)
        out = torch.conv2d(X_relu,kernel,stride=1,bias=None)
        X = torch.cat([X,out],dim=1)
    
    return X.permute(0,2,3,1)
        

def main():
    np.random.seed(42)
    X = np.random.randn(1, 1, 1, 2)
    kernels_np = [np.random.randn(3, 3, 2 + i*1, 1) * 0.01 for i in range(2)]
    X_t = torch.tensor(X, dtype=torch.float64)
    kernels_t = [torch.tensor(k, dtype=torch.float64) for k in kernels_np]
    result = dense_net_block(X_t, 2, 1, kernels_t)
    print(result.numpy())
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-10 16:10:48
#
