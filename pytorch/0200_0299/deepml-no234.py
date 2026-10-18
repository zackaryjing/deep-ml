# Block-wise FP8 Quantization

import torch


def fp8_block_quantize(
        tensor: torch.Tensor,
        block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to FP8-E4M3 format using block-wise scaling.
    
    Args:
        tensor_copy: Input tensor of shape (N,) where N is divisible by block_size
        block_size: Number of elements per quantization block
        
    Returns:
        quantized: Quantized values of shape (N,), clipped to [-448, 448]
        scales: Per-block scale factors of shape (N // block_size,)
    """
    # Your code here
    N = tensor.shape[0]
    G = N // block_size
    tensor_copy = tensor.clone()
    tensor_copy = tensor_copy.reshape(shape=[G, block_size])
    maximum, _ = tensor_copy.abs().max(dim=1)
    tensor_copy /= maximum.unsqueeze(-1)
    max_val = torch.finfo(torch.float8_e4m3fn).max
    tensor_copy = tensor_copy.clamp(min=-max_val, max=max_val)
    tensor_copy *= max_val
    # tensor_copy = tensor.to(dtype=torch.float8_e4m3fn)
    return tensor_copy.reshape(-1), maximum / max_val


def fp8_block_dequantize(
        quantized: torch.Tensor,
        scales: torch.Tensor,
        block_size: int = 128
) -> torch.Tensor:
    """
    Dequantize FP8-E4M3 values back to full precision.
    
    Args:
        quantized: Quantized values of shape (N,)
        scales: Per-block scale factors of shape (N // block_size,)
        block_size: Number of elements per quantization block
        
    Returns:
        Dequantized tensor of shape (N,)
    """
    # Your code here
    scales = scales.unsqueeze(-1).repeat(1,block_size).reshape(-1)
    quantized = quantized.to(dtype=torch.float32)
    quantized *= scales
    return quantized


def main():
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    quantized, scales = fp8_block_quantize(tensor, block_size=4)
    print(len(scales), round(scales[0].item(), 6), round(quantized[3].item(), 1))
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])

    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    quantized, scales = fp8_block_quantize(tensor, block_size=4)
    print(tensor)
    dequantized = fp8_block_dequantize(quantized, scales, block_size=4)
    print(quantized,scales,dequantized)
    print(tensor,dequantized)
    print(torch.allclose(tensor, dequantized, rtol=0.01))


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-10-14 22:54:06
#
