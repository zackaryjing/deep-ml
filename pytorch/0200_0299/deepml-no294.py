# Implement INT8 Quantization

import torch
import math

def int8_quantize(x: list[float]) -> dict:
    """
    Perform symmetric INT8 quantization on a floating-point array using PyTorch.
    
    Args:
        x: Input list of floating-point values
        
    Returns:
        Dictionary with 'quantized', 'scale', and 'dequantized' keys
    """
    # Your code here
    x_t = torch.tensor(x,dtype=torch.float32)
    mx = x_t.abs().max().item()
    scale = mx / 127
    epsilion = 7e-15
    if scale < epsilion:
        q = x_t.to(dtype=torch.int8)
        scale = 1.0
    else:
        q = torch.round(x_t / scale).to(dtype=torch.int8)
    dq = q * scale
    return {"quantized": q.tolist(),"scale":round(scale,6),"dequantized":[round(i,4) for i in dq.tolist()]}
    

def main():
    result = int8_quantize([1.0, -1.0, 0.5, -0.5])
    print(result)
    result = int8_quantize([0.0, 0.0, 0.0])
    print(result)
    result = int8_quantize([0.01, -0.02, 0.015])
    print(result)

if __name__ == "__main__":
    main()
    


#
# Created By jing At 2026-10-17 00:37:04
#
