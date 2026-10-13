# Flash Attention v1 - Forward Pass

import numpy as np
import math
import torch

def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                           block_size: int = 2) -> torch.Tensor:
    """
    Compute attention output using Flash Attention v1 algorithm.
    
    Args:
        Q: Query matrix (seq_len, d_model)
        K: Key matrix (seq_len, d_model)
        V: Value matrix (seq_len, d_model)
        block_size: Size of blocks for tiled computation
    
    Returns:
        Output matrix (seq_len, d_model)
    """
    seq_len = Q.shape[0]
    d_model = Q.shape[-1]
    
    scale = 1.0 / math.sqrt(d_model)
    outputs = []
    for i in range(0,seq_len,block_size):
        Q_block = Q[i:i + block_size] # [block,d_model]
        m = torch.full((block_size,1),-float('inf'),dtype=Q.dtype)
        l = torch.zeros((block_size,1),dtype=Q.dtype)
        a = torch.zeros((block_size,d_model),dtype=Q.dtype)
        for j in range(0,seq_len,block_size):
            K_block = K[j:j + block_size] # [block,d_model]
            V_block = V[j:j + block_size] # [block,d_model]
            S = Q_block @ K_block.T * scale# [block,block]
            block_max = S.max(dim=-1,keepdim=True).values
            m_new = torch.maximum(m,block_max)
            l = torch.exp(m - m_new) * l + torch.exp(S - m_new).sum(dim=-1,keepdim=True)
            a = torch.exp(m - m_new) * a + torch.exp(S - m_new) @ V_block
            m = m_new
        out = a / l
        outputs.append(out)
    return torch.cat(outputs,dim=0)
            
            

def main():
    np.random.seed(42)
    Q = torch.tensor(np.random.randn(4, 4))
    K = torch.tensor(np.random.randn(4, 4))
    V = torch.tensor(np.random.randn(4, 4))
    out = flash_attention_forward(Q, K, V, block_size=2)
    print([round(x, 4) for x in out[0].tolist()])
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-10 20:55:07
#
