# problem: KV Cache for Efficient Autoregressive Attention

import torch
import math


def kv_cache_attention_step(x_new: torch.Tensor, W_Q: torch.Tensor, W_K: torch.Tensor, W_V: torch.Tensor,
                            cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.

    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) of tensors or None if first step

    Returns:
        Tuple (output, updated_cache) where output is shape (d_v,)
        and updated_cache is (K_new, V_new)
    """
    d_model, d_k = W_K.shape
    V_new = x_new @ W_V
    Q_new = x_new @ W_Q
    K_new = x_new @ W_K
    if cache == None:
        K = K_new.unsqueeze(0)
        V = V_new.unsqueeze(0)
    else:
        K_cache,V_cache = cache
        K = torch.cat([K_cache,K_new.unsqueeze(0)],dim=0)
        V = torch.cat([V_cache,V_new.unsqueeze(0)],dim=0) # [seq_len,d_v]
    scores = (Q_new @ K.T) / math.sqrt(d_k)
    # [1,d_k] @ [seq_len,d_k].T -> [1,seq_len]

    alpha = torch.softmax(scores,dim=0) # [1,seq_len]
    return alpha @ V,(K,V)# [1,d_v]

def main():
    W_Q = torch.eye(2)
    W_K = torch.eye(2)
    W_V = torch.eye(2)
    x1 = torch.tensor([1.0, 0.0])
    out1, c1 = kv_cache_attention_step(x1, W_Q, W_K, W_V, None)
    print([round(x, 4) for x in out1.numpy().tolist()])

    W_Q = torch.eye(2)
    W_K = torch.eye(2)
    W_V = torch.eye(2)
    x1 = torch.tensor([1.0, 0.0])
    _, c1 = kv_cache_attention_step(x1, W_Q, W_K, W_V, None)
    x2 = torch.tensor([0.0, 1.0])
    _, c2 = kv_cache_attention_step(x2, W_Q, W_K, W_V, c1)
    x3 = torch.tensor([1.0, 1.0])
    out3, c3 = kv_cache_attention_step(x3, W_Q, W_K, W_V, c2)
    print([round(x, 4) for x in out3.numpy().tolist()])
    print(tuple(c3[0].shape))
if __name__ == "__main__":
    main()

#
# Created By jing At 2026-10-19 14:58:39
#
