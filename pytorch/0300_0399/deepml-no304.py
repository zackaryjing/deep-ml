# Forward & Backward Diffusion Process

import torch

def diffusion_process(x_0: torch.Tensor,
                      betas: torch.Tensor,
                      timestep: int,
                      forward_noise: torch.Tensor,
                      predicted_noise: torch.Tensor,
                      backward_noise: torch.Tensor = None) -> tuple:
    """
    Implement forward and backward diffusion processes.

    Args:
        x_0: Original clean data (any shape), torch.Tensor
        betas: Noise schedule tensor of shape (T,)
        timestep: Current timestep t (1-indexed)
        forward_noise: Noise epsilon for forward diffusion
        predicted_noise: Model's predicted noise for backward diffusion
        backward_noise: Random noise z for stochastic backward step

    Returns:
        tuple: (x_t, x_t_minus_1) - noisy and denoised samples as torch.Tensors
    """
    t = timestep - 1
    alphas = 1 - betas
    alpha_bar_t = torch.prod(alphas[:timestep])
    alpha_bar_prev = torch.prod(alphas[:t])
    alpha_t = alphas[t]
    beta_t = betas[t]
    
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1.0 - alpha_bar_t) * forward_noise
    mean = (x_t - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) * 1 / torch.sqrt(alpha_t) 
    
    beta_tilde = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t
    
    if timestep > 1 and backward_noise is not None:
        x_t_minus_1 = mean + torch.sqrt(beta_tilde) * backward_noise
    else:
        x_t_minus_1 = mean
    return x_t ,x_t_minus_1
    
    

def main():
    x_0 = torch.tensor([1.0])
    betas = torch.tensor([0.1])
    forward_noise = torch.tensor([0.5])
    predicted_noise = torch.tensor([0.5])
    x_t, x_t_minus_1 = diffusion_process(x_0, betas, 1, forward_noise, predicted_noise)
    print(f'{round(x_t[0].item(), 4)}, {round(x_t_minus_1[0].item(), 4)}')
    
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-17 21:04:11
#
