# Forward Diffusion Process

import torch

def forward_diffusion(x_0: torch.Tensor, t: int, beta_start: float, beta_end: float, num_timesteps: int, noise: torch.Tensor) -> torch.Tensor:
    """
    Apply forward diffusion process to add noise to input data.
    
    Args:
        x_0: Original input data (torch.Tensor)
        t: Timestep (1-indexed, from 1 to num_timesteps)
        beta_start: Starting value of linear beta schedule
        beta_end: Ending value of linear beta schedule
        num_timesteps: Total number of diffusion timesteps
        noise: Noise tensor (same shape as x_0)
    
    Returns:
        Noisy sample x_t as torch.Tensor
    """
    betas = torch.linspace(beta_start,beta_end,num_timesteps)
    alphas = 1.0 - betas
    alpha_bar_t = torch.prod(alphas[:t])
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
    return x_t

def main():
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-17 20:27:36
#
