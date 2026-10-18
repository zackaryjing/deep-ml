# Diffusion Reconstruction Loss

import torch
import torch.nn.functional as F

def diffusion_loss(x_0: torch.Tensor, t: int, beta_start: float, beta_end: float, num_timesteps: int, noise: torch.Tensor, predicted_noise: torch.Tensor) -> float:
    """
    Compute the reconstruction loss for diffusion model training.
    
    Args:
        x_0: Original input data (torch.Tensor)
        t: Timestep (1-indexed, from 1 to num_timesteps)
        beta_start: Starting value of linear beta schedule
        beta_end: Ending value of linear beta schedule
        num_timesteps: Total number of diffusion timesteps
        noise: True noise tensor (same shape as x_0)
        predicted_noise: Model's predicted noise (same shape as x_0)
    
    Returns:
        Mean squared error loss (float)
    """
    betas = torch.linspace(beta_start,beta_end,num_timesteps)
    alphas = 1.0 - betas
    alpha_bar_t = torch.prod(alphas[:t])
    
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
    x_0_hat = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
    loss = F.mse_loss(x_0,x_0_hat).item()
    
    return loss
    
    
    

def main():
    result = diffusion_loss(torch.tensor([1.0]), 1, 0.1, 0.2, 5, torch.tensor([1.0]), torch.tensor([0.9]))
    print(result)
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-17 16:25:03
#
