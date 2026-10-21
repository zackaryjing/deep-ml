# problem: Implement Variational Autoencoder (VAE) Loss (ELBO)

import torch
from torch.distributions import Normal, kl_divergence as kl_div_fn
import torch.nn.functional as F


def vae_loss(x: torch.Tensor, x_reconstructed: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> tuple:
    """
    Compute the VAE loss (negative ELBO).

    Args:
        x: torch.Tensor of shape (batch_size, features), original input
        x_reconstructed: torch.Tensor of shape (batch_size, features), reconstructed input
        mu: torch.Tensor of shape (batch_size, latent_dim), latent mean
        log_var: torch.Tensor of shape (batch_size, latent_dim), latent log-variance

    Returns:
        tuple: (total_loss, reconstruction_loss, kl_divergence) as floats
    """
    # Your code here
    batch_size = x.size(0)
    std = torch.exp(0.5 * log_var)

    q = Normal(mu, std)
    p = Normal(torch.zeros_like(mu), torch.ones_like(std))

    recon_loss = F.mse_loss(x_reconstructed, x, reduction='sum') / batch_size
    kl_div = kl_div_fn(q, p).sum() / batch_size

    return (recon_loss + kl_div).item(), recon_loss.item(), kl_div.item()


def main():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    x_hat = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    mu = torch.tensor([[0.0], [0.0]])
    log_var = torch.tensor([[0.0], [0.0]])
    result = vae_loss(x, x_hat, mu, log_var)
    print((round(result[0], 4), round(result[1], 4), round(result[2], 4)))

if __name__ == "__main__":
    main()

#
# Created By jing At 2026-10-21 13:15:28
#
