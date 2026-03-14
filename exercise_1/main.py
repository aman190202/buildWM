# libraries
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

def sinusoidal_embedding(t , dim = 32):
    half_dim = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half_dim) / half_dim)
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)],dim = -1)


class NoisePredictor(nn.module):
    def __init__(self,hidden_dim = 128):
        super().__init__()

    
    def forward():
        pass


def forward_diffusion(x_0, t, alpha_bars):
    """
    Add noise to x_0 at timestep t.
    Returns: x_t (noised data), epsilon (the noise that was added)
    """

    # Use the formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
    epsilon = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bars[t]) * x_0 + torch.sqrt(1 - alpha_bars[t]) * epsilon
    return x_t, epsilon

def get_noise_schedule(T=1000):
    """Return alpha_bar for each timestep 0..T-1"""
    betas = torch.linspace(0.0001, 0.02, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alpha_bars

# main file
if __name__=='__main__':

    theta = torch.linspace(0, 4 * math.pi, 1000)
    x = theta * torch.cos(theta)
    y = theta * torch.sin(theta)
    data = torch.stack([x, y], dim=1)  # shape [1000, 2]

    alpha_bars = get_noise_schedule()

    timesteps = [0, 250, 500, 750, 999]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for i, t in enumerate(timesteps):
        x_t, _ = forward_diffusion(data, t, alpha_bars)
        axes[i].scatter(x_t[:, 0], x_t[:, 1], s=2, alpha=0.5)
        axes[i].set_title(f't = {t}')
        axes[i].set_xlim(-15, 15)
        axes[i].set_ylim(-15, 15)
        axes[i].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('exercise_1/forward_diffusion.png')
    plt.show()
    