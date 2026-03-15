# libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

@torch.no_grad()
def sample(model, alpha_bars, T=1000, num_samples=5000):
    betas = torch.linspace(0.0001, 0.02, T)
    alphas = 1.0 - betas
    x_t = torch.randn(num_samples, 2)
    
    for t in reversed(range(T)):
        t_batch = torch.full((num_samples,), t, dtype=torch.long)
        eps_pred = model(x_t, t_batch)
        z = torch.randn_like(x_t) if t > 0 else 0
        x_t = (1/torch.sqrt(alphas[t])) * (x_t - (betas[t]/torch.sqrt(1 - alpha_bars[t])) * eps_pred) + torch.sqrt(betas[t]) * z

    return x_t

    

def sinusoidal_embedding(t , dim = 32):
    half_dim = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half_dim) / half_dim)
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)],dim = -1)


class NoisePredictor(nn.Module):
    def __init__(self,hidden_dim = 128):
        super().__init__()
        # 34 because 32 of positonal encoded t and 2 of the point location
        self.layer_1 = nn.Linear(34,hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim,hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 2)
    
    def forward(self, x_t, t):
        embedding = sinusoidal_embedding(t)
        x = torch.cat([x_t,embedding],dim=-1)
        x1 = F.relu(self.layer_1(x))
        x2 = F.relu(self.layer_2(x1))
        x3 = self.layer_3(x2)
        return x3


def forward_diffusion(x_0, t, alpha_bars):
    """
    Add noise to x_0 at timestep t.
    Returns: x_t (noised data), epsilon (the noise that was added)
    """

    # Use the formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
    epsilon = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bars[t][:,None]) * x_0 + torch.sqrt(1 - alpha_bars[t][:,None]) * epsilon
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

    # timesteps = [0, 250, 500, 750, 999]
    # fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # for i, t in enumerate(timesteps):
    #     x_t, _ = forward_diffusion(data, t, alpha_bars)
    #     axes[i].scatter(x_t[:, 0], x_t[:, 1], s=2, alpha=0.5)
    #     axes[i].set_title(f't = {t}')
    #     axes[i].set_xlim(-15, 15)
    #     axes[i].set_ylim(-15, 15)
    #     axes[i].set_aspect('equal')

    # plt.tight_layout()
    # plt.savefig('exercise_1/forward_diffusion.png')
    # plt.show()

    model = NoisePredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    num_epochs = 10000

    for epoch in range(num_epochs):
        batch = data[torch.randint(0,999,size=(64,))]
        t_batch = torch.randint(0,999,size=(64,))
        x_t, epsilon = forward_diffusion(batch, t_batch, alpha_bars)
        epsilon_pred = model(x_t, t_batch)
        loss = F.mse_loss(epsilon_pred, epsilon)

        if(epoch % 100 == 0) :
            print(f"Epoch : {epoch} Loss : {loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



    # generate and plot
    generated = sample(model, alpha_bars)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(data[:, 0], data[:, 1], s=2, alpha=0.5)
    axes[0].set_title('Original spiral')
    axes[0].set_aspect('equal')

    axes[1].scatter(generated[:, 0], generated[:, 1], s=2, alpha=0.5)
    axes[1].set_title('Generated (sampled)')
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('exercise_1/generated_spiral.png')
    plt.show()





    



