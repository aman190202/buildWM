import torch 
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plot

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

@torch.no_grad()
def sample(model, alpha_bars, T = 1000,num_samples = 10):
    betas = torch.linspace(0.0001, 0.02, T).to(device)
    alphas = 1.0 - betas
    x_t = torch.randn(size=(num_samples,1,28,28)).to(device)

    for t in reversed(range(T)):
        t_batch = torch.full((num_samples,), t, dtype=torch.long).to(device)
        eps_pred = model(x_t, t_batch)
        z = torch.randn_like(x_t) if t > 0 else 0
        x_t = (1/torch.sqrt(alphas[t])) * (x_t - (betas[t]/torch.sqrt(1 - alpha_bars[t])) * eps_pred) + torch.sqrt(betas[t]) * z

    return x_t


def sinusoidal_embedding(t, dim=32):
    half_dim = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half_dim) / half_dim).to(device)
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

def forward_diffusion(x_0, t, alpha_bars): # for each batch
    """
    Add noise to x_0 at timestep t.
    Returns: x_t (noised data), epsilon (the noise that was added)
    alpha_bars -> [B]
    """
    # Use the formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
    epsilon = torch.randn_like(x_0) # B, 1, 28, 28
    x_t = torch.sqrt(alpha_bars[t][:, None, None, None]) * x_0 + torch.sqrt(1 - alpha_bars[t][:, None,None,None]) * epsilon
    return x_t, epsilon

def get_noise_schedule(T=1000):
    """Return alpha_bar for each timestep 0..T-1"""
    betas = torch.linspace(0.0001, 0.02, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0).to(device)
    return alpha_bars

class SimpleUNet(nn.Module):
    def __init__(self, time_emb_dim=32):
        super().__init__()
        self.ec1 = nn.Conv2d(1 , 32, 3, stride=2, padding=1)
        self.ec2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bottleneck = nn.Conv2d(64,64,3, padding=1)
        self.dec1 = nn.ConvTranspose2d(128,32, 4, stride = 2, padding = 1) # because of skip connections
        self.dec2 = nn.ConvTranspose2d(64 ,1 , 4, stride = 2, padding = 1) # because of skip connections
        self.time_mlp_64 = nn.Sequential(
            nn.Linear(time_emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.time_mlp_32 = nn.Sequential(
            nn.Linear(time_emb_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

    def forward(self,x,t):
        emb = sinusoidal_embedding(t)
        t_emb_32 = self.time_mlp_32(emb)
        t_emb_64 = self.time_mlp_64(emb)
        x1 = F.relu(self.ec1(x) + t_emb_32[:,:,None,None])
        x2 = F.relu(self.ec2(x1) + t_emb_64[:,:,None,None])
        x3 = F.relu(self.bottleneck(x2) + t_emb_64[:,:,None,None])
        x4 = F.relu(self.dec1(torch.cat([x3,x2],dim=1)) + t_emb_32[:,:,None,None])
        x5 = self.dec2(torch.cat([x4,x1],dim=1))
        return x5

if __name__ == '__main__':
    dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
  ]))

    
    # Load entire dataset into GPU once — MNIST is ~190MB, no reason to keep hitting CPU
    loader = DataLoader(dataset, batch_size=4096, num_workers=4, pin_memory=True)
    all_images = torch.cat([imgs for imgs, _ in loader]).to(device)  # [60000, 1, 28, 28]
    N = len(all_images)
    batch_size = 1024

    alpha_bars = get_noise_schedule()

    model = SimpleUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 2e-4)
    num_epochs = 1000

    for epoch in range(num_epochs):
        total_loss = 0
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch_size):
            batch = all_images[perm[i:i+batch_size]]
            t = torch.randint(0, 999, size=(len(batch),), device=device)
            x_t, epsilon = forward_diffusion(batch, t, alpha_bars)
            epsilon_pred = model(x_t, t)
            loss = F.mse_loss(epsilon_pred, epsilon)
            total_loss += loss.item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        num_steps = math.ceil(N / batch_size)
        if epoch % 1 == 0:
            print(f"Epoch : {epoch} | Loss: {total_loss / num_steps}")

        # generate and plot
    generated = sample(model, alpha_bars).cpu()  # [10, 1, 28, 28]

    fig, axes = plot.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        axes[i].imshow(generated[i, 0].numpy(), cmap='gray')
        axes[i].axis('off')
    plot.suptitle('Generated MNIST digits')
    plot.tight_layout()
    plot.savefig('diffusion_MNIST/generated_digits.png')
    plot.show()








    
    