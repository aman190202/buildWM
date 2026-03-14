import torch
import math
if __name__=='__main__':
    # Hint: use parametric equations
    theta = torch.linspace(0, 4 * math.pi, 1000)
    x = theta * torch.cos(theta)
    y = theta * torch.sin(theta)
    data = torch.stack([x, y], dim=1)  # shape [1000, 2]